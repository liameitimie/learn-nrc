#include <luisa/luisa-compute.h>
#include <global.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

namespace pcg32 {
    ulong state = 0x853c49e6748fea9bull;
    ulong inc = 0xda3e39cb94b95bdbull;
    ulong mul = 0x5851f42d4c957f2dull;

    uint next_uint() {
        uint t1 = ((state >> 18u) ^ state) >> 27u;
        uint t2 = state >> 59u;
        state = state * mul + inc;
        return (t1 >> t2) | (t1 << ((~t2 + 1u) & 31));
    }
    float next_float() {
        union {
            uint u;
            float f;
        } x;
        x.u = (next_uint() >> 9) | 0x3f800000u;
        return x.f - 1;
    }
}


$int thread_reduce($int *input, int len) {
// $int thread_reduce($int4 *input, int len) {
    $int res = 0;
    for (int i = 0; i < len; i++) {
        res += input[i];
    }
    return res;
}
void thread_scan_in($int *input, $int *output, int len, $int prefix = 0) {
// void thread_scan_in($int4 *input, $int4 *output, int len, $int prefix = 0) {
    $int s = prefix;
    for (int i = 0; i < len; i++) {
        s += input[i];
        output[i] = s;
    }
}
void block_scan_ex($int input, $int &output, $int &block_sum, $shared<int> &smem, $uint wid, $uint lid, int block_size) {
    $int t = warp_prefix_sum(input);
    $if (lid == 31) {
        smem[wid] = t + input;
    };
    sync_block();
    block_sum = smem[0];
    $int t1 = 0;
    for (int i = 1; i < block_size/32; i++) {
        $if (wid == i) { t1 = block_sum; };
        block_sum += smem[i];
    }
    output = t + t1;
}

// void block_scan_ex1($int input, $int &output, $shared<int> &smem, $uint tid, int block_size) {
//     smem[tid] = input;
//     sync_block();
//     for (int i = 1; i < block_size; i <<= 1) {
//         $if ((tid & i) != 0) {
//             smem[tid] += smem[(tid | (i - 1)) ^ i];
//         };
//         sync_block();
//     }
//     output = smem[tid] - input;
// }

const int n = 1 << 22;
Buffer<int> input;
Buffer<int> output;

namespace decoupled_look_back {
    const int block_size = 256;
    const int thread_items = 4;

    // tile state
    const uint invalid = 0;
    const uint partial = 0x40000000;
    const uint inclusive = 0x80000000;
    const uint content_mask = 0x3fffffff;

    Buffer<uint> counter;
    Buffer<uint> tile_state;

    Kernel1D init_tile_state = []($uint num_tile, $buffer<uint> tile_state, $buffer<uint> counter) {
        $uint tid = $dispatch_x;
        $if (tid < num_tile) {
            tile_state.write(32 + tid, invalid);
        };
        // padding
        $if (tid < 32) {
            tile_state.write(tid, partial);
        };
        $if (tid == 0) {
            counter.write(0, 0u);
        };
    };
    void tile_state_read($int tile_id, $uint &val, $buffer<uint> &tile_state) {
        val = tile_state.read(32 + tile_id);
    }
    void tile_state_write($int tile_id, $uint val, $buffer<uint> &tile_state) {
        // tile_state.write(32 + tile_id, val);
        tile_state.atomic(32 + tile_id).exchange(val);
    }

    Kernel1D scan_kernel = []($buffer<int4> input, $buffer<int4> output, $buffer<uint> tile_state, $buffer<uint> global_counter) {
        set_block_size(block_size);

        $int4 tmp;
        // $int4 items[thread_items/4];
        $int items[thread_items];
        $uint tid = $thread_x;
        $uint lid = tid % 32;
        $uint wid = tid / 32;

        // get tile id by atomic
        // $shared<uint> share_counter{1};
        // $if (tid == 0) {
        //     share_counter[0] = global_counter.atomic(0).fetch_add(1u);
        // };
        // sync_block();
        // $uint tile_id = share_counter[0];

        // get tile id by block_id
        $uint tile_id = $block_x;

        // load
        for (int i = 0; i < thread_items/4; i++) {
            // items[i] = input.read(i + tid*(thread_items/4) + tile_id*(block_size*thread_items/4));
            tmp = input.read(i + tid*(thread_items/4) + tile_id*(block_size*thread_items/4));
            for (int j = 0; j < 4; j++) items[j+i*4] = tmp[j];
        }

        // block reduce-then-scan
        $shared<int> smem{block_size/32};
        $int block_sum;
        // $shared<int> smem{block_size};
        // $int t = thread_reduce(items, thread_items/4);
        $int t = thread_reduce(items, thread_items);
        block_scan_ex(t, t, block_sum, smem, wid, lid, block_size);
        // block_scan_ex1(t, t, smem, tid, block_size);

        $uint state;
        $int tile_prefix = 0;

        $if (tile_id == 0) {
            $if (tid == 0) {
                state = inclusive | (block_sum & content_mask);
                tile_state_write(0, state, tile_state);
                all_memory_barrier();
            };
        }
        $else {
            $if (wid == 0) {
                $if (tid == 0) {
                    state = partial | (block_sum & content_mask);
                    tile_state_write(tile_id, state, tile_state);
                    all_memory_barrier();
                };

                auto proc_window = [&]($int idx, $uint &state, $int &window_sum) {
                    // wait for all state in window is no invalid
                    tile_state_read(idx, state, tile_state);
                    $while (warp_active_any((state & ~content_mask) == invalid)) {
                        // sync_block();
                        all_memory_barrier();
                        tile_state_read(idx, state, tile_state);
                    };
                    $uint f = warp_active_bit_mask((state & inclusive) == inclusive).x;
                    $uint mask = (f & -f) | ((f & -f) - 1);
                    window_sum = warp_active_sum((mask >> tid) * (state & content_mask));
                };

                $int idx = tile_id - tid - 1;
                $int window_sum;

                proc_window(idx, state, window_sum);
                tile_prefix = window_sum;

                // $while (warp_active_any((state & inclusive) != inclusive)) {
                //     idx -= 32;
                //     proc_window(idx, state, window_sum);
                //     tile_prefix += window_sum;
                // };
                $if (tid == 0) {
                    state = inclusive | ((tile_prefix + block_sum) & content_mask);
                    tile_state_write(tile_id, state, tile_state);
                    all_memory_barrier();
                    smem[0] = tile_prefix;
                };
            };
            sync_block();
            tile_prefix = smem[0];
        };

        thread_scan_in(items, items, thread_items, t + tile_prefix);

        // store
        for (int i = 0; i < thread_items/4; i++) {
            for (int j = 0; j < 4; j++) tmp[j] = items[j+i*4];
            output.write(i + tid*(thread_items/4) + tile_id*(block_size*thread_items/4), tmp);
        }
    };
    Shader1D<uint, Buffer<uint>, Buffer<uint>> init_tile_state_shader;
    Shader1D<Buffer<int4>, Buffer<int4>, Buffer<uint>, Buffer<uint>> scan_shader;

    void init() {
        counter = global::device().create_buffer<uint>(1);
        tile_state = global::device().create_buffer<uint>(n/(block_size*thread_items));

        init_tile_state_shader = global::device().compile(init_tile_state);
        scan_shader = global::device().compile(scan_kernel);
    }
    void scan(BufferView<int> input, BufferView<int> output) {
        global::cmd_list()
            << init_tile_state_shader(n/(block_size*thread_items), tile_state, counter).dispatch(max(n/(block_size*thread_items), 32))
            << scan_shader(input.as<int4>(), output.as<int4>(), tile_state, counter).dispatch(n/thread_items);
    }
}

template<typename T>
void print_vec(vector<T> &v, string name, int n = -1) {
    if (n == -1) n = v.size();
    n = min(n, (int)v.size());
    print("{}: [", name);
    for (int i = 0; i < n; i++) {
        print("{}, ", v[i]);
    }
    print("]\n");
}

template<typename T1, typename T2>
void compare_vec(vector<T1> &v1, vector<T2> &v2) {
    if (v1.size() != v2.size()) {
        print("compare different size vec\n");
        exit(0);
    }
    int n = v1.size();
    float f_err = 0;
    int err_c = 0;
    for (int i = 0; i < n; i++) {
        float t1 = v1[i];
        float t2 = v2[i];
        float err = abs(t1 - t2);
        f_err = max(f_err, err);
        if (err > 0.01) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, t1, t2);
            }
            err_c++;
        }
    }
    print("f_err: {}\n", f_err);
    print("err_c: {}\n", err_c);
    print("ok\n");
}


int main(int argc, char** argv) {
    global::init(argv[0]);
    decoupled_look_back::init();

    auto in = global::device().create_buffer<int>(n);
    auto out = global::device().create_buffer<int>(n);

    vector<int> in_h(n);
    vector<int> out_h(n);
    vector<int> out_d(n);

    for (int &x: in_h) x = pcg32::next_uint() % 32;

    // int tile_size = decoupled_look_back::block_size * decoupled_look_back::thread_items;
    // for (int t = 0; t < n / tile_size; t++) {
    //     out_h[t*tile_size] = in_h[t*tile_size];
    //     for (int i = 1; i < tile_size; i++) {
    //         out_h[t*tile_size+i] = out_h[t*tile_size+i-1] + in_h[t*tile_size+i];
    //     }
    // }
    out_h[0] = in_h[0];
    for (int i = 1; i < n; i++) {
        out_h[i] = out_h[i-1] + in_h[i];
    }

    global::stream() << in.copy_from(in_h.data()) << synchronize();

    Clock timer;
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            decoupled_look_back::scan(in, out);
        }
        timer.tic();
        global::stream() << global::cmd_list().commit();
        global::stream().synchronize();
        print("{}\n", timer.toc());
    }

    global::stream() << out.copy_to(out_d.data()) << synchronize();

    print_vec(out_h, "out_h", 32);
    print_vec(out_d, "out_d", 32);

    compare_vec(out_h, out_d);

    return 0;
}