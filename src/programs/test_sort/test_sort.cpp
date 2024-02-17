#include <luisa/luisa-compute.h>
#include <global.h>
#include <algorithm>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

constexpr uint log2i(uint x) {
    uint res = (x & (x - 1)) != 0;
    uint t = 16;
    uint y = 0;

    y = -((x>>t)!=0), res |= y&t, x >>= y&t, t >>= 1;
    y = -((x>>t)!=0), res |= y&t, x >>= y&t, t >>= 1;
    y = -((x>>t)!=0), res |= y&t, x >>= y&t, t >>= 1;
    y = -((x>>t)!=0), res |= y&t, x >>= y&t, t >>= 1;
    y = (x>>t)!=0, res += y;
    return res;
}

template<typename T>
class BlockScan {
    $shared<T> *smem;
    bool own_smem = false;

    $uint warp_id;
    $uint lane_id;
public:
    // create shared memory own for single use, if use multi time, please pass the smem
    BlockScan():
        warp_id($thread_x / 32),
        lane_id($thread_x % 32)
    {
        smem = new $shared<T>(shared_size());
        own_smem = true;
    }
    BlockScan($shared<T> *smem):
        smem(smem),
        warp_id($thread_x / 32),
        lane_id($thread_x % 32)
    {
        if (smem == nullptr) {
            print("invalid shared memory pass to block scan\n");
            exit(1);
        }
        if (smem->size() < shared_size()) {
            print("not enough shared memory for block scan\n");
            exit(1);
        }
    }
    ~BlockScan() { if (own_smem) delete smem; }

    static uint shared_size() {
        return block_size().x / 32;
    }

    void exclusive_scan($<T> input, $<T> &output, $<T> &block_sum) {
        $<T> t = warp_prefix_sum(input);
        $if (lane_id == 31) {
            (*smem)[warp_id] = t + input;
        };
        sync_block();
        block_sum = (*smem)[0];
        $<T> t1 = 0;
        for (int i = 1; i < block_size().x / 32; i++) {
            $if (warp_id == i) { t1 = block_sum; };
            block_sum += (*smem)[i];
        }
        output = t + t1;
    }

    void inclusive_scan($<T> input, $<T> &output, $<T> &block_sum) {
        exclusive_scan(input, output, block_sum);
        output += input;
    }
};

template<typename T>
class BlockRadixRank {
    $shared<T> *counter_smem;
    $shared<T> *scan_smem;
    bool own_smem = false;
public:
    // create shared memory own for single use, if use multi time, please pass the smem
    BlockRadixRank()
    {
        scan_smem = new $shared<T>(BlockScan<T>::shared_size());
        counter_smem = new $shared<T>(counter_lines * block_size().x);
        own_smem = true;
    }
    BlockRadixRank($shared<T> *counter_smem, $shared<T> *scan_smem):
        counter_smem(counter_smem),
        scan_smem(scan_smem)
    {
        if (counter_smem == nullptr || scan_smem == nullptr) {
            print("invalid shared memory pass to block radix rank\n");
            exit(1);
        }
        if (counter_smem->size() < counter_lines * block_size().x || scan_smem->size() < BlockScan<T>::shared_size()) {
            print("not enough shared memory for block radix rank\n");
            exit(1);
        }
    }
    ~BlockRadixRank() {
        if (own_smem) {
            delete counter_smem;
            delete scan_smem;
        }
    }

    static uint counter_shared_size() {
        return pad_counter_lines * block_size().x;
    }

    static const uint radix_bits = 4;
    static const uint radix_digits = 1 << radix_bits;
    // static const uint max_rank = 1024;
    // static const uint log_max_rank = log2i(max_rank);
    static const uint max_rank = 65536;
    static const uint log_max_rank = 16;
    static const uint packing_radio = 32 / log_max_rank;
    // static const uint counter_lines = ((1 << radix_bits) + packing_radio - 1) / packing_radio;
    static const uint counter_lines = 8;
    static const uint log_counter_lines = 3;
    static const uint pad_counter_lines = counter_lines + 1;

    $uint extract_counter($uint line_id, $uint sub_id) {
        return (*counter_smem)[$thread_x + line_id*block_size().x] >> (sub_id * log_max_rank) & (max_rank - 1);
    }
    void counter_add($uint line_id, $uint sub_id) {
        (*counter_smem)[$thread_x + line_id*block_size().x] += 1u << (sub_id * log_max_rank);
    }

    void scan_counters() {
        // simply scan the counter, the lower bit can get it high bit prefix from block_sum
        $uint partial_sum = 0;
        // store in local to avoid some shared access
        $array<uint, pad_counter_lines> seg_cache;

        for (int i = 0; i < pad_counter_lines; i++) {
            seg_cache[i] = (*counter_smem)[i + $thread_x * pad_counter_lines];
        }
        for (int i = 0; i < pad_counter_lines; i++)
            partial_sum += seg_cache[i];

        $uint prefix = 0;
        $uint block_sum = 0;
        BlockScan<uint>(scan_smem).exclusive_scan(partial_sum, prefix, block_sum);

        // adding the lower bit count to higher bit
        for (int i = 1; i < packing_radio; i++) {
            prefix += block_sum << (i * log_max_rank);
        }

        $uint tmp;
        for (int i = 0; i < pad_counter_lines; i++) {
            tmp = seg_cache[i];
            seg_cache[i] = prefix;
            prefix += tmp;
        }
        for (int i = 0; i < pad_counter_lines; i++) {
            (*counter_smem)[i + $thread_x * pad_counter_lines] = seg_cache[i];
        }
    }

    template<int thread_items>
    void rank($uint (&keys)[thread_items], $uint (&ranks)[thread_items], $uint begin_bit) {
        if (thread_items * block_size().x > max_rank) {
            print("ranking keys is more than {}\n", max_rank);
            exit(0);
        }
        
        // smem layout: (counter_lines * block_size * packing_radio)
        // last dim of smem "packing_radio" is packed in one uint
        $ushort thread_prefix[thread_items];
        // $uint *counter_ptr[thread_items];

        for (int i = 0; i < pad_counter_lines; i++)
            (*counter_smem)[$thread_x + i*block_size().x] = 0;

        // counting digits
        for (int i = 0; i < thread_items; i++) {
            $uint digit = (keys[i] >> begin_bit) & (radix_digits - 1);

            // high bit as sub_id, can easy to get high bit prefix after block scan
            $uint sub_id = digit >> log_counter_lines;
            $uint line_id = digit & (counter_lines - 1);

            // counter_ptr[i] = &(*counter_smem)[$thread_x + line_id*block_size().x];

            // thread_prefix[i] = *counter_ptr[i] >> (sub_id * log_max_rank) & (max_rank - 1);
            // *counter_ptr[i] += 1u << (sub_id * log_max_rank);

            thread_prefix[i] = extract_counter(line_id, sub_id);
            counter_add(line_id, sub_id);
        }
        sync_block();

        scan_counters();

        sync_block();
        for (int i = 0; i < thread_items; i++) {
            $uint digit = (keys[i] >> begin_bit) & (radix_digits - 1);
            $uint sub_id = digit >> log_counter_lines;
            $uint line_id = digit & (counter_lines - 1);

            ranks[i] = thread_prefix[i].cast<uint>() + extract_counter(line_id, sub_id);
            // ranks[i] = thread_prefix[i].cast<uint>() + (*counter_ptr[i] >> (sub_id * log_max_rank) & (max_rank - 1));
        }
    }

};

template<uint thread_items>
class BlockRadixSort {
    $shared<uint> *smem; // use for rank counter and exchange keys
    $shared<uint> *scan_smem;
    bool own_smem = false;
public:
    BlockRadixSort() {
        scan_smem = new $shared<uint>(BlockScan<uint>::shared_size());
        smem = new $shared<uint>(max(thread_items * block_size().x, BlockRadixRank<uint>::counter_shared_size()));
        own_smem = true;
    }
    ~BlockRadixSort() {
        if (own_smem) {
            delete smem;
            delete scan_smem;
        }
    }
    static const uint radix_bits = BlockRadixRank<uint>::radix_bits;

    void sort($uint (&keys)[thread_items], uint begin_bit, uint end_bit) {
        if ((end_bit - begin_bit) % radix_bits != 0) {
            print("range of sorting bit must multiple of radix_bits:{}\n", radix_bits);
            exit(0);
        }
        $uint ranks[thread_items];
        $for (t, begin_bit, end_bit, radix_bits) {
            BlockRadixRank<uint>(smem, scan_smem).rank(keys, ranks, t);
            sync_block();
            for (int i = 0; i < thread_items; i++) {
                (*smem)[ranks[i]] = keys[i];
            }
            sync_block();
            for (int i = 0; i < thread_items; i++) {
                keys[i] = (*smem)[i + $thread_x * thread_items];
            }
            sync_block();
        };
    }
};

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

    Kernel1D test_scan = []($buffer<int> input, $buffer<int> output) {
        set_block_size(256);
        $int val = input.read($dispatch_x);
        $int block_sum;
        BlockScan<int>().inclusive_scan(val, val, block_sum);
        output.write($dispatch_x, val);
    };
    Kernel1D test_rank = []($buffer<uint4> input, $buffer<uint4> output) {
        set_block_size(128);
        $uint4 tmp = input.read($dispatch_x);
        $uint keys[4] = {tmp[0], tmp[1], tmp[2], tmp[3]};
        $uint ranks[4];

        BlockRadixRank<uint>().rank(keys, ranks, 0);

        tmp = {ranks[0], ranks[1], ranks[2], ranks[3]};
        output.write($dispatch_x, tmp);
    };
    Kernel1D test_sort = []($buffer<uint4> input, $buffer<uint4> output) {
        set_block_size(256);
        $uint4 tmp = input.read($dispatch_x);
        $uint keys[4] = {tmp[0], tmp[1], tmp[2], tmp[3]};

        BlockRadixSort<4>().sort(keys, 0, 32);

        tmp = {keys[0], keys[1], keys[2], keys[3]};
        output.write($dispatch_x, tmp);
    };

    // auto shader = global::device().compile(test_scan);
    auto shader = global::device().compile(test_rank);
    // auto shader = global::device().compile(test_sort);

    const uint n = 1 << 22;

    auto in = global::device().create_buffer<int>(n);
    auto out = global::device().create_buffer<int>(n);

    vector<int> in_h(n);
    vector<int> out_h(n);
    vector<int> out_d(n);

    for (int &x: in_h) x = pcg32::next_uint() % 16;
    global::stream() << in.copy_from(in_h.data()) << synchronize();

    // test scan
    // int tile_size = 256;
    // for (int t = 0; t < n / tile_size; t++) {
    //     out_h[t*tile_size] = in_h[t*tile_size];
    //     // out_h[t*tile_size] = 0;
    //     for (int i = 1; i < tile_size; i++) {
    //         out_h[t*tile_size+i] = out_h[t*tile_size+i-1] + in_h[t*tile_size+i];
    //     }
    // }

    // test rank
    int tile_size = 128*4;
    for (int t = 0; t < n / tile_size; t++) {
        vector<uint> counter(16);
        for (int i = 0; i < tile_size; i++) {
            counter[in_h[t*tile_size+i]]++;
        }
        int sum = 0, tmp;
        for (int i = 0; i < 16; i++) {
            tmp = counter[i];
            counter[i] = sum;
            sum += tmp;
        }
        for (int i = 0; i < tile_size; i++) {
            tmp = in_h[t*tile_size+i];
            out_h[t*tile_size+i] = counter[tmp];
            counter[tmp]++;
        }
    }

    // test sort
    // int tile_size = 256*4;
    // for (int t = 0; t < n / tile_size; t++) {
    //     std::sort(in_h.data()+t*tile_size, in_h.data()+(t+1)*tile_size);
    //     for (int i = 0; i < tile_size; i++) {
    //         out_h[t*tile_size+i] = in_h[t*tile_size+i];
    //     }
    // }

    Clock timer;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 100; j++) {
            // global::cmd_list() << shader(in, out).dispatch(n);
            global::cmd_list() << shader(in.view().as<uint4>(), out.view().as<uint4>()).dispatch(n/4);
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