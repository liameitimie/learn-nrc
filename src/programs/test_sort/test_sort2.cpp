#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

namespace pcg32 {
    uint64 state = 0x853c49e6748fea9bull;
    uint64 inc = 0xda3e39cb94b95bdbull;
    uint64 mul = 0x5851f42d4c957f2dull;

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

    bool enter = n > 32;
    print("{}: [", name);
    for (int i = 0; i < n; i++) {
        if (enter && (i % 32 == 0)) print("\n");
        print("{}, ", v[i]);
    }
    if (enter) print("\n");
    print("]\n");
}

template<typename T1, typename T2>
void compare_vec(vector<T1> &v1, vector<T2> &v2) {
    if (v1.size() != v2.size()) {
        print("compare different size vec\n");
        exit(0);
    }
    int n = v1.size();
    int f_err = 0;
    int err_c = 0;
    for (int i = 0; i < n; i++) {
        int t1 = v1[i];
        int t2 = v2[i];
        int err = abs(t1 - t2);

        if (err > f_err) {
            print("!inc error {}: {}, {}; f_err: {}\n", i, t1, t2, err);
        }

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

void block_exclusive_scan($uint input, $uint &output, $uint &block_sum, $uint tid, $shared<uint> &smem) {
    // $shared<uint> smem{7};
    $uint t = warp_prefix_sum(input);
    $if (tid % 32 == 31) {
        smem[tid / 32] = t + input;
    };
    sync_block();
    block_sum = smem[0];
    $uint t1 = 0;
    for (int i = 1; i < 7; i++) {
        $if (tid / 32 == i) { t1 = block_sum; };
        block_sum += smem[i];
    }
    output = t + t1;
}

int main(int argc, char** argv) {

    const int block_size = 128;

    const int radix_bits = 4;
    const int radix_digits = 1 << radix_bits;
    const int begin_bit = 0;
    // const int thread_items = 4;
    // const int tile_size = block_size * thread_items;

    Kernel1D block_sort = [&]($buffer<uint> keys, $buffer<uint> sorted_keys) {
        set_block_size(block_size);

        $uint tid = $thread_x;
        $uint bid = $block_x;

        // $uint key[1];
        // key[0] = keys.read(tid + bid * block_size);

        // BlockRadixSort<1>().sort(key, 0, 4);

        // sorted_keys.write(tid + bid * block_size, key[0]);

        $uint key = keys.read(tid + bid * block_size);

        $shared<uint> key_scratch{block_size};
        $shared<uint> scan_smem{block_size}; /// ??? smem size 设为7会出错

        // uint p = 0;
        for (uint p = 0; p < radix_bits; p += 2) {
            $uint digit = (key >> begin_bit) & (radix_digits - 1);
            $uint sub_d = (digit >> p) & 3;

            // block内计数一定小于256，用一个uint表示4个counter
            $uint pack_counter = 1 << (sub_d * 8);
            $uint block_sum = 0;
            block_exclusive_scan(pack_counter, pack_counter, block_sum, tid, scan_smem);

            // sorted_keys.write(tid + bid * block_size, block_sum);
            // sorted_keys.write(tid + bid * block_size, pack_counter);

            // 将低位计数加到高位
            pack_counter += (block_sum << 8) + (block_sum << 16) + (block_sum << 24);

            // sorted_keys.write(tid + bid * block_size, pack_counter);

            $uint key_offset = (pack_counter >> (sub_d * 8)) & 0xff;

            // sorted_keys.write(tid + bid * block_size, key_offset);

            // ???
            key_scratch[key_offset] = key;
            sync_block();

            // sorted_keys.write(tid + bid * block_size, key_scratch[tid]);

            key = key_scratch[tid];
            sync_block();
        }
        sorted_keys.write(tid + bid * block_size, key);
    };

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    Clock timer;

    const int n = 1 << 19;

    vector<uint> keys_h(n);
    vector<uint> sorted_keys_h(n, ~0u);
    vector<uint> sorted_keys_d(n, ~0u);

    for (uint &x: keys_h) x = pcg32::next_uint() % 16;

    auto keys = device.create_buffer<uint>(n);
    auto sorted_keys = device.create_buffer<uint>(n);

    stream
        << keys.copy_from(keys_h.data())
        << sorted_keys.copy_from(sorted_keys_h.data())
        << synchronize();

    timer.tic();
    {
        sorted_keys_h = keys_h;
        for (int t = 0; t < n / block_size; t++) {
            // std::sort(sorted_keys_h.data() + i * block_size, sorted_keys_h.data() + (i + 1) * block_size);
            // const int bit = 2;
            const int bit = radix_bits;
            const int bins = 1 << bit;
            const int mask = bins - 1;
            vector<int> counter(bins);

            int l = t * block_size;
            int r = (t + 1) * block_size;

            for (int i = l; i < r; i++) {
                // uint pack_counter = counter[0] + (counter[1] << 8) + (counter[2] << 16) + (counter[3] << 24);
                // sorted_keys_h[i] = pack_counter;

                counter[keys_h[i] & mask]++;
            }

            // uint pack_counter = counter[0] + (counter[1] << 8) + (counter[2] << 16) + (counter[3] << 24);
            // for (int i = l; i < r; i++) {
            //     sorted_keys_h[i] = pack_counter;
            // }

            int sum = 0;
            for (int i = 0; i < bins; i++) {
                int tsum = sum;
                sum += counter[i];
                counter[i] = tsum;
            }
            for (int i = l; i < r; i++) {
                // uint pack_counter = counter[0] + (counter[1] << 8) + (counter[2] << 16) + (counter[3] << 24);
                // sorted_keys_h[i] = pack_counter;

                int &cnt = counter[keys_h[i] & mask];

                // sorted_keys_h[i] = cnt;
                sorted_keys_h[l + cnt] = keys_h[i];
                cnt++;
            }
        }
    }
    print("calc ref res: {}\n", timer.toc());

    timer.tic();
    auto shader = device.compile(block_sort);
    print("compiled shader: {}\n", timer.toc());

    for (int t = 0; t < 1; t++) {
        CommandList cmd_list;
        for (int i = 0; i < 1; i++) {
            cmd_list << shader(keys, sorted_keys).dispatch(n);
        }
        timer.tic();
        stream << cmd_list.commit();
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << sorted_keys.copy_to(sorted_keys_d.data()) << synchronize();

    print_vec(keys_h, "keys_h", block_size);
    print_vec(sorted_keys_h, "out_h", block_size);
    print_vec(sorted_keys_d, "out_d", block_size);

    compare_vec(sorted_keys_h, sorted_keys_d);

    return 0;
}

/*

*/