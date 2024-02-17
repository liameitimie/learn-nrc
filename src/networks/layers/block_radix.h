#pragma once
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

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

class BlockRadixRank {
    $shared<uint> *counter_smem;
    $shared<uint> *scan_smem;
    bool own_smem = false;
public:
    // create shared memory own for single use, if use multi time, please pass the smem
    BlockRadixRank()
    {
        scan_smem = new $shared<uint>(BlockScan<uint>::shared_size());
        counter_smem = new $shared<uint>(counter_lines * block_size().x);
        own_smem = true;
    }
    BlockRadixRank($shared<uint> *counter_smem, $shared<uint> *scan_smem):
        counter_smem(counter_smem),
        scan_smem(scan_smem)
    {
        if (counter_smem == nullptr || scan_smem == nullptr) {
            print("invalid shared memory pass to block radix rank\n");
            exit(1);
        }
        if (counter_smem->size() < counter_lines * block_size().x || scan_smem->size() < BlockScan<uint>::shared_size()) {
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
        // store in local to save shared
        $array<uint, pad_counter_lines> seg_cache;

        $for (i, pad_counter_lines) {
            seg_cache[i] = (*counter_smem)[i + $thread_x * pad_counter_lines];
        };
        $for (i, pad_counter_lines) {
            partial_sum += seg_cache[i];
        };

        $uint prefix = 0;
        $uint block_sum = 0;
        BlockScan<uint>(scan_smem).exclusive_scan(partial_sum, prefix, block_sum);

        // adding the lower bit count to higher bit
        for (int i = 1; i < packing_radio; i++) {
            prefix += block_sum << (i * log_max_rank);
        }

        $uint tmp;
        $for (i, pad_counter_lines) {
            tmp = seg_cache[i];
            seg_cache[i] = prefix;
            prefix += tmp;
        };
        $for (i, pad_counter_lines) {
            (*counter_smem)[i + $thread_x * pad_counter_lines] = seg_cache[i];
        };
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

        for (int i = 0; i < pad_counter_lines; i++)
            (*counter_smem)[$thread_x + i*block_size().x] = 0;

        // counting digits
        for (int i = 0; i < thread_items; i++) {
            $uint digit = (keys[i] >> begin_bit) & (radix_digits - 1);

            // high bit as sub_id, can easy to get high bit prefix after block scan
            $uint sub_id = digit >> log_counter_lines;
            $uint line_id = digit & (counter_lines - 1);

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
        }
    }

    template<size_t thread_items>
    void rank($array<uint, thread_items> &keys, $array<uint, thread_items> &ranks, $uint begin_bit) {
        if (thread_items * block_size().x > max_rank) {
            print("ranking keys is more than {}\n", max_rank);
            exit(0);
        }
        
        // smem layout: (counter_lines * block_size * packing_radio)
        // last dim of smem "packing_radio" is packed in one uint

        // $array<ushort, thread_items> thread_prefix;
        $array<uint, thread_items> thread_prefix;

        $for (i, pad_counter_lines) {
            (*counter_smem)[$thread_x + i*block_size().x] = 0;
        };

        // counting digits
        $for (i, (uint)thread_items) {
            $uint digit = (keys[i] >> begin_bit) & (radix_digits - 1);

            // high bit as sub_id, can easy to get high bit prefix after block scan
            $uint sub_id = digit >> log_counter_lines;
            $uint line_id = digit & (counter_lines - 1);

            thread_prefix[i] = extract_counter(line_id, sub_id);
            counter_add(line_id, sub_id);
        };
        sync_block();

        scan_counters();

        sync_block();
        $for (i, (uint)thread_items) {
            $uint digit = (keys[i] >> begin_bit) & (radix_digits - 1);
            $uint sub_id = digit >> log_counter_lines;
            $uint line_id = digit & (counter_lines - 1);

            ranks[i] = thread_prefix[i].cast<uint>() + extract_counter(line_id, sub_id);
        };
    }

};

template<size_t thread_items>
class BlockRadixSort {
    $shared<uint> *smem; // use for rank counter and exchange keys
    $shared<uint> *scan_smem;
    bool own_smem = false;
public:
    BlockRadixSort() {
        scan_smem = new $shared<uint>(BlockScan<uint>::shared_size());
        smem = new $shared<uint>(max((uint)thread_items * block_size().x, BlockRadixRank::counter_shared_size()));
        own_smem = true;
    }
    ~BlockRadixSort() {
        if (own_smem) {
            delete smem;
            delete scan_smem;
        }
    }
    static const uint radix_bits = BlockRadixRank::radix_bits;

    void sort($uint (&keys)[thread_items], uint begin_bit, uint end_bit) {
        if ((end_bit - begin_bit) % radix_bits != 0) {
            print("range of sorting bit must multiple of radix_bits:{}\n", radix_bits);
            exit(0);
        }
        $uint ranks[thread_items];
        $for (t, begin_bit, end_bit, radix_bits) {
            BlockRadixRank(smem, scan_smem).rank(keys, ranks, t);
            sync_block();
            for (int i = 0; i < thread_items; i++) {
                (*smem)[ranks[i]] = keys[i];
            }
            sync_block();
            for (int i = 0; i < thread_items; i++) {
                keys[i] = (*smem)[i + $thread_x * (uint)thread_items];
            }
        };
    }

    void sort($array<uint, thread_items> &keys, uint begin_bit, uint end_bit) {
        if ((end_bit - begin_bit) % radix_bits != 0) {
            print("range of sorting bit must multiple of radix_bits:{}\n", radix_bits);
            exit(0);
        }
        $array<uint, thread_items> ranks;
        $for (t, begin_bit, end_bit, radix_bits) {
            BlockRadixRank(smem, scan_smem).rank(keys, ranks, t);
            sync_block();
            for (int i = 0; i < thread_items; i++) {
                (*smem)[ranks[i]] = keys[i];
            }
            sync_block();
            for (int i = 0; i < thread_items; i++) {
                keys[i] = (*smem)[i + $thread_x * (uint)thread_items];
            }
        };
    }

    // 排序后keys为striped排列，即先按tid顺序，再按thread_items顺序
    void sort_to_striped($uint (&keys)[thread_items], uint begin_bit, uint end_bit) {
        if ((end_bit - begin_bit) % radix_bits != 0) {
            print("range of sorting bit must multiple of radix_bits:{}\n", radix_bits);
            exit(0);
        }
        $uint ranks[thread_items];
        $for (t, begin_bit, end_bit, radix_bits) {
            BlockRadixRank(smem, scan_smem).rank(keys, ranks, t);
            sync_block();
            for (int i = 0; i < thread_items; i++) {
                (*smem)[ranks[i]] = keys[i];
            }
            sync_block();
            $if (t + radix_bits < end_bit) {
                for (int i = 0; i < thread_items; i++) {
                    keys[i] = (*smem)[i + $thread_x * (uint)thread_items];
                }
            }
            $else {
                //最后一次按striped排
                for (int i = 0; i < thread_items; i++) {
                    keys[i] = (*smem)[$thread_x + i * block_size().x];
                }
            };
        };
    }

    // 排序后keys为striped排列，即先按tid顺序，再按thread_items顺序
    void sort_to_striped($array<uint, thread_items> &keys, uint begin_bit, uint end_bit) {
        if ((end_bit - begin_bit) % radix_bits != 0) {
            print("range of sorting bit must multiple of radix_bits:{}\n", radix_bits);
            exit(0);
        }
        $array<uint, thread_items> ranks;
        $for (t, begin_bit, end_bit, radix_bits) {
            BlockRadixRank(smem, scan_smem).rank(keys, ranks, t);
            sync_block();
            $for (i, (uint)thread_items) {
                (*smem)[ranks[i]] = keys[i];
            };
            sync_block();
            $if (t + radix_bits < end_bit) {
                $for (i, (uint)thread_items) {
                    keys[i] = (*smem)[i + $thread_x * (uint)thread_items];
                };
            }
            $else {
                //最后一次按striped排
                $for (i, (uint)thread_items) {
                    keys[i] = (*smem)[$thread_x + i * block_size().x];
                };
            };
        };
    }
};
