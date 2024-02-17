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

struct RadixSortContext {
    uint num_key;
    uint num_block;
    uint num_tile_per_block;
    uint num_extra_tile;
    uint num_reduce_block;
    uint begin_bit;
};
LUISA_BINDING_GROUP(RadixSortContext, num_key, num_block, num_tile_per_block, num_extra_tile, num_reduce_block, begin_bit) {};


// $uint block_reduce($uint val, $uint tid) {
//     $shared<uint> reduce_smem{32};

//     $uint warp_sum = warp_active_sum(val);
//     $uint warp_id = tid / warp_lane_count();

//     $if (warp_is_first_active_lane()) {
//         reduce_smem[warp_id] = warp_sum;
//     };
//     sync_block();

//     $uint sum = 0;
//     $if (warp_id == 0) {
//         $if (tid < block_size().x / warp_lane_count()) {
//             sum = reduce_smem[tid];
//         };
//         sum = warp_active_sum(sum);
//     };

//     // 只有warp_id == 0才返回正确值
//     return sum;
// }

$uint block_reduce($uint val, $uint tid) {
    $shared<uint> reduce_smem{8};

    $uint warp_id = tid / 32;
    $uint lane_id = tid % 32;

    $uint warp_sum = warp_active_sum(val);
    
    $if (lane_id == 0) {
        reduce_smem[warp_id] = warp_sum;
    };
    sync_block();

    $uint sum = 0;
    $if (tid == 0) {
        for (int i = 0; i < 7; i++) {
            sum += reduce_smem[i];
        }
    };

    // 只有tid == 0才返回正确值
    return sum;
}

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
// void block_exclusive_scan($uint4 input, $uint4 &output, $uint4 &block_sum, $uint tid) {
//     $shared<uint4> smem{7};
//     $uint4 t = warp_prefix_sum(input);
//     $if (tid % 32 == 31) {
//         smem[tid / 32] = t + input;
//     };
//     sync_block();
//     block_sum = smem[0];
//     $uint4 t1;
//     for (int i = 1; i < 7; i++) {
//         $if (tid / 32 == i) { t1 = block_sum; };
//         block_sum += smem[i];
//     }
//     output = t + t1;
// }

int main(int argc, char** argv) {

    const int block_size = 128;
    const int max_num_block = 1024;

    const int radix_bits = 4;
    const int radix_digits = 1 << radix_bits;
    const int thread_items = 4;
    const int tile_size = block_size * thread_items;

    // Kernel1D counting_keys = [&]($buffer<uint> keys, $uint begin_bit, $buffer<uint> out) {
    //     set_block_size(block_size);
    //     $shared<uint> counter{block_size * radix_digits};
        
    //     $uint tid = $thread_x;
    //     $uint bid = $block_x;

    //     for (int i = 0; i < radix_digits; i++)
    //         counter[tid + i * block_size] = 0;
    //     sync_block();

    //     $uint k[thread_items];
    //     for (int i = 0; i < thread_items; i++) {
    //         k[i] = keys.read(tid + i * block_size + bid * tile_size);
    //     }
    //     for (int i = 0; i < thread_items; i++) {
    //         $uint digit = (k[i] >> begin_bit) & (radix_digits - 1);
    //         counter.atomic(tid + digit * block_size).fetch_add(1u);
    //     }
    //     sync_block();

    //     $if (tid < radix_digits) {
    //         $uint sum = 0;
    //         $for (i, block_size) {
    //         // for (int i = 0; i < block_size; i++) {
    //             sum += counter[i + tid * block_size];
    //         };
    //         $uint num_block = $dispatch_size_x / block_size;
    //         out.write(bid + tid * num_block, sum);
    //     };
    // };
    Kernel1D counting_keys1 = [&]($buffer<uint> keys, $buffer<uint> digit_counter, $<RadixSortContext> ctx) {
        set_block_size(block_size);
        $shared<uint> counter{block_size};

        const int bin_size = block_size / radix_digits;
        $uint tid = $thread_x;
        $uint bid = $block_x;

        counter[tid] = 0;
        sync_block();

        $uint num_proc_tile = ctx.num_tile_per_block;
        $uint base_idx = bid * num_proc_tile * tile_size;

        // 剩余tile被尾部block平分
        $if (bid >= ctx.num_block - ctx.num_extra_tile) {
            base_idx += (bid - (ctx.num_block - ctx.num_extra_tile)) * tile_size;
            num_proc_tile += 1;
        };

        $for (t, num_proc_tile) {
            $uint offset = tid + base_idx + t * tile_size;
            // for (int i = 0; i < thread_items; i++) {
            //     $uint item_idx = offset + i * block_size;
            //     $if (item_idx < ctx.num_key) {
            //         $uint key = keys.read(item_idx);
            //         $uint digit = (key >> ctx.begin_bit) & (radix_digits - 1);
            //         counter.atomic(tid % bin_size + digit * bin_size).fetch_add(1u);
            //     };
            // }
            $uint4 key;
            for (int i = 0; i < thread_items; i++) {
                key[i] = keys.read(offset + i * block_size);
            }
            for (int i = 0; i < thread_items; i++) {
                $uint item_idx = offset + i * block_size;
                $if (item_idx < ctx.num_key) {
                    $uint digit = (key[i] >> ctx.begin_bit) & (radix_digits - 1);
                    counter.atomic(tid % bin_size + digit * bin_size).fetch_add(1u);
                };
            }
        };
        sync_block();

        $if (tid < radix_digits) {
            $uint sum = 0;
            for (int i = 0; i < bin_size; i++) {
                sum += counter[i + tid * bin_size];
            }
            digit_counter.write(bid + tid * ctx.num_block, sum);
        };
    };

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    Clock timer;

    const int n = 1 << 19;

    RadixSortContext ctx{};
    {
        const int num_tile = (n + tile_size - 1) / tile_size;
        ctx.num_key = n;
        ctx.num_block = max_num_block;
        ctx.num_tile_per_block = num_tile / ctx.num_block;
        ctx.num_extra_tile = num_tile % ctx.num_block;

        if (num_tile < max_num_block) {
            ctx.num_block = num_tile;
            ctx.num_tile_per_block = 1;
            ctx.num_extra_tile = 0;
        }

        int num_scan_val = radix_digits * ctx.num_block; // < 16 * 1024
        ctx.num_reduce_block = (num_scan_val + tile_size - 1) / tile_size; // < 32
    }
    ctx.begin_bit = 0;

    print("## ctx:\n");
    print("num_key:{}\nnum_block:{}\nnum_tile_per_block:{}\nnum_extra_tile:{}\nnum_reduce_block:{}\n", ctx.num_key, ctx.num_block, ctx.num_tile_per_block, ctx.num_extra_tile, ctx.num_reduce_block);

    vector<uint> keys_h(n);
    vector<uint> digit_counter_h(radix_digits * ctx.num_block);
    vector<uint> digit_counter_d(radix_digits * ctx.num_block);

    for (uint &x: keys_h) x = pcg32::next_uint() % 16;

    print("## counting digit\n");
    timer.tic();
    // for (int t = 0; t < ctx.num_block; t++) {
    //     for (int i = 0; i < tile_size; i++) {
    //         uint key = keys_h[i + t * tile_size];
    //         uint digit = (key >> ctx.begin_bit) & (radix_digits - 1);
    //         digit_counter_h[t + digit * ctx.num_block] += 1;
    //     }
    // }
    for (int t = 0; t < ctx.num_block; t++) {
        uint num_proc_tile = ctx.num_tile_per_block;
        uint base_idx = t * num_proc_tile * tile_size;

        // 剩余tile被尾部block平分
        if (t >= ctx.num_block - ctx.num_extra_tile) {
            base_idx += (t - (ctx.num_block - ctx.num_extra_tile)) * tile_size;
            num_proc_tile += 1;
        }

        for (int i = 0; i < num_proc_tile * tile_size; i++) {
            if (i + base_idx < n) {
                uint key = keys_h[i + base_idx];
                uint digit = (key >> ctx.begin_bit) & (radix_digits - 1);
                digit_counter_h[t + digit * ctx.num_block] += 1;
            }
        }
    }
    print("calc ref res: {}\n", timer.toc());

    timer.tic();
    // auto shader = device.compile(counting_keys);
    auto counting_keys_shader = device.compile(counting_keys1);
    print("compiled shader: {}\n", timer.toc());

    auto keys = device.create_buffer<uint>(n);
    // auto keys1 = device.create_buffer<uint>(n);
    auto digit_counter = device.create_buffer<uint>(radix_digits * ctx.num_block);

    stream << keys.copy_from(keys_h.data()) << synchronize();

    for (int t = 0; t < 5; t++) {
        CommandList cmd_list;
        for (int i = 0; i < 1000; i++) {
            cmd_list << counting_keys_shader(keys, digit_counter, ctx).dispatch(ctx.num_block * block_size);
            // cmd_list << shader(keys.view().as<uint4>(), begin_bit, out).dispatch(num_block * block_size);
            // cmd_list << keys1.copy_from(keys); // 测试带宽
        }
        timer.tic();
        stream << cmd_list.commit();
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << digit_counter.copy_to(digit_counter_d.data()) << synchronize();

    print_vec(digit_counter_h, "out_h", 32);
    print_vec(digit_counter_d, "out_d", 32);

    compare_vec(digit_counter_h, digit_counter_d);

    

    Kernel1D reduce = [&]($buffer<uint> digit_counter, $buffer<uint> reduce_scratch, $<RadixSortContext> ctx) {
        set_block_size(block_size);

        $uint tid = $thread_x;
        $uint bid = $block_x;
        $uint num_scan_val = radix_digits * ctx.num_block; // < 16 * 1024

        $uint sum = 0;
        $uint val[4];
        for (int i = 0; i < thread_items; i++) {
            $uint idx = tid + i * block_size + bid * tile_size;
            // val[i] = digit_counter.read(idx);
            $if (idx < num_scan_val) { sum += digit_counter.read(idx); };
            // $if (idx < num_scan_val) { sum += val[i]; };
        }
        // for (int i = 0; i < thread_items; i++) {
        //     $uint idx = tid + i * block_size + bid * tile_size;
        //     $if (idx < num_scan_val) { sum += val[i]; };
        // }

        // sum = block_reduce(sum, tid);
        $shared<uint> reduce_smem{8};

        $uint warp_sum = warp_active_sum(sum);
        
        $if (tid % 32 == 0) {
            reduce_smem[tid / 32] = warp_sum;
        };
        sync_block();

        sum = 0;
        $if (tid / 32 == 0) {
            $if (tid < 7) { sum = reduce_smem[tid]; };
            sum = warp_active_sum(sum);
        };
        $if (tid == 0) {
            // for (int i = 0; i < 7; i++) {
            //     sum += reduce_smem[i];
            // }
            reduce_scratch.write(bid, sum);
        };
    };

    vector<uint> reduce_scratch_h(ctx.num_reduce_block);
    vector<uint> reduce_scratch_d(ctx.num_reduce_block);

    print("## reduce\n");
    timer.tic();
    for (int t = 0; t < ctx.num_reduce_block; t++) {
        const int num_scan_val = radix_digits * ctx.num_block;
        for (int i = 0; i < tile_size; i++) {
            int idx = i + t * tile_size;
            if (idx < num_scan_val) {
                reduce_scratch_h[t] += digit_counter_h[idx];
            }
        }
    }
    print("calc ref res: {}\n", timer.toc());

    timer.tic();
    auto reduce_shader = device.compile(reduce);
    print("compiled shader: {}\n", timer.toc());

    auto reduce_scratch = device.create_buffer<uint>(ctx.num_reduce_block);
    // auto digit_counter1 = device.create_buffer<uint>(radix_digits * ctx.num_block);

    for (int t = 0; t < 5; t++) {
        CommandList cmd_list;
        for (int i = 0; i < 1000; i++) {
            cmd_list << reduce_shader(digit_counter, reduce_scratch, ctx).dispatch(ctx.num_reduce_block * block_size);
            // cmd_list << digit_counter1.copy_from(digit_counter); // 测试带宽
        }
        timer.tic();
        stream << cmd_list.commit();
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << reduce_scratch.copy_to(reduce_scratch_d.data()) << synchronize();

    print_vec(reduce_scratch_h, "out_h", 32);
    print_vec(reduce_scratch_d, "out_d", 32);

    compare_vec(reduce_scratch_h, reduce_scratch_d);


    Kernel1D scan = [&]($buffer<uint> digit_counter, $buffer<uint> reduce_scratch, $buffer<uint> scan_res, $<RadixSortContext> ctx) {
        set_block_size(block_size);

        $shared<uint> coarse_scan{32};
        $uint tid = $thread_x;
        $uint bid = $block_x;

        $if (tid < ctx.num_reduce_block) { // < 32
            $uint sum = reduce_scratch.read(tid);
            coarse_scan[tid] = warp_prefix_sum(sum);
        };
        sync_block();

        $uint num_scan_val = radix_digits * ctx.num_block; // < 16 * 1024

        $uint val[4];
        $uint sum = 0;
        for (int i = 0; i < thread_items; i++) {
            $uint idx = i + tid * 4 + bid * tile_size;
            val[i] = digit_counter.read(idx);
            $if (idx < num_scan_val) { sum += val[i]; };
        }
        // for (int i = 0; i < thread_items; i++) {
        //     $uint idx = tid + i * block_size + bid * tile_size;
        //     $if (idx < num_scan_val) { sum += digit_counter.read(idx); };
        // }

        $shared<uint> scan_smem{32};

        $uint warp_prefix = warp_prefix_sum(sum);
        $if (tid % 32 == 31) {
            scan_smem[tid / 32] = warp_prefix + sum;
        };
        sync_block();

        $if (tid / 32 == 0) {
            scan_smem[tid] = warp_prefix_sum(scan_smem[tid]);
        };
        sync_block();

        $uint block_prefix = scan_smem[tid / 32] + warp_prefix;
        $uint global_prefix = block_prefix + coarse_scan[bid];

        for (int i = 0; i < thread_items; i++) {
            $uint idx = i + tid * 4 + bid * tile_size;
            $if (idx < num_scan_val) {
                // digit_counter.write(idx, global_prefix);
                scan_res.write(idx, global_prefix);
                global_prefix += val[i];
            };
        }
    };

    print("## scan\n");
    timer.tic();
    {
        int sum = 0;
        for (uint &x: digit_counter_h) {
            int t = sum;
            sum += x;
            x = t;
        }
    }
    print("calc ref res: {}\n", timer.toc());

    timer.tic();
    auto scan_shader = device.compile(scan);
    print("compiled shader: {}\n", timer.toc());

    auto scan_res = device.create_buffer<uint>(radix_digits * ctx.num_block);

    for (int t = 0; t < 5; t++) {
        CommandList cmd_list;
        for (int i = 0; i < 1000; i++) {
            cmd_list << scan_shader(digit_counter, reduce_scratch, scan_res, ctx).dispatch(ctx.num_reduce_block * block_size);
        }
        timer.tic();
        stream << cmd_list.commit();
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << scan_res.copy_to(digit_counter_d.data()) << synchronize();

    print_vec(digit_counter_h, "out_h", 32);
    print_vec(digit_counter_d, "out_d", 32);

    compare_vec(digit_counter_h, digit_counter_d);


    print("## total scan\n");
    for (int t = 0; t < 5; t++) {
        CommandList cmd_list;
        for (int i = 0; i < 1000; i++) {
            cmd_list
                << reduce_shader(digit_counter, reduce_scratch, ctx).dispatch(ctx.num_reduce_block * block_size)
                << scan_shader(digit_counter, reduce_scratch, scan_res, ctx).dispatch(ctx.num_reduce_block * block_size);
        }
        timer.tic();
        stream << cmd_list.commit();
        stream.synchronize();
        print("{}\n", timer.toc());
    }


    Kernel1D scatter = [&]($buffer<uint> keys, $buffer<uint> sorted_keys, $buffer<uint> digit_counter, $<RadixSortContext> ctx) {
        set_block_size(block_size);

        $uint tid = $thread_x;
        $uint bid = $block_x;

        // ??? 只要出现就会出错
        // $shared<uint> global_offsets{radix_digits};
        // $if (tid < 32) {
        //     global_offsets[tid] = digit_counter.read(tid * ctx.num_block + bid);
        // };
        // sync_block();

        $uint num_proc_tile = ctx.num_tile_per_block;
        $uint base_idx = bid * num_proc_tile * tile_size;

        // 剩余tile被尾部block平分
        $if (bid >= ctx.num_block - ctx.num_extra_tile) {
            base_idx += (bid - (ctx.num_block - ctx.num_extra_tile)) * tile_size;
            num_proc_tile += 1;
        };

        $shared<uint> scan_smem{block_size}; /// ??? smem size 设为7会出错
        $shared<uint> key_scratch{block_size};
        $shared<uint> counter{block_size}; /// ??? smem size 设为32会出错
        // $shared<uint> counter_prefix{32};

        // for (int t = 0; t < 1; t++) {
        $for (t, num_proc_tile) {
            $uint offset = base_idx + t * tile_size;
            $uint4 key;
            for (int i = 0; i < thread_items; i++) {
                $uint idx = tid + i * block_size + offset;
                key[i] = keys.read(idx);
                $if (idx >= ctx.num_key) { key[i] = ~0u; };
            }
            for (int i = 0; i < thread_items; i++) {
                // $if (tid < radix_digits) {
                //     counter[tid] = 0;
                // };

                $uint local_key = key[i];

                // 统计整个block的digit出现次数
                // $uint digit = (local_key >> ctx.begin_bit) & (radix_digits - 1);
                // counter.atomic(digit).fetch_add(1u);
                // sync_block();

                for (uint p = 0; p < radix_bits; p += 2) {
                    $uint digit = (local_key >> ctx.begin_bit) & (radix_digits - 1);
                    $uint sub_d = (digit >> p) & 0x3;

                    // block内计数一定小于256，用一个uint表示4个counter
                    $uint pack_counter = 1 << (sub_d * 8);
                    $uint block_sum;
                    block_exclusive_scan(pack_counter, pack_counter, block_sum, tid, scan_smem);

                    // 将低位计数加到高位
                    pack_counter += (block_sum << 8) + (block_sum << 16) + (block_sum << 24);

                    $uint key_offset = (pack_counter >> (sub_d * 8)) & 0xff;
                    key_scratch[key_offset] = local_key;
                    sync_block();

                    local_key = key_scratch[tid];
                    sync_block();
                }

                $uint idx = tid + i * block_size + offset;
                $if (idx < ctx.num_key) {
                    sorted_keys.write(idx, local_key);
                };
                sync_block();

                // 统计整个block的digit出现次数
                // $uint digit = (local_key >> ctx.begin_bit) & (radix_digits - 1);
                // counter.atomic(digit).fetch_add(1u);
                // sync_block();

                // $if (tid / 32 == 0) {
                //     counter_prefix[tid] = warp_prefix_sum(counter[tid]);
                // };
                // $uint global_offset = global_offsets[digit];
                // sync_block();

                // $if (tid < radix_digits) {
                //     global_offsets[tid] += counter[tid];
                // };

                // // global offset中包含digit小的所有个数，local offset中需要减去
                // $uint local_offset = tid - counter_prefix[digit];

                // $uint idx = local_offset + global_offset;
                // $if (idx < ctx.num_key) {
                //     sorted_keys.write(idx, local_key);
                // };

                // 假设block内相同digit数量小于256，用一个uint表示4个counter，4个uint即可表示radix_digits个counter
                // $uint4 pack_counter{};
                // $uint digit = (key[i] >> ctx.begin_bit) & (radix_digits - 1);
                // pack_counter[digit >> 2] += 1 << ((digit & 0x3) * 8);

                // $uint4 block_sum;
                // block_exclusive_scan(pack_counter, pack_counter, block_sum, tid);
            }
        };
    };

    vector<uint> sorted_keys_h(n, -1);
    vector<uint> sorted_keys_d(n, -1);
    print("## scatter\n");
    timer.tic();
    // for (int t = 0; t < ctx.num_block; t++) {
    //     uint num_proc_tile = ctx.num_tile_per_block;
    //     uint base_idx = t * num_proc_tile * tile_size;

    //     // 剩余tile被尾部block平分
    //     if (t >= ctx.num_block - ctx.num_extra_tile) {
    //         base_idx += (t - (ctx.num_block - ctx.num_extra_tile)) * tile_size;
    //         num_proc_tile += 1;
    //     }

    //     vector<uint> global_offsets(radix_digits);
    //     for (int i = 0; i < radix_digits; i++) {
    //         global_offsets[i] = digit_counter_d[t + i * ctx.num_block];
    //     }
    //     for (int i = 0; i < num_proc_tile * tile_size; i++) {
    //         if (i + base_idx < n) {
    //             uint key = keys_h[i + base_idx];
    //             uint digit = (key >> ctx.begin_bit) & (radix_digits - 1);

    //             sorted_keys_h[global_offsets[digit]] = key;
    //             global_offsets[digit] += 1;
    //         }
    //     }
    // }
    {
        // test sort block
        sorted_keys_h = keys_h;
        int num_tile = n / block_size;
        int num_extra = n % block_size;
        for (int i = 0; i < num_tile; i++) {
            std::sort(sorted_keys_h.data() + i * block_size, sorted_keys_h.data() + (i + 1) * block_size);
        }
        if (num_extra) {
            std::sort(sorted_keys_h.begin() + num_tile * block_size, sorted_keys_h.end());
        }
    }
    print("calc ref res: {}\n", timer.toc());

    // for (int i = 0; i < n - 1; i++) {
    //     if (sorted_keys_h[i + 1] < sorted_keys_h[i]) {
    //         print("ref unsort!\n");
    //         exit(1);
    //     }
    // }

    timer.tic();
    auto scatter_shader = device.compile(scatter);
    print("compiled shader: {}\n", timer.toc());

    auto sorted_keys = device.create_buffer<uint>(n);

    for (int t = 0; t < 1; t++) {
        CommandList cmd_list;
        for (int i = 0; i < 1; i++) {
            cmd_list << scatter_shader(keys, sorted_keys, scan_res, ctx).dispatch(ctx.num_block * block_size);
        }
        timer.tic();
        stream << cmd_list.commit();
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << sorted_keys.copy_to(sorted_keys_d.data()) << synchronize();

    print_vec(sorted_keys_h, "out_h", 32);
    print_vec(sorted_keys_d, "out_d", 32);

    compare_vec(sorted_keys_h, sorted_keys_d);

    return 0;
}