#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <ngp_encode_layer.h>
#include <global.h>
#include <gpu_rands.h>
#include <block_radix.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

const int N_min = 16;

slong powll(slong x, int p) {
    slong res = 1;
    for (int i = 0; i < p; i++) {
        res *= x;
    }
    return res;
}
int powi(int x, int p) {
    int res = 1;
    for (int i = 0; i < p; i++) {
        res *= x;
    }
    return res;
}
$uint powi($uint x, int p) {
    $uint res = 1;
    for (int i = 0; i < p; i++) {
        res *= x;
    }
    return res;
}

NGPEncodeLayer::NGPEncodeLayer(int input_dim, int output_dim, int max_level_table_size, int levels, AdamConfig optim_cfg):
    DiffLayer(input_dim, output_dim),
    optim(optim_cfg),
    // optim_config(optim_cfg),
    max_level_table_size(max_level_table_size),
    levels(levels)
    // feature_per_level(feature_per_level)
{
    if (input_dim > 3) {
        fmt::print("error: ngp_encoder not impl for input_dim > 3\n");
        exit(1);
    }
    if (levels * feature_per_level > output_dim) {
        fmt::print("error: feature num > output_dim\n");
        exit(1);
    }

    init_level_offset();
    _feature_table = global::device().create_buffer<half2>(_table_size);
    // _feature_table = global::device().create_byte_buffer(table_size * sizeof(half2));
    _feature_gradient = global::device().create_buffer<half2>(_table_size);

    reset_parameters();
    optim.init(_feature_table.view().as<half4>());
}

void NGPEncodeLayer::init_level_offset() {
    level_offset.resize(levels + 1);
    int offset = 0;
    for (int i = 0; i < levels; i++) {
        int res = (1 << i) * N_min;
        int level_size = min(powll(res, input_dim()), (slong)max_level_table_size);
        level_offset[i] = offset;
        offset += level_size;
    }
    level_offset[levels] = offset;
    _table_size = offset;

    level_offset_buffer = global::device().create_buffer<int>(levels + 1);
    global::stream() << level_offset_buffer.copy_from(level_offset.data()) << synchronize();
}

void trans_input($float4 *in, int input_dim) {
    $float tmp;
    auto swap = [&]($float &a, $float &b) {
        tmp = a;
        a = b;
        b = tmp;
    };
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < min(input_dim, j); i++) {
            swap(in[i][j], in[j][i]);
        }
    }
}
void calc_pos($float4 &in, $float4 &pos, $uint4 &grid_idx, $uint &grid_res) {
    pos = in * (grid_res - 1).cast<float>();
    $float4 tmp = floor(pos);
    grid_idx = tmp;
    pos -= tmp;
}
$uint table_idx(int dim, $uint4 &grid_idx, $uint &grid_res, $uint &level_size, $bool &use_hash) {
    $uint idx = 0;
    $if (use_hash) {
        uint prime[4] = {1u, 2654435761u, 805459861u, 3674653429u};
        for (int i = 0; i < dim; i++) {
            idx ^= grid_idx[i] * prime[i];
        }
    }
    $else {
        $uint s = 1;
        for (int i = 0; i < dim; i++) {
            idx += grid_idx[i] * s;
            s *= grid_res;
        }
    };
    return idx % level_size;
}

template<int input_dim, int num_item = (1 << input_dim) * 4>
void calc_table_idx($array<uint, num_item> &table_idx, $array<uint4, input_dim> &grid_idx, $uint grid_res, $uint level_size, $bool use_hash) {
    uint primes[4] = {1u, 2654435761u, 805459861u, 3674653429u};
    $if (use_hash) {
        // for (int t = 0; t < 4; t++) {
        $for (t, 4) {
            for (int s = 0; s < (1 << input_dim); s++) {
                for (int d = 0; d < input_dim; d++) {
                    table_idx[s + t*(1<<input_dim)] ^= (grid_idx[d][t] + (s >> d & 1)) * primes[d];
                }
            }
        };
        // for (int i = 0; i < num_item; i++) {
        $for (i, num_item) {
            table_idx[i] %= level_size;
        };
    }
    $else {
        // for (int t = 0; t < 4; t++) {
        $for (t, 4) {
            for (int s = 0; s < (1 << input_dim); s++) {
                $uint st = 1;
                for (int d = 0; d < input_dim; d++) {
                    table_idx[s + t*(1<<input_dim)] += (grid_idx[d][t] + (s >> d & 1)) * st;
                    st *= grid_res;
                }
            }
        };
    };
}

template<int num_item>
void sort_idx($array<uint, num_item> &table_idx) {
    const uint sort_num = num_item / 2;
    for (int i = 0; i < sort_num; i++) {
    // $for (i, num_item) {
        for (int j = i & 1; j < sort_num - 1; j += 2) {
        // $for (j, i & 1, num_item - 1, 2u) {
            $if (table_idx[(j + 1)*2] < table_idx[j*2]) {
                $uint tmp = table_idx[j*2];
                table_idx[j*2] = table_idx[(j + 1)*2];
                table_idx[(j + 1)*2] = tmp;

                tmp = table_idx[j*2 + 1];
                table_idx[j*2 + 1] = table_idx[(j + 1)*2 + 1];
                table_idx[(j + 1)*2 + 1] = tmp;
            };
        };
    };
}

template<int input_dim>
void ngp_encode_kernel_impl(
    $uint &batch_size,
    $buffer<half4> &input, 
    $buffer<half4> &output, 
    $buffer<half2> &feature_table, 
    $buffer<int> &level_offsets
) {
    set_block_size(256, 1);
    $uint tid = $dispatch_x;
    $uint level = $dispatch_y;
    $uint level_offset = level_offsets.read(level);
    $uint level_size = level_offsets.read(level + 1) - level_offset;

    $uint grid_res = (1u << level) * N_min;
    $bool use_hash = (pow(grid_res.cast<float>(), (float)input_dim) > level_size + 50) | (powi(grid_res, input_dim) > level_size);

    $array<float4, input_dim> in;
    $array<float4, input_dim> pos;
    $array<uint4, input_dim> grid_idx;

    // for (int i = 0; i < input_dim; i++) {
    $for (i, input_dim) {
        in[i] = input.read(tid + i*batch_size/4);
        in[i] = fract(fract(in[i]) + 1.f);
    };
    // for (int i = 0; i < input_dim; i++) {
    $for (i, input_dim) {
        $float4 tmp = in[i] * (grid_res-1).cast<float>();
        grid_idx[i] = floor(tmp);
        pos[i] = fract(tmp);
    };

    // trans_input(in, input_dim);
    // for (int i = 0; i < 4; i++) {
    //     calc_pos(in[i], pos[i], grid_idx[i], grid_res);
    // }

    constexpr const uint num_item = (1 << input_dim) * 4;
    $array<uint, num_item> table_idx;

    calc_table_idx<input_dim, num_item>(table_idx, grid_idx, grid_res, level_size, use_hash);

    // for (int i = 0; i < num_item; i++) {
    $for (i, num_item) {
        table_idx[i] = (table_idx[i] << 5) | i;
    };

    // odd even sort
    if (input_dim == 3) {
        sort_idx<num_item>(table_idx);
    }

    $float4 feature[2];
    $for (ti, num_item) {
        $float2 v = feature_table.read(level_offset + (table_idx[ti] >> 5));
        $uint iid = table_idx[ti] & 31;

        $uint s = iid & ((1 << input_dim) - 1);
        $uint t = iid >> input_dim;

        $float w = 1;
        for (int d = 0; d < input_dim; d++) {
            $int f = (s & (1 << d)) == 0;
            w *= f + (1 - f*2) * pos[d][t];
            // if (f) w *= 1 - pos[d][t];
            // else w *= pos[d][t];
        }
        for (int i = 0; i < 2; i++) {
            feature[i][t] += w * v[i];
        }
    };

    for (int i = 0; i < 2; i++) {
        $half4 tmp = feature[i];
        output.write(tid + (level*2+i)*batch_size/4, tmp);
    }
}

// template<int input_dim>
// void ngp_encode_kernel_impl1(
//     $uint &batch_size,
//     $buffer<half4> &input, 
//     $buffer<half4> &output, 
//     $bytebuffer &feature_table, 
//     $buffer<int> &level_offsets
// ) {
//     set_block_size(256, 1);
//     $uint tid = $dispatch_x;
//     $uint level = $dispatch_y;
//     $uint level_offset = level_offsets.read(level);
//     $uint level_size = level_offsets.read(level + 1) - level_offset;

//     $uint grid_res = (1u << level) * N_min;
//     $bool use_hash = (pow(grid_res.cast<float>(), (float)input_dim) > level_size + 50) | (powi(grid_res, input_dim) > level_size);

//     $float4 in[input_dim];
//     $uint4 grid_idx[input_dim];
//     $float4 pos[input_dim];
//     // $array<float4, input_dim> pos;

//     for (int i = 0; i < input_dim; i++) {
//         in[i] = input.read(tid + i*batch_size/4);
//     }
//     for (int i = 0; i < input_dim; i++) {
//         $float4 tmp = in[i] * (grid_res-1).cast<float>();
//         grid_idx[i] = floor(tmp);
//         pos[i] = fract(tmp);
//     }

//     // 假设hash后x坐标仍然连续（第一个hash素数为1）
//     // 则只需要存格子中x较小的角的hash坐标，然后一次取两个feature（float4）
//     // $uint table_idx[(1 << input_dim) / 2 * 4];
//     constexpr const uint num_item = (1 << input_dim) / 2 * 4;
//     $array<uint, num_item> table_idx;
//     uint primes[4] = {1u, 2654435761u, 805459861u, 3674653429u};

//     $if (use_hash) {
//         for (int t = 0; t < 4; t++) {
//             for (int s = 0; s < (1 << input_dim); s += 2) {
//                 for (int d = 0; d < input_dim; d++) {
//                     table_idx[t + s/2*4] ^= (grid_idx[d][t] + (s >> d & 1)) * primes[d];
//                 }
//             }
//         }
//     }
//     $else {
//         for (int t = 0; t < 4; t++) {
//             for (int s = 0; s < (1 << input_dim); s += 2) {
//                 $uint st = 1;
//                 for (int d = 0; d < input_dim; d++) {
//                     table_idx[t + s/2*4] += (grid_idx[d][t] + (s >> d & 1)) * st;
//                     st *= grid_res;
//                 }
//             }
//         }
//     };
//     for (int i = 0; i < num_item; i++) {
//         auto &x = table_idx[i];
//         x %= level_size - 1;
//         x = (x << 4) | i;
//     }

//     // odd even sort
//     // for (int i = 0; i < num_item; i++) {
//     //     for (int j = i & 1; j < num_item - 1; j += 2) {
//     //         $if (table_idx[j + 1] < table_idx[j]) {
//     //             $uint tmp = table_idx[j];
//     //             table_idx[j] = table_idx[j + 1];
//     //             table_idx[j + 1] = tmp;
//     //         };
//     //     }
//     // }

//     $float4 feature[2];
//     // $array<half4, 2> feature;

//     // $shared<half4> val{256};
//     // $shared<ushort> item_id{256};
//     // for (int t = 0; t < num_item; t++) {
//     $for (ti, num_item) {
//         // val[tid] = feature_table.read<half4>((level_offset + (table_idx[ti] >> 4)) * (uint)sizeof(half2));
//         // item_id[tid] = table_idx[ti] & 15;

//         // $float4 v = val[tid];
//         // $uint iid = item_id[tid];
//         $float4 v = feature_table.read<half4>((level_offset + (table_idx[ti] >> 4)) * (uint)sizeof(half2));
//         $uint iid = table_idx[ti] & 15;

//         $uint s = iid >> 2;
//         $uint t = iid & 3;

//         $float w = 1;
//         for (int d = 1; d < input_dim; d++) {
//             $int f = (s & (1 << (d - 1))) == 0;
//             w *= f + (1 - f*2) * pos[d][t];
//             // if (f) w *= 1 - pos[d][t];
//             // else w *= pos[d][t];
//         }
//         for (int i = 0; i < 2; i++) {
//             feature[i][t] += (1 - pos[0][t]) * w * v[i];
//             feature[i][t] += pos[0][t] * w * v[i + 2];
//         }
//     };
//     // $for (t, num_item) {
//     //     $float4 v = feature_table.read<half4>((level_offset + (table_idx[t] >> 4)) * (uint)sizeof(half2));
        
//     //     $uint iid = table_idx[t] & 15;
//     //     $uint s = iid >> 2;
//     //     $uint t = iid & 3;

//     //     $float w = 1;
//     //     for (int d = 1; d < input_dim; d++) {
//     //         $uint f = (s & (1 << (d - 1))) == 0;
//     //         w *= f + (1 - f*2) * pos[d][t];
//     //         // if (f) w *= 1 - pos[d][t];
//     //         // else w *= pos[d][t];
//     //     }
//     //     for (int i = 0; i < 2; i++) {
//     //         feature[i][t] += (1 - pos[0][t]) * w * v[i];
//     //         feature[i][t] += pos[0][t] * w * v[i + 2];
//     //     }
//     // };
//     for (int i = 0; i < 2; i++) {
//         $half4 tmp = feature[i];
//         output.write(tid + (level*2+i)*batch_size/4, tmp);
//     }
// }


// template<int input_dim>
// void ngp_encode_kernel_impl1(
//     $uint &batch_size,
//     $buffer<half4> &input, 
//     $buffer<half4> &output, 
//     // $bytebuffer &feature_table, 
//     $buffer<half2> &feature_table, 
//     $buffer<int> &level_offsets
// ) {
//     set_block_size(256, 1);
//     $uint tid = $dispatch_x;
//     $uint level = $dispatch_y;
//     $uint level_offset = level_offsets.read(level);
//     $uint level_size = level_offsets.read(level + 1) - level_offset;

//     $uint grid_res = (1u << level) * N_min;
//     $bool use_hash = (pow(grid_res.cast<float>(), (float)input_dim) > level_size + 50) | (powi(grid_res, input_dim) > level_size);

//     $float4 in[input_dim];
//     $uint4 grid_idx[input_dim];
//     $float4 pos[input_dim];
//     // $array<float4, input_dim> pos;

//     for (int i = 0; i < input_dim; i++) {
//         in[i] = input.read(tid + i*batch_size/4);
//     }
//     for (int i = 0; i < input_dim; i++) {
//         $float4 tmp = in[i] * (grid_res-1).cast<float>();
//         grid_idx[i] = floor(tmp);
//         pos[i] = fract(tmp);
//     }

//     // 假设hash后x坐标仍然连续（第一个hash素数为1）
//     // 则只需要存格子中x较小的角的hash坐标，然后一次取两个feature（float4）
//     // $uint table_idx[(1 << input_dim) / 2 * 4];
//     constexpr const uint num_item = (1 << input_dim) / 2 * 4;
//     $array<uint, num_item> table_idx;
//     uint primes[4] = {1u, 2654435761u, 805459861u, 3674653429u};

//     $if (use_hash) {
//         for (int t = 0; t < 4; t++) {
//             for (int s = 0; s < (1 << input_dim); s += 2) {
//                 for (int d = 0; d < input_dim; d++) {
//                     table_idx[t + s/2*4] ^= (grid_idx[d][t] + (s >> d & 1)) * primes[d];
//                 }
//             }
//         }
//     }
//     $else {
//         for (int t = 0; t < 4; t++) {
//             for (int s = 0; s < (1 << input_dim); s += 2) {
//                 $uint st = 1;
//                 for (int d = 0; d < input_dim; d++) {
//                     table_idx[t + s/2*4] += (grid_idx[d][t] + (s >> d & 1)) * st;
//                     st *= grid_res;
//                 }
//             }
//         }
//     };
//     for (int i = 0; i < num_item; i++) {
//         auto &x = table_idx[i];
//         x %= level_size - 1;

//         // assert (x < 1<<20), (tid < 1<<8), (i < 1<<4)
//         x = (x << 12) | (tid << 4) | i;
//     }

//     // odd even sort
//     // for (int i = 0; i < num_item; i++) {
//     //     for (int j = i & 1; j < num_item - 1; j += 2) {
//     //         $if (table_idx[j + 1] < table_idx[j]) {
//     //             $uint tmp = table_idx[j];
//     //             table_idx[j] = table_idx[j + 1];
//     //             table_idx[j + 1] = tmp;
//     //         };
//     //     }
//     // }

//     // BlockRadixSort<(1 << input_dim) / 2 * 4>().sort(table_idx, 16, 32);

//     // 链式前向星
//     $shared<uint> head{256};
//     $shared<ushort> next{256};
//     $shared<half4> val{256};
//     $shared<ushort> item_id{256};

//     $float4 feature[2];
//     // $array<half4, 2> feature;

//     // for (int t = 0; t < (1 << input_dim) / 2 * 4; t++) {
//     $for (ti, num_item) {
//         sync_block();
//         head[tid] = uint(-1);
//         sync_block();
//         // val[tid] = feature_table.read<half4>((level_offset + (table_idx[ti] >> 12)) * (uint)sizeof(half2));
//         val[tid] = make_float4((table_idx[ti] >> 12).cast<float>());
//         item_id[tid] = table_idx[ti] & 15;
//         next[tid] = head.atomic(table_idx[ti] >> 4 & 255).exchange(tid);
//         sync_block();

//         // $uint cur = head[tid];
//         $uint cur = tid;
//         $while (cur != uint(-1)) {
//         // $for (_, num_item) { // 让dxc知道有上界，否则报错
//             // $if (cur == uint(-1)) {
//             //     $break;
//             // };
//             $float4 v = val[cur];
//             $uint iid = item_id[cur];
//             $uint s = iid >> 2;
//             $uint t = iid & 3;

//             $float w = 1;
//             for (int d = 1; d < input_dim; d++) {
//                 $uint f = (s & (1 << (d - 1))) == 0;
//                 w *= f + (1 - f*2) * pos[d][t];
//                 // if (f) w *= 1 - pos[d][t];
//                 // else w *= pos[d][t];
//             }
//             for (int i = 0; i < 2; i++) {
//                 feature[i][t] += (1 - pos[0][t]) * w * v[i];
//                 feature[i][t] += pos[0][t] * w * v[i + 2];
//             }
//             cur = next[cur];
//         };
//     };
//     for (int i = 0; i < 2; i++) {
//         $half4 tmp = feature[i];
//         output.write(tid + (level*2+i)*batch_size/4, tmp);
//     }
// }

Shader2D<uint, Buffer<half4>, Buffer<half4>, Buffer<half2>, Buffer<int>> ngp_encode_shader[5];
// Shader2D<uint, Buffer<half4>, Buffer<half4>, ByteBuffer, Buffer<int>> ngp_encode_shader[5];

void NGPEncodeLayer::forward(const BufferView<half4> input, BufferView<half4> output) {
    if (!ngp_encode_shader[input_dim()]) {
        Kernel2D ngp_encode_kernel = [&](
            $uint batch_size,
            $buffer<half4> input, 
            $buffer<half4> output, 
            $buffer<half2> feature_table, 
            // $bytebuffer feature_table, 
            $buffer<int> level_offsets
        ) {
            // ngp_encode_kernel_impl(batch_size, input_dim(), input, output, feature_table, level_offsets);
            switch (input_dim()) {
                case 1: ngp_encode_kernel_impl<1>(batch_size, input, output, feature_table, level_offsets); break;
                case 2: ngp_encode_kernel_impl<2>(batch_size, input, output, feature_table, level_offsets); break;
                case 3: ngp_encode_kernel_impl<3>(batch_size, input, output, feature_table, level_offsets); break;
                default: {
                    print("error ngp encode input dim: {}\n", input_dim());
                    exit(1);
                }
            }
            // switch (input_dim()) {
            //     case 1: ngp_encode_kernel_impl1<1>(batch_size, input, output, feature_table, level_offsets); break;
            //     case 2: ngp_encode_kernel_impl1<2>(batch_size, input, output, feature_table, level_offsets); break;
            //     case 3: ngp_encode_kernel_impl1<3>(batch_size, input, output, feature_table, level_offsets); break;
            //     default: {
            //         print("error ngp encode input dim: {}\n", input_dim());
            //         exit(0);
            //     }
            // }
        };
        ngp_encode_shader[input_dim()] = global::device().compile(ngp_encode_kernel);
    }
    const uint batch_size = input.size()*4 / input_dim();
    global::cmd_list() << ngp_encode_shader[input_dim()](batch_size, input, output, _feature_table, level_offset_buffer).dispatch(batch_size/4, levels);
}

Shader1D<Buffer<half2>> init_feature_shader;
// Shader1D<ByteBuffer> init_feature_shader;

void NGPEncodeLayer::reset_parameters() {
    if (!init_feature_shader) {
        Kernel1D init_feature_kernel = []($buffer<half2> feature_table) {
        // Kernel1D init_feature_kernel = []($bytebuffer feature_table) {
            set_block_size(256);
            $uint tid = $dispatch_x;
            $half2 f;
            $uint2 s = tea(tid, 233);
            f.x = 1e-4f * as_uniform(s.x);
            f.y = 1e-4f * as_uniform(s.y);
            feature_table.write(tid, f);
            // feature_table.write(tid * (uint)sizeof(half2), f);
        };
        init_feature_shader = global::device().compile(init_feature_kernel);
    }
    global::cmd_list() << init_feature_shader(_feature_table).dispatch(_table_size);
}

template<int input_dim>
void ngp_calc_gradient(
    $uint &batch_size, 
    // int input_dim, 
    $buffer<half4> &input, 
    $buffer<half4> &output_grad, 
    $buffer<half2> &feature_grad, 
    $buffer<int> &level_offsets
) {
    set_block_size(256, 1);
    $uint tid = $dispatch_x;
    $uint level = $dispatch_y;
    $uint level_offset = level_offsets.read(level);
    $uint level_size = level_offsets.read(level + 1) - level_offset;

    $uint grid_res = (1u << level) * N_min;
    $bool use_hash = (pow(grid_res.cast<float>(), (float)input_dim) > level_size + 50) | (powi(grid_res, input_dim) > level_size);

    $array<float4, input_dim> in;
    $array<float4, input_dim> pos;
    $array<uint4, input_dim> grid_idx;

    // for (int i = 0; i < input_dim; i++) {
    $for (i, input_dim) {
        in[i] = input.read(tid + i*batch_size/4);
        in[i] = fract(fract(in[i]) + 1.f);
    };
    // for (int i = 0; i < input_dim; i++) {
    $for (i, input_dim) {
        $float4 tmp = in[i] * (grid_res-1).cast<float>();
        grid_idx[i] = floor(tmp);
        pos[i] = fract(tmp);
    };

    constexpr const uint num_item = (1 << input_dim) * 4;
    $array<uint, num_item> table_idx;

    calc_table_idx<input_dim, num_item>(table_idx, grid_idx, grid_res, level_size, use_hash);

    // for (int i = 0; i < num_item; i++) {
    $for (i, num_item) {
        table_idx[i] = (table_idx[i] << 5) | i;
    };

    // odd even sort
    if (input_dim == 3) {
        sort_idx<num_item>(table_idx);
    }

    $float4 g_out[2];
    g_out[0] = output_grad.read(tid + level*2*batch_size/4);
    g_out[1] = output_grad.read(tid + (level*2+1)*batch_size/4);

    $for (ti, num_item) {
        // $float2 v = feature_table.read(level_offset + (table_idx[ti] >> 5));
        $uint iid = table_idx[ti] & 31;

        $uint s = iid & ((1 << input_dim) - 1);
        $uint t = iid >> input_dim;

        $float w = 1;
        for (int d = 0; d < input_dim; d++) {
            $int f = (s & (1 << d)) == 0;
            w *= f + (1 - f*2) * pos[d][t];
            // if (f) w *= 1 - pos[d][t];
            // else w *= pos[d][t];
        }
        $half2 g;
        g[0] = w * g_out[0][t];
        g[1] = w * g_out[1][t];

        feature_grad.write(level_offset + (table_idx[ti] >> 5), g);
    };



    // $float4 in[4];
    // $float4 pos[4];
    // $uint4 grid_idx[4];

    // for (int i = 0; i < input_dim; i++) {
    //     in[i] = input.read(tid + i*batch_size/4);
    // }
    // trans_input(in, input_dim);
    // for (int i = 0; i < 4; i++) {
    //     calc_pos(in[i], pos[i], grid_idx[i], grid_res);
    // }

    // $half4 g_out[2];
    // g_out[0] = output_grad.read(tid + level*2*batch_size/4);
    // g_out[1] = output_grad.read(tid + (level*2+1)*batch_size/4);

    // $float w;
    // $float2 g;
    // $half2 tg;
    // $uint4 idx;
    // for (int i = 0; i < 4; i++) {
    //     g[0] = g_out[0][i];
    //     g[1] = g_out[1][i];
    //     for (int t = 0; t < powi(2, input_dim); t++) {
    //         w = 1;
    //         for (int d = 0; d < input_dim; d++) {
    //             if ((t & (1 << d)) == 0) {
    //                 w *= 1 - pos[i][d];
    //                 idx[d] = grid_idx[i][d];
    //             }
    //             else {
    //                 w *= pos[i][d];
    //                 idx[d] = grid_idx[i][d] + 1;
    //             }
    //         }
    //         // tg = (w*batch_size)*g;
    //         tg = max(batch_size.cast<float>() / level_size, 1.0f) * w * g;
    //         feature_grad.write(level_offset + table_idx(input_dim, idx, grid_res, level_size, use_hash), tg);
    //     }
    // }
}

Shader2D<uint, Buffer<half4>, Buffer<half4>, Buffer<half2>, Buffer<int>> ngp_calc_gradient_shader[5];
Shader1D<Buffer<half2>> clear_grad_shader;

void NGPEncodeLayer::backward(
    const BufferView<half4> fwd_input,
    const BufferView<half4> fwd_output,
    BufferView<half4> output_grad,
    BufferView<half4> input_grad,
    BufferView<half4> arena
) {
    if (!ngp_calc_gradient_shader[input_dim()]) {
        Kernel2D ngp_calc_gradient_kernel = [&](
            $uint batch_size,
            $buffer<half4> input, 
            $buffer<half4> output_grad, 
            $buffer<half2> feature_grad, 
            $buffer<int> level_offsets
        ) {
            // ngp_calc_gradient(batch_size, input_dim(), input, output_grad, feature_grad, level_offsets);
            switch (input_dim()) {
                case 1: ngp_calc_gradient<1>(batch_size, input, output_grad, feature_grad, level_offsets); break;
                case 2: ngp_calc_gradient<2>(batch_size, input, output_grad, feature_grad, level_offsets); break;
                case 3: ngp_calc_gradient<3>(batch_size, input, output_grad, feature_grad, level_offsets); break;
                default: {
                    print("error ngp encode input dim: {}\n", input_dim());
                    exit(1);
                }
            }
        };
        ngp_calc_gradient_shader[input_dim()] = global::device().compile(ngp_calc_gradient_kernel);
    }
    if (!clear_grad_shader) {
        Kernel1D clear_grad_kernel = []($buffer<half2> feature_grad) {
            set_block_size(256);
            $uint tid = $dispatch_x;
            feature_grad.write(tid, $half2(0, 0));
        };
        clear_grad_shader = global::device().compile(clear_grad_kernel);
    }
    const uint batch_size = output_grad.size()*4 / output_dim();
    
    global::cmd_list()
        << clear_grad_shader(_feature_gradient).dispatch(_feature_gradient.size())
        << ngp_calc_gradient_shader[input_dim()](batch_size, fwd_input, output_grad, _feature_gradient, level_offset_buffer).dispatch(batch_size/4, levels);
}

void NGPEncodeLayer::optimize() {
    optim.optimize(_feature_table.view().as<half4>(), _feature_gradient.view().as<half4>());
}