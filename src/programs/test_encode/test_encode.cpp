#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <frequency_encode_layer.h>
#include <ngp_encode_layer.h>
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
            print("inc error {}: {}, {}\n", i, t1, t2);
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

int main(int argc, char** argv) {
    log_level_verbose();
    global::init(argv[0]);
    
    Clock timer;

    const uint input_dim = 3;
    const uint batch_size = 1920*1080;

    NGPEncodeLayer layer(input_dim, 32);

    auto f_in = global::device().create_buffer<half4>(layer.input_dim() * batch_size / 4);
    auto f_out = global::device().create_buffer<half4>(layer.output_dim() * batch_size / 4);

    vector<half> f_in_h(layer.input_dim() * batch_size);
    vector<float> f_out_h(layer.output_dim() * batch_size);
    vector<half> f_out_buffer(layer.output_dim() * batch_size);

    vector<half2> feature_table(layer.table_size());

    for (auto &x : f_in_h) x = pcg32::next_float() - 0.5;

    global::stream()
        << global::cmd_list().commit()
        << f_in.copy_from(f_in_h.data())
        << layer.feature_table().copy_to(feature_table.data())
        << synchronize();

    // for (int i = 0; i < 32; i++) {
    //     print("{}\n", feature_table[i]);
    // }

    print("calc ref result:\n");
    timer.tic();

    auto powll = [](slong x, int p) {
        slong res = 1;
        for (int i = 0; i < p; i++) {
            res *= x;
        }
        return res;
    };
    uint prime[4] = {1u, 2654435761u, 805459861u, 3674653429u};
    auto table_idx = [&](uint *grid_idx, uint grid_res, uint level_size, bool use_hash) -> uint {
        if (use_hash) {
            uint idx = 0;
            for (int i = 0; i < input_dim; i++) {
                idx ^= grid_idx[i] * prime[i];
            }
            return idx % level_size;
        }
        else {
            ulong idx = 0;
            ulong s = 1;
            for (int i = 0; i < input_dim; i++) {
                idx += grid_idx[i] * s;
                s *= grid_res;
            }
            return idx;
        }
    };

    for (int level = 0; level < 16; level++) {
        int level_offset = layer.level_offsets()[level];
        int level_size = layer.level_offsets()[level + 1] - level_offset;
        int grid_res = (1u << level) * 16;
        bool use_hash = powll(grid_res, input_dim) > level_size;

        // print("level{}: offset: {}, size: {}, grid_res: {}, use_hash: {}\n", level, level_offset, level_size, grid_res, use_hash);

        for (int i = 0; i < batch_size; i++) {
            float in[input_dim];
            float pos[input_dim];
            int grid_idx[input_dim];

            for (int j = 0; j < input_dim; j++) {
                in[j] = f_in_h[i + j*batch_size];
                in[j] = fract(fract(in[j]) + 1.f);

                float tmp = in[j] * (grid_res - 1);
                grid_idx[j] = tmp;
                pos[j] = tmp - grid_idx[j];
            }
            
            float2 feature;

            for (int t = 0; t < powll(2, input_dim); t++) {
                float w = 1;
                uint idx[input_dim];
                for (int d = 0; d < input_dim; d++) {
                    if ((t & (1 << d)) == 0) {
                        w *= 1 - pos[d];
                        idx[d] = grid_idx[d];
                    }
                    else {
                        w *= pos[d];
                        idx[d] = grid_idx[d] + 1;
                    }
                }
                half2 f = feature_table[level_offset + table_idx(idx, grid_res, level_size, use_hash)];

                feature[0] += w * (float)f[0];
                feature[1] += w * (float)f[1];
            }
            f_out_h[i + level*2*batch_size] = feature[0];
            f_out_h[i + (level*2+1)*batch_size] = feature[1];
        }
    }
    print("{}\n\n", timer.toc());

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 100; j++) {
            layer.forward(f_in, f_out);
        }
        timer.tic();
        global::stream() << global::cmd_list().commit();
        global::stream().synchronize();
        print("{}\n", timer.toc());
    }

    global::stream() << f_out.copy_to(f_out_buffer.data()) << synchronize();

    print_vec(f_out_h, "out_h", 32);
    print_vec(f_out_buffer, "out_d", 32);

    compare_vec(f_out_h, f_out_buffer);

    return 0;
}