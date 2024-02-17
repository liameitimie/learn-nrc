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

void unpack_half($half x[8], $uint4 pack) {
    for (int i = 0; i < 4; i++) {
        x[i * 2] = (pack[i] & ((1 << 16) - 1)).as<half>();
        x[i * 2 + 1] = (pack[i] >> 16).as<half>();
    }
}
$uint4 pack_half($half x[8]) {
    $uint4 pack;
    for (int i = 0; i < 4; i++) {
        pack[i] = x[i * 2].as<ushort>().cast<uint>()
            + (x[i * 2 + 1].as<ushort>().cast<uint>() << 16);
    }
    return pack;
}

int main(int argc, char** argv) {
    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    Clock timer;

    const uint dim_x = 32;
    const uint dim_y = 1920 * 1080;
    const uint dim_k = 32;

    const uint block_tile_x = 32;
    const uint block_tile_y = 128;
    const uint block_tile_k = 32;

    const uint line_items_a = block_tile_x / 8;
    const uint line_items_b = block_tile_y / 8;

    const uint block_size = (block_tile_x / 32) * (block_tile_y / 64) * 32;

    Kernel1D kernel = [&]($buffer<uint4> a, $buffer<uint4> b, $buffer<uint4> d) {
        set_block_size(block_size);

        $shared<uint4> a_tile{line_items_a * block_tile_k};
        $shared<uint4> b_tile{(line_items_b + 1) * block_tile_k};

        $half a_frag[8];
        $half b_frag[8];
        $half acc[8][8];

        $uint tid = $thread_x;
        $uint bid = $block_x;

        $uint x_ofs = tid % 4;
        $uint y_ofs = tid / 4;

        $for (k, 0u, dim_k, block_tile_k) {
            sync_block();

            for (int i = 0; i < line_items_a * block_tile_k; i += block_size) {
                if (line_items_a * block_tile_k - i < block_size) {
                    $if (tid < line_items_a * block_tile_k - i) {
                        a_tile[tid + i] = a.read((tid % line_items_a) + ((tid + i) / line_items_a + k) * (dim_x / 8));
                    };
                }
                else {
                    a_tile[tid + i] = a.read((tid % line_items_a) + ((tid + i) / line_items_a + k) * (dim_x / 8));
                }
            }
            for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
                b_tile[(tid % line_items_b) + ((tid + i) / line_items_b) * (line_items_b + 1)] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b + k) * (dim_y / 8));
            }
            // $if (tid < line_items_a * block_tile_k) {
            //     a_tile[tid] = a.read((tid % line_items_a) + (tid / line_items_a + k) * (dim_x / 8));
            // };
            // b_tile[tid] = b.read((tid % line_items_b) + bid * line_items_b + (tid / line_items_b + k) * (dim_y / 8));

            sync_block();

            for (int kk = 0; kk < block_tile_k; kk++) {
            // $for (kk, block_tile_k) {
                $uint4 a_packfrag = a_tile[x_ofs + kk * line_items_a];
                $uint4 b_packfrag = b_tile[y_ofs + kk * (line_items_b + 1)];
                unpack_half(a_frag, a_packfrag);
                unpack_half(b_frag, b_packfrag);

                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        acc[x][y] += a_frag[x] * b_frag[y];
                    }
                }
            };
        };

        // sync_block();
        // for (int i = 0; i < 8; i++) {
        //     $uint4 pack = pack_half(acc[i]);
        //     b_tile[y_ofs + (i + x_ofs * 8) * line_items_b] = pack;
        // }
        // sync_block();

        // for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
        //     // d.write((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8), b_tile[(tid % line_items_b) + ((tid + i) / line_items_b) * (line_items_b + 1)]);
        //     d.write((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8), b_tile[tid + i]);
        // }

        for (int i = 0; i < 8; i++) {
            $uint4 pack = pack_half(acc[i]);
            d.write(y_ofs + bid * line_items_b + (i + x_ofs * 8) * (dim_y / 8), pack);
        }
    };

    // assert block_tile_k == dim_k == 32
    
    Kernel1D kernel1 = [&]($buffer<uint4> a, $buffer<uint4> b, $buffer<uint4> d, $uint a_offset) {
        set_block_size(block_size);

        $shared<uint4> a_tile{line_items_a * block_tile_k};
        $shared<uint4> b_tile{(line_items_b + 1) * block_tile_k};
        // $shared<uint4> b_tile{line_items_b * block_tile_k};
        // $shared<uint> b_tile{(line_items_b * 4 + 4) * block_tile_k}; // 每32 uint补一位padding

        $half a_frag[8];
        $half b_frag[8];
        $half acc[8][8];

        $uint tid = $thread_x;
        $uint bid = $block_x;

        $uint x_ofs = tid % 4;
        $uint y_ofs = tid / 4;

        for (int i = 0; i < line_items_a * block_tile_k; i += block_size) {
            a_tile[tid + i] = a.read((tid % line_items_a) + ((tid + i) / line_items_a) * (dim_x / 8) + a_offset);
        }
        for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
            b_tile[(tid % line_items_b) + ((tid + i) / line_items_b) * (line_items_b + 1)] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
            // b_tile[tid + i] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
            // $uint4 t = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
            // $uint unpad_idx = (tid + i) * 4; // 未添加padding的下标
            // for (int j = 0; j < 4; j++) {
            //     b_tile[unpad_idx % 32 + unpad_idx / 32 * 33 + j] = t[j];
            // }
        }
        sync_block();
        $for (kk, block_tile_k) {
            $uint4 a_packfrag = a_tile[x_ofs + kk * line_items_a];
            $uint4 b_packfrag = b_tile[y_ofs + kk * (line_items_b + 1)];
            // $uint4 b_packfrag = b_tile[y_ofs + kk * line_items_b];
            // $uint unpad_idx = (y_ofs + kk * line_items_b) * 4;
            // $uint4 b_packfrag;
            // for (int j = 0; j < 4; j++) {
            //     b_packfrag[j] = b_tile[unpad_idx % 32 + unpad_idx / 32 * 33 + j];
            // }

            unpack_half(a_frag, a_packfrag);
            unpack_half(b_frag, b_packfrag);

            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    acc[x][y] += a_frag[x] * b_frag[y];
                }
            }
        };
        // $shared<uint4> c_tile{block_size};
        for (int i = 0; i < 8; i++) {
            $uint4 pack = pack_half(acc[i]);
            d.write(y_ofs + bid * line_items_b + (i + x_ofs * 8) * (dim_y / 8), pack);
        }
    };

    const uint size_a = dim_x * dim_k;
    const uint num_a = 3;

    // 预取下次所需数据（负优化）
    Kernel1D kernel2 = [&]($buffer<uint4> a, $buffer<uint4> b, $buffer<uint4> d) {
        set_block_size(block_size);

        $shared<uint4> a_tile{line_items_a * block_tile_k * 2};
        $shared<uint4> b_tile{line_items_b * block_tile_k * 2};

        $half a_frag[8];
        $half b_frag[8];
        $half acc[8][8];

        $uint tid = $thread_x;
        $uint bid = $block_x;

        $uint x_ofs = tid % 4;
        $uint y_ofs = tid / 4;

        for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
            // b_tile[(tid % line_items_b) + ((tid + i) / line_items_b) * (line_items_b + 1)] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
            b_tile[tid + i] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
        }
        for (int i = 0; i < line_items_a * block_tile_k; i += block_size) {
            a_tile[tid + i] = a.read((tid % line_items_a) + ((tid + i) / line_items_a) * (dim_x / 8));
        }

        $for (t, num_a) {

            sync_block();

            $if (t < num_a - 1) {
                // 下次循环所需a矩阵
                for (int i = 0; i < line_items_a * block_tile_k; i += block_size) {
                    a_tile[tid + i + (~t & 1) * (line_items_a * block_tile_k)] = a.read((tid % line_items_a) + ((tid + i) / line_items_a) * (dim_x / 8) + size_a / 8 * (t + 1));
                }
            };

            for (int kk = 0; kk < block_tile_k; kk++) {
            // $for (kk, block_tile_k) {
                $uint4 a_packfrag = a_tile[x_ofs + kk * line_items_a + (t & 1) * (line_items_a * block_tile_k)];
                // $uint4 b_packfrag = b_tile[y_ofs + kk * (line_items_b + 1)];
                $uint4 b_packfrag = b_tile[y_ofs + kk * line_items_b + (t & 1) * (line_items_b * block_tile_k)];

                unpack_half(a_frag, a_packfrag);
                unpack_half(b_frag, b_packfrag);

                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        acc[x][y] += a_frag[x] * b_frag[y];
                    }
                }
            };
            // sync_block();

            $if (t < num_a - 1) {
                for (int i = 0; i < 8; i++) {
                    $uint4 pack = pack_half(acc[i]);
                    // b_tile[y_ofs + (i + x_ofs * 8) * (line_items_b + 1)] = pack;
                    b_tile[y_ofs + (i + x_ofs * 8) * line_items_b + (~t & 1) * (line_items_b * block_tile_k)] = pack;
                    for (int j = 0; j < 8; j++) acc[i][j] = 0;
                }
            };
        };
        // sync_block();
        // for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
        //     d.write((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8), b_tile[tid + i]);
        // }

        for (int i = 0; i < 8; i++) {
            $uint4 pack = pack_half(acc[i]);

            sync_block();
            b_tile[tid] = pack;
            sync_block();

            d.write(y_ofs + bid * line_items_b + (i + x_ofs * 8) * (dim_y / 8), b_tile[tid]);
        }
    };

    Kernel1D kernel3 = [&]($buffer<uint4> a, $buffer<uint4> b, $buffer<uint4> d) {
        set_block_size(block_size);

        $shared<uint4> a_tile{line_items_a * block_tile_k * num_a};
        $shared<uint4> b_tile{line_items_b * block_tile_k};
        // $shared<uint4> b_tile{(line_items_b + 1) * block_tile_k};

        $half a_frag[8];
        $half b_frag[8];
        $half acc[8][8];

        $uint tid = $thread_x;
        $uint bid = $block_x;

        $uint x_ofs = tid % 4;
        $uint y_ofs = tid / 4;

        for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
            // b_tile[(tid % line_items_b) + ((tid + i) / line_items_b) * (line_items_b + 1)] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
            b_tile[tid + i] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
        }
        for (int t = 0; t < num_a; t++) {
            for (int i = 0; i < line_items_a * block_tile_k; i += block_size) {
                a_tile[tid + i + t * line_items_a * block_tile_k] = a.read((tid % line_items_a) + ((tid + i) / line_items_a) * (dim_x / 8) + size_a / 8 * t);
            }
        }
        // sync_block();

        // for (int t = 0; t < num_a; t++) {
        $for (t, num_a) {
            // for (int i = 0; i < line_items_a * block_tile_k; i += block_size) {
            //     a_tile[tid + i] = a.read((tid % line_items_a) + ((tid + i) / line_items_a) * (dim_x / 8) + size_a / 8 * t);
            // }
            sync_block();
            for (int kk = 0; kk < block_tile_k; kk++) {
            // $for (kk, block_tile_k) {
                $uint4 a_packfrag = a_tile[x_ofs + kk * line_items_a + t * line_items_a * block_tile_k];
                // $uint4 b_packfrag = b_tile[y_ofs + kk * (line_items_b + 1)];
                $uint4 b_packfrag = b_tile[y_ofs + kk * line_items_b];

                unpack_half(a_frag, a_packfrag);
                unpack_half(b_frag, b_packfrag);

                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        acc[x][y] += a_frag[x] * b_frag[y];
                    }
                }
            };
            // sync_block();

            // $if (t < num_a - 1) {
                for (int i = 0; i < 8; i++) {
                    $uint4 pack = pack_half(acc[i]);
                    // b_tile[y_ofs + (i + x_ofs * 8) * (line_items_b + 1)] = pack;
                    b_tile[y_ofs + (i + x_ofs * 8) * line_items_b] = pack;
                    for (int j = 0; j < 8; j++) acc[i][j] = 0;
                }
            // };
        };

        sync_block();
        for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
            // d.write((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8), b_tile[(tid % line_items_b) + ((tid + i) / line_items_b) * (line_items_b + 1)]);
            d.write((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8), b_tile[tid + i]);
        }

        // for (int i = 0; i < 8; i++) {
        //     $uint4 pack = pack_half(acc[i]);
        //     d.write(y_ofs + bid * line_items_b + (i + x_ofs * 8) * (dim_y / 8), pack);
        // }
    };

    Kernel1D kernel4 = [&]($buffer<uint4> a, $buffer<uint4> b, $buffer<uint4> d) {
        set_block_size(block_size);

        $shared<uint4> smem_a{line_items_a * block_tile_k * num_a};
        $shared<uint4> smem_b{line_items_b * block_tile_k};

        $uint tid = $thread_x;
        $uint bid = $block_x;

        $uint x_ofs = tid % 4;
        $uint y_ofs = tid / 4;

        $uint4 reg_a_pack[2]; // 对shared预取
        $uint4 reg_b_pack[2];

        $half reg_a[8];
        $half reg_b[8];
        $half acc[8][8];

        for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
            smem_b[tid + i] = b.read((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8));
        }
        for (int t = 0; t < num_a; t++) {
            for (int i = 0; i < line_items_a * block_tile_k; i += block_size) {
                smem_a[tid + i + t * line_items_a * block_tile_k] = a.read((tid % line_items_a) + ((tid + i) / line_items_a) * (dim_x / 8) + size_a / 8 * t);
            }
        }
        sync_block();

        reg_a_pack[0] = smem_a[x_ofs];
        reg_b_pack[0] = smem_b[y_ofs];
        
        $for (t, num_a) {
            for (int kk = 0; kk < block_tile_k - 1; kk++) {
                reg_a_pack[~kk & 1] = smem_a[x_ofs + (kk + 1) * line_items_a + t * line_items_a * block_tile_k];
                reg_b_pack[~kk & 1] = smem_b[y_ofs + (kk + 1) * line_items_b];
                
                unpack_half(reg_a, reg_a_pack[kk & 1]);
                unpack_half(reg_b, reg_b_pack[kk & 1]);
                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        acc[x][y] += reg_a[x] * reg_b[y];
                    }
                }
            };

            unpack_half(reg_a, reg_a_pack[1]);
            unpack_half(reg_b, reg_b_pack[1]);
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    acc[x][y] += reg_a[x] * reg_b[y];
                }
            }

            $if (t < num_a - 1) {
                for (int i = 0; i < 8; i++) {
                    $uint4 pack = pack_half(acc[i]);
                    smem_b[y_ofs + (i + x_ofs * 8) * line_items_b] = pack;
                    for (int j = 0; j < 8; j++) acc[i][j] = 0;
                }
                sync_block();
                
                reg_a_pack[0] = smem_a[x_ofs + (t + 1) * line_items_a * block_tile_k];
                reg_b_pack[0] = smem_b[y_ofs];
            };
        };

        // for (int i = 0; i < line_items_b * block_tile_k; i += block_size) {
        //     d.write((tid % line_items_b) + bid * line_items_b + ((tid + i) / line_items_b) * (dim_y / 8), smem_b[tid + i]);
        // }

        for (int i = 0; i < 8; i++) {
            $uint4 pack = pack_half(acc[i]);
            d.write(y_ofs + bid * line_items_b + (i + x_ofs * 8) * (dim_y / 8), pack);
        }
    };


    auto a = device.create_buffer<uint4>(size_a * num_a / 8); // num_a 个矩阵
    auto b = device.create_buffer<uint4>(dim_k * dim_y / 8);
    auto d = device.create_buffer<uint4>(dim_x * dim_y / 8);

    vector<half> a_h(size_a * num_a); // num_a 个矩阵
    vector<half> b_h(dim_k * dim_y);
    vector<float> d_h[num_a];
    vector<half> d_d(dim_x * dim_y);

    for (auto &v: d_h) v.resize(dim_x * dim_y);

    for (auto &x : a_h) x = pcg32::next_float() - 0.5;
    for (auto &x : b_h) x = pcg32::next_float() - 0.5;

    timer.tic();
    for (int x = 0; x < dim_x; x++) {
        for (int k = 0; k < dim_k; k++) {
            float tmp = a_h[x + k * dim_x];
            for (int y = 0; y < dim_y; y++) {
                d_h[0][y + x * dim_y] += tmp * (float)b_h[y + k * dim_y];
            }
        }
    }
    for (int t = 1; t < num_a; t++) {
        for (auto &x: d_h[t & 1]) x = 0;
        for (int x = 0; x < dim_x; x++) {
            for (int k = 0; k < dim_k; k++) {
                float tmp = a_h[x + k * dim_x + t * size_a];
                for (int y = 0; y < dim_y; y++) {
                    d_h[t & 1][y + x * dim_y] += tmp * (float)d_h[~t & 1][y + k * dim_y];
                }
            }
        }
    }
    // for (int i = 0; i < dim_x * dim_y; i++) {
    //     d_h[i] = b_h[i];
    // }
    print("calc ref res: {}\n", timer.toc());

    timer.tic();
    // auto shader = device.compile(kernel);
    // auto shader = device.compile(kernel1);
    // auto shader = device.compile(kernel2);
    auto shader = device.compile(kernel3);
    // auto shader = device.compile(kernel4);
    print("compiled shader: {}\n", timer.toc());

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    for (int i = 0; i < 10; i++) {
        CommandList cmd_list;
        for (int j = 0; j < 100; j++) {
            // cmd_list << shader(a, b, d, 0).dispatch(dim_y / 2);
            // for (int t = 1; t < num_a; t++) {
            //     cmd_list << shader(a, d, d, size_a / 8 * t).dispatch(dim_y / 2);
            // }
            cmd_list << shader(a, b, d).dispatch(dim_y / 2);
            // cmd_list << d.copy_from(b); // 测试带宽
        }
        timer.tic();
        stream << cmd_list.commit();
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << d.copy_to(d_d.data()) << synchronize();

    print_vec(d_h[~num_a & 1], "d_h", 32);
    print_vec(d_d, "d_d", 32);

    compare_vec(d_h[~num_a & 1], d_d);
    return 0;
}