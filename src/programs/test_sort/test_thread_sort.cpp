#include <luisa/luisa-compute.h>
#include <algorithm>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

$uint2 tea($uint v0, $uint v1) {
    $uint s0 = 0;
    for (int i = 0; i < 8; i++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return $uint2(v0, v1);
};

uint2 tea(uint v0, uint v1) {
    uint s0 = 0;
    for (int i = 0; i < 8; i++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return uint2(v0, v1);
};

void swap($uint &v1, $uint &v2) {
    $uint tmp = v1;
    v1 = v2;
    v2 = tmp;
}

struct Cluster {
    static const int max_vertices = 64;
    static const int max_triangles = 124;

    luisa::vector<luisa::float4> positions;
    luisa::vector<luisa::float2> texcoords;
    luisa::vector<int> indices;

    // the index to group buffer in VirtualMesh
    int group;

    Cluster() = default;
    Cluster(Cluster &) = default;
    Cluster(Cluster &&) = default;
    Cluster& operator=(Cluster &rhs) = default;
    Cluster& operator=(Cluster &&rhs) = default;
};

void print_cluster(Cluster &cluster) {
    print("{}, {}, {}, {}\n", cluster.positions.size(), cluster.texcoords.size(), cluster.indices.size(), cluster.group);
}

template<>
struct equal_to<float4> {
    bool operator()(const float4 &a, const float4 &b) const {
        return memcmp(&a, &b, sizeof(float4)) == 0;
    }
};

struct alignas(16) Node {
    float val;
    uint ls;
    uint rs;
};

LUISA_STRUCT(Node, val, ls, rs) {};

int main(int argc, char** argv) {

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    // auto buffer = device.create_buffer<uint2>(6);

    // Kernel1D kernel = [&]() {
    //     buffer->write(1, make_uint2(~0u));
    //     // buffer->atomic(1).val.fetch_add(1.0f);
    // };

    // auto shader = device.compile(kernel);

    // stream << shader().dispatch(1) << synchronize();



    const int threads = 1 << 17;
    const int n = 32;

    Kernel1D test_thread_sort = [&]($buffer<uint> output) {
        $uint tid = $dispatch_x;

        $array<uint, n> vals;

        $for (i, n) {
            vals[i] = tea(i + tid * n, 233).x;
        };

        // odd even sort
        for (int i = 0; i < n; i++) {
            for (int j = i & 1; j < n - 1; j += 2) {
                $if (vals[j + 1] < vals[j]) {
                    swap(vals[j], vals[j + 1]);
                };
            }
        }
        // $for (i, n) {
        //     $for (j, i & 1, n - 1, 2) {
        //         $if (vals[j + 1] < vals[j]) {
        //             swap(vals[j], vals[j + 1]);
        //         };
        //     };
        // };
        // $for (i, n) {
        //     for (int t = 0; t < n / 2 - 1; t++) {
        //         $uint j = (i & 1) + t * 2;
        //         $if (vals[j + 1] < vals[j]) {
        //             swap(vals[j], vals[j + 1]);
        //         };
        //     }
        //     if (n % 2 == 0) {
        //         $if ((i & 1) == 0 & vals[n - 1] < vals[n - 2]) {
        //             swap(vals[n - 2], vals[n - 1]);
        //         };
        //     } else {
        //         $uint j = (i & 1) + (n / 2 - 1) * 2;
        //         $if (vals[j + 1] < vals[j]) {
        //             swap(vals[j], vals[j + 1]);
        //         };
        //     }
        // };

        $for (i, n) {
            output.write(i + tid * n, vals[i]);
        };
    };

    auto shader = device.compile(test_thread_sort);

    auto out = device.create_buffer<uint>(threads * n);

    vector<uint> out_h(threads * n);
    vector<uint> out_d(threads * n);

    for (int i = 0; i < threads * n; i++) {
        out_h[i] = tea(i, 233).x;
    }
    for (int i = 0; i < threads * n; i += n) {
        std::sort(out_h.data() + i, out_h.data() + i + n);
    }

    Clock timer;

    for (int i = 0; i < 10; i++) {
        CommandList cmd_list;
        for (int j = 0; j < 100; j++) {
            cmd_list << shader(out).dispatch(threads);
        }
        timer.tic();
        stream << cmd_list.commit() << synchronize();
        print("{}\n", timer.toc());
    }

    // stream << shader(out).dispatch(len / n) << synchronize();

    stream << out.copy_to(out_d.data()) << synchronize();

    for (int i = 0; i < threads * n; i++) {
        uint h = out_h[i];
        uint d = out_d[i];

        if (i < 32) {
            print("{}: {}, {}\n", i, h, d);
        }

        if (h != d) {
            print("error!\n");
            break;
        }
    }
    print("ok\n");

    return 0;
}