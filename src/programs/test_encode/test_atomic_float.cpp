#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

$float as_uniform($uint x) {
    return ((x >> 9) | 0x3f800000u).as<float>() - 1.0f;
}
float as_uniform(uint x) {
    union {
        float f;
        uint u;
    } t;
    t.u = (x >> 9) | 0x3f800000u;

    return t.f - 1;
}

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
    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    const int threads = 1 << 20;
    const int n = 1 << 22;

    Kernel1D kernel = [&]($buffer<float> buf) {
        $uint tid = $dispatch_x;

        for (int i = 0; i < 4; i++) {
            $uint idx = tea(tid, i + 233).x % n;
            $float v = as_uniform(tea(tid, i + 19).x);

            buf.atomic(idx).fetch_add(v);
            // buf.write(idx, v);
        }
    };
    Kernel1D clear_kernel = [&]($buffer<float> buf) {
        $uint tid = $dispatch_x;
        buf.write(tid, 0.f);
    };

    auto shader = device.compile(kernel);
    auto clear = device.compile(clear_kernel);

    auto out = device.create_buffer<float>(n);

    vector<float> out_h(n);
    vector<float> out_d(n);

    for (int tid = 0; tid < threads; tid++) {
        for (int i = 0; i < 4; i++) {
            uint idx = tea(tid, i + 233).x % n;
            float v = as_uniform(tea(tid, i + 19).x);
            out_h[idx] += v;
        }
    }

    Clock timer;

    for (int i = 0; i < 10; i++) {
        CommandList cmd_list;
        for (int j = 0; j < 100; j++) {
            cmd_list << clear(out).dispatch(n) << shader(out).dispatch(threads);
        }
        timer.tic();
        stream << cmd_list.commit() << synchronize();
        print("{}\n", timer.toc());
    }

    stream << out.copy_to(out_d.data()) << synchronize();

    print_vec(out_h, "out_h", 32);
    print_vec(out_d, "out_d", 32);

    compare_vec(out_h, out_d);

    return 0;
}