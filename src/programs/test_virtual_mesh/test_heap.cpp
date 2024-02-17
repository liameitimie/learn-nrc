#include <index_heap.h>
#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl.h>
#include <unordered_set>

using namespace luisa;
using namespace fmt;

template<>
struct equal_to<float4> {
    bool operator()(const float4 &a, const float4 &b) const {
        // return memcmp(&a, &b, sizeof(float4)) == 0;
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }
};

template<>
struct less<float4> {
    bool operator()(const float4 &a, const float4 &b) const {
        return memcmp(&a, &b, sizeof(float4)) < 0;
    }
};


int main() {
    srand(time(0));
    const int n = 2002154;
    vector<double> res(n);

    Clock timer;

    // std::unordered_set<float4, hash<float4>, equal_to<float4>> mp;
    unordered_set<float4, hash<float4>, equal_to<float4>> mp;
    // set<float4, less<float4>> mp;

    float3 x;

    timer.tic();
    for (int i = 0; i < n; i++) {
        mp.insert(float4(rand(), rand(), rand(), rand()));
    }
    print("{}\n", timer.toc());
    

    // timer.tic();
    // IndexHeap h(n);
    // print("{}\n", timer.toc());

    // // timer.tic();
    // // for (int i = 0; i < n; i++) {
    // //     h.update_unadjust(i, rand());
    // // }
    // // h.build_heap();
    // // fmt::print("{}\n", timer.toc());

    // timer.tic();
    // for (int i = 0; i < n; i++) {
    //     h.update(i, rand() | (int(rand()) << 16));
    // }
    // print("{}\n", timer.toc());

    // timer.tic();
    // for (int i = 0; i < n; i++) {
    //     // res[i] = h.keys()[h.top()];
    //     // res[i] = h.top();
    //     // h.pop();

    //     auto [key, idx] = h.pop();
    //     res[i] = key;

    //     if (i < 32) {
    //         print("{} ", key);
    //     }
    // }
    // print("\n");
    // print("{}\n", timer.toc());

    // // timer.tic();
    // // priority_queue<double, vector<double>, greater<double>> h1;
    // // h1.reserve(n);
    // // print("{}\n", timer.toc());

    // // timer.tic();
    // // for (int i = 0; i < n; i++) {
    // //     h1.push(rand());
    // // }
    // // print("{}\n", timer.toc());

    // // timer.tic();
    // // for (int i = 0; i < n; i++) {
    // //     h1.pop(res[i]);
    // // }
    // // print("{}\n", timer.toc());

    // for (int i = 1; i < n; i++) {
    //     if (res[i] < res[i - 1]) {
    //         print("error: {}\n", i);
    //         exit(1);
    //     }
    // }
    // print("ok\n");

    return 0;
}