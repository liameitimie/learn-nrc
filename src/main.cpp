#include <iostream>
#include <vector>

using namespace std;

using ll = long long;
using ull = unsigned long long;
using LD = double;

//constexpr LD eps = 1e-10;
const int N = 1e5 + 5;

void solve() {
    int n; cin >> n;
    vector<vector<LD>>M(n, vector<LD>(n+1));

    for(int i = 0; i<n; ++i){
        for(int j = 0; j<n+1; ++j) cin >> M[i][j];
    }

    vector<bool>eli(n, false);
    for(int j = 0; j<n; ++j){
        int p = -1;
        LD mx = 1e-12;
        for(int i = 0; i<n; ++i){
            if(fabs(M[i][j]) > mx and !eli[i]) mx = fabs(M[i][j]), p = i;
        }

        if(p == -1) {  //该列没有主元
            cout << "No Solution";
            return;
        }
        eli[p] = true;

        for(int i = 0; i<n; ++i){
            if(p == i) continue;
            LD ratio = -M[i][j] / M[p][j];
            M[i][j] = 0;  //被消除
            for(int col = j+1; col < n+1; ++col){
                M[i][col] += M[p][col]*ratio;
            }
        }

        printf("## %d:\n", j);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n + 1; j++)
                printf("%lld ", *(ull*)&M[i][j]);
            printf("\n");
        }
    }

    for(int i = 0; i<n; ++i){  //化为行阶梯形式
        if(M[i][i] != 0) continue;
        for(int r = i+1, f = -1; r<n and f<0; ++r){
            if(M[r][i] != 0) f = 1, swap(M[i], M[r]);
        }
    }

    for(int i = 0; i<n; ++i) {
        printf("%.2f\n", M[i][n]/M[i][i]);
        // cout << fixed << setprecision(2) << M[i][n]/M[i][i] << '\n';
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int t = 1;
//    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}

/*
2
1 1 2
3 3 6
*/

// #include <luisa/dsl/sugar.h>
// #include <luisa/runtime/device.h>
// #include <luisa/runtime/stream.h>
// #include <luisa/runtime/buffer.h>
// #include <luisa/core/clock.h>
// #include <luisa/core/logging.h>
// #include <global.h>
// #include <gpu_rands.h>

// using namespace luisa;
// using namespace luisa::compute;
// using namespace fmt;

// namespace pcg32 {
//     ulong state = 0x853c49e6748fea9bull;
//     ulong inc = 0xda3e39cb94b95bdbull;
//     ulong mul = 0x5851f42d4c957f2dull;

//     uint next_uint() {
//         uint t1 = ((state >> 18u) ^ state) >> 27u;
//         uint t2 = state >> 59u;
//         state = state * mul + inc;
//         return (t1 >> t2) | (t1 << ((~t2 + 1u) & 31));
//     }
//     float next_float() {
//         union {
//             uint u;
//             float f;
//         } x;
//         x.u = (next_uint() >> 9) | 0x3f800000u;
//         return x.f - 1;
//     }
// }

// int main(int argc, char** argv) {
//     global::init(argv[0]);

//     Kernel1D kernel = []($buffer<int> steps, $buffer<int> out) {
//         set_block_size(256);
//         $shared<int> cnt{1};

//         $int tid = $dispatch_x % 256;
//         $int step = steps.read(tid);
//         $int idx;

//         $for (t, step) {
//             for (int i = 0; i < 8; i++) {
//                 $if (tid/32 == i) {
//                     idx = cnt.atomic(0).fetch_add(1);
//                     out.write(idx, tid);
//                 };
//                 sync_block();
//             }
//         };
//     };
//     auto shader = global::device().compile(kernel);
//     auto steps = global::device().create_buffer<int>(256);
//     auto out = global::device().create_buffer<int>(256 * 5);

//     vector<int> steps_h(256);
//     vector<int> out_h(256 * 5, -1);
//     for (int &x: steps_h) x = pcg32::next_uint() % 4;

//     for (int i = 0; i < steps_h.size(); i++) {
//         print("{}:{}, ", i, steps_h[i]);
//     }
//     print("\n\n");
    
//     global::stream()
//         << steps.copy_from(steps_h.data())
//         << out.copy_from(out_h.data())
//         << shader(steps, out).dispatch(256)
//         << out.copy_to(out_h.data())
//         << synchronize();

//     int lst = -1;
//     for (int x: out_h) {
//         if (x == -1) break;
//         if (x < lst) print("\n\n");
//         print("{}, ", x);
//         lst = x;
//     }
//     print("\n");
//     return 0;
// }
