#include <luisa/core/logging.h>

using namespace luisa;
using namespace fmt;

unsigned char arr[1 << 20];

int main() {
    FILE* p = fopen("ext/LuisaCompute/src/backends/common/hlsl/builtin/indirect", "rb");
    if (!p) {
        print("can't open file\n");
        exit(1);
    }
    int cnt = fread(arr, 1, 1 << 20, p);
    print("{}\n", cnt);
    printf("%s\n", arr);

    for (int i = 0; i < cnt; i++) {
        int t = arr[i];
        print("{},", t);
    }
    return 0;
}