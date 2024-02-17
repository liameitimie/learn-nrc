#include <luisa/core/stl/vector.h>

// 压缩图表示（csr）
struct MetisGraph{
    int num_vert;
    luisa::vector<int> adj_offset; // 顶点对应起始边
    luisa::vector<int> adj_id; // 邻接顶点id
    luisa::vector<int> adj_weight; // 边权重

    void clear() {
        num_vert = 0;
        adj_offset.clear();
        adj_id.clear();
        adj_weight.clear();
    }
};

struct PartitionResult {
    int num_part;
    luisa::vector<int> part_id; // 每个顶点被划分到的块id
};

PartitionResult partition(MetisGraph &graph, int max_part_size);