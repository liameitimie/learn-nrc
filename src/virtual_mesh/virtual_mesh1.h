#pragma once
#include <mesh.h>

namespace virtual_mesh {

struct VirtualMeshLevel {
    Mesh mesh; // 每个lod层一个mesh
    int num_cluster;
    int num_group;
    luisa::vector<int> triangle_part_id; // 每个三角形一个编号 (0 <= id <= num_cluster)，代表被划分到的cluster
    luisa::vector<int> cluster_part_id; // 每个cluster一个编号 (0 <= id <= num_group)，代表被划分到的group
};

}