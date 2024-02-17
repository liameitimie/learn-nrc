#pragma once
#include <mesh.h>

namespace virtual_mesh {

struct Cluster {
    static const int max_vertices = 128;
    static const int max_triangles = 128;

    luisa::vector<float3> positions;
    luisa::vector<luisa::float2> texcoords;
    luisa::vector<int> indices;

    // the index to group buffer in VirtualMesh
    int group_id;

    Cluster() = default;
    Cluster(Cluster &) = default;
    Cluster(Cluster &&) = default;
    Cluster& operator=(Cluster &rhs) = default;
    Cluster& operator=(Cluster &&rhs) = default;
};

struct ClusterGroup {
    static const int max_group_size = 32;

    // the index to cluster buffer in VirtualMesh
    luisa::vector<int> clusters;

    ClusterGroup() = default;
    ClusterGroup(ClusterGroup &) = default;
    ClusterGroup(ClusterGroup &&) = default;
    ClusterGroup& operator=(ClusterGroup &rhs) = default;
    ClusterGroup& operator=(ClusterGroup &&rhs) = default;
};

struct VirtualMeshLevel {
    // the index to VirtualMesh
    int cluster_offset;
    int group_offset;
    int cluster_count;
    int group_count;
};

struct VirtualMesh {
    luisa::vector<Cluster> clusters;
    luisa::vector<ClusterGroup> cluster_groups;
    luisa::vector<VirtualMeshLevel> levels;

    VirtualMesh() = default;
    VirtualMesh(VirtualMesh &) = default;
    VirtualMesh(VirtualMesh &&) = default;
    VirtualMesh& operator=(VirtualMesh &rhs) = default;
    VirtualMesh& operator=(VirtualMesh &&rhs) = default;
};

VirtualMesh build_virtual_mesh(Mesh &mesh);

}