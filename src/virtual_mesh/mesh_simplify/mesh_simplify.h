#pragma once

#include <mesh.h>
#include <mesh_structure.h>
#include <quadric.h>
#include <index_heap.h>
// #include <luisa/core/stl/unordered_map.h>

namespace virtual_mesh {

using Corner = MeshStructure::Corner;
using Vertex = MeshStructure::Vertex;

struct MergeContext;

struct MeshSimplifier {
    MeshStructure m;

    double position_scale;
    double attr_scale;

    int num_attr;
    luisa::vector<luisa::ubyte> face_quadrics; // 使用字节数组并使用强制转换，将可变属性数量的矩阵存在线性内存中

    luisa::vector<int> edge_set;
    
    luisa::vector<int> vert_tag;
    luisa::vector<int> face_tag;

    MeshSimplifier(Mesh &mesh);

private:
    void init_face_quadric();
    void calc_face_quadric(int face_idx);
    void get_adjacent_face(MergeContext &ctx);
    void calc_wedge_quadric(MergeContext &ctx);
    Quadric calc_edge_quadric(MergeContext &ctx);
    void evaluate_merge(MergeContext &ctx, int e_idx);
    void perform_merge(MergeContext &ctx, IndexHeap &heap);
    void compact_mesh();
    bool link_condition(MergeContext &ctx);

public:
    double simplify(int target_face_num);
    void lock_position(float3 p);
};

// double mesh_simplify(Mesh &mesh, int target_face_num, luisa::unordered_set<float3> &locked_pos);
// double mesh_simplify1(Mesh &mesh, int target_face_num);


}