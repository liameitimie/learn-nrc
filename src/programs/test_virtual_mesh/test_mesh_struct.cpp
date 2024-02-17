#include <mesh_structure.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

int main(int argc, char** argv) {
    Clock timer;

    vector<virtual_mesh::float3> pos = {
        {0, 0, 0}, // p0
        {0, 0, 0},
        {2, 0, 0}, // p1
        {1, 1, 0}, // adj0
        {1, -1, 0}, // adj1

        {-1, 1, 0}, // 5
        {-1, 0, 0},
        {-1, -1, 0},

        {3, 1, 0}, // 8
        {3, -1, 0},
    };
    vector<int> idx = {
        // 0, 2, 3,
        // 1, 4, 2,

        // 0, 3, 5,
        // 0, 5, 6,
        // 1, 6, 7,
        // 1, 7, 4,

        // 2, 8, 3,
        // 2, 9, 8,
        // 2, 4, 9,

        0, 3, 5,
        2, 9, 8,
        2, 4, 9,
        0, 5, 6,
        1, 6, 7,
        1, 7, 4,
        2, 8, 3,
        0, 2, 3,
        1, 4, 2,
    };

    virtual_mesh::Mesh mesh;
    mesh.positions = pos;
    mesh.indices = idx;

    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/Nature_Rock_wldhdhxva_8K_3d_ms/wldhdhxva_High.fbx");
    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/spot_triangulated_good.obj");
    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/SM_Gate_A.FBX");

    // print("build mesh structure\n");
    // timer.tic();
    virtual_mesh::MeshStructure mesh_struct{mesh};
    mesh_struct.init_vert_link();
    mesh_struct.init_corner_link();
    mesh_struct.init_edge_link();
    // print("{}\n", timer.toc());

    // int cnt = 0;
    // for (int x: mesh_struct.opposite_edge) {
    //     if (x == -1 || (mesh_struct.edge_flag[x] & virtual_mesh::MeshStructure::Seam)) {
    //         cnt++;
    //     }
    // }
    // print("{}\n", cnt);

    // for (int x: mesh_struct.unique_vert)
    //     print("{} ", x);
    // print("\n");

    // for (int x: mesh_struct.vert_link)
    //     print("{} ", x);
    // print("\n");

    // for (int x: mesh_struct.corner_head)
    //     print("{} ", x);
    // print("\n");

    // int id = 0;
    // for (int x: mesh_struct.corner_link)
    //     print("{}: {}\n", id++, x);
    // print("\n");

    int id = 0;
    for (int x: mesh_struct.opposite_edge) {
        print("{}: {}, {}, {}\n", id, x, mesh_struct.edge_flag[x] > 0, mesh_struct.edge_id[id]);
        id++;
    }
    print("\n");
    return 0;
}