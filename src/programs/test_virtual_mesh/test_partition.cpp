#include <mesh_structure.h>
#include <partitioner.h>
#include <luisa/luisa-compute.h>
#include <luisa/gui/window.h>
#include <gpu_rands.h>
#include "camera.h"

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

using Corner = virtual_mesh::MeshStructure::Corner;
using Vertex = virtual_mesh::MeshStructure::Vertex;

struct v2p {
    float4 pos;
    float2 uv;
    luisa::float3 color;
};
LUISA_STRUCT(v2p, pos, uv, color) {};

struct MeshData {
    Buffer<float> positions;
    Buffer<uint> indices;
    Buffer<float2> texcoords;
};
LUISA_BINDING_GROUP(MeshData, positions, indices, texcoords) {};

RasterStageKernel vert = [](Var<AppData> var, Var<MeshData> mesh_data, $float4x4 vp_mat, $uint show_state, $uint seed, $buffer<uint> part_id) {
    Var<v2p> out;
    $uint vert_id = vertex_id();

    $uint vid = mesh_data.indices.read(vert_id);
    $float3 pos;
    $float2 uv = mesh_data.texcoords.read(vid);

    for (int i = 0; i < 3; i++) {
        $float x = mesh_data.positions.read(vid * 3 + i);
        pos[i] = x;
    }

    out.pos = vp_mat * make_float4(pos, 1.0f);
    out.uv = {uv.x, 1 - uv.y};

    $uint s;
    $switch (show_state) {
        $case (0) { s = tea(vert_id / 3, seed).x; };
        $case (1) { s = tea(part_id.read(vert_id / 3), seed).x; };
    };

    out.color = make_float3(
        ((s >> 0) & 255) / 255.0f,
        ((s >> 8) & 255) / 255.0f,
        ((s >> 16) & 255) / 255.0f
    );

    return out;
};
RasterStageKernel pixel = [](Var<v2p> in) {
    $float4 color = make_float4(in.color, 1.0f);
    return color;
};

RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};

Kernel2D clear_kernel = []($image<float> image) {
    image.write(dispatch_id().xy(), make_float4(0.1f));
};

const uint width = 1920;
const uint height = 1080;

int is_key_switch(Window &window, Key key) {
    static bool key_state[500] = {};

    int res = 0;
    bool cur_state = window.is_key_down(key);
    if (cur_state != key_state[key]) {
        key_state[key] = cur_state;
        res = cur_state ? 1 : -1;
    }
    return res;
}


int main(int argc, char** argv) {
    Clock timer;

    virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/spot_triangulated_good.obj");
    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/SM_Gate_A.FBX");
    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/Nature_Rock_wldhdhxva_8K_3d_ms/wldhdhxva_High.fbx");
    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/Font_Reconstructed.stl");

    print("init mesh struct: ");
    timer.tic();

    virtual_mesh::MeshStructure m{mesh};
    m.init_vert_link();
    m.init_corner_link();
    m.init_edge_link();

    print("{}\n", timer.toc());

    vector<int> face_tag(m.face_count, -1);

    MetisGraph g;

    g.clear();
    print("build graph1: ");
    timer.tic();

    g.num_vert = m.face_count;
    g.adj_offset.reserve(m.face_count + 1);
    g.adj_id.reserve(m.edge_count * 2);
    g.adj_weight.reserve(m.edge_count * 2);

    for (int i = 0; i < m.face_count; i++) {
        g.adj_offset.push_back(g.adj_id.size());

        for (int k = 0; k < 3; k++) {
            int cid = i * 3 + k;
            int link_f = m.opposite_edge[cid];
            if (link_f != -1) {
                g.adj_id.push_back(link_f / 3);
                g.adj_weight.push_back(1);
            }
        }
    }
    g.adj_offset.push_back(g.adj_id.size());

    print("{}\n", timer.toc());
    print("adj_id size: {}, edge count: {}\n", g.adj_id.size(), m.edge_count);


    // g.clear();
    // print("build graph: ");
    // timer.tic();

    // g.num_vert = m.face_count;
    // for (int i = 0; i < m.face_count; i++) {
    //     g.adj_offset.push_back(g.adj_id.size());

    //     for (int k = 0; k < 3; k++) {
    //         int cid = i * 3 + k;
    //         int vid = m.mesh.indices[cid];
    //         Vertex v{m, vid};
    //         for (Corner wc: v.wedge_corners()) {
    //             int f = wc.cid / 3;
    //             if (f != i) {
    //                 int &idx = face_tag[f];
    //                 if (idx != -1) {
    //                     g.adj_weight[idx]++;
    //                 }
    //                 else {
    //                     idx = g.adj_id.size();
    //                     g.adj_id.push_back(f);
    //                     g.adj_weight.push_back(1);
    //                 }
    //             }
    //         }
    //     }
    //     for (int j = g.adj_offset[i]; j < g.adj_id.size(); j++) {
    //         face_tag[g.adj_id[j]] = -1;
    //     }
    //     // for (int x: face_tag) {
    //     //     if (x != -1) {
    //     //         print("error\n");
    //     //         exit(1);
    //     //     }
    //     // }
    // }
    // g.adj_offset.push_back(g.adj_id.size());
    // print("{}\n", timer.toc());

    // g.num_vert = 7;
    // g.adj_id = {
    //     4, 2, 1,
    //     0, 2, 3,
    //     4, 3, 1, 0,
    //     1, 2, 5, 6,
    //     0, 2, 5,
    //     4, 3, 6,
    //     5, 3,
    // };
    // g.adj_weight = {
    //     1, 2, 1,
    //     1, 2, 1,
    //     3, 2, 2, 2,
    //     1, 2, 2, 5,
    //     1, 3, 2,
    //     2, 2, 6,
    //     6, 5,
    // };
    // g.adj_offset = {
    //     0, 3, 6, 10, 14, 17, 20, 22
    // };

    print("partition: \n");
    timer.tic();

    auto [n_part, part_id] = partition(g, 128);
    print("{}\n", timer.toc());

    print("n_part: {}\n", n_part);
    // print("part_id: [");
    // for (int x: part_id) print("{} ", x);
    // print("]\n");

    vector<int> part_size(n_part);
    for (int x: part_id) {
        if (x < 0 || x >= n_part) {
            print("error part id\n");
            exit(1);
        }
        part_size[x]++;
    }
    // print("part_size: [");
    // for (int x: part_size) print("{} ", x);
    // print("]\n");
    int cnt = 0;
    for (int x: part_size) {
        if (x > 128) cnt++;
    }
    print("part size > 128 : {}\n", cnt);

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    MeshFormat mesh_format;

    auto clear_shader = device.compile(clear_kernel);
    auto raster_shader = device.compile(kernel, mesh_format);

    Window window{"", width, height};
    Swapchain swap_chain = device.create_swapchain(
        window.native_handle(),
        stream,
        uint2(width, height),
        false, false,
        3
    );
    Image<float> out_img = device.create_image<float>(swap_chain.backend_storage(), width, height);
    auto depth = device.create_depth_buffer(DepthFormat::D32, {width, height});

    RasterState state {
        .cull_mode = CullMode::None,
        .depth_state = {
            .enable_depth = true,
            .comparison = Comparison::Greater,
            .write = true
        }
    };

    vector<RasterMesh> meshes;

    MeshData mesh_data;
    mesh_data.positions = device.create_buffer<float>(mesh.positions.size() * 3);
    mesh_data.indices = device.create_buffer<uint>(mesh.indices.size());
    mesh_data.texcoords = device.create_buffer<float2>(mesh.texcoords.size());

    auto part_id_buffer = device.create_buffer<uint>(part_id.size());

    stream << mesh_data.positions.copy_from(mesh.positions.data())
        << mesh_data.indices.copy_from(mesh.indices.data())
        << mesh_data.texcoords.copy_from(mesh.texcoords.data())
        << part_id_buffer.copy_from(part_id.data());

    Camera camera {
        .position = luisa::float3(0, 0, 10),
        .yaw = -90,
        .pitch = 0,
        .move_speed = 5000
    };
    uint frame_cnt = 0;
    float2 lst_cursor_pos;
    uint seed = 233;

    window.set_cursor_disabled();
    bool is_cursor_disabled = true;
    int show_state = 0; // 0: triangle, 1: part_id

    timer.tic();
    
    while (!window.should_close()) {
        float tick_time = timer.toc() / 1000;
        timer.tic();

        // switch show level (0: triangle, 1: part_id)
        if (is_key_switch(window, KEY_J) == 1) show_state = 0;
        if (is_key_switch(window, KEY_K) == 1) show_state = 1;
        if (is_key_switch(window, KEY_L) == 1) seed = rand();

        if (window.is_key_down(KEY_W)) camera.move_front(tick_time);
        if (window.is_key_down(KEY_S)) camera.move_front(-tick_time);
        if (window.is_key_down(KEY_A)) camera.move_right(-tick_time);
        if (window.is_key_down(KEY_D)) camera.move_right(tick_time);

        if (is_key_switch(window, KEY_B) == 1) {
            is_cursor_disabled = !is_cursor_disabled;
            if (is_cursor_disabled) window.set_cursor_disabled();
            else window.set_cursor_normal();
        }
        // switch (is_key_switch(window, KEY_B)) {
        //     case 1: {
        //         window.set_cursor_normal();
        //         is_cursor_disabled = false;
        //     } break;
        //     case -1: {
        //         window.set_cursor_disabled();
        //         is_cursor_disabled = true;
        //     } break;
        // }

        float2 cursor_pos = window.get_cursor_pos();
        if (frame_cnt == 0) lst_cursor_pos = cursor_pos;
        if (is_cursor_disabled) {
            camera.rotate_view(cursor_pos.x - lst_cursor_pos.x, cursor_pos.y - lst_cursor_pos.y);
        }
        lst_cursor_pos = cursor_pos;
        frame_cnt++;
        
        float4x4 v_mat = camera.view_mat();
        float4x4 p_mat = camera.projection_mat(radians(40), (float)width / (float)height);
        float4x4 vp_mat = p_mat * v_mat;

        meshes.emplace_back(span<VertexBufferView>{}, mesh.indices.size(), 1, 0);
        stream
            << depth.clear(0)
            << clear_shader(out_img).dispatch(width, height)
            << raster_shader(mesh_data, vp_mat, show_state, seed, part_id_buffer).draw(
                std::move(meshes),
                Viewport{0.f, 0.f, float(width), float(height)}, 
                state, 
                &depth,
                out_img
            )
            << swap_chain.present(out_img);

        window.poll_events();
    }
    stream << synchronize();

    return 0;
}

/*
7 11
5 1 3 2 2 1
1 1 3 2 4 1
5 3 4 2 2 2 1 2
2 1 3 2 6 2 7 5
1 1 3 3 6 2
5 2 4 2 7 6
6 6 4 5

0
4 2 1
1 2 1
1
0 2 3
1 2 1
2
4 3 1 0
3 2 2 2
3
1 2 5 6
1 2 2 5
4
0 2 5
1 3 2
5
4 3 6
2 2 6
6
5 3
6 5
*/