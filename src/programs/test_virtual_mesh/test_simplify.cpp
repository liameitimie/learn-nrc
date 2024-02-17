#include <mesh.h>
#include <mesh_simplify.h>
#include <luisa/luisa-compute.h>
#include <luisa/gui/window.h>
#include <gpu_rands.h>
#include "camera.h"
#include <double3.h>
#include <stb/stb_image.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

struct Img {
    int width = 0;
    int height = 0;
    int channel = 0;
    unsigned char* pixels;
    
    void load(const char* file) {
        Clock timer;
        print("load texture: ");
        timer.tic();
        pixels = stbi_load(file, &width, &height, &channel, 4);
        print("{} ms\n", timer.toc());
    }
};

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

RasterStageKernel vert = [](Var<AppData> var, Var<MeshData> mesh_data, $float4x4 vp_mat) {
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

    $uint s = tea(vert_id / 3, 233).x;
    out.color = make_float3(
        ((s >> 0) & 255) / 255.0f,
        ((s >> 8) & 255) / 255.0f,
        ((s >> 16) & 255) / 255.0f
    );

    return out;
};
RasterStageKernel pixel = [](Var<v2p> in, $image<float> texture, $uint2 texture_dim, $int show_state) {
    $float4 color;
    $switch (show_state) {
        $case (0) { color = make_float4(in.color, 1.f); };
        $case (1) { color = texture.read($uint2(in.uv * $float2(texture.size()))); };
        $case (2) { color = make_float4(in.uv, 0.f, 1.f); };
    };
    return color;
};

RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};

Kernel2D clear_kernel = []($image<float> image) {
    image.write(dispatch_id().xy(), make_float4(0.1f));
};

Kernel1D set_indirect_draw = [](Var<IndirectDispatchBuffer> indirect_buffer, $uint index_count) {
    set_block_size(32);
    // indirect_buffer.set_dispatch_count(1u);
    indirect_buffer.set_draw_argument($uint4{index_count, 1u, 0u, 0u});
};
/*
typedef struct D3D12_DRAW_ARGUMENTS
    {
    UINT VertexCountPerInstance;
    UINT InstanceCount;
    UINT StartVertexLocation;
    UINT StartInstanceLocation;
    } 	D3D12_DRAW_ARGUMENTS;

*/
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

// struct Quadric {
// 	// double xx, yy, zz;
// 	// double xy, xz, yz;
// 	// double dx, dy, dz;
// 	// double d2;
	
// 	double a;
// };

// struct QuadricGrad {
//     // virtual_mesh::double3 g;
// 	double d;
// };

// struct QuadricAttr {
// 	Quadric m;
// 	QuadricGrad g[];
// };

template<>
struct equal_to<float3> {
    bool operator()(const float3 &a, const float3 &b) const {
        return memcmp(&a, &b, sizeof(float) * 3) == 0;
    }
};

template <>
struct luisa::hash<eastl::pair<virtual_mesh::float3, virtual_mesh::float3>> {
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(const eastl::pair<virtual_mesh::float3, virtual_mesh::float3> &e, uint64_t seed = hash64_default_seed) const noexcept {
        return hash64(&e, sizeof(float) * 6, seed);
    }
};


int main(int argc, char** argv) {
    
    // unordered_set<eastl::pair<virtual_mesh::float3, virtual_mesh::float3>> s;
    // unordered_set<float3, luisa::hash<float3>, equal_to<float3>> s;
    
    // s.insert({{0, 0, 1}, {1, 1, 0}});
    // s.insert({{0, 1, 1}, {1, 0, 0}});
    // s.insert({{1, 0, 1}, {1, 1, 1}});
    // s.insert({{1, 0, 0}, {0, 1, 0}});
    // s.insert({{1, 0, 0}, {0, 1, 0}});
    // s.insert({{1, 0, 0}, {1, 1, 0}});

    // luisa::hash<eastl::pair<virtual_mesh::float3, virtual_mesh::float3>> hasher;

    // eastl::pair<virtual_mesh::float3, virtual_mesh::float3> e = {{0, 0, 1}, {1, 1, 0}};
    // hasher(e);

    // s.insert({1, 1, 0});
    // s.insert({0, 0, 1});

    // s.insert({0, 1, 1});
    // s.insert({1, 0, 0});

    // s.insert({1, 0, 1});
    // s.insert({1, 1, 1});

    // s.insert({1, 0, 0});
    // s.insert({0, 1, 0});

    // print("{}\n", s.size());

    // for (auto e: s) {
    //     print("{}\n", hasher(e));
    // }

    // double2x2 t{{4, 1}, {1, -1}};
    // double2 b{100, 100};

    // double2x2 t1;
    // bool res = inverse(t, t1);

    // print("{}, {}\n", t, res);
    // print("{}\n", t1);
    // print("{}\n", t1 * b);

    // double4x4 a = {
    //     {1, 1, 2, 5},
    //     {2, 5, -1, -9},
    //     {2, 1, -1, 3},
    //     {1, 3, 2, 7}
    // };
    // double4x4 a1;
    // res = inverse(a, a1);
    // double4 b1 = {3, -3, -11, -5};
    // double4 x = a1 * b1;

    // print("{}, {}\n", a, res);
    // print("{}\n", a1);
    // print("{}\n", x);
    // print("{}\n", a * x);


    // vector<ubyte> data;
    // data.resize(sizeof(Quadric) + sizeof(QuadricGrad) * 4);

    // QuadricAttr &q = *(QuadricAttr*)(&data[0]);

    // q.m.a = 1;
    // q.g[3].d = 1;

    // for (auto x: data) print("{} ", x);
    // print("\n");

    // print("{}\n", q.g[3].d);

    // auto mesh = virtual_mesh::load_mesh("assets/SM_Gate_A.FBX");

    // vector<virtual_mesh::float3> pos = {
    // // virtual_mesh::float3 pos[] = {
    //     {0, 0, 0}, // p0
    //     {0, 0, 0},
    //     {2, 0, 0}, // p1
    //     {1, 1, 0}, // adj0
    //     {1, -1, 0}, // adj1

    //     {-1, 1, 0}, // 5
    //     {-1, 0, 0},
    //     {-1, -1, 0},

    //     {3, 1, 0}, // 8
    //     {3, -1, 0},
    // };
    // vector<int> idx = {
    //     // 0, 2, 3,
    //     // 1, 4, 2,

    //     // 0, 3, 5,
    //     // 0, 5, 6,
    //     // 1, 6, 7,
    //     // 1, 7, 4,

    //     // 2, 8, 3,
    //     // 2, 9, 8,
    //     // 2, 4, 9,

    //     0, 3, 5,
    //     2, 9, 8,
    //     2, 4, 9,
    //     0, 5, 6,
    //     1, 6, 7,
    //     1, 7, 4,
    //     2, 8, 3,
    //     0, 2, 3,
    //     1, 4, 2,
    // };

    Clock timer;

    virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/spot_triangulated_good.obj");
    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/SM_Gate_A.FBX");
    // virtual_mesh::Mesh mesh = virtual_mesh::load_mesh("assets/Nature_Rock_wldhdhxva_8K_3d_ms/wldhdhxva_High.fbx");

    // virtual_mesh::Mesh mesh;
    // mesh.positions = pos;
    // mesh.indices = idx;
    // virtual_mesh::mesh_simplify1(mesh, 1000000);

    // virtual_mesh::debug_in(mesh, "debug_mesh.txt");

    print("simplify:\n");
    timer.tic();
    virtual_mesh::MeshSimplifier simplifier(mesh);

    int f_cnt = mesh.indices.size() / 3;
    double error = simplifier.simplify(1000000);
    // unordered_set<virtual_mesh::float3> locked_pos;
    // double error = virtual_mesh::mesh_simplify(mesh, 1000000, locked_pos);
    // double error = virtual_mesh::mesh_simplify1(mesh, 1000000);
    print("{}\n", timer.toc());

    print("error: {}\n", error);
    print("v_cnt:{}, f_cnt:{}\n", mesh.positions.size(), mesh.indices.size() / 3);

    Img img;
    // img.load("assets/Nature_Rock_wldhdhxva_8K_3d_ms/wldhdhxva_8K_Albedo.jpg");
    img.load("assets/spot_texture.png");

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    MeshFormat mesh_format;

    auto clear_shader = device.compile(clear_kernel);
    auto raster_shader = device.compile(kernel, mesh_format);
    auto set_indirect_shader = device.compile(set_indirect_draw);

    auto indirect_buffer = device.create_indirect_dispatch_buffer(1);

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

    auto texture = device.create_image<float>(PixelStorage::BYTE4, img.width, img.height);

    stream << mesh_data.positions.copy_from(mesh.positions.data())
        << mesh_data.indices.copy_from(mesh.indices.data())
        << mesh_data.texcoords.copy_from(mesh.texcoords.data())
        << texture.copy_from(img.pixels)
        << set_indirect_shader(indirect_buffer, mesh.indices.size()).dispatch(1);

    Camera camera {
        .position = luisa::float3(0, 0, 10),
        .yaw = -90,
        .pitch = 0,
        // .move_speed = 5000
    };
    uint frame_cnt = 0;
    float2 lst_cursor_pos;

    window.set_cursor_disabled();
    bool is_cursor_disabled = true;
    int show_state = 0; // 0: triangle, 1: texture, 2: uv

    timer.tic();
    
    while (!window.should_close()) {
        float tick_time = timer.toc() / 1000;
        timer.tic();

        // switch show level (0: triangle, 1: texture, 2: uv)
        if (is_key_switch(window, KEY_J) == 1) show_state = 0;
        if (is_key_switch(window, KEY_K) == 1) show_state = 1;
        if (is_key_switch(window, KEY_L) == 1) show_state = 2;

        if (window.is_key_down(KEY_W)) camera.move_front(tick_time);
        if (window.is_key_down(KEY_S)) camera.move_front(-tick_time);
        if (window.is_key_down(KEY_A)) camera.move_right(-tick_time);
        if (window.is_key_down(KEY_D)) camera.move_right(tick_time);

        switch (is_key_switch(window, KEY_B)) {
            case 1: {
                window.set_cursor_normal();
                is_cursor_disabled = false;
            } break;
            case -1: {
                window.set_cursor_disabled();
                is_cursor_disabled = true;
            } break;
        }

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

        // meshes.emplace_back(span<VertexBufferView>{}, mesh.indices.size(), 1, 0);
        stream
            << depth.clear(0)
            << clear_shader(out_img).dispatch(width, height)
            // << raster_shader(mesh_data, vp_mat, texture, texture.size(), show_state).draw(
            //     std::move(meshes),
            //     Viewport{0.f, 0.f, float(width), float(height)}, 
            //     state, 
            //     &depth,
            //     out_img
            // )
            << raster_shader(mesh_data, vp_mat, texture, texture.size(), show_state).draw_indirect(
                indirect_buffer,
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
v_cnt:1907, i_cnt:9072
p0: (-13065.237, 4327.8203, -6676.622)
i0: 0, i1: 1, i2: 2
simplify:
v0:1417, v1:1383
## ctx
wedge vert: [1419 1417 1418 1383 ]
wedge union: [0 1 0 1 ]
wedge face: [(7933, 0) (7877, 1) (7844, 3) (7634, 3) (7596, 3) ]
remove face: [7932 7875 ]
opt_p:(-11917.429, 4979.129, -6822.6396)
opt_a: 0:0.5122663, 1:0.40030828, 2:0.9349835, 3:0.011624124, 4:0, 5:0, 6:0, 7:0,
error: 0.16043260793477138
penalty: 0
verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:1383, vlink:1383, chead:7932, flag:
1419:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
  uni:1417, vlink:1417, chead:7933, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:7934, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:1383, vlink:1418, chead:7875, flag:
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7876, flag:
1417:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
  uni:1417, vlink:1419, chead:7877, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:1418, v1:1419, v2:1420, flag:
c:7877,2, f:2625, v0:1383, v1:1385, v2:1417, flag:
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:1418, clink:-1
7933: v:1419, clink:-1
7934: v:1420, clink:-1
7875: v:1383, clink:7844
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:1385, clink:7770
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:1417, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:7877, flag:seam,
7934: e_id:-1, op_e:7634, flag:seam,
7933: e_id:4043, op_e:-1, flag:border,
7875: e_id:-1, op_e:7598, flag:
7877: e_id:4031, op_e:7932, flag:seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:7934, flag:seam,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:3895, op_e:7875, flag:
7876: e_id:4030, op_e:-1, flag:border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


 ### after remove vert ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:1419, vlink:1417, chead:7932, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:7934, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del,
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7876, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1419, vlink:1419, chead:7596, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:1419, v1:1419, v2:1420, flag:
c:7877,2, f:2625, v0:1417, v1:1385, v2:1417, flag:
c:7844,2, f:2614, v0:1210, v1:1405, v2:1417, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1417, flag:
c:7596,0, f:2532, v0:1417, v1:1405, v2:1385, flag:
corners:
7932: v:1419, clink:7933
7933: v:1419, clink:-1
7934: v:1420, clink:-1
7875: v:1417, clink:7877
7844: v:1417, clink:7875
7634: v:1417, clink:7844
7596: v:1417, clink:7634
7876: v:1385, clink:7770
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:1417, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:7877, flag:seam,
7934: e_id:-1, op_e:7634, flag:seam,
7933: e_id:4043, op_e:-1, flag:border,
7875: e_id:-1, op_e:7598, flag:
7877: e_id:4031, op_e:7932, flag:seam, 
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:7934, flag:seam,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:3895, op_e:7875, flag:
7876: e_id:4030, op_e:-1, flag:border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


 ### after remove face ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:-1, vlink:-1, chead:-1, flag: del,
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:-1, vlink:-1, chead:-1, flag: del,
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del,
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1417, vlink:1417, chead:7596, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1400, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1417, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1417, flag:
c:7596,0, f:2532, v0:1417, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1417, clink:-1
7634: v:1417, clink:7844
7596: v:1417, clink:7634
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:



v_cnt:1907, i_cnt:9072
p0: (-13065.237, 4327.8203, -6676.622)
i0: 0, i1: 1, i2: 2
simplify:
v0:1417, v1:1383
## ctx
wedge vert: [1419 1417 1418 1383 ]
wedge union: [0 1 0 1 ]
wedge face: [(7933, 0) (7877, 1) (7844, 3) (7634, 3) (7596, 3) ]
remove face: [7932 7875 ]
opt_p:(-11917.429, 4979.129, -6822.6396)
opt_a: 0:0.5122663, 1:0.40030828, 2:0.9349835, 3:0.011624124, 4:0, 5:0, 6:0, 7:0,
error: 0.16043260793477138
penalty: 0
verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:1383, vlink:1383, chead:7932, flag:
1419:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
  uni:1417, vlink:1417, chead:7933, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:7934, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:1383, vlink:1418, chead:7875, flag:
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7876, flag:
1417:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
  uni:1417, vlink:1419, chead:7877, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:1418, v1:1419, v2:1420, flag:
c:7877,2, f:2625, v0:1383, v1:1385, v2:1417, flag:
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:1418, clink:-1
7933: v:1419, clink:-1
7934: v:1420, clink:-1
7875: v:1383, clink:7844
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:1385, clink:7770
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:1417, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:7877, flag:seam,
7934: e_id:-1, op_e:7634, flag:seam,
7933: e_id:4043, op_e:-1, flag:border,
7875: e_id:-1, op_e:7598, flag:
7877: e_id:4031, op_e:7932, flag:seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:7934, flag:seam,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:3895, op_e:7875, flag:
7876: e_id:4030, op_e:-1, flag:border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


 ### after remove vert ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:1419, vlink:1417, chead:7932, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:7934, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del, 
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7876, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1419, vlink:1419, chead:7596, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:1419, v1:1419, v2:1420, flag:
c:7877,2, f:2625, v0:1417, v1:1385, v2:1417, flag:
c:7844,2, f:2614, v0:1210, v1:1405, v2:1417, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1417, flag:
c:7596,0, f:2532, v0:1417, v1:1405, v2:1385, flag:
corners:
7932: v:1419, clink:7933
7933: v:1419, clink:-1
7934: v:1420, clink:-1
7875: v:1417, clink:7877
7844: v:1417, clink:7875
7634: v:1417, clink:7844
7596: v:1417, clink:7634
7876: v:1385, clink:7770
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:1417, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:7877, flag:seam,
7934: e_id:-1, op_e:7634, flag:seam,
7933: e_id:4043, op_e:-1, flag:border,
7875: e_id:-1, op_e:7598, flag:
7877: e_id:4031, op_e:7932, flag:seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:7934, flag:seam,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:3895, op_e:7875, flag:
7876: e_id:4030, op_e:-1, flag:border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


 ### after remove face ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:-1, vlink:-1, chead:-1, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:-1, vlink:-1, chead:-1, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del,
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1417, vlink:1417, chead:7596, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1400, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1417, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1417, flag:
c:7596,0, f:2532, v0:1417, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1417, clink:-1
7634: v:1417, clink:7844
7596: v:1417, clink:7634
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


*/



/*
v_cnt:1907, i_cnt:9072
p0: (-13065.237, 4327.8203, -6676.622)
i0: 0, i1: 1, i2: 2
simplify:
v0:1417, v1:1383
## ctx
wedge vert: [1419 1417 1418 1383 ]
wedge union: [0 1 0 1 ]
wedge face: [(7933, 0) (7877, 1) (7844, 3) (7634, 3) (7596, 3) ]
remove face: [7932 7875 ]
opt_p:(-11917.429, 4979.129, -6822.6396)
opt_a: 0:0.5122663, 1:0.40030828, 2:0.9349835, 3:0.011624124, 4:0, 5:0, 6:0, 7:0,
error: 0.16043260793477138
penalty: 0
verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:1383, vlink:1383, chead:7932, flag:
1419:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
  uni:1417, vlink:1417, chead:7933, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:7934, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:1383, vlink:1418, chead:7875, flag:
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7876, flag:
1417:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
  uni:1417, vlink:1419, chead:7877, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:1418, v1:1419, v2:1420, flag:
c:7877,2, f:2625, v0:1383, v1:1385, v2:1417, flag:
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:1418, clink:-1
7933: v:1419, clink:-1
7934: v:1420, clink:-1
7875: v:1383, clink:7844
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:1385, clink:7770
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:1417, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:7877, flag:seam,
7934: e_id:-1, op_e:7634, flag:seam,
7933: e_id:4043, op_e:-1, flag:border,
7875: e_id:-1, op_e:7598, flag:
7877: e_id:4031, op_e:7932, flag:seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:7934, flag:seam,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:3895, op_e:7875, flag:
7876: e_id:4030, op_e:-1, flag:border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


 ### after remove face ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:1383, vlink:1383, chead:-1, flag:
1419:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
  uni:1417, vlink:1417, chead:-1, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:-1, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:1383, vlink:1418, chead:7844, flag:
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
  uni:1417, vlink:1419, chead:-1, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


 ### after remove vert ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:1419, vlink:1417, chead:-1, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:-1, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del,
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1419, vlink:1419, chead:-1, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:


 ### after update mesh ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:1419, vlink:1417, chead:-1, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:-1, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del,
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1419, vlink:1419, chead:7596, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1417, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1417, flag:
c:7596,0, f:2532, v0:1417, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1417, clink:-1
7634: v:1417, clink:7844
7596: v:1417, clink:7634
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7597: e_id:3894, op_e:7713, flag:
7842: e_id:-1, op_e:7831, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7830: e_id:-1, op_e:7654, flag:
7714: e_id:-1, op_e:7653, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:







v_cnt:1907, i_cnt:9072
p0: (-13065.237, 4327.8203, -6676.622)
i0: 0, i1: 1, i2: 2
simplify:
v0:1417, v1:1383
## ctx
wedge vert: [1419 1417 1418 1383 ]
wedge union: [0 1 0 1 ]
wedge face: [(7933, 0) (7877, 1) (7844, 3) (7634, 3) (7596, 3) ]
remove face: [7932 7875 ]
opt_p:(-11917.429, 4979.129, -6822.6396)
opt_a: 0:0.5122663, 1:0.40030828, 2:0.9349835, 3:0.011624124, 4:0, 5:0, 6:0, 7:0,
error: 0.16043260793477138
penalty: 0
verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:1383, vlink:1383, chead:7932, flag:
1419:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
  uni:1417, vlink:1417, chead:7933, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:7934, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:1383, vlink:1418, chead:7875, flag:
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7876, flag:
1417:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
  uni:1417, vlink:1419, chead:7877, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:1418, v1:1419, v2:1420, flag:
c:7877,2, f:2625, v0:1383, v1:1385, v2:1417, flag:
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:1418, clink:-1
7933: v:1419, clink:-1
7934: v:1420, clink:-1
7875: v:1383, clink:7844
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:1385, clink:7770
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:1417, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:7877, flag:seam,
7934: e_id:-1, op_e:7634, flag:seam,
7933: e_id:4043, op_e:-1, flag:border,
7932: e_id:-1, op_e:7877, flag:seam,
7934: e_id:-1, op_e:7634, flag:seam,
7933: e_id:4043, op_e:-1, flag:border,
7875: e_id:-1, op_e:7598, flag:
7877: e_id:4031, op_e:7932, flag:seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:7934, flag:seam,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:3895, op_e:7875, flag:
7876: e_id:4030, op_e:-1, flag:border,
7875: e_id:-1, op_e:7598, flag:
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7598: e_id:3895, op_e:7875, flag:
7597: e_id:3894, op_e:7713, flag:
7877: e_id:4031, op_e:7932, flag:seam,
7876: e_id:4030, op_e:-1, flag:border,
7842: e_id:-1, op_e:7831, flag:
7844: e_id:-1, op_e:7633, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7633: e_id:3922, op_e:7844, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7597: e_id:3894, op_e:7713, flag:
7596: e_id:3893, op_e:7843, flag:
7831: e_id:4026, op_e:7842, flag:
7830: e_id:-1, op_e:7654, flag:
7843: e_id:-1, op_e:7596, flag:
7842: e_id:-1, op_e:7831, flag:
7714: e_id:-1, op_e:7653, flag:
7713: e_id:-1, op_e:7597, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:
7632: e_id:3921, op_e:-1, flag:border,
7634: e_id:3923, op_e:7934, flag:seam,


 ### after remove face ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:1383, vlink:1383, chead:-1, flag:
1419:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
  uni:1417, vlink:1417, chead:-1, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:-1, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:1383, vlink:1418, chead:7844, flag: 
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
  uni:1417, vlink:1419, chead:-1, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7598: e_id:4030, op_e:-1, flag:border,
7597: e_id:3894, op_e:7713, flag:
7877: e_id:-1, op_e:-1, flag:del, seam,
7876: e_id:-1, op_e:-1, flag:del, border,
7842: e_id:-1, op_e:7831, flag:
7844: e_id:-1, op_e:7633, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7633: e_id:3922, op_e:7844, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7597: e_id:3894, op_e:7713, flag:
7596: e_id:3893, op_e:7843, flag:
7831: e_id:4026, op_e:7842, flag:
7830: e_id:-1, op_e:7654, flag:
7843: e_id:-1, op_e:7596, flag:
7842: e_id:-1, op_e:7831, flag:
7714: e_id:-1, op_e:7653, flag:
7713: e_id:-1, op_e:7597, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:
7632: e_id:3921, op_e:-1, flag:border,
7634: e_id:3923, op_e:-1, flag:seam, border,


 ### after remove vert ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:1419, vlink:1417, chead:-1, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:-1, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del,
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1419, vlink:1419, chead:-1, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1383, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1383, flag:
c:7596,0, f:2532, v0:1383, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1383, clink:7634
7634: v:1383, clink:7596
7596: v:1383, clink:-1
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7598: e_id:4030, op_e:-1, flag:border,
7597: e_id:3894, op_e:7713, flag:
7877: e_id:-1, op_e:-1, flag:del, seam,
7876: e_id:-1, op_e:-1, flag:del, border,
7842: e_id:-1, op_e:7831, flag:
7844: e_id:-1, op_e:7633, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7633: e_id:3922, op_e:7844, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7597: e_id:3894, op_e:7713, flag:
7596: e_id:3893, op_e:7843, flag:
7831: e_id:4026, op_e:7842, flag:
7830: e_id:-1, op_e:7654, flag:
7843: e_id:-1, op_e:7596, flag:
7842: e_id:-1, op_e:7831, flag:
7714: e_id:-1, op_e:7653, flag:
7713: e_id:-1, op_e:7597, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:
7632: e_id:3921, op_e:-1, flag:border,
7634: e_id:3923, op_e:-1, flag:seam, border,


 ### after update mesh ###

verts:
1418:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
  uni:-1, vlink:-1, chead:-1, flag: del,
1419:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
  uni:1419, vlink:1417, chead:-1, flag:
1420:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.51317537, 0.40018588)
  uni:1400, vlink:1400, chead:-1, flag:
1383:
  p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
  uni:-1, vlink:-1, chead:-1, flag: del,
1385:
  p:(-11958.764, 4944.409, -6834.79), uv:(0.9352733, 0.012176621)
  uni:1385, vlink:1385, chead:7770, flag:
1417:
  p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
  uni:1419, vlink:1419, chead:7596, flag:
1210:
  p:(-11893.508, 4902.713, -6919.989), uv:(0.9366327, 0.010935204)
  uni:1210, vlink:1210, chead:7842, flag:
1405:
  p:(-11932.581, 4925.8706, -6870.6626), uv:(0.93587285, 0.011722636)
  uni:1405, vlink:1405, chead:7597, flag:
1400:
  p:(-11889.021, 4930.594, -6885.1353), uv:(0.9360168, 0.011192593)
  uni:1400, vlink:1420, chead:7632, flag:
faces:
c:7933,1, f:2644, v0:-1, v1:-1, v2:-1, flag: del,
c:7877,2, f:2625, v0:-1, v1:-1, v2:-1, flag: del,
c:7844,2, f:2614, v0:1210, v1:1405, v2:1417, flag:
c:7634,2, f:2544, v0:1400, v1:1210, v2:1417, flag:
c:7596,0, f:2532, v0:1417, v1:1405, v2:1385, flag:
corners:
7932: v:-1, clink:-1
7933: v:-1, clink:-1
7934: v:-1, clink:-1
7875: v:-1, clink:-1
7844: v:1417, clink:-1
7634: v:1417, clink:7844
7596: v:1417, clink:7634
7876: v:-1, clink:-1
7770: v:1385, clink:7743
7743: v:1385, clink:7713
7713: v:1385, clink:7598
7598: v:1385, clink:-1
7877: v:-1, clink:-1
7842: v:1210, clink:7832
7832: v:1210, clink:7633
7633: v:1210, clink:6646
6646: v:1210, clink:6560
6560: v:1210, clink:-1
7597: v:1405, clink:7831
7831: v:1405, clink:7843
7843: v:1405, clink:7714
7714: v:1405, clink:7654
7654: v:1405, clink:-1
7632: v:1400, clink:-1
edges:
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7932: e_id:-1, op_e:-1, flag:del, seam,
7934: e_id:-1, op_e:-1, flag:del, seam,
7933: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7877: e_id:-1, op_e:-1, flag:del, seam,
7844: e_id:-1, op_e:7633, flag:
7843: e_id:-1, op_e:7596, flag:
7634: e_id:3923, op_e:-1, flag:seam, border,
7633: e_id:3922, op_e:7844, flag:
7596: e_id:3893, op_e:7843, flag:
7598: e_id:4030, op_e:-1, flag:border,
7876: e_id:-1, op_e:-1, flag:del, border,
7875: e_id:-1, op_e:-1, flag:del,
7770: e_id:-1, op_e:7745, flag:
7772: e_id:4007, op_e:-1, flag:border,
7743: e_id:-1, op_e:7715, flag:
7745: e_id:3994, op_e:7770, flag:
7713: e_id:-1, op_e:7597, flag:
7715: e_id:3980, op_e:7743, flag:
7598: e_id:4030, op_e:-1, flag:border,
7597: e_id:3894, op_e:7713, flag:
7877: e_id:-1, op_e:-1, flag:del, seam,
7876: e_id:-1, op_e:-1, flag:del, border,
7842: e_id:-1, op_e:7831, flag:
7844: e_id:-1, op_e:7633, flag:
7832: e_id:-1, op_e:6559, flag:
7831: e_id:4026, op_e:7842, flag:
7633: e_id:3922, op_e:7844, flag:
7632: e_id:3921, op_e:-1, flag:border,
6646: e_id:3439, op_e:-1, flag:border,
6645: e_id:-1, op_e:6560, flag:
6560: e_id:3393, op_e:6645, flag:
6559: e_id:3392, op_e:7832, flag:
7597: e_id:3894, op_e:7713, flag:
7596: e_id:3893, op_e:7843, flag:
7831: e_id:4026, op_e:7842, flag:
7830: e_id:-1, op_e:7654, flag:
7843: e_id:-1, op_e:7596, flag:
7842: e_id:-1, op_e:7831, flag:
7714: e_id:-1, op_e:7653, flag:
7713: e_id:-1, op_e:7597, flag:
7654: e_id:3937, op_e:7830, flag:
7653: e_id:3936, op_e:7714, flag:
7632: e_id:3921, op_e:-1, flag:border,
7634: e_id:3923, op_e:-1, flag:seam, border,






v_cnt:1907, i_cnt:9072
p0: (-13065.237, 4327.8203, -6676.622)
i0: 0, i1: 1, i2: 2
simplify:
## ctx
wedge vert: [1419 1417 1418 1383 ]
wedge union: [0 1 0 1 ]
wedge face: [(7933, 0) (7877, 1) (7844, 3) (7634, 3) (7596, 3) ]
remove face: [7932 7875 ]
opt_p:(-11917.429, 4979.129, -6822.6396)
opt_a: 0:0.5122663, 1:0.40030828, 2:0.9349835, 3:0.011624124, 4:0, 5:0, 6:0, 7:0, 
error: 0.16043260793477138
penalty: 0
v0:1417, v1:1383
1419: p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
1417: p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
1418: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
1383: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
1419: uni:1417, vlink:1417, chead:7933, flag:
1417: uni:1417, vlink:1419, chead:7877, flag:
1418: uni:1383, vlink:1383, chead:7932, flag:
1383: uni:1383, vlink:1418, chead:7875, flag:
c:7933,1, f:2644, i0:1418, i1:1419, i2:1420, flag:
p0:(-11903.24, 4945.5938, -6867.6978), p1:(-11917.12, 4979.8994, -6821.3794), p2:(-11889.021, 4930.594, -6885.1353)
c:7877,2, f:2625, i0:1383, i1:1385, i2:1417, flag:
p0:(-11903.24, 4945.5938, -6867.6978), p1:(-11958.764, 4944.409, -6834.79), p2:(-11917.12, 4979.8994, -6821.3794)
c:7844,2, f:2614, i0:1210, i1:1405, i2:1383, flag:
p0:(-11893.508, 4902.713, -6919.989), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11903.24, 4945.5938, -6867.6978)
c:7634,2, f:2544, i0:1400, i1:1210, i2:1383, flag:
p0:(-11889.021, 4930.594, -6885.1353), p1:(-11893.508, 4902.713, -6919.989), p2:(-11903.24, 4945.5938, -6867.6978)
c:7596,0, f:2532, i0:1383, i1:1405, i2:1385, flag:
p0:(-11903.24, 4945.5938, -6867.6978), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11958.764, 4944.409, -6834.79)
7932: clink:-1, e_id:-1, op_e:7877, flag: seam,
7933: clink:-1, e_id:4043, op_e:-1, flag: border,
7934: clink:-1, e_id:-1, op_e:7634, flag: seam,
7875: clink:7844, e_id:-1, op_e:7598, flag:
7876: clink:7770, e_id:4030, op_e:-1, flag: border,
7877: clink:-1, e_id:4031, op_e:7932, flag: seam,
7842: clink:7832, e_id:-1, op_e:7831, flag:
7843: clink:7714, e_id:-1, op_e:7596, flag:
7844: clink:7634, e_id:-1, op_e:7633, flag:
7632: clink:-1, e_id:3921, op_e:-1, flag: border,
7633: clink:6646, e_id:3922, op_e:7844, flag:
7634: clink:7596, e_id:3923, op_e:7934, flag: seam,
7596: clink:-1, e_id:3893, op_e:7843, flag:
7597: clink:7831, e_id:3894, op_e:7713, flag:
7598: clink:-1, e_id:3895, op_e:7875, flag:


 ### after remove face ###

v0:1016, v1:1016
1419: p:(-11917.12, 4979.8994, -6821.3794), uv:(0.5122511, 0.40030482)
1417: p:(-11917.12, 4979.8994, -6821.3794), uv:(0.9350026, 0.011492229)
1418: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
1383: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
1419: uni:1417, vlink:1417, chead:-1, flag:
1417: uni:1417, vlink:1419, chead:-1, flag:
1418: uni:1383, vlink:1383, chead:-1, flag:
1383: uni:1383, vlink:1418, chead:7844, flag:
c:7933,1, f:2644, i0:-1, i1:-1, i2:-1, flag: del,
p0:(0, 0, 0), p1:(0, 0, 0), p2:(0, 0, 0)
c:7877,2, f:2625, i0:-1, i1:-1, i2:-1, flag: del,
p0:(0, 0, 0), p1:(0, 0, 0), p2:(0, 0, 0)
c:7844,2, f:2614, i0:1210, i1:1405, i2:1383, flag:
p0:(-11893.508, 4902.713, -6919.989), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11903.24, 4945.5938, -6867.6978)
c:7634,2, f:2544, i0:1400, i1:1210, i2:1383, flag:
p0:(-11889.021, 4930.594, -6885.1353), p1:(-11893.508, 4902.713, -6919.989), p2:(-11903.24, 4945.5938, -6867.6978)
c:7596,0, f:2532, i0:1383, i1:1405, i2:1385, flag:
p0:(-11903.24, 4945.5938, -6867.6978), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11958.764, 4944.409, -6834.79)
7932: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7933: clink:-1, e_id:-1, op_e:-1, flag: del, border,
7934: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7875: clink:-1, e_id:-1, op_e:-1, flag: del,
7876: clink:-1, e_id:-1, op_e:-1, flag: del, border,
7877: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7842: clink:7832, e_id:-1, op_e:7831, flag:
7843: clink:7714, e_id:-1, op_e:7596, flag:
7844: clink:7634, e_id:-1, op_e:7633, flag:
7632: clink:-1, e_id:3921, op_e:-1, flag: border,
7633: clink:6646, e_id:3922, op_e:7844, flag:
7634: clink:7596, e_id:3923, op_e:-1, flag: seam, border,
7596: clink:-1, e_id:3893, op_e:7843, flag:
7597: clink:7831, e_id:3894, op_e:7713, flag:
7598: clink:-1, e_id:4030, op_e:-1, flag: border,


 ### after remove vert ###

v0:0, v1:0
1419: p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
1417: p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
1418: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
1383: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
1419: uni:1419, vlink:1417, chead:-1, flag:
1417: uni:1419, vlink:1419, chead:-1, flag:
1418: uni:-1, vlink:-1, chead:-1, flag: del,
1383: uni:-1, vlink:-1, chead:-1, flag: del,
c:7933,1, f:2644, i0:-1, i1:-1, i2:-1, flag: del,
p0:(0, 0, 0), p1:(0, 0, 0), p2:(0, 0, 0)
c:7877,2, f:2625, i0:-1, i1:-1, i2:-1, flag: del,
p0:(0, 0, 0), p1:(0, 0, 0), p2:(0, 0, 0)
c:7844,2, f:2614, i0:1210, i1:1405, i2:1383, flag:
p0:(-11893.508, 4902.713, -6919.989), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11903.24, 4945.5938, -6867.6978)
c:7634,2, f:2544, i0:1400, i1:1210, i2:1383, flag:
p0:(-11889.021, 4930.594, -6885.1353), p1:(-11893.508, 4902.713, -6919.989), p2:(-11903.24, 4945.5938, -6867.6978)
c:7596,0, f:2532, i0:1383, i1:1405, i2:1385, flag:
p0:(-11903.24, 4945.5938, -6867.6978), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11958.764, 4944.409, -6834.79)
7932: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7933: clink:-1, e_id:-1, op_e:-1, flag: del, border,
7934: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7875: clink:-1, e_id:-1, op_e:-1, flag: del,
7876: clink:-1, e_id:-1, op_e:-1, flag: del, border,
7877: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7842: clink:7832, e_id:-1, op_e:7831, flag:
7843: clink:7714, e_id:-1, op_e:7596, flag:
7844: clink:7634, e_id:-1, op_e:7633, flag:
7632: clink:-1, e_id:3921, op_e:-1, flag: border,
7633: clink:6646, e_id:3922, op_e:7844, flag:
7634: clink:7596, e_id:3923, op_e:-1, flag: seam, border,
7596: clink:-1, e_id:3893, op_e:7843, flag:
7597: clink:7831, e_id:3894, op_e:7713, flag:
7598: clink:-1, e_id:4030, op_e:-1, flag: border,


 ### after update mesh ###

v0:0, v1:0
1419: p:(-11917.429, 4979.129, -6822.6396), uv:(0.5122663, 0.40030828)
1417: p:(-11917.429, 4979.129, -6822.6396), uv:(0.9349835, 0.011624124)
1418: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.5128995, 0.4002623)
1383: p:(-11903.24, 4945.5938, -6867.6978), uv:(0.9357311, 0.011334724)
1419: uni:1419, vlink:1417, chead:-1, flag:
1417: uni:1419, vlink:1419, chead:7596, flag:
1418: uni:-1, vlink:-1, chead:-1, flag: del,
1383: uni:-1, vlink:-1, chead:-1, flag: del,
c:7933,1, f:2644, i0:-1, i1:-1, i2:-1, flag: del,
p0:(0, 0, 0), p1:(0, 0, 0), p2:(0, 0, 0)
c:7877,2, f:2625, i0:-1, i1:-1, i2:-1, flag: del,
p0:(0, 0, 0), p1:(0, 0, 0), p2:(0, 0, 0)
c:7844,2, f:2614, i0:1210, i1:1405, i2:1417, flag:
p0:(-11893.508, 4902.713, -6919.989), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11917.429, 4979.129, -6822.6396)
c:7634,2, f:2544, i0:1400, i1:1210, i2:1417, flag:
p0:(-11889.021, 4930.594, -6885.1353), p1:(-11893.508, 4902.713, -6919.989), p2:(-11917.429, 4979.129, -6822.6396)
c:7596,0, f:2532, i0:1417, i1:1405, i2:1385, flag:
p0:(-11917.429, 4979.129, -6822.6396), p1:(-11932.581, 4925.8706, -6870.6626), p2:(-11958.764, 4944.409, -6834.79)
7932: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7933: clink:-1, e_id:-1, op_e:-1, flag: del, border,
7934: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7875: clink:-1, e_id:-1, op_e:-1, flag: del,
7876: clink:-1, e_id:-1, op_e:-1, flag: del, border,
7877: clink:-1, e_id:-1, op_e:-1, flag: del, seam,
7842: clink:7832, e_id:-1, op_e:7831, flag:
7843: clink:7714, e_id:-1, op_e:7596, flag:
7844: clink:-1, e_id:-1, op_e:7633, flag:
7632: clink:-1, e_id:3921, op_e:-1, flag: border,
7633: clink:6646, e_id:3922, op_e:7844, flag:
7634: clink:7844, e_id:3923, op_e:-1, flag: seam, border,
7596: clink:7634, e_id:3893, op_e:7843, flag:
7597: clink:7831, e_id:3894, op_e:7713, flag:
7598: clink:-1, e_id:4030, op_e:-1, flag: border,



*/

/*
simplify:
init_vert_link: 0.0572
init_corner_link: 0.0819
init_edge_link: 0.205
num edge: 8784
avg tri area: 0.0009749861162656414
avg uv area: 8.400447360721317e-05
c:0, v0:0, v1:1
## ctx
wedge vert: [0 1 ]
wedge union: [0 0 ]
wedge face: [(8793, 0) (8788, 0) (8786, 0) (9, 0) (0, 0) (8837, 1) (8833, 1) (57, 1) (5, 1) ]
remove face: [8787 1 ]
opt_p:(0.31406885, -0.4040115, 0.423396)
opt_a: 0:0.7897065, 1:0.668319, 2:0, 3:0,
error: 0.00039723559504739977
penalty: 0


c:1, v0:1, v1:2
## ctx
wedge vert: [1 2 ]
wedge union: [0 0 ]
wedge face: [(8837, 0) (8833, 0) (8787, 0) (57, 0) (5, 0) (1, 0) (8805, 1) (8794, 1) (8792, 1) (18, 1) ]
remove face: [4 2 ]
opt_p:(0.30907768, -0.40754646, 0.42777577)
opt_a: 0:0.78860456, 1:0.66753435, 2:0, 3:0,
error: 0.007557730975124972
penalty: 0


c:2, v0:2, v1:0
## ctx
wedge vert: [2 0 ]
wedge union: [0 0 ]
wedge face: [(8805, 0) (8794, 0) (8792, 0) (18, 0) (4, 0) (2, 0) (8788, 1) (8786, 1) (9, 1) ]
remove face: [8793 0 ]
opt_p:(0.3100639, -0.4091178, 0.4530468)
opt_a: 0:0.7839734, 1:0.6681718, 2:0, 3:0,
error: 0.008021350412244246
penalty: 0


c:3, v0:3, v1:2
## ctx
wedge vert: [3 2 ]
wedge union: [0 0 ]
wedge face: [(87, 0) (59, 0) (55, 0) (23, 0) (19, 0) (3, 0) (8805, 1) (8794, 1) (8792, 1) (2, 1) ]
remove face: [18 4 ]
opt_p:(0.2897807, -0.41660547, 0.42331913)
opt_a: 0:0.78897923, 1:0.6651497, 2:0, 3:0,
error: 0.0010799675506846281
penalty: 0


c:5, v0:1, v1:3
## ctx
wedge vert: [1 3 ]
wedge union: [0 0 ]
wedge face: [(8837, 0) (8833, 0) (8787, 0) (57, 0) (5, 0) (1, 0) (87, 1) (55, 1) (23, 1) (19, 1) ]
remove face: [59 3 ]
opt_p:(0.30716336, -0.40917972, 0.4246489)
opt_a: 0:0.7892995, 1:0.6671944, 2:0, 3:0,
error: 0.0095183126815239
penalty: 0
19.5531
error: 0
v_cnt:3225, f_cnt:5856

## 1, 3
p0:float3(0.313121, -0.40468, 0.424303), p1:float3(0.287063, -0.417912, 0.42339)
opt_p:float3(0.30716336, -0.40917972, 0.4246489)
opt_attr: [0.7892995, 0.6671944, ]
q error:0.009518312681064966
num wedge:1
wedge face:[
        {w:0, wv:1, c:1, v:1}
        {w:0, wv:1, c:5, v:1}
        {w:0, wv:1, c:57, v:1}
        {w:0, wv:1, c:8787, v:1}
        {w:0, wv:1, c:8833, v:1}
        {w:0, wv:1, c:8837, v:1}
        {w:0, wv:1, c:19, v:3}
        {w:0, wv:1, c:23, v:3}
        {w:0, wv:1, c:55, v:3}
        {w:0, wv:1, c:87, v:3}
]




simplify:
init simplifier
compact: 0.1724
unique edge: 2.4665
num edge: 8784
calc area: 0.037399999999999996
avg tri area: 0.0009749861162656414
avg uv area: 8.400447360721317e-05
init_face_quadric: 1.1169
## 0, 1
p0:float3(0.317288, -0.397295, 0.364448), p1:float3(0.313121, -0.40468, 0.424303)
opt_p:float3(0.31406885, -0.4040115, 0.423396)
opt_attr: [0.7897065, 0.668319, ]
q error:0.0003972355959467386
num wedge:1
wedge face:[
        {w:0, wv:0, c:0, v:0}
        {w:0, wv:0, c:9, v:0}
        {w:0, wv:0, c:8786, v:0}
        {w:0, wv:0, c:8788, v:0}
        {w:0, wv:0, c:8793, v:0}
        {w:0, wv:0, c:5, v:1}
        {w:0, wv:0, c:57, v:1}
        {w:0, wv:0, c:8833, v:1}
        {w:0, wv:0, c:8837, v:1}
]
## 1, 2
p0:float3(0.313121, -0.40468, 0.424303), p1:float3(0.289638, -0.411984, 0.363044)
opt_p:float3(0.30907768, -0.40754646, 0.42777577)
opt_attr: [0.78860456, 0.66753435, ]
q error:0.00755773097465358
num wedge:1
wedge face:[
        {w:0, wv:1, c:1, v:1}
        {w:0, wv:1, c:5, v:1}
        {w:0, wv:1, c:57, v:1}
        {w:0, wv:1, c:8787, v:1}
        {w:0, wv:1, c:8833, v:1}
        {w:0, wv:1, c:8837, v:1}
        {w:0, wv:1, c:18, v:2}
        {w:0, wv:1, c:8792, v:2}
        {w:0, wv:1, c:8794, v:2}
        {w:0, wv:1, c:8805, v:2}
]
## 2, 0
p0:float3(0.289638, -0.411984, 0.363044), p1:float3(0.317288, -0.397295, 0.364448)
opt_p:float3(0.3100639, -0.4091178, 0.4530468)
opt_attr: [0.7839734, 0.6681718, ]
q error:0.00802135041179493
num wedge:1
wedge face:[
        {w:0, wv:2, c:2, v:2}
        {w:0, wv:2, c:4, v:2}
        {w:0, wv:2, c:18, v:2}
        {w:0, wv:2, c:8792, v:2}
        {w:0, wv:2, c:8794, v:2}
        {w:0, wv:2, c:8805, v:2}
        {w:0, wv:2, c:9, v:0}
        {w:0, wv:2, c:8786, v:0}
        {w:0, wv:2, c:8788, v:0}
]
## 3, 2
p0:float3(0.287063, -0.417912, 0.42339), p1:float3(0.289638, -0.411984, 0.363044)
opt_p:float3(0.2897807, -0.41660547, 0.42331913)
opt_attr: [0.78897923, 0.6651497, ]
q error:0.0010799675520153605
num wedge:1
wedge face:[
        {w:0, wv:3, c:3, v:3}
        {w:0, wv:3, c:19, v:3}
        {w:0, wv:3, c:23, v:3}
        {w:0, wv:3, c:55, v:3}
        {w:0, wv:3, c:59, v:3}
        {w:0, wv:3, c:87, v:3}
        {w:0, wv:3, c:2, v:2}
        {w:0, wv:3, c:8792, v:2}
        {w:0, wv:3, c:8794, v:2}
        {w:0, wv:3, c:8805, v:2}
]
## 1, 3
p0:float3(0.313121, -0.40468, 0.424303), p1:float3(0.287063, -0.417912, 0.42339)
opt_p:float3(0.30716336, -0.40917972, 0.4246489)
opt_attr: [0.7892995, 0.6671944, ]
q error:0.009518312681064966
num wedge:1
wedge face:[
        {w:0, wv:1, c:1, v:1}
        {w:0, wv:1, c:5, v:1}
        {w:0, wv:1, c:57, v:1}
        {w:0, wv:1, c:8787, v:1}
        {w:0, wv:1, c:8833, v:1}
        {w:0, wv:1, c:8837, v:1}
        {w:0, wv:1, c:19, v:3}
        {w:0, wv:1, c:23, v:3}
        {w:0, wv:1, c:55, v:3}
        {w:0, wv:1, c:87, v:3}
]
73.2389
error: 0
v_cnt:3225, f_cnt:5856
*/