#include <luisa/luisa-compute.h>
#include <stb/stb_image.h>
#include <luisa/gui/window.h>
#include <meshoptimizer.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "camera.h"
#include <gpu_rands.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

struct Mesh {
    vector<float4> positions;
    vector<uint> indices;
    vector<float2> texcoords;

    void load_mesh(const char* file) {
        assimp_load(file);
        compact();
    }
    void assimp_load(const char* file) {
        Clock timer;
        Assimp::Importer importer;

        print("load assimp scene: ");
        timer.tic();

        // assimp 读取时去重花太多内存，我8g会寄
        auto load_flag = 0 
            // | aiProcess_JoinIdenticalVertices 
            | aiProcess_Triangulate
            ;
        const aiScene* scene = importer.ReadFile(file, load_flag);
        print("{} ms\n", timer.toc());

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            print("failed to load scene: {}\n", importer.GetErrorString());
            exit(1);
        }
        auto mesh = scene->mMeshes[0];

        if (!mesh->HasPositions() || !mesh->HasFaces() || !mesh->HasTextureCoords(0)) {
            print("invaild mesh\n");
            exit(1);
        }

        print("convert from assimp scene: ");
        timer.tic();

        positions.resize(mesh->mNumVertices);
        indices.resize(mesh->mNumFaces * 3);
        texcoords.resize(mesh->mNumVertices);

        for(int i = 0; i < mesh->mNumVertices; i++){
            positions[i] = {
                mesh->mVertices[i].x,
                mesh->mVertices[i].y,
                mesh->mVertices[i].z,
                0.f
            };
        }
        for(int i = 0; i < mesh->mNumFaces; i++){
            if (mesh->mFaces[i].mNumIndices != 3) {
                print("not a triangle mesh\n");
                exit(1);
            }
            indices[i * 3] = mesh->mFaces[i].mIndices[0];
            indices[i * 3 + 1] = mesh->mFaces[i].mIndices[1];
            indices[i * 3 + 2] = mesh->mFaces[i].mIndices[2];
        }
        for(int i = 0; i < mesh->mNumVertices; i++){
            texcoords[i] = {
                mesh->mTextureCoords[0][i].x,
                mesh->mTextureCoords[0][i].y,
            };
        }
        print("{} ms\n", timer.toc());
        print("num vertex: {}, num triangle: {}\n", positions.size(), indices.size() / 3);
    }
    void compact() {
        Clock timer;
        print("compact: ");
        timer.tic();

        meshopt_Stream stream[] = {
            {&positions[0], sizeof(float)*4, sizeof(float)*4},
            {&texcoords[0], sizeof(float)*2, sizeof(float)*2},
        };
        uint stream_count = sizeof(stream) / sizeof(stream[0]);
        uint index_count = indices.size();
        vector<uint> remap(index_count);
        uint vertex_count = meshopt_generateVertexRemapMulti(&remap[0], &indices[0], index_count, index_count, stream, stream_count);

        vector<float4> new_positions(vertex_count);
        vector<float2> new_texcoords(vertex_count);

        meshopt_remapIndexBuffer(&indices[0], &indices[0], index_count, &remap[0]);
        meshopt_remapVertexBuffer(&new_positions[0], &positions[0], positions.size(), sizeof(float4), &remap[0]);
        meshopt_remapVertexBuffer(&new_texcoords[0], &texcoords[0], texcoords.size(), sizeof(float2), &remap[0]);

        positions.swap(new_positions);
        texcoords.swap(new_texcoords);

        print("{} ms\n", timer.toc());
        print("num vertex: {}, num triangle: {}\n", positions.size(), indices.size() / 3);
    }
};

struct MeshRenderData {
    Buffer<float3> positions;
    Buffer<uint> indices;
    Buffer<float2> texcoords;
};

LUISA_BINDING_GROUP(MeshRenderData, positions, indices, texcoords) {};

struct MeshletRenderData {
    Buffer<uint4> meshlet;
    Buffer<uint> meshlet_vertices;
    Buffer<uint> meshlet_indices;
};

LUISA_BINDING_GROUP(MeshletRenderData, meshlet, meshlet_vertices, meshlet_indices) {};

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

// struct v2p {
//     float4 pos;
// };
// LUISA_STRUCT(v2p, pos) {};

// RasterStageKernel vert = [](Var<AppData> var) {
//     Var<v2p> out;
//     $uint vid = vertex_id();

//     $array<float3, 6> pos;

//     pos[0] = {-0.5f, 0.5f, 0.5f};
//     pos[1] = {0.5f, 0.5f, 0.5f};
//     pos[2] = {0.0f, -0.5f, 0.5f};

//     pos[3] = {-0.7f, 0.5f, 0.2f};
//     pos[4] = {0.5f, 0.2f, 0.8f};
//     pos[5] = {0.2f, -0.5f, 0.3f};

//     out.pos = make_float4(pos[vid], 1.f);
//     return out;
// };
// RasterStageKernel pixel = [](Var<v2p> in) {
//     return float4(1);
// };
// RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};

struct v2p {
    float4 pos;
    float2 uv;
    float3 color;
};
LUISA_STRUCT(v2p, pos, uv, color) {};

// RasterStageKernel vert = [](Var<AppData> var, $buffer<float3> positions, $buffer<uint> indices, $buffer<float2> texcoords, $float4x4 vp_mat) {
RasterStageKernel vert = [](Var<AppData> var, Var<MeshRenderData> mesh, Var<MeshletRenderData> meshlet, $float4x4 vp_mat, $uint seed) {
    Var<v2p> out;
    $uint vid = vertex_id();
    $uint iid = instance_id();

    $uint4 meshlet_info = meshlet.meshlet.read(iid);
    $if (vid >= meshlet_info.w * 3) {
        out.pos.z = $float(0.0f)/0.0f;
    }
    $else {
        $uint miid = meshlet.meshlet_indices.read(meshlet_info.y + vid);
        $uint mvid = meshlet.meshlet_vertices.read(meshlet_info.x + miid);
        $float3 pos = mesh.positions.read(mvid);
        // $float2 uv = mesh.texcoords.read(mvid);

        out.pos = vp_mat * make_float4(pos, 1.f);
        // out.uv = uv;
        $uint s = tea(iid, seed).x;
        // $uint s = tea(iid * 128 + vid / 3, seed).x;
        out.color = make_float3(
            ((s >> 0) & 255) / 255.0f,
            ((s >> 8) & 255) / 255.0f,
            ((s >> 16) & 255) / 255.0f
        );
    };
    // $uint idx = mesh.indices.read(vid);
    // $float3 pos = mesh.positions.read(idx);
    // $float2 uv = mesh.texcoords.read(idx);
    // // $uint idx = indices.read(vid);
    // // $float3 pos = positions.read(idx);
    // // $float2 uv = texcoords.read(idx);

    // out.pos = vp_mat * make_float4(pos, 1.f);
    // // out.pos = make_float4(pos, 1.f);
    // out.uv = uv;

    // $uint s = tea(vid / 3, 233).x;
    // out.color = make_float3(
	// 	((s >> 0) & 255) / 255.0f,
	// 	((s >> 8) & 255) / 255.0f,
	// 	((s >> 16) & 255) / 255.0f
	// );
    
    return out;
};
RasterStageKernel pixel = [](Var<v2p> in, $image<float> texture, $uint2 texture_dim) {
    // return texture.read($uint2(in.uv.x * texture_dim.x, (1 - in.uv.y) * texture_dim.y));
    // return make_float4(in.uv, 0.f, 1.f);
    return make_float4(in.color, 1.f);
};
RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};

Kernel2D clear_kernel = []($image<float> image) {
    image.write(dispatch_id().xy(), make_float4(0.1f));
};

const uint width = 1920;
const uint height = 1080;

int main(int argc, char *argv[]) {
    Clock timer;

    ::Mesh mesh;

    // mesh.load_mesh("assets/spot_triangulated_good.obj");
    mesh.load_mesh("assets/Nature_Rock_wldhdhxva_8K_3d_ms/wldhdhxva_High.fbx");

    float4 pmin, pmax;
    for (float4 p: mesh.positions) {
        pmin = min(pmin, p);
        pmax = max(pmax, p);
    }
    print("pmin: {}, pmax: {}\n", pmin, pmax);

    // vector<uint> simplify_indices(mesh.indices.size());
    vector<uint> simplify_indices = mesh.indices;

    float threshold = 0.5f;
    uint index_count = mesh.indices.size();
    uint target_index_count = index_count * threshold;
    float target_error = 2.f;
    float lod_error = 0.f;

    float attribute_weights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    uint simplify_index_count = meshopt_simplifyWithAttributes(
        &simplify_indices[0], 
        &mesh.indices[0], 
        index_count, 
        &mesh.positions[0].x, 
        mesh.positions.size(),
        sizeof(float4),
        &mesh.texcoords[0].x,
        sizeof(float2),
        attribute_weights,
        2,
        target_index_count,
        target_error,
        0,
        &lod_error
    );

    // uint simplify_index_count = meshopt_simplify(
    //     &simplify_indices[0], 
    //     &mesh.indices[0], 
    //     index_count, 
    //     &mesh.positions[0].x, 
    //     mesh.positions.size(), 
    //     sizeof(float4), 
    //     target_index_count, 
    //     target_error, 
    //     0, 
    //     &lod_error
    // );
    float error_scale = meshopt_simplifyScale(&mesh.positions[0].x, mesh.positions.size(), sizeof(float4));

    simplify_indices.resize(simplify_index_count);

    print("simplify target: {}\n", simplify_index_count);
    print("simplify error: {}, error: {}, scale: {}\n", lod_error * error_scale, lod_error, error_scale);

    print("building cluster: \n");
    timer.tic();

    const size_t max_vertices = 64;
    const size_t max_triangles = 124;
    const float cone_weight = 0.5f;
    size_t max_meshlets = meshopt_buildMeshletsBound(simplify_indices.size(), max_vertices, max_triangles);
    print("max_meshlets: {}\n", max_meshlets);

    vector<meshopt_Meshlet> meshlets(max_meshlets);
    vector<uint> meshlet_vertices(max_meshlets * max_vertices);
    vector<ubyte> meshlet_triangles(max_meshlets * max_triangles * 3);
    size_t meshlet_count = meshopt_buildMeshlets(
        meshlets.data(), 
        meshlet_vertices.data(), 
        meshlet_triangles.data(), 
        simplify_indices.data(),
        simplify_indices.size(), 
        &mesh.positions[0].x, 
        mesh.positions.size(), 
        sizeof(float4), 
        max_vertices, 
        max_triangles, 
        cone_weight
    );
    print("meshlet_count: {}\n", meshlet_count);
    print("{} ms\n", timer.toc());

    const meshopt_Meshlet& last = meshlets[meshlet_count - 1];
    meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
    meshlet_triangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
    meshlets.resize(meshlet_count);

    vector<uint> meshlet_indices(meshlet_triangles.size());
    for (int i = 0; i < meshlet_indices.size(); i++) {
        meshlet_indices[i] = meshlet_triangles[i];
    } 

    Img texture;
    texture.load("assets/spot_texture.png");

    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

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

    MeshFormat mesh_format;
    auto clear_shader = device.compile(clear_kernel);
    auto shader = device.compile(kernel, mesh_format);

    RasterState state {
        .cull_mode = CullMode::None,
        .depth_state = {
            .enable_depth = true,
            .comparison = Comparison::Greater,
            .write = true
        }
    };

    vector<RasterMesh> meshes;

    MeshRenderData mesh_renderdata;
    mesh_renderdata.positions = device.create_buffer<float3>(mesh.positions.size());
    // mesh_renderdata.indices = device.create_buffer<uint>(mesh.indices.size());
    mesh_renderdata.indices = device.create_buffer<uint>(simplify_indices.size());
    mesh_renderdata.texcoords = device.create_buffer<float2>(mesh.texcoords.size());
    // auto positions = device.create_buffer<float3>(mesh.positions.size());
    // // auto indices = device.create_buffer<uint>(mesh.indices.size());
    // auto indices = device.create_buffer<uint>(simplify_index_count);
    // auto texcoords = device.create_buffer<float2>(mesh.texcoords.size());

    MeshletRenderData meshlet_renderdata;
    meshlet_renderdata.meshlet = device.create_buffer<uint4>(meshlet_count);
    meshlet_renderdata.meshlet_vertices = device.create_buffer<uint>(meshlet_vertices.size());
    meshlet_renderdata.meshlet_indices = device.create_buffer<uint>(meshlet_indices.size());

    auto texture_img = device.create_image<float>(PixelStorage::BYTE4, texture.width, texture.height);

    stream
        << mesh_renderdata.positions.copy_from(mesh.positions.data())
        // << mesh_renderdata.indices.copy_from(mesh.indices.data())
        << mesh_renderdata.indices.copy_from(simplify_indices.data())
        << mesh_renderdata.texcoords.copy_from(mesh.texcoords.data())
        // << positions.copy_from(mesh.positions.data())
        // // << indices.copy_from(mesh.indices.data())
        // << indices.copy_from(simplify_indices.data())
        // << texcoords.copy_from(mesh.texcoords.data())
        << meshlet_renderdata.meshlet.copy_from(meshlets.data())
        << meshlet_renderdata.meshlet_vertices.copy_from(meshlet_vertices.data())
        << meshlet_renderdata.meshlet_indices.copy_from(meshlet_indices.data())
        << texture_img.copy_from(texture.pixels)
        << synchronize();

    Camera camera {
        .position = float3(0, 0, 10),
        .yaw = -90,
        .pitch = 0
    };

    timer.tic();

    uint frame_cnt = 0;
    float2 lst_cursor_pos;
    float lst_print_time = 0;
    float speed = 1;
    uint seed = 19;

    window.set_cursor_disabled();
    bool is_cursor_disabled=true;

    while (!window.should_close()) {
        float tick_time = timer.toc() / 1000;
        // print("time: {}, tick: {}\n", timer.toc(), tick_time);
        timer.tic();

        if (window.is_key_down(KEY_W)) camera.move_front(tick_time * speed);
        if (window.is_key_down(KEY_S)) camera.move_front(-tick_time * speed);
        if (window.is_key_down(KEY_A)) camera.move_right(-tick_time * speed);
        if (window.is_key_down(KEY_D)) camera.move_right(tick_time * speed);

        if (window.is_key_down(KEY_C)) speed *= 0.9;
        if (window.is_key_down(KEY_V)) speed *= 1.1;

        if (window.is_key_down(KEY_I)) seed = rand();

        if (is_cursor_disabled) {
            if (window.is_key_down(KEY_B)) {
                window.set_cursor_normal();
                lst_cursor_pos=window.get_cursor_pos();
                is_cursor_disabled = false;
            }
        }
        else {
            if (!window.is_key_down(KEY_B)) {
                window.set_cursor_disabled();
                is_cursor_disabled = true;
            }
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

        // luisa::float3 up = { 0, 1, 0 };
        // luisa::float3 front = camera.direction();
        // luisa::float3 right = normalize(cross(front, up));
        // up = luisa::cross(right, front);
        
        // luisa::float4x4 rotate;
        // rotate[0] = luisa::make_float4(right, 0);
        // rotate[1] = luisa::make_float4(up, 0);
        // rotate[2] = luisa::make_float4(-front, 0);
        // rotate[3] = luisa::make_float4(0, 0, 0, 1);

        // luisa::float4x4 move;
        // move[0] = luisa::make_float4(1, 0, 0, 0);
        // move[1] = luisa::make_float4(0, 1, 0, 0);
        // move[2] = luisa::make_float4(0, 0, 1, 0);
        // move[3] = luisa::make_float4(-camera.position, 1);

        // lst_print_time += tick_time;
        // if (lst_print_time > 0.5) {
        //     float3 p = mesh.positions[0];
        //     float4 vp = v_mat * make_float4(p, 1);
        //     float4 pp = vp_mat * make_float4(p, 1);
        //     print("pos: {}, dir: {}\n", camera.position, camera.direction());
        //     print("r_mat: {}\n", rotate);
        //     print("m_mat: {}\n", move);
        //     print("v_mat: {}\n", v_mat);
        //     print("p_mat: {}\n", p_mat);
        //     print("p:  {}\n", p);
        //     print("vp: {}\n", vp);
        //     print("pp: {}\n\n", pp);
        //     lst_print_time -= 0.5;
        // }

        // meshes.emplace_back(span<VertexBufferView>{}, mesh.indices.size(), 1, 0);
        // meshes.emplace_back(span<VertexBufferView>{}, simplify_indices.size(), 1, 0);
        meshes.emplace_back(span<VertexBufferView>{}, 128*3, meshlets.size(), 0);
        stream
            << depth.clear(0)
            << clear_shader(out_img).dispatch(width, height)
            // << shader(positions, indices, texcoords, vp_mat, texture_img, uint2(texture.width, texture.height)).draw(
            // << shader(positions, indices, texcoords, vp_mat).draw(
            << shader(mesh_renderdata, meshlet_renderdata, vp_mat, seed, texture_img, uint2(texture.width, texture.height)).draw(
                std::move(meshes),
                Viewport{0.f, 0.f, float(width), float(height)}, state, 
                &depth,
                out_img
            )
            << swap_chain.present(out_img);
            // << synchronize();
        window.poll_events();
        // break;
    }
    stream << synchronize();
    return 0;
}