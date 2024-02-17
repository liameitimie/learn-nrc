#include <virtual_mesh.h>
#include <luisa/luisa-compute.h>
#include <luisa/gui/window.h>
#include <gpu_rands.h>
#include "camera.h"

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;
using namespace virtual_mesh;

template<typename T>
void pack_vector(const vector<T>& v, vector<uint>& buf) {
    buf.push_back(v.size());
    int offset = buf.size();
    buf.push_back_uninitialized(v.size() * sizeof(T) / sizeof(uint));
    memcpy(buf.data() + offset, v.data(), v.size() * sizeof(T));
}

template<typename T>
void unpack_vector(vector<T>& v, const vector<uint>& buf, int &offset) {
    int size = buf[offset];
    v.resize_uninitialized(size);
    memcpy(v.data(), buf.data() + offset, size * sizeof(T));
    offset += size * sizeof(T) / sizeof(uint);
}

vector<uint> pack_virtual_mesh(const VirtualMesh& virtual_mesh) {
    vector<uint> buf;

    buf.push_back(virtual_mesh.clusters.size());
    for (auto& cluster: virtual_mesh.clusters) {
        pack_vector(cluster.positions, buf);
        pack_vector(cluster.texcoords, buf);
        pack_vector(cluster.indices, buf);

        buf.push_back(cluster.group_id);
    }

    buf.push_back(virtual_mesh.cluster_groups.size());
    for (auto& group: virtual_mesh.cluster_groups) {
        pack_vector(group.clusters, buf);
    }

    pack_vector(virtual_mesh.levels, buf);

    return buf;
}

VirtualMesh unpack_virtual_mesh(const vector<uint>& buf) {
    VirtualMesh virtual_mesh;
    int offset = 0;

    virtual_mesh.clusters.resize(buf[offset]);
    offset++;
    for (auto& cluster: virtual_mesh.clusters) {
        unpack_vector(cluster.positions, buf, offset);
        unpack_vector(cluster.texcoords, buf, offset);
        unpack_vector(cluster.indices, buf, offset);

        cluster.group_id = buf[offset];
        offset++;
    }

    virtual_mesh.cluster_groups.resize(buf[offset]);
    offset++;
    for (auto& group: virtual_mesh.cluster_groups) {
        unpack_vector(group.clusters, buf, offset);
    }

    unpack_vector(virtual_mesh.levels, buf, offset);

    return virtual_mesh;
}

void virtual_mesh_info(const VirtualMesh& virtual_mesh) {
    {
        int mf = 999999, Mf = 0;
        int mv = 999999, Mv = 0;
        int cv = 0, cf = 0;
        for (auto& cluster: virtual_mesh.clusters) {
            cf += cluster.indices.size() / 3;
            mf = min(mf, (int)cluster.indices.size() / 3);
            Mf = max(Mf, (int)cluster.indices.size() / 3);

            cv += cluster.positions.size();
            mv = min(mv, (int)cluster.positions.size());
            Mv = max(Mv, (int)cluster.positions.size());
        }
        print("{}\n", virtual_mesh.clusters.size());
        print("Mf: {}, mf: {}, avg_f: {}\n", Mf, mf, (double)cf / virtual_mesh.clusters.size());
        print("Mv: {}, mv: {}, avg_v: {}\n", Mv, mv, (double)cv / virtual_mesh.clusters.size());
    }
    {
        int m = 999999, M = 0;
        for (auto& group: virtual_mesh.cluster_groups) {
            m = min(m, (int)group.clusters.size());
            M = max(M, (int)group.clusters.size());
        }
        print("{}\n", virtual_mesh.cluster_groups.size());
        print("M: {}, m: {}\n", M, m);
    }
    print("{}\n", virtual_mesh.levels.size());
}

void write_file(const char* file, const VirtualMesh& virtual_mesh) {

}

struct v2p {
    float4 pos;
    float2 uv;
    luisa::float3 color;
};
LUISA_STRUCT(v2p, pos, uv, color) {};

struct ClusterData {
    Buffer<float4> positions;
    Buffer<uint> indices;
    Buffer<float2> texcoords;
};
LUISA_BINDING_GROUP(ClusterData, positions, indices, texcoords) {};

struct ClusterInfo {
    int vertex_offset;
	int triangle_offset;
    int vertex_count;
	int triangle_count;
    int group_id;
};
LUISA_STRUCT(ClusterInfo, vertex_offset, triangle_offset, vertex_count, triangle_count, group_id) {};

RasterStageKernel vert = [](Var<AppData> var, Var<ClusterData> cluster_data, $buffer<ClusterInfo> cluster_info, $uint level_offset, $float4x4 vp_mat, $int seed, $int show_state) {
    Var<v2p> out;
    $uint vert_id = vertex_id();
    $uint inst_id = instance_id();

    Var<ClusterInfo> info = cluster_info.read(inst_id + level_offset);
    $if (vert_id >= info.triangle_count * 3) {
        out.pos.z = $float(0.0f)/0.0f;
    }
    $else {
        $int v_idx = cluster_data.indices.read(info.triangle_offset + vert_id);
        $float4 pos = cluster_data.positions.read(info.vertex_offset + v_idx);
        $float2 uv = cluster_data.texcoords.read(info.vertex_offset + v_idx);

        pos.w = 1;
        out.pos = vp_mat * pos;

        $uint seed0;
        $switch (show_state) {
            $case (0) { seed0 = inst_id * 128 + vert_id / 3; };
            $case (1) { seed0 = inst_id; };
            $case (2) { seed0 = info.group_id; };
        };

        $if (show_state <= 2) {
            $uint s = tea(seed0, seed).x;
            out.color = make_float3(
                ((s >> 0) & 255) / 255.0f,
                ((s >> 8) & 255) / 255.0f,
                ((s >> 16) & 255) / 255.0f
            );
        } $else {
            out.color = make_float3(uv, 0.0f);
        };
    };
    return out;
};
RasterStageKernel pixel = [](Var<v2p> in) {
    return make_float4(in.color, 1.f);
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

template<typename T>
void push_back_vec(vector<T>& dst, const vector<T>& src) {
    int offset = dst.size();
    dst.push_back_uninitialized(src.size());
    memcpy(dst.data() + offset, src.data(), src.size() * sizeof(T));
}

int main(int argc, char** argv) {
    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    ClusterData cluster_data;
    Buffer<ClusterInfo> cluster_info;
    vector<VirtualMeshLevel> level_infos;
    {
        vector<float4> positions;
        vector<float2> texcoords;
        vector<int> indices;
        vector<ClusterInfo> infos;
        
        {
            VirtualMesh virtual_mesh;
            {
                ::Mesh mesh = load_mesh("assets/spot_triangulated_good.obj");
                // virtual_mesh::Mesh mesh = load_mesh("assets/Nature_Rock_wldhdhxva_8K_3d_ms/wldhdhxva_High.fbx");
                // virtual_mesh::Mesh mesh = load_mesh("assets/Font_Reconstructed.stl");
                // virtual_mesh::Mesh mesh = load_mesh("assets/SM_Gate_A.FBX");
                virtual_mesh = build_virtual_mesh(mesh);
            }
            virtual_mesh_info(virtual_mesh);

            for (auto& cluster: virtual_mesh.clusters) {
                infos.push_back(
                    ClusterInfo {
                        .vertex_offset = (int)positions.size(),
                        .triangle_offset = (int)indices.size(),
                        .vertex_count = (int)cluster.positions.size(),
                        .triangle_count = (int)cluster.indices.size() / 3,
                        .group_id = cluster.group_id
                    }
                );
                // push_back_vec(positions, cluster.positions);
                for (auto p: cluster.positions) {
                    positions.push_back({p.x, p.y, p.z, 1});
                }
                push_back_vec(texcoords, cluster.texcoords);
                push_back_vec(indices, cluster.indices);
            }
            level_infos.swap(virtual_mesh.levels);
        }

        cluster_data.positions = device.create_buffer<float4>(positions.size());
        cluster_data.texcoords = device.create_buffer<float2>(texcoords.size());
        cluster_data.indices = device.create_buffer<uint>(indices.size());
        cluster_info = device.create_buffer<ClusterInfo>(infos.size());

        stream
            << cluster_data.positions.copy_from(positions.data())
            << cluster_data.texcoords.copy_from(texcoords.data())
            << cluster_data.indices.copy_from(indices.data())
            << cluster_info.copy_from(infos.data());
    }

    MeshFormat mesh_format;
    auto clear_shader = device.compile(clear_kernel);
    auto shader = device.compile(kernel, mesh_format);

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

    Camera camera {
        .position = luisa::float3(0, 0, 10),
        .yaw = -90,
        .pitch = 0,
        .move_speed = 5000
    };

    uint frame_cnt = 0;
    float2 lst_cursor_pos;
    float speed = 1;
    uint seed = 19;
    int show_state = 0; // 0: triangle, 1: cluster, 2: group, 3: uv
    int level = 0;

    window.set_cursor_disabled();
    bool is_cursor_disabled=true;

    Clock timer;
    timer.tic();

    vector<RasterMesh> meshes;

    while (!window.should_close()) {
        float tick_time = timer.toc() / 1000;
        timer.tic();

        if (window.is_key_down(KEY_W)) camera.move_front(tick_time * speed);
        if (window.is_key_down(KEY_S)) camera.move_front(-tick_time * speed);
        if (window.is_key_down(KEY_A)) camera.move_right(-tick_time * speed);
        if (window.is_key_down(KEY_D)) camera.move_right(tick_time * speed);

        if (window.is_key_down(KEY_C)) speed *= 0.9;
        if (window.is_key_down(KEY_V)) speed *= 1.1;

        if (is_key_switch(window, KEY_I) == 1) seed = rand();

        // switch show level (0: triangle, 1: cluster, 2: group, 3: uv)
        if (is_key_switch(window, KEY_J) == 1) show_state = 0;
        if (is_key_switch(window, KEY_K) == 1) show_state = 1;
        if (is_key_switch(window, KEY_L) == 1) show_state = 2;
        if (is_key_switch(window, KEY_M) == 1) show_state = 3;

        if (is_key_switch(window, KEY_0) == 1) level = 0;
        if (is_key_switch(window, KEY_1) == 1) level = 1;
        if (is_key_switch(window, KEY_2) == 1) level = 2;
        if (is_key_switch(window, KEY_3) == 1) level = 3;
        if (is_key_switch(window, KEY_4) == 1) level = 4;
        if (is_key_switch(window, KEY_5) == 1) level = 5;
        if (is_key_switch(window, KEY_6) == 1) level = 6;
        if (is_key_switch(window, KEY_7) == 1) level = 7;
        if (is_key_switch(window, KEY_8) == 1) level = 8;
        if (is_key_switch(window, KEY_9) == 1) level = 9;

        if (is_key_switch(window, KEY_MINUS) == 1) level--;
        if (is_key_switch(window, KEY_EQUAL) == 1) level++;

        level = (level % level_infos.size() + level_infos.size()) % level_infos.size();

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

        meshes.emplace_back(span<VertexBufferView>{}, 128*3, level_infos[level].cluster_count, 0);
        stream
            << depth.clear(0)
            << clear_shader(out_img).dispatch(width, height)
            << shader(cluster_data, cluster_info, level_infos[level].cluster_offset, vp_mat, seed, show_state).draw(
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