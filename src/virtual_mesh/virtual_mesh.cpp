#include <virtual_mesh.h>
#include <partitioner.h>
#include <meshoptimizer.h>
#include <metis.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/core/mathematics.h>

#include <algorithm>
#include <mesh_simplify/mesh_simplify.h>

using namespace luisa;
using namespace fmt;

// template<>
// struct equal_to<float4> {
//     bool operator()(const float4 &a, const float4 &b) const {
//         return memcmp(&a, &b, sizeof(float4)) == 0;
//     }
// };

namespace virtual_mesh {

vector<Cluster> cluster_triangles(Mesh &mesh) {
    vector<Cluster> clusters;

    // int max_meshlets = meshopt_buildMeshletsBound(mesh.indices.size(), Cluster::max_vertices, Cluster::max_triangles);
    // int max_meshlets = mesh.indices.size() / 3 / Cluster::max_triangles * 2;

    // vector<meshopt_Meshlet> meshlets(max_meshlets);
    // vector<uint> meshlet_vertices(max_meshlets * Cluster::max_vertices);
    // vector<ubyte> meshlet_triangles(max_meshlets * Cluster::max_triangles * 3);

    // const float cone_weight = 0.0f; // 三角形法线方向影响分簇贪心权重，考虑法线方向似乎效果不是很好
    // int meshlet_count = meshopt_buildMeshlets(
    //     meshlets.data(), 
    //     meshlet_vertices.data(), 
    //     meshlet_triangles.data(), 
    //     mesh.indices.data(),
    //     mesh.indices.size(), 
    //     &mesh.positions[0].x, 
    //     mesh.positions.size(), 
    //     sizeof(float3), 
    //     Cluster::max_vertices, 
    //     Cluster::max_triangles, 
    //     cone_weight
    // );

    // if (meshlet_count > max_meshlets) {
    //     print("error: meshlet_count > max_meshlets\n");
    //     exit(1);
    // }

    // clusters.resize(meshlet_count);
    // for (int i = 0; i < meshlet_count; i++) {
    //     auto &cluster = clusters[i];
    //     meshopt_Meshlet cluster_info = meshlets[i];

    //     cluster.indices.resize(cluster_info.triangle_count * 3);
    //     cluster.positions.resize(cluster_info.vertex_count);
    //     cluster.texcoords.resize(cluster_info.vertex_count);

    //     for (int j = 0; j < cluster_info.triangle_count * 3; j++) {
    //         cluster.indices[j] = meshlet_triangles[j + cluster_info.triangle_offset];
    //     }
    //     for (int j = 0; j < cluster_info.vertex_count; j++) {
    //         int vertex_id = meshlet_vertices[j + cluster_info.vertex_offset];
    //         cluster.positions[j] = mesh.positions[vertex_id];
    //         cluster.texcoords[j] = mesh.texcoords[vertex_id];
    //     }
    // }

    MeshStructure m{mesh};
    m.init_vert_link();
    m.init_corner_link();
    m.init_edge_link();

    MetisGraph g;
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

    auto [n_part, part_id] = partition(g, 128);

    clusters.resize(n_part);

    vector<int> part_size(n_part);
    vector<int> vert_head(n_part, -1);
    vector<int> next_vert(g.num_vert, -1); // 每个part一个链表，连接该part所有的顶点

    for (int i = g.num_vert - 1; i >= 0; i--) {
        int id = part_id[i];
        next_vert[i] = vert_head[id];
        vert_head[id] = i;

        part_size[id]++;
    }
    for (int i = 0; i < n_part; i++) {
        clusters[i].indices.reserve(part_size[i] * 3);
        clusters[i].positions.reserve(part_size[i] * 0.75);
        clusters[i].texcoords.reserve(part_size[i] * 0.75);
    }

    vector<int> vert_tag(m.vertex_count, -1);
    vector<int> verts;
    verts.reserve(Cluster::max_triangles);

    for (int i = 0; i < n_part; i++) {
        auto &cluster = clusters[i];
        for (int f = vert_head[i]; f != -1; f = next_vert[f]) {
            for (int k = 0; k < 3; k++) {
                int v = m.mesh.indices[f * 3 + k];
                if (vert_tag[v] == -1) {
                    vert_tag[v] = verts.size();
                    verts.push_back(v);
                    cluster.positions.push_back(m.mesh.positions[v]);
                    cluster.texcoords.push_back(m.mesh.texcoords[v]);
                }
                v = vert_tag[v];
                cluster.indices.push_back(v);
            }
        }
        for (int v: verts) vert_tag[v] = -1;
        verts.clear();
    }

    return clusters;
}

struct MetisGraph1 {
    int nvtxs;
    vector<int> xadj;
    vector<int> adjncy; //压缩图表示
    vector<int> adjwgt; //边权重
};

vector<ClusterGroup> grouping_clusters(span<Cluster> clusters) {
    vector<ClusterGroup> groups;

    if (clusters.size() < ClusterGroup::max_group_size) {
        groups.resize(1);
        groups[0].clusters.resize(clusters.size());
        for (int i = 0; i < clusters.size(); i++) {
            groups[0].clusters[i] = i;
        }
        return groups;
    }

    MetisGraph1 metis_graph; // cluster连接关系图，用于metis图划分
    vector<int> part; // 划分结果，下标表示顶点，值表示顶点划分后所在的组

    Clock timer;
    timer.tic();

    print("building graph: ");
    {
        // 假设cluster每个顶点邻接关系较少，使用vector_map/vector_set
        vector<vector_map<int, int>> cluster_graph(clusters.size());
        int edge_count = 0;
        {
            // 记录position到cluster id的映射
            unordered_map<float3, vector_set<int>> cluster_id;
            int id = 0;
            for (auto& cluster: clusters) {
                for (auto position: cluster.positions) {
                    cluster_id[position].insert(id);
                }
                id++;
            }

            // 建立cluster邻接关系，以相邻顶点为边权
            id = 0;
            for (auto& cluster: clusters) {
                for (auto position: cluster.positions) {
                    for (int link_id: cluster_id[position]) {
                        if (link_id != id) {
                            cluster_graph[id][link_id]++;
                        }
                    }
                }
                edge_count += cluster_graph[id].size();
                id++;
            }
        }

        metis_graph.nvtxs = clusters.size();
        metis_graph.xadj.reserve(clusters.size() + 1);
        metis_graph.adjncy.reserve(edge_count);
        metis_graph.adjwgt.reserve(edge_count);

        for (auto& map: cluster_graph) {
            metis_graph.xadj.push_back(metis_graph.adjncy.size());
            for (auto [to, cost]: map) {
                metis_graph.adjncy.push_back(to);
                metis_graph.adjwgt.push_back(cost);
            }
        }
        metis_graph.xadj.push_back(metis_graph.adjncy.size());
    }
    print("{}\n", timer.toc());

    print("part graph: ");
    timer.tic();
    {
        part.resize(clusters.size());

        int nparts = 1.25 * clusters.size() / ClusterGroup::max_group_size;
        nparts = std::max(nparts, 2);
        int ncon = 1, ncut = 0;

        // kway方法会有很多被独立出来的cluster，使用递归划分方法

        int result = METIS_PartGraphRecursive(
            &metis_graph.nvtxs,
            &ncon,
            metis_graph.xadj.data(),
            metis_graph.adjncy.data(),
            nullptr,  // vertex weights
            nullptr,  // vertex size
            metis_graph.adjwgt.data(),
            &nparts,
            nullptr,  // partition weight
            nullptr,  // constraint
            nullptr,  // options
            &ncut,
            part.data()
        );

        if (result != METIS_OK) [[unlikely]] {
            print("error in metis graph partioner: ");
            switch (result) {
                case METIS_ERROR_INPUT: print("error input\n"); break;
                case METIS_ERROR_MEMORY: print("insufficient memory\n"); break;
                default: print("unknown error\n");
            }
            exit(1);
        }
        print("finish part: {}, ", timer.toc());
        timer.tic();

        groups.resize(nparts);

        print("copy: {}\n", timer.toc());

        int id = 0;
        for (int group_id: part) {
            groups[group_id].clusters.push_back(id);
            clusters[id].group_id = group_id;
            id++;
        }
    }

    return groups;
}

// float simplify(Mesh &mesh) {
//     vector<int> simplify_indices(mesh.indices.size());

//     float threshold = 0.5f;
//     uint index_count = mesh.indices.size();
//     uint target_index_count = index_count * threshold;
//     float target_error = 2.f;
//     float lod_error = 0.f;

//     float attribute_weights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
//     uint simplify_index_count = meshopt_simplifyWithAttributes(
//         &simplify_indices[0],
//         &mesh.indices[0],
//         index_count,
//         &mesh.positions[0].x,
//         mesh.positions.size(),
//         sizeof(float3),
//         &mesh.texcoords[0].x,
//         sizeof(luisa::float2),
//         attribute_weights,
//         2,
//         target_index_count,
//         target_error,
//         meshopt_SimplifyLockBorder,
//         &lod_error
//     );

//     simplify_indices.resize(simplify_index_count);
//     mesh.indices.swap(simplify_indices);

//     float error_scale = meshopt_simplifyScale(&mesh.positions[0].x, mesh.positions.size(), sizeof(float3));
//     return lod_error * error_scale;
// }

VirtualMeshLevel build_parent_level(VirtualMesh &virtual_mesh, VirtualMeshLevel level) {
    Clock timer;
    VirtualMeshLevel cur_level = {
        .cluster_offset = (int)virtual_mesh.clusters.size(),
        .group_offset = (int)virtual_mesh.cluster_groups.size(),
        .cluster_count = 0,
        .group_count = 0
    };

    print("simplify and clustering: \n");
    timer.tic();

    double time1 = 0;
    double time2 = 0;
    double time3 = 0;

    print("find locked pos: ");

    unordered_set<float3> locked_pos;
    {
        unordered_map<float3, vector_set<int>> group_id;
        for (int i = 0; i < level.group_count; i++) {
            auto& group = virtual_mesh.cluster_groups[level.group_offset + i];
            for (int cluster_id: group.clusters) {
                auto& cluster = virtual_mesh.clusters[cluster_id];
                for (float3 p: cluster.positions) {
                    group_id[p].insert(i);
                }
            }
        }
        for (auto &[p, set]: group_id) {
            if (set.size() > 1) {
                locked_pos.insert(p);
            }
        }
    }

    print("{}\n", timer.toc());
    timer.tic();

    // 将每个组包含的簇组合为mesh，用于简化并生成上一级level
    for (int i = 0; i < level.group_count; i++) {
        const auto& group = virtual_mesh.cluster_groups[level.group_offset + i];
        Mesh mesh;

        // copy time
        timer.tic();

        int vertex_count = 0;
        int index_count = 0;
        for (int cluster_id: group.clusters) {
            const auto& cluster = virtual_mesh.clusters[cluster_id];
            vertex_count += cluster.positions.size();
            index_count += cluster.indices.size();
        }
        mesh.positions.resize(vertex_count);
        mesh.texcoords.resize(vertex_count);
        mesh.indices.resize(index_count);

        int vertex_offset = 0;
        int index_offset = 0;
        for (int cluster_id: group.clusters) {
            const auto& cluster = virtual_mesh.clusters[cluster_id];
            memcpy(mesh.positions.data() + vertex_offset, cluster.positions.data(), cluster.positions.size() * sizeof(float3));
            memcpy(mesh.texcoords.data() + vertex_offset, cluster.texcoords.data(), cluster.texcoords.size() * sizeof(float2));

            for (int j = 0; j < cluster.indices.size(); j++) {
                mesh.indices[j + index_offset] = cluster.indices[j] + vertex_offset;
            }

            vertex_offset += cluster.positions.size();
            index_offset += cluster.indices.size();
        }

        time1 += timer.toc();

        // if (i == 225) {
        //     debug_out(mesh, "debug_mesh.txt");
        // }

        // 将group的mesh简化后，再拆分为cluster，作为上一级level
        // mesh.compact();

        // simplify time
        timer.tic();

        MeshSimplifier simplifier(mesh);

        for (float3 p: mesh.positions) {
            if (locked_pos.contains(p)) {
                simplifier.lock_position(p);
            }
        }
        
        simplifier.simplify(mesh.indices.size() / 3 / 2);

        // mesh_simplify(mesh, mesh.indices.size() / 3 / 2, locked_pos);

        time2 += timer.toc();

        // cluster time
        timer.tic();

        auto clusters = cluster_triangles(mesh);
        cur_level.cluster_count += clusters.size();

        time3 += timer.toc();

        for (auto &cluster: clusters) {
            virtual_mesh.clusters.push_back(std::move(cluster));
        }
    }
    // print("{} ms\n", timer.toc());
    print("copy mesh: {}\n", time1);
    print("simplify: {}\n", time2);
    print("cluster: {}\n", time3);

    print("grouping clusters\n");
    timer.tic();

    span<Cluster> level_clusters(virtual_mesh.clusters.data() + cur_level.cluster_offset, cur_level.cluster_count);

    auto groups = grouping_clusters(level_clusters);
    cur_level.group_count = groups.size();

    

    for (auto &group: groups) {
        for (int &cluster_id: group.clusters) {
            level_clusters[cluster_id].group_id += cur_level.group_offset;
            cluster_id += cur_level.cluster_offset;
        }
        virtual_mesh.cluster_groups.push_back(std::move(group));
    }
    // print("{} ms\n", timer.toc());
    print("level, cluster num: {}, group num: {}\n", cur_level.cluster_count, cur_level.group_count);

    // exit(1);

    // logging info
    {
        vector<int> cluster_vertex_sizes;
        vector<int> cluster_index_sizes;
        vector<int> group_sizes;

        int sum_v = 0;
        int sum_f = 0;
        int sum_g = 0;

        for (auto &cluster: level_clusters) {
            cluster_vertex_sizes.push_back(cluster.positions.size());
            cluster_index_sizes.push_back(cluster.indices.size() / 3);

            sum_v += cluster.positions.size();
            sum_f += cluster.indices.size() / 3;
        }
        for (auto &group: span<ClusterGroup>(virtual_mesh.cluster_groups.data() + cur_level.group_offset, cur_level.group_count)) {
            group_sizes.push_back(group.clusters.size());

            sum_g += group.clusters.size();
        }

        std::sort(cluster_vertex_sizes.begin(), cluster_vertex_sizes.end());
        std::sort(cluster_index_sizes.begin(), cluster_index_sizes.end());
        std::sort(group_sizes.begin(), group_sizes.end());

        print("vertex size: ");
        print("min3: ");
        for (int i = 0; i < 3; i++) print("{}, ", cluster_vertex_sizes[i]);
        print("mid3: ");
        for (int i = 0; i < 3; i++) print("{}, ", cluster_vertex_sizes[cluster_vertex_sizes.size() / 2 - 1 + i]);
        print("max3: ");
        for (int i = 0; i < 3; i++) print("{}, ", cluster_vertex_sizes[cluster_vertex_sizes.size() - 3 + i]);
        print("avg: {}\n", (float)sum_v / cluster_vertex_sizes.size());

        print("index size: ");
        print("min3: ");
        for (int i = 0; i < 3; i++) print("{}, ", cluster_index_sizes[i]);
        print("mid3: ");
        for (int i = 0; i < 3; i++) print("{}, ", cluster_index_sizes[cluster_index_sizes.size() / 2 - 1 + i]);
        print("max3: ");
        for (int i = 0; i < 3; i++) print("{}, ", cluster_index_sizes[cluster_index_sizes.size() - 3 + i]);
        print("avg: {}\n", (float)sum_f / cluster_index_sizes.size());

        print("group size: ");
        print("min3: ");
        for (int i = 0; i < 3; i++) print("{}, ", group_sizes[i]);
        print("mid3: ");
        for (int i = 0; i < 3; i++) print("{}, ", group_sizes[group_sizes.size() / 2 - 1 + i]);
        print("max3: ");
        for (int i = 0; i < 3; i++) print("{}, ", group_sizes[group_sizes.size() - 3 + i]);
        print("avg: {}\n", (float)sum_g / group_sizes.size());

    }
    // exit(1);

    return cur_level;
}

VirtualMeshLevel build_first_level(VirtualMesh &virtual_mesh, Mesh &mesh) {
    Clock timer;

    print("clustering triangles: ");
    timer.tic();
    virtual_mesh.clusters = cluster_triangles(mesh);
    print("{} ms\n", timer.toc());

    print("grouping clusters: ");
    timer.tic();
    virtual_mesh.cluster_groups = grouping_clusters(virtual_mesh.clusters);
    print("{} ms\n", timer.toc());

    VirtualMeshLevel level0 = {
        .cluster_offset = 0,
        .group_offset = 0,
        .cluster_count = (int)virtual_mesh.clusters.size(),
        .group_count = (int)virtual_mesh.cluster_groups.size()
    };

    print("level, cluster num: {}, group num: {}\n", level0.cluster_count, level0.group_count);

    return level0;
}

VirtualMesh build_virtual_mesh(Mesh &mesh) {
    print("\n# begin building virtual mesh\n");
    print("mesh verts: {}, tris: {}\n\n", mesh.positions.size(), mesh.indices.size() / 3);

    Clock timer;
    VirtualMesh vertual_mesh;

    print("## building level 0\n");
    timer.tic();
    vertual_mesh.levels.push_back(build_first_level(vertual_mesh, mesh));
    print("{} ms\n\n", timer.toc());

    while (true) {
        // if (vertual_mesh.levels.size() >= 6) {
        //     break;
        // }
        VirtualMeshLevel last_level = vertual_mesh.levels.back();
        if (last_level.cluster_count == 1)
            break;

        print("## building level {}\n", vertual_mesh.levels.size());
        timer.tic();
        vertual_mesh.levels.push_back(build_parent_level(vertual_mesh, last_level));
        print("{} ms\n\n", timer.toc());
    }

    return vertual_mesh;
}

}