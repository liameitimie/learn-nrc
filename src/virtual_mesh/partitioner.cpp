#include <partitioner.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/logging.h>
// #include <luisa/core/clock.h>
#include <metis.h>

using namespace luisa;
using namespace fmt;

void check_metis(int result) {
    if (result != METIS_OK) [[unlikely]] {
        print("error in metis graph partioner: ");
        switch (result) {
            case METIS_ERROR_INPUT: print("error input\n"); break;
            case METIS_ERROR_MEMORY: print("insufficient memory\n"); break;
            default: print("unknown error\n");
        }
        exit(1);
    }
}

PartitionResult metis_partition(MetisGraph &graph, int part_size) {
    vector<int> part_id(graph.num_vert);
    int num_part = (graph.num_vert + part_size - 1) / part_size;
    // int num_part = 2;

    int num_constraint = 1;
    int edge_cut = 0;
    // int option[METIS_NOPTIONS];
    // option[METIS_OPTION_UFACTOR] = 200;

    int result = METIS_PartGraphRecursive(
        &graph.num_vert,
        &num_constraint,
        graph.adj_offset.data(),
        graph.adj_id.data(),
        nullptr,  // vertex weights
        nullptr,  // vertex size
        graph.adj_weight.data(),
        &num_part,
        nullptr,  // partition weight
        nullptr,  // constraint
        nullptr,  // options
        &edge_cut,
        part_id.data()
    );

    check_metis(result);

    return {num_part, std::move(part_id)};
}

// 严格控制每个分块的最大顶点个数
PartitionResult fine_partition(MetisGraph &graph, int max_part_size) {
    auto [n_part, part_id] = metis_partition(graph, max_part_size);

    // 目前直接划分都可以满足
    vector<int> part_sizes(n_part);
    for (int p: part_id) part_sizes[p]++;

    for (int sz: part_sizes) {
        if (sz > max_part_size) {
            print("error in fine partition\n");
            exit(1);
        }
    }

    return {n_part, std::move(part_id)};
}

PartitionResult partition(MetisGraph &graph, int max_part_size) {
    // Clock timer;
    if (graph.num_vert <= max_part_size) {
        vector<int> part_id(graph.num_vert, 0);
        return {1, std::move(part_id)};
    }

    // print("coarse partition: ");
    // timer.tic();
    // 粗划分
    auto [n_part, part_id] = metis_partition(graph, max_part_size - 2);
    // print("{}\n", timer.toc());

    // print("build link: ");
    // timer.tic();
    vector<int> sub_id(graph.num_vert); // 顶点在分块内的编号

    vector<int> part_size(n_part);
    vector<int> vert_head(n_part, -1);
    vector<int> next_vert(graph.num_vert, -1); // 每个part一个链表，连接该part所有的顶点

    for (int i = graph.num_vert - 1; i >= 0; i--) {
        int id = part_id[i];
        next_vert[i] = vert_head[id];
        vert_head[id] = i;

        part_size[id]++;
    }
    for (int i = 0; i < graph.num_vert; i++) {
        if (next_vert[i] != -1) {
            sub_id[next_vert[i]] = sub_id[i] + 1;
        }
    }
    // print("{}\n", timer.toc());

    int part_offset = 0;
    vector<int> fine_part_id(graph.num_vert); // part_id不能就地修改

    MetisGraph g;

    int num_subpart = 0;

    // print("build sub graph and part: ");
    // timer.tic();
    
    for (int pid = 0; pid < n_part; pid++) {
        if (part_size[pid] <= max_part_size) {
            // 不需要再次划分
            for (int vid = vert_head[pid]; vid != -1; vid = next_vert[vid]) {
                fine_part_id[vid] = part_offset;
            }
            part_offset++;
        }
        else {
            num_subpart++;

            g.clear();

            for (int vid = vert_head[pid]; vid != -1; vid = next_vert[vid]) {
                g.adj_offset.push_back(g.adj_id.size());

                // 枚举边，出点在同一分组则加入
                for (int i = graph.adj_offset[vid]; i < graph.adj_offset[vid + 1]; i++) {
                    if (part_id[graph.adj_id[i]] == pid) {
                        g.adj_id.push_back(sub_id[graph.adj_id[i]]);
                        g.adj_weight.push_back(graph.adj_weight[i]);
                    }
                }
            }
            g.num_vert = g.adj_offset.size();
            g.adj_offset.push_back(g.adj_id.size());

            auto [sub_n_part, sub_part_id] = fine_partition(g, max_part_size);

            int id = 0;
            for (int vid = vert_head[pid]; vid != -1; vid = next_vert[vid]) {
                fine_part_id[vid] = part_offset + sub_part_id[id];
                id++;
            }

            part_offset += sub_n_part;
        }
    }
    // print("{}\n", timer.toc());
    // print("num_subpart: {}\n", num_subpart);

    return {part_offset, std::move(fine_part_id)};
}