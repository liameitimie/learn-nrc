#include <mesh.h>
#include <luisa/core/stl.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <quadric.h>
#include <index_heap.h>
#include <luisa/core/mathematics.h>

using namespace luisa;
using namespace fmt;

using Edge = eastl::pair<virtual_mesh::float3, virtual_mesh::float3>;

Clock timer;

template <>
struct luisa::hash<Edge> {
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(const Edge &e, uint64_t seed = hash64_default_seed) const noexcept {
        return hash64(&e, sizeof(float) * 6, seed);
    }
};

template<>
struct fmt::formatter<virtual_mesh::float3> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }
    template<typename FormatContext>
    auto format(const virtual_mesh::float3 &v, FormatContext &ctx) -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), FMT_STRING("float3({}, {}, {})"), v.x, v.y, v.z);
    }
};


namespace virtual_mesh {

inline int cycle3(int i, int ofs) {
    return i - i % 3 + (i + ofs) % 3;
}

// 简易并查集
struct DisjointSet {
    vector_map<int, int> fa;
    
    int find(int x) {
        if (fa.find(x) == fa.end()) { fa[x] = x; return x; }
        int tx = x;
        while (fa[tx] != tx) tx = fa[tx];
        while (fa[x] != x) { int t = fa[x]; fa[x] = tx; x = t; }
        return tx;
    }
    void merge(int x, int y) {
        fa[find(y)] = find(x);
    };
};

struct WedgeFace {
    int wedge_id;
    int wedge_vert;
    int corner_id;
    // int face_id;
    int vert_id;
};

struct MergeContext {
    float3 p0, p1;
    bool locked;
    float3 opt_p;
    vector<float> opt_attr;
    double quadric_error;
    double penalty;

    int num_wedge;
    vector<WedgeFace> wedge_faces;
    vector<ubyte> wedge_quadrics; // 使用字节数组并使用强制转换，将可变属性数量的矩阵存在线性内存中
    vector_set<int> wedge_vert_ids;

    vector<int> recalc_edges;
};

void print_ctx(MergeContext &ctx) {
    print("p0:{}, p1:{}\n", ctx.p0, ctx.p1);
    print("opt_p:{}\n", ctx.opt_p);

    print("opt_attr: [");
    for (float t: ctx.opt_attr) print("{}, ", t);
    print("]\n");

    print("q error:{}\n", ctx.quadric_error);

    print("num wedge:{}\n", ctx.num_wedge);

    print("wedge face:[\n");
    for (auto f: ctx.wedge_faces) {
        print("\t{{w:{}, wv:{}, c:{}, v:{}}}\n", f.wedge_id, f.wedge_vert, f.corner_id, f.vert_id);
    }
    print("]\n");
}

struct MeshSimplifier1 {

    double position_scale;
    double attr_scale;

    unordered_map<float3, vector_set<int>> corner_lut;
    unordered_map<float3, int> vert_cnt;
    Mesh &mesh;
    unordered_set<float3> &locked_pos;
    int num_face;
    
    int num_attr;
    vector<ubyte> face_quadrics; // 使用字节数组并使用强制转换，将可变属性数量的矩阵存在线性内存中

    unordered_map<Edge, int> edge_id;
    vector<Edge> edges;

    void init_face_quadric() {
        print("init_face_quadric: ");
        timer.tic();

        const int quadric_size = sizeof(Quadric) + sizeof(QuadricGrad) * num_attr;
        
        face_quadrics.resize(num_face * quadric_size);
        for (int i = 0; i < num_face; i++) {
            if (is_face_invaild(i) || is_face_duplicate(i)) {
                remove_face(i);
            }
            else {
                calc_face_quadric(i);
            }
        }
        print("{}\n", timer.toc());
    }

    MeshSimplifier1(Mesh &mesh, unordered_set<float3> &locked_pos): mesh(mesh), locked_pos(locked_pos) {
        print("init simplifier\n");
        print("compact: ");
        timer.tic();
        mesh.compact();
        print("{}\n", timer.toc());

        num_attr = 2;
        num_face = mesh.indices.size() / 3;


        print("unique edge: ");
        timer.tic();

        int exp_num_edge = std::min({mesh.indices.size(), 3 * mesh.positions.size() - 6, num_face + mesh.positions.size()});
        edge_id.reserve(exp_num_edge);
        edges.reserve(exp_num_edge);
        
        for (int i = 0; i < mesh.indices.size() / 3 * 3; i++) {
            float3 p0 = mesh.positions[mesh.indices[i]];
            float3 p1 = mesh.positions[mesh.indices[cycle3(i, 1)]];
            
            corner_lut[p0].insert(i);
            vert_cnt[p0]++;

            if (p0 == p1) continue; //

            if (p1 < p0) std::swap(p0, p1);
            if (!edge_id.contains({p0, p1})) {
                edge_id[{p0, p1}] = edges.size();
                edges.push_back({p0, p1});
            }
        }

        print("{}\n", timer.toc());
        print("num edge: {}\n", edges.size());

        print("calc area: ");
        timer.tic();

        double area = 0;
        double uv_area = 0;

        for (int i = 0; i < num_face; i++) {
            int i0 = mesh.indices[i * 3];
            int i1 = mesh.indices[i * 3 + 1];
            int i2 = mesh.indices[i * 3 + 2];
            float3 tp0 = mesh.positions[i0];
            float3 tp1 = mesh.positions[i1];
            float3 tp2 = mesh.positions[i2];

            double3 p0 = {tp0.x, tp0.y, tp0.z};
            double3 p1 = {tp1.x, tp1.y, tp1.z};
            double3 p2 = {tp2.x, tp2.y, tp2.z};

            area += abs(0.5 * length(cross(p1 - p0, p2 - p0)));

            float2 tuv0 = mesh.texcoords[i0];
            float2 tuv1 = mesh.texcoords[i1];
            float2 tuv2 = mesh.texcoords[i2];

            double2 uv0 = {tuv0.x, tuv0.y};
            double2 uv1 = {tuv1.x, tuv1.y};
            double2 uv2 = {tuv2.x, tuv2.y};

            uv_area += abs(0.5 * cross(uv1 - uv0, uv2 - uv0));
        }

        area /= num_face;
        uv_area /= num_face;

        position_scale = 1 / sqrt(area);
        attr_scale = 1 / (128 * sqrt(uv_area));

        print("{}\n", timer.toc());
        print("avg tri area: {}\navg uv area: {}\n", area, uv_area);

        init_face_quadric();
    }

    bool is_face_duplicate(int face_idx) {
        int i0 = mesh.indices[face_idx * 3];
        int i1 = mesh.indices[face_idx * 3 + 1];
        int i2 = mesh.indices[face_idx * 3 + 2];

        for (int corner: corner_lut[mesh.positions[i0]]) {
            if (corner != face_idx * 3) {
                if (
                    i0 == mesh.indices[corner]
                    && i1 == mesh.indices[cycle3(corner, 1)]
                    && i2 == mesh.indices[cycle3(corner, 2)]
                ) {
                    return true;
                }
            }
        }
        return false;
    }

    bool is_face_invaild(int face_idx) {
        float3 p0 = mesh.positions[mesh.indices[face_idx * 3]];
        float3 p1 = mesh.positions[mesh.indices[face_idx * 3 + 1]];
        float3 p2 = mesh.positions[mesh.indices[face_idx * 3 + 2]];

        return p0 == p1 || p1 == p2 || p2 == p0;
    }

    void remove_face(int face_idx) {
        num_face--;

        for (int k = 0; k < 3; k++) {
            int corner = face_idx * 3 + k;
            int vid = mesh.indices[corner];
            float3 p = mesh.positions[vid];

            corner_lut[p].erase(corner);
            mesh.indices[corner] = -1;
        }
    }

    double time15;
    double time16;
    double time17;
    double time18;

    void get_adjacent_face(MergeContext& ctx) {
        Clock timer;
        
        ctx.wedge_faces.clear();

        DisjointSet wedge_verts;
        wedge_verts.fa.reserve(8);

        vector_set<int> adjverts;
        adjverts.reserve(2);

        timer.tic();
        for (int corner: corner_lut[ctx.p0]) {
            // int face_id = corner / 3;
            int vert_id = mesh.indices[corner];
            // ctx.wedge_faces.push_back({-1, corner, face_id, vert_id});
            ctx.wedge_faces.push_back({-1, -1, corner, vert_id});

            // 如果收缩边是三角形的边，那么两个顶点应合并为一个wedge
            int adjvert = -1;
            if (mesh.positions[mesh.indices[cycle3(corner, 1)]] == ctx.p1) {
                adjvert = cycle3(corner, 1);
                adjverts.insert(adjvert);
            }
            if (mesh.positions[mesh.indices[cycle3(corner, 2)]] == ctx.p1) {
                adjvert = cycle3(corner, 2);
                adjverts.insert(adjvert);
            }
            if (adjvert != -1) {
                int adjvert_id = mesh.indices[adjvert];
                wedge_verts.merge(vert_id, adjvert_id);
            }
        }
        time15 += timer.toc();

        timer.tic();
        for (int corner: corner_lut[ctx.p1]) {
            if (adjverts.find(corner) != adjverts.end())
                continue;
            // int face_id = corner / 3;
            int vert_id = mesh.indices[corner];
            // ctx.wedge_faces.push_back({-1, corner, face_id, vert_id});
            ctx.wedge_faces.push_back({-1, -1, corner, vert_id});
        }
        time16 += timer.toc();

        // vector_set<int> vert_ids;
        // vert_ids.reserve(8);
        ctx.wedge_vert_ids.clear();

        timer.tic();
        for (auto& wedge_face: ctx.wedge_faces) {
            ctx.wedge_vert_ids.insert(wedge_verts.find(wedge_face.vert_id));
        }

        time17 += timer.toc();
        timer.tic();

        for (auto& wedge_face: ctx.wedge_faces) {
            int wedge_vert = wedge_verts.find(wedge_face.vert_id);
            int index = ctx.wedge_vert_ids.find(wedge_vert) - ctx.wedge_vert_ids.begin();
            wedge_face.wedge_id = index;
            wedge_face.wedge_vert = wedge_vert;
        }
        time18 += timer.toc();

        ctx.num_wedge = ctx.wedge_vert_ids.size();
        
    }

    void calc_wedge_quadric(MergeContext& ctx) {
        const int quadric_size = sizeof(Quadric) + sizeof(QuadricGrad) * num_attr;
        ctx.wedge_quadrics.resize(ctx.num_wedge * quadric_size);
        memset(ctx.wedge_quadrics.data(), 0, ctx.num_wedge * quadric_size);

        for (auto f: ctx.wedge_faces) {
            QuadricAttr& wedge_quadric = QuadricAttr::get(ctx.wedge_quadrics, num_attr, f.wedge_id);
            QuadricAttr& face_quadric = QuadricAttr::get(face_quadrics, num_attr, f.corner_id / 3);

            wedge_quadric.add(face_quadric, num_attr);
        }
    }

    Quadric calc_edge_quadric(MergeContext& ctx) {
        Quadric q = {};

        for (auto wf: ctx.wedge_faces) {
            float3 p0 = mesh.positions[mesh.indices[wf.corner_id]];
            if (vert_cnt[p0] <= 1) continue;

            for (int i = 1; i <= 2; i++) {
                float3 p1 = mesh.positions[mesh.indices[cycle3(wf.corner_id, i)]];
                if (vert_cnt[p1] <= 1) continue;
                float3 p2 = mesh.positions[mesh.indices[cycle3(wf.corner_id, 3 - i)]];

                double3 tp0 = {p0.x, p0.y, p0.z};
                double3 tp1 = {p1.x, p1.y, p1.z};
                double3 tp2 = {p2.x, p2.y, p2.z};

                tp0 *= position_scale;
                tp1 *= position_scale;
                tp2 *= position_scale;

                // 两个顶点都有大于一个wedge，则当前边是纹理接缝（错误的）
                Quadric eq;
                eq.from_edge(tp0, tp1, tp2);

                q += eq;
            }
        }
        return q;
    }

    double time11;
    double time12;
    double time13;
    double time14;
    double time19;

    void evaluate_merge(MergeContext &ctx, float3 p0, float3 p1) {
        Clock timer;
        timer.tic();

        ctx.locked = false;
        if (locked_pos.contains(p0) && locked_pos.contains(p1)) {
            ctx.locked = true;
            ctx.penalty = 1e9;

            time11 += timer.toc();
            return;
        }

        ctx.p0 = p0;
        ctx.p1 = p1;
        ctx.penalty = 0;

        timer.tic();
        get_adjacent_face(ctx);
        time12 += timer.toc();

        timer.tic();
        calc_wedge_quadric(ctx);
        time13 += timer.toc();

        if (ctx.wedge_faces.size() > 24) {
            ctx.penalty += 0.5 * (ctx.wedge_faces.size() - 24);
        }

        timer.tic();
        QuadricOptimizer opt(ctx.num_wedge, num_attr, ctx.wedge_quadrics);
        time14 += timer.toc();

        timer.tic();
        Quadric edge_quadric = {};
        if (ctx.num_wedge > 1) {
            edge_quadric = calc_edge_quadric(ctx);
            opt.add_edge_quadric(edge_quadric);
        }
        time19 += timer.toc();
        timer.tic();

        double3 p;
        if (locked_pos.contains(p0)) {
            p = {p0.x, p0.y, p0.z};
            p *= position_scale;
        }
        else if (locked_pos.contains(p1)) {
            p = {p1.x, p1.y, p1.z};
            p *= position_scale;
        }
        else {
            if (!opt.optimize(p)) {
                float3 tp = (p0 + p1) * 0.5;
                p = {tp.x, tp.y, tp.z};
                p *= position_scale;
            }
        }
        
        ctx.opt_attr.resize(ctx.num_wedge * num_attr);
        ctx.quadric_error = opt.calc_attr_with_error(p, ctx.opt_attr.data());

        if (ctx.num_wedge > 1) {
            ctx.quadric_error += edge_quadric.eval(p);
        }
        time14 += timer.toc();

        p /= position_scale;
        for (auto& x: ctx.opt_attr) x /= attr_scale;

        ctx.opt_p = {(float)p.x, (float)p.y, (float)p.z};
    }

    double time4;
    double time5;
    double time6;
    double time7;
    double time8;
    double time9;
    double time10;

    void perform_merge(MergeContext &ctx, IndexHeap &heap) {
        Clock timer;
        timer.tic();
        { // 更新 edge set
            for (auto wf: ctx.wedge_faces) {
                float3 p0 = mesh.positions[mesh.indices[wf.corner_id]];

                // 只需要修改与corner连接的边
                for (int i = 1; i <= 2; i++) {
                    float3 p1 = mesh.positions[mesh.indices[cycle3(wf.corner_id, i)]];
                    
                    int e_id = -1;
                    {
                        Edge e = {p0, p1};
                        if (e.second < e.first) std::swap(e.first, e.second);

                        // 边已被修改或不存在
                        if (!edge_id.contains(e)) continue;

                        e_id = edge_id[e];
                        edge_id.erase(e); // 将待修改边删除
                    }

                    Edge& e = edges[e_id];
                    if (e.first == ctx.p0 || e.first == ctx.p1) e.first = ctx.opt_p;
                    if (e.second == ctx.p0 || e.second == ctx.p1) e.second = ctx.opt_p;

                    if (e.second < e.first) std::swap(e.first, e.second);

                    // 边被合并或边重复，从堆中删除
                    if (e.first == e.second || edge_id.contains(e)) {
                        heap.remove(e_id);
                        continue;
                    }
                    // 将修改后的边重新加入
                    edge_id[e] = e_id;
                }
            }
        }
        time4 += timer.toc();

        timer.tic();
        { // 修改mesh
            for (auto wf: ctx.wedge_faces) {
                int face_id = wf.corner_id / 3;
                for (int k = 0; k < 3; k++) {
                    int corner = face_id * 3 + k;
                    int &vert_idx = mesh.indices[corner];
                    float3 p = mesh.positions[vert_idx];

                    if (p == ctx.p0 || p == ctx.p1) {
                        vert_idx = wf.wedge_vert;
                    }
                }
            }
            for (int i = 0; i < ctx.num_wedge; i++) {
                int v_id = ctx.wedge_vert_ids[i];

                mesh.positions[v_id] = ctx.opt_p;
                for (int j = 0; j < num_attr; j++) {
                    mesh.texcoords[v_id][j] = ctx.opt_attr[j + i * num_attr];
                }
            }
        }
        time5 += timer.toc();

        timer.tic();
        { // 对 corner_lut 更新，将两点的 corner set 合并
            auto set0 = std::move(corner_lut[ctx.p0]);
            auto set1 = std::move(corner_lut[ctx.p1]);

            corner_lut.erase(ctx.p0);
            corner_lut.erase(ctx.p1);

            vector_set<int> set;
            set.reserve(set0.size() + set1.size());

            auto iter0 = set0.begin();
            auto iter1 = set1.begin();
            while (iter0 != set0.end() || iter1 != set1.end()) {
                if (iter0 == set0.end()) { set.push_back_unsorted(*iter1); iter1++; }
                else if (iter1 == set1.end()) { set.push_back_unsorted(*iter0); iter0++; }
                else {
                    if (*iter0 == *iter1) {
                        set.push_back_unsorted(*iter0);
                        iter0++, iter1++;
                    }
                    else if (*iter0 < *iter1) { set.push_back_unsorted(*iter0); iter0++; }
                    else { set.push_back_unsorted(*iter1); iter1++; }
                }
            }
            corner_lut[ctx.opt_p] = std::move(set);
        }
        time6 += timer.toc();

        timer.tic();
        {
            vert_cnt.erase(ctx.p0);
            vert_cnt.erase(ctx.p1);
            vert_cnt[ctx.opt_p] = ctx.num_wedge;
        }
        time7 += timer.toc();

        timer.tic();
        { // 重新计算面的qem
            for (auto wf: ctx.wedge_faces) {
                int face_id = wf.corner_id / 3;
                if (is_face_invaild(face_id) || is_face_duplicate(face_id)) {
                    remove_face(face_id);
                }
                else {
                    calc_face_quadric(face_id);
                }
            }
        }
        time8 += timer.toc();

        timer.tic();
        { // 更新受影响的边

            ctx.recalc_edges.clear();

            for (auto wf: ctx.wedge_faces) {
                if (mesh.indices[wf.corner_id] == -1) continue;
                // 枚举相邻顶点，与这些顶点连接的边都需要重新计算
                for (int i = 1; i <= 2; i++) {
                    float3 p0 = mesh.positions[mesh.indices[cycle3(wf.corner_id, i)]];

                    for (int corner: corner_lut[p0]) {
                        // 枚举连接边
                        for (int i = 1; i <= 2; i++) {
                            float3 p1 = mesh.positions[mesh.indices[cycle3(corner, i)]];
                            Edge e = {p0, p1};
                            if (e.second < e.first) std::swap(e.first, e.second);

                            if (edge_id.contains(e)) {
                                int e_id = edge_id[e];
                                if (heap.is_present(e_id)) {
                                    heap.remove(e_id);
                                    ctx.recalc_edges.push_back(e_id);
                                }
                            }
                        }
                    }
                }
            }

            time9 += timer.toc();
            timer.tic();

            for (auto e_id: ctx.recalc_edges) {
                auto [p0, p1] = edges[e_id];
                evaluate_merge(ctx, p0, p1);
                if (ctx.locked) {
                    heap.remove(e_id);
                }
                else {
                    heap.update(e_id, ctx.quadric_error + ctx.penalty);
                }
            }
            time10 += timer.toc();
        }
    }

    void calc_face_quadric(int face_idx) {
        QuadricAttr& quadric = QuadricAttr::get(face_quadrics, num_attr, face_idx);

        int i0 = mesh.indices[face_idx * 3];
        int i1 = mesh.indices[face_idx * 3 + 1];
        int i2 = mesh.indices[face_idx * 3 + 2];
        float3 tp0 = mesh.positions[i0];
        float3 tp1 = mesh.positions[i1];
        float3 tp2 = mesh.positions[i2];

        double3 p0 = {tp0.x, tp0.y, tp0.z};
        double3 p1 = {tp1.x, tp1.y, tp1.z};
        double3 p2 = {tp2.x, tp2.y, tp2.z};

        p0 *= position_scale;
        p1 *= position_scale;
        p2 *= position_scale;

        vector<double> attr(num_attr * 3);
        for (int i = 0; i < num_attr; i++) {
            attr[i] = mesh.texcoords[i0][i] * attr_scale;
            attr[i + num_attr] = mesh.texcoords[i1][i] * attr_scale;
            attr[i + num_attr * 2] = mesh.texcoords[i2][i] * attr_scale;
        }
        double* attr0 = attr.data();
        double* attr1 = attr.data() + num_attr;
        double* attr2 = attr.data() + num_attr * 2;

        quadric.from_plane(p0, p1, p2, attr0, attr1, attr2, num_attr);
    }

    void compact_mesh() {
        int cur_idx = 0;
        for (int i = 0; i < mesh.indices.size() / 3; i++) {
            int i0 = mesh.indices[i * 3];
            int i1 = mesh.indices[i * 3 + 1];
            int i2 = mesh.indices[i * 3 + 2];

            int f = int(i0 == -1) + int(i1 == -1) + int(i2 == -1);
            assert(f == 0 || f == 3);

            // 三角形保留
            if (f == 0) {
                if (i != cur_idx) {
                    mesh.indices[cur_idx * 3] = i0;
                    mesh.indices[cur_idx * 3 + 1] = i1;
                    mesh.indices[cur_idx * 3 + 2] = i2;
                }
                cur_idx++;
            }
        }

        assert(cur_idx == num_face);
        mesh.indices.resize(cur_idx * 3);
        mesh.compact();
    }

    double simplify(int target_face_num) {
        MergeContext ctx;

        // print("## 0, 1\n");
        // float3 p0 = mesh.positions[0];
        // float3 p1 = mesh.positions[1];
        // evaluate_merge(ctx, p0, p1);
        // print_ctx(ctx);

        // print("## 1, 2\n");
        // p0 = mesh.positions[1];
        // p1 = mesh.positions[2];
        // evaluate_merge(ctx, p0, p1);
        // print_ctx(ctx);

        // print("## 2, 0\n");
        // p0 = mesh.positions[2];
        // p1 = mesh.positions[0];
        // evaluate_merge(ctx, p0, p1);
        // print_ctx(ctx);

        // print("## 3, 2\n");
        // p0 = mesh.positions[3];
        // p1 = mesh.positions[2];
        // evaluate_merge(ctx, p0, p1);
        // print_ctx(ctx);

        // print("## 1, 3\n");
        // p0 = mesh.positions[1];
        // p1 = mesh.positions[3];
        // evaluate_merge(ctx, p0, p1);
        // print_ctx(ctx);


        IndexHeap heap(edges.size());
        
        // print("init merge quadric: ");
        // timer.tic();

        for (int i = 0; i < edges.size(); i++) {
            auto [p0, p1] = edges[i];
            evaluate_merge(ctx, p0, p1);
            if (ctx.locked) {
                heap.remove(i);
            }
            else {
                heap.update(i, ctx.quadric_error + ctx.penalty);
            }
        }
        // print("{}\n", timer.toc());

        double max_error = 0;

        double time1 = 0;
        double time2 = 0;
        double time3 = 0;

        // time4 = 0;
        // time5 = 0;
        // time6 = 0;
        // time7 = 0;
        // time8 = 0;
        // time9 = 0;
        // time10 = 0;
        // time11 = 0;
        // time12 = 0;
        // time13 = 0;
        // time14 = 0;
        // time15 = 0;
        // time16 = 0;
        // time17 = 0;
        // time18 = 0;
        // time19 = 0;

        while (heap.size()) {
            timer.tic();
            auto [error, edge_id] = heap.pop();
            time1 += timer.toc();
            auto [p0, p1] = edges[edge_id];

            // print("error: {}\n", error);

            timer.tic();
            evaluate_merge(ctx, p0, p1);
            time2 += timer.toc();

            if (ctx.locked) {
                print("error: merge a locked edge\n");
                exit(0);
            }

            max_error = max(max_error, ctx.quadric_error);

            timer.tic();
            perform_merge(ctx, heap);
            time3 += timer.toc();

            if (num_face <= target_face_num) break;
        }
        // print("max_error: {}\n", max_error);

        if (num_face > target_face_num) {
            print("warning: simplify result doesn't reach target face num\n");
        }

        // print("heap pop: {}\n", time1);
        // print("evaluate_merge: {}\n", time2);
        // print("perform_merge: {}\n", time3);
        // print("    update edge set: {}\n", time4);
        // print("    update mesh: {}\n", time5);
        // print("    update corner lut: {}\n", time6);
        // print("    update vert cnt: {}\n", time7);
        // print("    recalc face quadric: {}\n", time8);
        // print("    find recalc edge: {}\n", time9);
        // print("    recalc edge: {}\n", time10);
        // print("total evaluate_merge:\n");
        // print("    lock condiction: {}\n", time11);
        // print("    get adj face: {}\n", time12);
        // print("        get corners0: {}\n", time15);
        // print("        get corners1: {}\n", time16);
        // print("        calc wedge id0: {}\n", time17);
        // print("        calc wedge id1: {}\n", time18);
        // print("    calc wedge quadric: {}\n", time13);
        // print("    calc edge quadric: {}\n", time19);
        // print("    optimize: {}\n", time14);

        compact_mesh();

        return sqrt(max_error) / position_scale;
        // return 0;
    }
};

double mesh_simplify(Mesh &mesh, int target_face_num, luisa::unordered_set<float3> &locked_pos) {
    MeshSimplifier1 simplifier(mesh, locked_pos);
    return simplifier.simplify(target_face_num);
}


}

/*

        // unordered_set<float3> vert_set;
        // for (float3 p: mesh.positions) {
        //     if (vert_set.contains(p)) {
        //         for (int corner: corner_lut[p]) {
        //             for (int i = 1; i <= 2; i++) {
        //                 float3 p0 = mesh.positions[mesh.indices[corner]];
        //                 float3 p1 = mesh.positions[mesh.indices[cycle3(corner, i)]];

        //                 evaluate_merge(ctx, p0, p1);
        //                 print_ctx(ctx);

        //                 double error = 0;
        //                 double3 p = {ctx.opt_p.x, ctx.opt_p.y, ctx.opt_p.z};

        //                 for (auto wf: ctx.wedge_faces) {
        //                     int f = wf.corner_id / 3;
        //                     print("f: {}\n", f);

        //                     int i0 = mesh.indices[f * 3];
        //                     int i1 = mesh.indices[f * 3 + 1];
        //                     int i2 = mesh.indices[f * 3 + 2];
        //                     float3 tp0 = mesh.positions[i0];
        //                     float3 tp1 = mesh.positions[i1];
        //                     float3 tp2 = mesh.positions[i2];

        //                     float2 uv0 = mesh.texcoords[i0];
        //                     float2 uv1 = mesh.texcoords[i1];
        //                     float2 uv2 = mesh.texcoords[i2];

        //                     // double2 uv0 = {tuv0.x, tuv0.y};
        //                     // double2 uv1 = {tuv1.x, tuv1.y};
        //                     // double2 uv2 = {tuv2.x, tuv2.y};

        //                     double3 p0 = {tp0.x, tp0.y, tp0.z};
        //                     double3 p1 = {tp1.x, tp1.y, tp1.z};
        //                     double3 p2 = {tp2.x, tp2.y, tp2.z};

        //                     print("i0:{},  i1:{},  i2:{}\n", i0, i1, i2);
        //                     print("p0:{},  p1:{},  p2:{}\n", tp0, tp1, tp2);
        //                     print("uv0:{},  uv1:{},  uv2:{}\n", uv0, uv1, uv2);

        //                     double3 p01 = p1 - p0;
        //                     double3 p02 = p2 - p0;

        //                     double3 n = normalize(cross(p01, p02));
        //                     double d = -dot(n, p0);

        //                     print("n:{}, d:{}\n", n, d);

        //                     double g_error = abs(dot(n, p) + d);

        //                     error += g_error * g_error;

        //                     QuadricAttr& q = QuadricAttr::get(face_quadrics, num_attr, f);

        //                     for (int i = 0; i < num_attr; i++) {
        //                         auto tg = q.gs[i];
        //                         double3 g = {tg.gx, tg.gy, tg.gz};
        //                         double s = dot(g, p) + tg.d;

        //                         print("g{}:{}, d:{}\n", i, g, tg.d);

        //                         double a_error = abs(s - ctx.opt_attr[i + wf.wedge_id * num_attr]);

        //                         error += a_error * a_error;
        //                     }
        //                 }

        //                 print("error: {}\n\n", error);
        //                 break;
        //             }
        //             break;
        //         }
        //         break;
        //     }
        //     else {
        //         vert_set.insert(p);
        //     }
        // }




p0:float3(0.194921, 0.221357, -0.173014), p1:float3(0.192003, 0.257627, -0.114007)
opt_p:float3(-0.7490821, -0.7257319, 0.69832385)
opt_attr: [0.58967006, 0.6034265, 0.21085367, -0.33919358, ]
q error:0.26104001382049447
num wedge:2
wedge face:[
        {w:0, wv:343, c:1068, v:343}
        {w:1, wv:360, c:1112, v:360}
        {w:1, wv:360, c:1114, v:360}
        {w:0, wv:343, c:9853, v:343}
        {w:0, wv:343, c:9857, v:343}
        {w:1, wv:360, c:9891, v:360}
        {w:0, wv:343, c:1073, v:344}
        {w:0, wv:343, c:1086, v:344}
        {w:1, wv:360, c:1142, v:361}
        {w:1, wv:360, c:1144, v:361}
]
f: 356
i0:343,  i1:344,  i2:345
p0:float3(0.194921, 0.221357, -0.173014),  p1:float3(0.192003, 0.257627, -0.114007),  p2:float3(0.175228, 0.201681, -0.146613)
uv0:float2(0.968754, 0.704629),  uv1:float2(0.964644, 0.717556),  uv2:float2(0.959599, 0.703413)
n:double3(0.8466833783401516, -0.43361003323139885, 0.3083984369668123), d:-0.015696507751665173
g0:double3(0.12570171395718424, 0.13939267556130544, -0.14911712785046882), d:0.8875971938510052
g1:double3(0.04566243030328187, 0.17157416738569511, 0.11587172756391745), d:0.6777967249624033
f: 370
i0:358,  i1:359,  i2:360
p0:float3(0.227725, 0.245742, -0.199837),  p1:float3(0.21819, 0.281995, -0.133552),  p2:float3(0.194921, 0.221357, -0.173014)
uv0:float2(0.15372, 0.172537),  uv1:float2(0.137511, 0.174005),  uv2:float2(0.154886, 0.15836)
n:double3(0.7350326547845475, -0.5447655487102429, 0.4036799392325656), d:0.047156648730266565
g0:double3(-0.038745458805147515, -0.16908522864113473, -0.1576313899423972), d:0.17259407308442784
g1:double3(0.20624828595364547, 0.22534953156371113, -0.07143422586267308), d:0.055916063849956496
f: 371
i0:361,  i1:360,  i2:359
p0:float3(0.192003, 0.257627, -0.114007),  p1:float3(0.194921, 0.221357, -0.173014),  p2:float3(0.21819, 0.281995, -0.133552)
uv0:float2(0.139434, 0.160894),  uv1:float2(0.154886, 0.15836),  uv2:float2(0.137511, 0.174005)
n:double3(0.765457768695897, -0.5306284493557833, 0.3640162813355683), d:0.03123444099517625
g0:double3(-0.03879017796364341, -0.16664629345196785, -0.1613527312526838), d:0.17141907025829675
g1:double3(0.2094516084517196, 0.23824762492068507, -0.09314251812160809), d:0.048680747253383616
f: 3284
i0:2834,  i1:343,  i2:347
p0:float3(0.184529, 0.181235, -0.238574),  p1:float3(0.194921, 0.221357, -0.173014),  p2:float3(0.160058, 0.168229, -0.195838)
uv0:float2(0.972864, 0.691703),  uv1:float2(0.968754, 0.704629),  uv2:float2(0.963128, 0.689798)
n:double3(0.7569290941231697, -0.60394212639643, 0.24962422645575974), d:0.029333927615169064
g0:double3(0.11506286946231818, 0.08839928249785045, -0.13502849216291193), d:0.9033962039474199
g1:double3(0.11385298767542272, 0.17297293281875875, 0.07325812198142088), d:0.6568225763239856
f: 3285
i0:345,  i1:347,  i2:343
p0:float3(0.175228, 0.201681, -0.146613),  p1:float3(0.160058, 0.168229, -0.195838),  p2:float3(0.194921, 0.221357, -0.173014)
uv0:float2(0.959599, 0.703413),  uv1:float2(0.963128, 0.689798),  uv2:float2(0.968754, 0.704629)
n:double3(0.7942630694633117, -0.5875917311709573, 0.15453845458731522), d:0.0019863075188192234
g0:double3(0.11500296919001331, 0.10797774791283252, -0.18051027853686005), d:0.8912050635534091
g1:double3(0.09062009155362621, 0.15940206574719915, 0.14033493314080372), d:0.6759603895815013
f: 3297
i0:360,  i1:2836,  i2:358
p0:float3(0.194921, 0.221357, -0.173014),  p1:float3(0.184529, 0.181235, -0.238574),  p2:float3(0.227725, 0.245742, -0.199837)
uv0:float2(0.154886, 0.15836),  uv1:float2(0.170337, 0.155827),  uv2:float2(0.15372, 0.172537)
n:double3(0.7101814107947803, -0.6450014704854724, 0.28216212863017937), d:0.05316431883640547
g0:double3(-0.05934238302808202, -0.129621245322585, -0.14694401669602125), d:0.1697222823151773
g1:double3(0.20878573384113427, 0.18324902733435203, -0.10660486247201287), d:0.05865569087622566
f: 357
i0:346,  i1:345,  i2:344
p0:float3(0.177164, 0.238112, -0.093383),  p1:float3(0.175228, 0.201681, -0.146613),  p2:float3(0.192003, 0.257627, -0.114007)
uv0:float2(0.956139, 0.717201),  uv1:float2(0.959599, 0.703413),  uv2:float2(0.964644, 0.717556)
n:double3(0.8791677375102603, -0.407533196119266, 0.24694287473402815), d:-0.0356580641510364
g0:double3(0.12209457180309657, 0.15643863446839493, -0.17650912908469357), d:0.8807753965393086
g1:double3(0.029632495935098986, 0.15567027146671675, 0.15140695557145295), d:0.6890230584256634
f: 362
i0:344,  i1:352,  i2:346
p0:float3(0.192003, 0.257627, -0.114007),  p1:float3(0.17918, 0.290497, -0.060974),  p2:float3(0.177164, 0.238112, -0.093383)
uv0:float2(0.964644, 0.717556),  uv1:float2(0.960533, 0.730482),  uv2:float2(0.956139, 0.717201)
n:double3(0.884335385877431, -0.2697612359361567, 0.38102467226048803), d:-0.05685778236134885
g0:double3(0.12178854657847167, 0.1766797651632931, -0.15757659251398162), d:0.8777779340776712
g1:double3(-0.007576263024526669, 0.16894123889987095, 0.13719257228513898), d:0.6911277527424408
f: 380
i0:359,  i1:372,  i2:361
p0:float3(0.21819, 0.281995, -0.133552),  p1:float3(0.200073, 0.314908, -0.073216),  p2:float3(0.192003, 0.257627, -0.114007)
uv0:float2(0.137511, 0.174005),  uv1:float2(0.120823, 0.175844),  uv2:float2(0.139434, 0.160894)
n:double3(0.7632212650673711, -0.4426909358881823, 0.4706570256900528), d:0.021166569428441515
g0:double3(-0.01233157734455317, -0.20206515633481037, -0.17006160906598936), d:0.17447092233831768
g1:double3(0.19792953185881906, 0.2764834418199486, -0.060909138752557795), d:0.04471727229067332
f: 381
i0:373,  i1:361,  i2:372
p0:float3(0.17918, 0.290497, -0.060974),  p1:float3(0.192003, 0.257627, -0.114007),  p2:float3(0.200073, 0.314908, -0.073216)
uv0:float2(0.123982, 0.163427),  uv1:float2(0.139434, 0.160894),  uv2:float2(0.120823, 0.175844)
n:double3(0.7758753264581374, -0.434823683722931, 0.45710594162305435), d:0.015165216574643297
g0:double3(-0.0137818788498597, -0.2024694604005441, -0.16920693830031006), d:0.17495098267439319
g1:double3(0.20945727205082407, 0.2889349613459601, -0.08067487583235364), d:0.037042631419186234
error: 0.2610400138204991
*/