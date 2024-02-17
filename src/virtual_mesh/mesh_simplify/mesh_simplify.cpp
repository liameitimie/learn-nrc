#include <mesh_simplify.h>
#include <index_heap.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include <quadric.h>

using namespace luisa;
using namespace fmt;

namespace virtual_mesh {

using Corner = MeshStructure::Corner;
using Vertex = MeshStructure::Vertex;

// 简易并查集
struct DisjointSet {
    luisa::vector<int> fa;
    
    int find(int x) {
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
    int corner;
    int wedge_vert;
};
struct MergeContext {
    int e_idx;
    bool lock_p0;
    bool lock_p1;

    vector<WedgeFace> wedge_faces;
    vector<int> wedge_verts;
    DisjointSet wedge_union;
    vector<ubyte> wedge_quadrics; // 使用字节数组并使用强制转换，将可变属性数量的矩阵存在线性内存中

    vector<int> remove_face;
    vector<int> recalc_edges;

    float3 opt_p;
    vector<float> opt_attr;
    double quadric_error;
    double penalty;

    void clear() {
        e_idx = -1;
        lock_p0 = false;
        lock_p1 = false;

        wedge_faces.clear();
        wedge_verts.clear();
        wedge_union.fa.clear();
        wedge_quadrics.clear();

        remove_face.clear();
        recalc_edges.clear();

        opt_p = {0};
        opt_attr.clear();

        quadric_error = 0;
        penalty = 0;
    }
};
void print_ctx(MergeContext &ctx);

MeshSimplifier::MeshSimplifier(Mesh &mesh): m(mesh) {
    m.init_vert_link();
    m.init_corner_link();
    m.init_edge_link();

    num_attr = 2;

    edge_set.resize(m.edge_count);
    for (int i = 0; i < m.face_count * 3; i++) {
        int e_id = m.edge_id[i];
        if (e_id != -1) {
            edge_set[e_id] = i;
        }
    }

    vert_tag.resize(m.vertex_count, -1);
    face_tag.resize(m.face_count, -1);

    double area = 0;
    double uv_area = 0;

    for (int i = 0; i < m.face_count; i++) {
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

    area /= m.face_count;
    uv_area /= m.face_count;

    // print("avg tri area: {}\navg uv area: {}\n", area, uv_area);

    position_scale = 1 / sqrt(area);
    attr_scale = 1 / (128 * sqrt(uv_area));

    init_face_quadric();
}

void MeshSimplifier::lock_position(float3 p) {
    m.lock_position(p);
}

void MeshSimplifier::init_face_quadric() {
    const int quadric_size = sizeof(Quadric) + sizeof(QuadricGrad) * num_attr;
    
    face_quadrics.resize(m.face_count * quadric_size);
    for (int i = 0; i < m.face_count; i++) {
        calc_face_quadric(i);
    }
}

void MeshSimplifier::calc_face_quadric(int face_idx) {
    QuadricAttr& quadric = QuadricAttr::get(face_quadrics, num_attr, face_idx);

    int i0 = m.mesh.indices[face_idx * 3];
    int i1 = m.mesh.indices[face_idx * 3 + 1];
    int i2 = m.mesh.indices[face_idx * 3 + 2];
    float3 tp0 = m.mesh.positions[i0];
    float3 tp1 = m.mesh.positions[i1];
    float3 tp2 = m.mesh.positions[i2];

    double3 p0 = {tp0.x, tp0.y, tp0.z};
    double3 p1 = {tp1.x, tp1.y, tp1.z};
    double3 p2 = {tp2.x, tp2.y, tp2.z};

    p0 *= position_scale;
    p1 *= position_scale;
    p2 *= position_scale;

    vector<double> attr(num_attr * 3);
    for (int i = 0; i < num_attr; i++) {
        attr[i] = m.mesh.texcoords[i0][i] * attr_scale;
        attr[i + num_attr] = m.mesh.texcoords[i1][i] * attr_scale;
        attr[i + num_attr * 2] = m.mesh.texcoords[i2][i] * attr_scale;
    }
    double* attr0 = attr.data();
    double* attr1 = attr.data() + num_attr;
    double* attr2 = attr.data() + num_attr * 2;

    quadric.from_plane(p0, p1, p2, attr0, attr1, attr2, num_attr);
}

void MeshSimplifier::get_adjacent_face(MergeContext &ctx) {
    Corner e{m, edge_set[ctx.e_idx]};
    Vertex v0 = e.v();
    Vertex v1 = e.v(1);

    auto add_wedge_vert = [&](int vid) {
        vert_tag[vid] = ctx.wedge_verts.size();
        ctx.wedge_union.fa.push_back(vert_tag[vid]);
        ctx.wedge_verts.push_back(vid);
    };

    for (Corner c: v0.wedge_corners()) {
        int fid = c.f().fid;
        int vid = c.v().vid;

        if (vert_tag[vid] == -1) add_wedge_vert(vid);

        assert(face_tag[fid] == -1);
        face_tag[fid] = ctx.wedge_faces.size();
        ctx.wedge_faces.push_back({c.cid, vert_tag[vid]});
    }
    for (Corner c: v1.wedge_corners()) {
        int fid = c.f().fid;
        int vid = c.v().vid;

        if (vert_tag[vid] == -1) add_wedge_vert(vid);

        if (face_tag[fid] == -1) {
            face_tag[fid] = ctx.wedge_faces.size();
            ctx.wedge_faces.push_back({c.cid, vert_tag[vid]});
        }
        else {
            int other_wedge_vert = ctx.wedge_faces[face_tag[fid]].wedge_vert;
            ctx.wedge_union.merge(other_wedge_vert, vert_tag[vid]);
            ctx.remove_face.push_back(c.cid);
        }
    }

    // 清除标记
    for (auto [c, wv]: ctx.wedge_faces) face_tag[c / 3] = -1;
    for (int v: ctx.wedge_verts) vert_tag[v] = -1;
}

void MeshSimplifier::calc_wedge_quadric(MergeContext &ctx) {
    const int quadric_size = sizeof(Quadric) + sizeof(QuadricGrad) * num_attr;
    ctx.wedge_quadrics.resize(ctx.wedge_verts.size() * quadric_size);
    memset(ctx.wedge_quadrics.data(), 0, ctx.wedge_verts.size() * quadric_size);

    for (auto [corner, wedge_id]: ctx.wedge_faces) {
        wedge_id = ctx.wedge_union.find(wedge_id);

        QuadricAttr& wedge_quadric = QuadricAttr::get(ctx.wedge_quadrics, num_attr, wedge_id);
        QuadricAttr& face_quadric = QuadricAttr::get(face_quadrics, num_attr, corner / 3);

        wedge_quadric.add(face_quadric, num_attr);
    }
}

Quadric MeshSimplifier::calc_edge_quadric(MergeContext &ctx) {
    using MeshStructure::Visited;
    using MeshStructure::Seam;
    using MeshStructure::Border;

    Quadric q = {};

    auto try_calc = [&](int corner) {
        for (int k: {0, 2}) {
            int e_id = cycle3(corner, k); // 连接corner的边
            if ((m.edge_flag[e_id] & int(Seam | Border)) && !(m.edge_flag[e_id] & Visited)) {
                // 若边为纹理接缝或边界且边未访问则计算
                m.edge_flag[e_id] |= Visited;

                Corner e{m, e_id};
                float3 p0 = e.v(0).p();
                float3 p1 = e.v(1).p();
                float3 p2 = e.v(2).p();
                double3 tp0 = {p0.x, p0.y, p0.z};
                double3 tp1 = {p1.x, p1.y, p1.z};
                double3 tp2 = {p2.x, p2.y, p2.z};

                tp0 *= position_scale;
                tp1 *= position_scale;
                tp2 *= position_scale;

                Quadric eq;
                eq.from_edge(tp0, tp1, tp2);

                q += eq;
            }
        }
    };
    auto clear_flag = [&](int corner) {
        int f = corner / 3;
        for (int k = 0; k < 3; k++) {
            int c = f * 3 + k;
            m.edge_flag[c] &= ~Visited;
        }
    };

    for (auto [c, wv]: ctx.wedge_faces) try_calc(c);
    for (int c: ctx.remove_face) try_calc(c);

    for (auto [c, wv]: ctx.wedge_faces) clear_flag(c);

    return q;
}

// int ttt = 0;

void MeshSimplifier::evaluate_merge(MergeContext &ctx, int e_idx) {
    ctx.clear();
    ctx.e_idx = e_idx;

    Corner e{m, edge_set[ctx.e_idx]};
    if (e.v().is_lock() && e.v(1).is_lock()) {
        ctx.lock_p0 = true;
        ctx.lock_p1 = true;
        ctx.penalty = 1e18;
        return;
    }

    get_adjacent_face(ctx);
    calc_wedge_quadric(ctx);

    QuadricOptimizer opt(ctx.wedge_verts.size(), num_attr, ctx.wedge_quadrics);
    Quadric edge_quadric = calc_edge_quadric(ctx);
    opt.add_edge_quadric(edge_quadric);

    double3 p;
    float3 tp0 = e.v().p();
    float3 tp1 = e.v(1).p();
    double3 p0 = {tp0.x, tp0.y, tp0.z};
    double3 p1 = {tp1.x, tp1.y, tp1.z};
    p0 *= position_scale;
    p1 *= position_scale;

    if (e.v().is_lock()) {
        p = p0;
    }
    else if (e.v(1).is_lock()) {
        p = p1;
    }
    else {
        bool is_valid = opt.optimize(p);

        if (is_valid) {
            if (length(p - p0) + length(p - p1) > 2 * length(p0 - p1))
                is_valid = false;
        }
        if (!is_valid) {
            // ttt++;
            p = p0;
            p += p1;
            p *= 0.5;
        }
    }

    ctx.opt_attr.resize(ctx.wedge_verts.size() * num_attr);
    ctx.quadric_error = opt.calc_attr_with_error(p, ctx.opt_attr.data());
    ctx.quadric_error += edge_quadric.eval(p);

    p /= position_scale;
    for (auto& x: ctx.opt_attr) x /= attr_scale;

    ctx.opt_p = {(float)p.x, (float)p.y, (float)p.z};
}

// bool debuging = false;

void MeshSimplifier::perform_merge(MergeContext &ctx, IndexHeap &heap) {
    using MeshStructure::Deleted;
    using MeshStructure::Visited;
    using MeshStructure::Visited1;
    using MeshStructure::Seam;
    using MeshStructure::Border;
    using MeshStructure::Locked;

    // debuging = ctx.e_idx == 4031;
    // vector<int> verts;
    // vector<int> corners;
    // vector<int> edges;
    // if (debuging) {
    //     using MeshStructure::Visited;
    //     for (auto [c, wv]: ctx.wedge_faces) {
    //         int f = c / 3;
    //         for (int k = 0; k < 3; k++) {
    //             Vertex v{m, m.mesh.indices[f * 3 + k]};
    //             if (!v.is_vis()) {
    //                 v.set_vis();
    //                 verts.push_back(v.vid);
    //             }
    //         }
    //     }
    //     for (int v: verts) m.vert_flag[v] &= ~Visited;
    //     for (int vid: verts) {
    //         Vertex v{m, vid};
    //         for (Corner c: v.vert_corners()) {
    //             corners.push_back(c.cid);
    //             for (int k: {0, 2}) {
    //                 int e = cycle3(c.cid, k);
    //                 if (!(m.edge_flag[e] & Visited)) {
    //                     m.edge_flag[e] |= Visited;
    //                     edges.push_back(cycle3(c.cid, k));
    //                 }
    //             }
    //         }
    //     }
    //     for (int e: edges) m.edge_flag[e] &= ~Visited;
    // }
    // auto debug_struct = [&]() {
    //     auto print_flag = [](int flag) {
    //         if (flag & Deleted) print("del, ");
    //         if (flag & Visited) print("vis, ");
    //         if (flag & Visited1) print("vis1, ");
    //         if (flag & Seam) print("seam, ");
    //         if (flag & Border) print("border, ");
    //     };
    //     print("verts:\n");
    //     for (int v: verts) {
    //         print("{}:\n", v);
    //         float3 p = m.mesh.positions[v];
    //         float2 uv = m.mesh.texcoords[v];
    //         print("  p:({}, {}, {}), uv:({}, {})\n", p.x, p.y, p.z, uv.x, uv.y);
    //         print("  uni:{}, vlink:{}, chead:{}, flag: ", m.unique_vert[v], m.vert_link[v], m.corner_head[v]);
    //         print_flag(m.vert_flag[v]);
    //         print("\n");
    //     }
    //     print("faces:\n");
    //     for (auto [c, wv]: ctx.wedge_faces) {
    //         int f = c / 3;
    //         int i0 = m.mesh.indices[f * 3];
    //         int i1 = m.mesh.indices[f * 3 + 1];
    //         int i2 = m.mesh.indices[f * 3 + 2];
    //         print("c:{},{}, f:{}, v0:{}, v1:{}, v2:{}, flag: ", c, c % 3, f, i0, i1, i2);
    //         print_flag(m.face_flag[f]);
    //         print("\n");
    //     }
    //     print("corners:\n");
    //     for (int c: corners) {
    //         print("{}: v:{}, clink:{}\n", c, m.mesh.indices[c], m.corner_link[c]);
    //     }
    //     print("edges:\n");
    //     for (int e: edges) {
    //         print("{}: e_id:{}, op_e:{}, flag:", e, m.edge_id[e], m.opposite_edge[e]);
    //         print_flag(m.edge_flag[e]);
    //         print("\n");
    //     }
    // };
    
    // if (debuging) exit(1);
    // if (debuging) {
    //     Corner e{m, edge_set[ctx.e_idx]};
    //     Vertex v0 = e.v().unique_v();
    //     Vertex v1 = e.v(1).unique_v();
    //     print("v0:{}, v1:{}\n", v0.vid, v1.vid);
    //     print_ctx(ctx);
    //     debug_struct();
    // }

    if (ctx.lock_p0 && ctx.lock_p1) return;

    if (!link_condition(ctx)) {
        return;
    }

    Corner e{m, edge_set[ctx.e_idx]};
    Vertex v0 = e.v().unique_v();
    Vertex v1 = e.v(1).unique_v();

    bool locked_newpos = false;
    if (v0.is_lock() || v1.is_lock()) locked_newpos = true;

    // 一条边最多与两个面相邻
    // assert(ctx.remove_face.size() <= 2);
    if (ctx.remove_face.size() > 2) {
        print("mesh will not manifold after merge!\n");
        exit(1);
    }

/*
		  o   v_corner0
		 / \
	v0  o---o  v1
		 \ /
		  o   v_corner1
*/
    // 每个待删除面一个，代表除v0 v1外的点，用于判断需要删除的边
    // int v_corner[2] = {-1, -1};

    // // 寻找待删除面的v_corner，因为需要未修改的index与unique_vert，因此首先执行
    // for (int i = 0; i < ctx.remove_face.size(); i++) {
    //     int f = ctx.remove_face[i] / 3;
    //     for (int k = 0; k < 3; k++) {
    //         int c = f * 3 + k;
    //         int v = m.mesh.indices[c];
    //         if (m.unique_vert[v] != v0.vid && m.unique_vert[v] != v1.vid) {
    //             assert(v_corner[i] == -1); // 只能有一个vc与v0 v1都不同
    //             v_corner[i] = c;
    //         }
    //     }
    //     assert(v_corner[i] != -1); // 必须存在与v0 v1不同的点
    // }

    // 将 vert 从 vert_link 中删除
    // 返回值：bool，当前节点是否为最后一个节点，用于判断边是否删除
    auto del_vert = [&](int v) -> bool {
        int uni_v = m.unique_vert[v];
        m.unique_vert[v] = -1;
        m.vert_flag[v] |= Deleted;
        m.vertex_count--;

        if (uni_v == v && m.vert_link[uni_v] == v) { // 只剩自己
            m.vert_link[v] = -1;
            return true;
        }
        if (m.vert_link[uni_v] == v) { // 由于上一步特判，此时 uni_v != v 一定成立
            m.vert_link[uni_v] = m.vert_link[v];
            m.vert_link[v] = -1;
        }
        else {
            int tv = m.vert_link[v];
            while (m.vert_link[tv] != v) {
                tv = m.vert_link[tv]; // 寻找v前一个节点
            }
            m.vert_link[tv] = m.vert_link[v]; // 当前节点前一个节点指向后一个节点
            m.vert_link[v] = -1;

            // 被删除点为unique，其他节点都需要更新unique_vert
            if (uni_v == v) {
                m.unique_vert[tv] = tv;
                int tv1 = m.vert_link[tv];
                while (tv1 != tv) {
                    m.unique_vert[tv1] = tv;
                    tv1 = m.vert_link[tv1];
                }
            }
        }
        return false;
    };

    // 将 corner 从 corner link 中删除
    // 返回值：bool，若删除corner后顶点没有邻接边，返回该顶点是否为相同pos的最后一个，用于判断边是否删除
    auto del_corner = [&](int c) -> bool {
        int v = m.mesh.indices[c];
        if (m.corner_head[v] == c) {
            m.corner_head[v] = m.corner_link[c];
            m.corner_link[c] = -1;

            if (m.corner_head[v] == -1) { // 该顶点没有面相邻
                return del_vert(v);
            }
        }
        else {
            int c1 = m.corner_head[v];
            int t = -1;

            while (c1 != -1) {
                t = c1;
                c1 = m.corner_link[c1];

                assert(c1 != -1); // 必须找到待删面
                if (c1 == c) { // 当前节点前一个节点指向后一个节点
                    m.corner_link[t] = m.corner_link[c];
                    m.corner_link[c] = -1;
                    break;
                }
            }
        }
        return false;
    };
    
    int lst_v = -1;
    for (int i = 0; i < ctx.wedge_verts.size(); i++) {
        int v = ctx.wedge_verts[i];
        int wi = ctx.wedge_union.find(i);
        if (wi != i) {
            int v1 = ctx.wedge_verts[wi];
            int vc = m.corner_head[v];
            // 将被合并顶点连接的面转移给合并点，用于后续删除面后判断点是否删除
            while (vc != -1) {
                int nxt = m.corner_link[vc];
                m.mesh.indices[vc] = v1; // 同时修改mesh index
                m.corner_link[vc] = m.corner_head[v1];
                m.corner_head[v1] = vc;
                vc = nxt;
            }
            m.corner_head[v] = -1;
            m.vert_flag[v] |= Deleted;
            m.vertex_count--;
            m.unique_vert[v] = -1;
            m.vert_link[v] = -1; // 此处可直接覆盖
        }
        else {
            // 重建vert_link，当前未删除的所有点建立一个环链
            if (lst_v == -1) {
                m.unique_vert[v] = v;
                m.vert_link[v] = v;
            }
            else {
                int uni_v = m.unique_vert[lst_v];
                m.unique_vert[v] = uni_v;
                m.vert_link[uni_v] = v;
                m.vert_link[v] = lst_v;
            }
            lst_v = v;

            if (locked_newpos) m.vert_flag[v] |= Locked;

            m.mesh.positions[v] = ctx.opt_p;
            for (int j = 0; j < num_attr; j++) {
                m.mesh.texcoords[v][j] = ctx.opt_attr[j + i * num_attr];
            }
        }
    }

    // if (debuging) {
    //     print("\n\n ### after remove vert ### \n\n");
    //     debug_struct();
    // }

    for (int cid: ctx.remove_face) {
        int fid = cid / 3;
/*
		  o   v_corner0
		 / \
	v0  o---o  v1
		 \ /
		  o   v_corner1
*/
        // 代表除v0 v1外的点，用于判断需要删除的边
        int v_corner = -1;
        bool is_v_remove = false;

        m.face_flag[fid] |= Deleted;
        m.face_count--;

        for (int k = 0; k < 3; k++) {
            Corner c{m, fid * 3 + k};
            // 另外两点被合并，当前点是v_corner
            if (c.v(1).unique_v() == c.v(2).unique_v()) {
                assert(v_corner == -1); // 只能有一个v_corner
                v_corner = c.cid;
            }
            // 从corner link分离待删除面
            bool is_remove = del_corner(c.cid);
            if (v_corner == c.cid) {
                // 若当前点为v_corner，且当前点被移除（且没有其他相同pos点）则v_corner连接的两条边都需要删除
                is_v_remove |= is_remove;
            }
        }
        assert(v_corner != -1); // 必须存在v_corner

        // 与vc相连的两条边需要合并
        int e0 = v_corner;
        int e1 = cycle3(v_corner, 2);
        int op_e0 = m.opposite_edge[e0];
        int op_e1 = m.opposite_edge[e1];
        // 若为border边则e一定是id，不需要特判
        int e_id0 = (m.edge_id[e0] != -1) ? m.edge_id[e0] : m.edge_id[op_e0];
        int e_id1 = (m.edge_id[e1] != -1) ? m.edge_id[e1] : m.edge_id[op_e1];

        if (is_v_remove) { // 两条边都需要删除
            edge_set[e_id0] = -1;
            edge_set[e_id1] = -1;
            heap.remove(e_id0);
            heap.remove(e_id1);
        }
        else {
            // 将两边的flag合并（合理吗?）
            const int flag = Border | Seam;
            int e_f = (m.edge_flag[e0] & flag) | (m.edge_flag[e1] & flag);

            // 修改对边，被合并的两边的对边相互指向
            if (op_e0 != -1) m.opposite_edge[op_e0] = op_e1, m.edge_flag[op_e0] |= e_f;
            if (op_e1 != -1) m.opposite_edge[op_e1] = op_e0, m.edge_flag[op_e1] |= e_f;
            
            // 删掉一条边
            edge_set[e_id1] = -1;
            heap.remove(e_id1);

            // 将e_id赋值给一条边，另一条赋值-1
            if (op_e0 != -1) {
                m.edge_id[op_e0] = e_id0;
                edge_set[e_id0] = op_e0;
                if (op_e1 != -1) m.edge_id[op_e1] = -1;
            }
            else {
                m.edge_id[op_e1] = e_id0;
                edge_set[e_id0] = op_e1;
            }
        }
        for (int k = 0; k < 3; k++) {
            int c = fid * 3 + k;
            m.opposite_edge[c] = -1;
            m.edge_id[c] = -1;
            m.edge_flag[c] |= Deleted;
            m.mesh.indices[c] = -1;
        }
    }

    // if (debuging) {
    //     print("\n\n ### after remove face ### \n\n");
    //     debug_struct();
    // }


    edge_set[ctx.e_idx] = -1;

    for (auto [c, wv]: ctx.wedge_faces) {
        int f = c / 3;
        if (m.face_flag[f] & Deleted) continue;

        // recalc face quadric
        calc_face_quadric(f);
    }



    // for (int cid: ctx.remove_face) {
    //     int fid = cid / 3;
    //     int vc = -1; // 不是v0 v1的corner

    //     m.face_flag[fid] |= MeshStructure::Deleted;
    //     m.face_count--;

    //     for (int k = 0; k < 3; k++) {
    //         int c = fid * 3 + k;
    //         int v = m.mesh.indices[c];

    //         if (m.unique_vert[v] != v0.vid && m.unique_vert[v] != v1.vid) {
    //             assert(vc == -1); // 只能有一个vc与v0 v1都不同
    //             vc = c;
    //         }

    //         // 从corner link分离待删除面
    //         if (m.corner_head[v] == c) {
    //             m.corner_head[v] = m.corner_link[c];
    //             m.corner_link[c] = -1;
    //         }
    //         else {
    //             int c1 = m.corner_head[v];
    //             int t = -1;

    //             while (c1 != -1) {
    //                 t = c1;
    //                 c1 = m.corner_link[c1];

    //                 assert(c1 != -1); // 必须找到待删面
    //                 if (c1 == c) { // 当前节点前一个节点指向后一个节点
    //                     m.corner_link[t] = m.corner_link[c];
    //                     m.corner_link[c] = -1;
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     assert(vc != -1); // 必须存在与v0 v1不同的点

    //     // 与vc相连的两条边需要合并
    //     int e0 = vc;
    //     int e1 = cycle3(vc, 2);
    //     int op_e0 = m.opposite_edge[e0];
    //     int op_e1 = m.opposite_edge[e1];
    //     // 若为border边则e一定是id，不需要特判
    //     int e_id0 = (m.edge_id[e0] != -1) ? m.edge_id[e0] : m.edge_id[op_e0];
    //     int e_id1 = (m.edge_id[e1] != -1) ? m.edge_id[e1] : m.edge_id[op_e1];

    //     // 将两边的flag合并（合理吗?）
    //     const int flag = MeshStructure::Border | MeshStructure::Seam;
    //     int e_f = (m.edge_flag[e0] & flag) | (m.edge_flag[e1] & flag);

    //     if (op_e0 != -1) m.opposite_edge[op_e0] = op_e1, m.edge_flag[op_e0] |= e_f;
    //     if (op_e1 != -1) m.opposite_edge[op_e1] = op_e0, m.edge_flag[op_e1] |= e_f;

    //     // 删掉一条边
    //     edge_set[e_id1] = -1;
    //     heap.remove(e_id1);

    //     // 将e_id赋值给一条边，另一条赋值-1
    //     if (op_e0 != -1) {
    //         m.edge_id[op_e0] = e_id0;
    //         edge_set[e_id0] = op_e0;
    //         if (op_e1 != -1) m.edge_id[op_e1] = -1;
    //     }
    //     else {
    //         m.edge_id[op_e1] = e_id0;
    //         edge_set[e_id0] = op_e1;
    //     }

    //     for (int k = 0; k < 3; k++) {
    //         int c = fid * 3 + k;
    //         m.opposite_edge[c] = -1;
    //         m.edge_id[c] = -1;
    //         m.edge_flag[c] |= MeshStructure::Deleted;
    //         m.mesh.indices[c] = -1;
    //     }
    // }

    // if (debuging) {
    //     print("\n\n ### after remove face ### \n\n");
    //     debug_struct();
    // }

    // edge_set[ctx.e_idx] = -1;

    // int lst_v = -1;
    // for (int i = 0; i < ctx.wedge_verts.size(); i++) {
    //     int v = ctx.wedge_verts[i];
    //     int wedge_id = ctx.wedge_union.find(i);

    //     if (wedge_id != i) { // 顶点被合并，删除该点
    //         m.vert_flag[v] |= MeshStructure::Deleted;
    //         m.vertex_count--;
    //         m.corner_head[v] = -1;
    //         m.unique_vert[v] = -1;
    //         m.vert_link[v] = -1;
    //     }
    //     else {
    //         // 重建vert_link，当前未删除的所有点建立一个环链
    //         if (lst_v == -1) {
    //             m.unique_vert[v] = v;
    //             m.vert_link[v] = v;
    //         }
    //         else {
    //             int uni_v = m.unique_vert[lst_v];
    //             m.unique_vert[v] = uni_v;
    //             m.vert_link[uni_v] = v;
    //             m.vert_link[v] = lst_v;
    //         }
    //         lst_v = v;

    //         if (locked_newpos) m.vert_flag[v] |= MeshStructure::Locked;

    //         m.mesh.positions[v] = ctx.opt_p;
    //         for (int j = 0; j < num_attr; j++) {
    //             m.mesh.texcoords[v][j] = ctx.opt_attr[j + i * num_attr];
    //         }
    //     }
    // }

    // if (debuging) {
    //     print("\n\n ### after remove vert ### \n\n");
    //     debug_struct();
    // }

    // for (auto [c, wv]: ctx.wedge_faces) {
    //     if (m.face_flag[c / 3] & MeshStructure::Deleted) continue;

    //     int wedge_id = ctx.wedge_union.find(wv);
    //     if (wedge_id != wv) { // 顶点被合并，将mesh index修改为合并后的点
    //         int v = ctx.wedge_verts[wedge_id];
    //         m.mesh.indices[c] = v;

    //         // 将当前面插入到新顶点的corner link
    //         m.corner_link[c] = m.corner_head[v];
    //         m.corner_head[v] = c;
    //     }
    //     // recalc face quadric
    //     calc_face_quadric(c / 3);
    // }

    // if (debuging) {
    //     print("\n\n ### after update mesh ### \n\n");
    //     debug_struct();
    // }

    // recalc merge
    Vertex v{m, lst_v};
    for (Corner c: v.wedge_corners()) {
        // 枚举相邻顶点
        for (int k: {1, 2}) {
            Vertex v1 = c.v(k);
            if (!v1.is_vis()) {
                v1.set_vis();
                // 枚举相邻顶点的连接边，这些边需要重新计算
                for (Corner c1: v1.wedge_corners()) {
                    int cid = c1.cid;
                    for (int k1: {0, 2}) {
                        int c = cycle3(cid, k1);
                        int e_id = m.edge_id[c];
                        if (e_id != -1 && heap.is_present(e_id)) {
                            ctx.recalc_edges.push_back(e_id);
                            heap.remove(e_id);
                        }
                    }
                }
            }
        }
    }
    for (Corner c: v.wedge_corners()) {
        for (int k: {1, 2}) c.v(k).clear_vis();
    }
    for (int e_id: ctx.recalc_edges) {
        evaluate_merge(ctx, e_id);
        heap.update(e_id, ctx.quadric_error + ctx.penalty);
    }

    // if (debuging) exit(1);
}

bool MeshSimplifier::link_condition(MergeContext &ctx) {
    Corner e{m, edge_set[ctx.e_idx]};
    Vertex v0 = e.v().unique_v();
    Vertex v1 = e.v(1).unique_v();

    int num_adj_vert = 0; // 与v0 v1都相邻的顶点数

    for (auto [cid, wv]: ctx.wedge_faces) {
        Corner c{m, cid};
        if (c.v().unique_v() == v0) {
            c.v(1).unique_v().set_vis();
            c.v(2).unique_v().set_vis();
        }
        else {
            for (int k = 1; k <= 2; k++) {
                if (!c.v(k).unique_v().is_vis1()) {
                    c.v(k).unique_v().set_vis1();
                    if (c.v(k).unique_v().is_vis()) num_adj_vert++;
                }
            }
        }
    }
    for (int cid: ctx.remove_face) {
        Corner c{m, cid};
        for (int k = 1; k <= 2; k++) {
            if (!c.v(k).unique_v().is_vis1()) {
                c.v(k).unique_v().set_vis1();
                if (c.v(k).unique_v().is_vis()) num_adj_vert++;
            }
        }
    }
    for (auto [cid, wv]: ctx.wedge_faces) {
        Corner c{m, cid};
        for (int k = 0; k < 3; k++) {
            c.v(k).unique_v().clear_vis();
            c.v(k).unique_v().clear_vis1();
        }
    }

    // print("num_adj_vert: {}, border: {}\n", num_adj_vert, bool(m.edge_flag[e.cid] & MeshStructure::Border));

    if (m.edge_flag[e.cid] & MeshStructure::Border) return num_adj_vert <= 1;
    else return num_adj_vert <= 2;
}

void MeshSimplifier::compact_mesh() {
    using MeshStructure::Deleted;
    int cur = 0;
    for (int i = 0; i < m.mesh.indices.size() / 3; i++) {
        // 三角形保留
        if (!(m.face_flag[i] & Deleted)) {
            if (i != cur) {
                m.mesh.indices[cur * 3] = m.mesh.indices[i * 3];
                m.mesh.indices[cur * 3 + 1] = m.mesh.indices[i * 3 + 1];
                m.mesh.indices[cur * 3 + 2] = m.mesh.indices[i * 3 + 2];
            }
            cur++;
        }
    }
    assert(cur == m.face_count);
    m.mesh.indices.resize(cur * 3);

    // 重映射顶点
    vector<int> remap(m.mesh.positions.size());
    cur = 0;
    for (int i = 0; i < m.mesh.positions.size(); i++) {
        remap[i] = cur;
        if (!(m.vert_flag[i] & Deleted)) {
            if (i != cur) {
                m.mesh.positions[cur] = m.mesh.positions[i];
                m.mesh.texcoords[cur] = m.mesh.texcoords[i];
            }
            cur++;
        }
    }
    assert(cur == m.vertex_count);
    m.mesh.positions.resize(cur);
    m.mesh.texcoords.resize(cur);

    for (int &i: m.mesh.indices) i = remap[i];
}

double MeshSimplifier::simplify(int target_face_num) {
    if (m.face_count <= target_face_num) return 0;

    MergeContext ctx;
    IndexHeap heap(edge_set.size());

    for (int i = 0; i < edge_set.size(); i++) {
        evaluate_merge(ctx, i);
        heap.update(i, ctx.quadric_error + ctx.penalty);
    }

    double max_error = 0;

    while (heap.size()) {
        auto [error, e_id] = heap.pop();

        evaluate_merge(ctx, e_id);
        max_error = std::max(max_error, ctx.quadric_error);
        
        perform_merge(ctx, heap);
        
        if (m.face_count <= target_face_num) break;
    }
    compact_mesh();

    if (m.mesh.indices.size() / 3 > target_face_num) {
        print("simplify warn: target_face_num {} not reach cur_face_num {}\n", target_face_num, m.mesh.indices.size() / 3);
    }

    // print("invaild: {}\n", ttt);

    return sqrt(max_error) / position_scale;
}

void print_ctx(MergeContext &ctx) {
    print("## ctx\n");
    print("wedge vert: [");
    for (int x: ctx.wedge_verts) print("{} ", x);
    print("]\n");

    print("wedge union: [");
    for (int x: ctx.wedge_union.fa) print("{} ", x);
    print("]\n");

    print("wedge face: [");
    for (auto [c, wv]: ctx.wedge_faces) {
        print("({}, {}) ", c, wv);
    }
    print("]\n");

    print("remove face: [");
    for (int x: ctx.remove_face) print("{} ", x);
    print("]\n");

    print("opt_p:({}, {}, {})\n", ctx.opt_p.x, ctx.opt_p.y, ctx.opt_p.z);
    print("opt_a: ");
    for (int i = 0; i < ctx.opt_attr.size(); i++) print("{}:{}, ", i, ctx.opt_attr[i]);
    print("\n");
    print("error: {}\n", ctx.quadric_error);
    print("penalty: {}\n", ctx.penalty);
}

void print_structure(MeshSimplifier &s) {
    print("## mesh struct\n");
    print("edge_set:\n");
    for (int i = 0; i < s.edge_set.size(); i++) {
        print("{}: {}\n", i, s.edge_set[i]);
    }

    auto print_flag = [](int flag) {
        using MeshStructure::Deleted;
        using MeshStructure::Visited;
        using MeshStructure::Seam;
        using MeshStructure::Border;

        if (flag & Deleted) print("del, ");
        if (flag & Visited) print("vis, ");
        if (flag & Seam) print("seam, ");
        if (flag & Border) print("border, ");
    };

    print("vert:\n");
    for (int i = 0; i < s.m.mesh.positions.size(); i++) {
        print("{}: uni:{}, vlink:{}, chead:{}, flag: ", i, s.m.unique_vert[i], s.m.vert_link[i], s.m.corner_head[i]);
        print_flag(s.m.vert_flag[i]);
        print("\n");
    }

    print("face:\n");
    for (int i = 0; i < s.m.mesh.indices.size() / 3; i++) {
        print("{}: v0:{}, v1:{}, v2:{}, flag: ", i, s.m.mesh.indices[i * 3], s.m.mesh.indices[i * 3 + 1], s.m.mesh.indices[i * 3 + 2]);
        print_flag(s.m.face_flag[i]);
        print("\n");
    }

    print("corner:\n");
    for (int i = 0; i < s.m.mesh.indices.size(); i++) {
        print("{}: clink:{}, e_id:{}, op_e:{}, flag: ", i, s.m.corner_link[i], s.m.edge_id[i], s.m.opposite_edge[i]);
        print_flag(s.m.edge_flag[i]);
        print("\n");
    }
}

// double mesh_simplify1(Mesh &mesh, int target_face_num) {
//     MeshSimplifier simplifier(mesh);
//     return simplifier.simplify(target_face_num);
// }

}
