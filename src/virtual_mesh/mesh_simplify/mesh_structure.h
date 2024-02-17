#include <mesh.h>
#include <assert.h>

namespace virtual_mesh {

inline int cycle3(int i, int ofs) {
    return i - i % 3 + (i + ofs) % 3;
}

struct MeshStructure {
    Mesh &mesh;

    int vertex_count;
    int face_count;
    int edge_count;

    size_t table_size;
    luisa::vector<unsigned int> vert_table;
    luisa::vector<int> unique_vert; // 每个下标指向第一个相同pos
    luisa::vector<int> vert_link; // 每个下标指向前一个相同pos，第一个pos指向最后一个

    luisa::vector<int> corner_head; // 每个顶点指向一个相邻三角形的corner，没有为-1
    luisa::vector<int> corner_link; // 每个三角形的每个corner指向下一个相同顶点的corener

    luisa::vector<int> opposite_edge; // 每个边相反边的下标，边为corner与下一个顶点
    luisa::vector<int> edge_id; // 每个边一个唯一id，若边重复，则该边edge id为-1，相反边为id

    luisa::vector<luisa::ubyte> vert_flag;
    luisa::vector<luisa::ubyte> face_flag;
    luisa::vector<luisa::ubyte> edge_flag;

    enum Flag {
        Deleted = 1 << 0,
        Visited = 1 << 1,
        Visited1 = 1 << 2,
        Seam = 1 << 3, // 纹理接缝
        Border = 1 << 4, // 边界
        Locked = 1 << 5,
    };

    struct VertCornerIterProxy;
    struct WedgeVertIterProxy;
    struct WedgeCornerIterProxy;

    struct Vertex {
        MeshStructure &m;
        int vid;

        VertCornerIterProxy vert_corners() { assert(vid != -1); return {m, vid}; }
        WedgeVertIterProxy wedge_verts() { assert(vid != -1); return {m, vid}; }
        WedgeCornerIterProxy wedge_corners() { assert(vid != -1); return {m, vid}; }

        auto& p() { assert(vid != -1); return m.mesh.positions[vid]; }
        Vertex unique_v() { assert(vid != -1); return {m, m.unique_vert[vid]}; }
        void to_unique() { assert(vid != -1); vid = m.unique_vert[vid]; }

        luisa::ubyte& flag() { assert(vid != -1); return m.vert_flag[vid]; }
        bool is_del() { assert(vid != -1); return flag() & Deleted; }
        bool is_vis() { assert(vid != -1); return flag() & Visited; }
        bool is_vis1() { assert(vid != -1); return flag() & Visited1; }
        bool is_lock() { assert(vid != -1); return flag() & Locked; }
        void set_del() { assert(vid != -1); flag() |= Deleted; }
        void set_vis() { assert(vid != -1); flag() |= Visited; }
        void set_vis1() { assert(vid != -1); flag() |= Visited1; }
        void set_lock() { assert(vid != -1); flag() |= Locked; }
        void clear_del() { assert(vid != -1); flag() &= ~Deleted; }
        void clear_vis() { assert(vid != -1); flag() &= ~Visited; }
        void clear_vis1() { assert(vid != -1); flag() &= ~Visited1; }
        void clear_lock() { assert(vid != -1); flag() &= ~Locked; }

        bool operator==(Vertex other) { return vid == other.vid; }
    };
    struct Face {
        MeshStructure &m;
        int fid;

        Vertex v(int i) { assert(fid != -1); return {m, m.mesh.indices[fid * 3 + i]}; }

        luisa::ubyte& flag() { assert(fid != -1); return m.face_flag[fid]; }
        bool is_del() { assert(fid != -1); return flag() & Deleted; }
        bool is_vis() { assert(fid != -1); return flag() & Visited; }
        void set_del() { assert(fid != -1); flag() |= Deleted; }
        void set_vis() { assert(fid != -1); flag() |= Visited; }
        void clear_del() { assert(fid != -1); flag() &= ~Deleted; }
        void clear_vis() { assert(fid != -1); flag() &= ~Visited; }
    };
    struct Corner {
        MeshStructure &m;
        int cid;

        // 以当前corner为三角形0下标，返回第i个顶点
        Vertex v(int i) {
            assert(cid != -1);
            assert(i >= 0 && i <= 2);
            return {m, m.mesh.indices[cycle3(cid, i)]};
        }
        Vertex v() { assert(cid != -1); return {m, m.mesh.indices[cid]}; }
        Face f() { assert(cid != -1); return {m, cid / 3}; }
    };

    // 遍历一个顶点对应的corner
    struct VertCornerIter {
        MeshStructure &m;
        int cid;

        void operator++() { assert(cid != -1); cid = m.corner_link[cid]; }
        Corner operator*() { assert(cid != -1); return {m, cid}; }
        bool operator==(VertCornerIter other) { return cid == other.cid; }
    };
    struct VertCornerIterProxy {
        MeshStructure &m;
        int vid;

        VertCornerIter begin() { assert(vid != -1); return {m, m.corner_head[vid]}; }
        VertCornerIter end() { return {m, -1}; }
    };

    // 遍历一个顶点相同pos的所有顶点
    struct WedgeVertIter {
        MeshStructure &m;
        int vid;

        void operator++() {
            assert(vid != -1);
            if (vid == m.unique_vert[vid]) vid = -1;
            else vid = m.vert_link[vid];
        }
        Vertex operator*() { assert(vid != -1); return {m, vid}; }
        bool operator==(WedgeVertIter other) { return vid == other.vid; }
    };
    struct WedgeVertIterProxy {
        MeshStructure &m;
        int vid;

        // 从最后一个开始遍历
        WedgeVertIter begin() { assert(vid != -1); return {m, m.vert_link[m.unique_vert[vid]]}; }
        WedgeVertIter end() { return {m, -1}; }
    };

    struct WedgeCornerIter {
        WedgeVertIter wv_iter;
        VertCornerIter vc_iter;

        void operator++() {
            ++vc_iter;
            if (vc_iter.cid == -1) {
                ++wv_iter;
                if (wv_iter.vid != -1) {
                    vc_iter.cid = vc_iter.m.corner_head[wv_iter.vid];
                }
            }
        }
        Corner operator*() { return *vc_iter; }
        bool operator==(WedgeCornerIter other) { return wv_iter == other.wv_iter; }
    };
    struct WedgeCornerIterProxy {
        MeshStructure &m;
        int vid;

        WedgeCornerIter begin() {
            assert(vid != -1);
            int v = m.vert_link[m.unique_vert[vid]];
            int c = m.corner_head[v];
            return {{m, v}, {m, c}};
        }
        WedgeCornerIter end() { return {{m, -1}, {m, -1}}; };
    };

    MeshStructure(Mesh &mesh);

    void init_vert_link();
    void init_corner_link();
    void init_edge_link();

    void lock_position(float3 p);
};


}