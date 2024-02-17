#include <mesh_structure.h>
#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <luisa/core/stl.h>
#include <luisa/core/mathematics.h>


using namespace luisa;
using namespace fmt;

namespace virtual_mesh {

/// hash table, copy from meshoptimizer

struct PositionHasher {
	const float* vertex_positions;
	size_t vertex_stride_float;

    size_t hash_key(const uint* key) const {
        // scramble bits to make sure that integer coordinates have entropy in lower bits
		uint x = key[0] ^ (key[0] >> 17);
		uint y = key[1] ^ (key[1] >> 17);
		uint z = key[2] ^ (key[2] >> 17);

		// Optimized Spatial Hashing for Collision Detection of Deformable Objects
		return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
    }
	size_t hash(uint index) const {
		const uint* key = (const uint*)(vertex_positions + index * vertex_stride_float);
        return hash_key(key);
	}
    size_t hash(const float3 &p) const {
        const uint* key = (const uint*)&p;
        return hash_key(key);
    }

	bool equal(uint lhs, uint rhs) const {
		return memcmp(vertex_positions + lhs * vertex_stride_float, vertex_positions + rhs * vertex_stride_float, sizeof(float) * 3) == 0;
	}
    bool equal(uint idx, const float3 &p) const {
        return memcmp(vertex_positions + idx * vertex_stride_float, &p, sizeof(float) * 3) == 0;
    }
};

size_t hash_buckets(size_t count) {
    return next_pow2(count + count / 4);
}

uint* hash_lookup(uint* table, size_t buckets, const PositionHasher& hash, const uint& key, const uint& empty) {
	assert(buckets > 0);
	assert((buckets & (buckets - 1)) == 0);

	size_t hashmod = buckets - 1;
	size_t bucket = hash.hash(key) & hashmod;

	for (size_t probe = 0; probe <= hashmod; ++probe) {
		uint& item = table[bucket];

		if (item == empty)
			return &item;

		if (hash.equal(item, key))
			return &item;

		// hash collision, quadratic probing
		bucket = (bucket + probe + 1) & hashmod;
	}

	assert(false && "Hash table is full"); // unreachable
	return NULL;
}

uint* hash_lookup(uint* table, size_t buckets, const PositionHasher& hash, const float3& p, const uint& empty) {
	assert(buckets > 0);
	assert((buckets & (buckets - 1)) == 0);

	size_t hashmod = buckets - 1;
	size_t bucket = hash.hash(p) & hashmod;

	for (size_t probe = 0; probe <= hashmod; ++probe) {
		uint& item = table[bucket];

		if (item == empty)
			return &item;

		if (hash.equal(item, p))
			return &item;

		// hash collision, quadratic probing
		bucket = (bucket + probe + 1) & hashmod;
	}

	assert(false && "Hash table is full"); // unreachable
	return NULL;
}

// Clock timer;

MeshStructure::MeshStructure(Mesh &mesh): mesh(mesh) {
    assert(mesh.indices.size() % 3 == 0);

    mesh.compact();

    vertex_count = mesh.positions.size();
    face_count = mesh.indices.size() / 3;
    edge_count = 0;

    vert_flag.resize(vertex_count);
    face_flag.resize(face_count);
    edge_flag.resize(face_count * 3);

    // init_vert_link();
    // init_corner_link();
    // init_edge_link();
    // init_edge_id();
}

void MeshStructure::init_vert_link() {
    // print("init_vert_link: ");
    // timer.tic();

    PositionHasher hasher{(float*)mesh.positions.data(), 3};

    table_size = hash_buckets(vertex_count);
    vert_table.resize(table_size, ~0u);

    unique_vert.resize(vertex_count);

    // 每个顶点指向第一个相同pos的下标
    for (int i = 0; i < vertex_count; i++) {
        uint* entry = hash_lookup(vert_table.data(), table_size, hasher, i, ~0u);

        if (*entry == ~0u)
            *entry = i;

        unique_vert[i] = *entry;
    }

    vert_link.resize(vertex_count);

    // 建立相同pos的顶点的链接，每个顶点指向前一个相同pos的顶点，第一个指向最后一个
    for (int i = 0; i < vertex_count; i++) {
        if (unique_vert[i] != i) {
            int v = unique_vert[i];
            vert_link[i] = vert_link[v];
            vert_link[v] = i;
        }
        else {
            vert_link[i] = i;
        }
    }
    // print("{}\n", timer.toc());
}

void MeshStructure::init_corner_link() {
    // print("init_corner_link: ");
    // timer.tic();

    int vert_cnt = mesh.positions.size();
    int index_cnt = mesh.indices.size() / 3 * 3;

    corner_head.resize(vert_cnt, -1);
    corner_link.resize(index_cnt, -1);

    for (int i = 0; i < index_cnt; i++) {
        int v_id = mesh.indices[i];

        corner_link[i] = corner_head[v_id];
        corner_head[v_id] = i;
    }
    // print("{}\n", timer.toc());
}

void MeshStructure::init_edge_link() {
    int index_cnt = mesh.indices.size() / 3 * 3;

    opposite_edge.resize(index_cnt, -1);
    edge_id.resize(index_cnt, -1);

    // print("init_edge_link: ");
    // timer.tic();

    edge_count = 0;
    for (int i = 0; i < index_cnt; i++) {
        if (opposite_edge[i] != -1) continue;

        edge_id[i] = edge_count++;

        Corner c{*this, i};

        for (Corner vc: c.v(1).vert_corners()) {
            if (vc.v(1) == c.v()) {
                if (opposite_edge[vc.cid] != -1) {
                    print("mesh not manifold\n");
                    exit(1);
                }
                opposite_edge[i] = vc.cid;
                opposite_edge[vc.cid] = i;
                break;
            }
        }
        if (opposite_edge[i] == -1) {
            // 寻找不同wedge的相反边
            for (Corner wc: c.v(1).wedge_corners()) {
                if (wc.v(1).unique_v() == c.v().unique_v()) {
                    if (opposite_edge[wc.cid] != -1) {
                        print("mesh not manifold\n");
                        exit(1);
                    }
                    opposite_edge[i] = wc.cid;
                    opposite_edge[wc.cid] = i;

                    // 不同wedge为纹理接缝
                    edge_flag[i] |= Seam;
                    edge_flag[wc.cid] |= Seam;

                    break;
                }
            }
        }
        if (opposite_edge[i] == -1) edge_flag[i] |= Border;
    }
    // print("{}\n", timer.toc());

    // print("num edge: {}\n", edge_count);
}

void MeshStructure::lock_position(float3 p) {
    PositionHasher hasher{(float*)mesh.positions.data(), 3};
    uint* entry = hash_lookup(vert_table.data(), table_size, hasher, p, ~0u);

    if (*entry != ~0u) {
        Vertex v{*this, (int)*entry};
        for (auto tv: v.wedge_verts()) {
            tv.set_lock();
        }
    }
}

}