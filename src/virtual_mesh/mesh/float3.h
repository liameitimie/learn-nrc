#pragma once

#include <luisa/core/stl/hash_fwd.h>

namespace virtual_mesh {

struct float3 {
    float x, y, z;
};

inline bool operator==(const float3 &a, const float3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool operator!=(const float3 &a, const float3 &b) {
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

inline bool operator<(const float3 &a, const float3 &b) {
    return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : (a.z < b.z));
}

inline float3 operator+(const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline float3 operator*(const float3 a, const float b) {
    return {a.x * b, a.y * b, a.z * b};
}

}

template<>
struct luisa::hash<virtual_mesh::float3> {
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(const virtual_mesh::float3 &v, uint64_t seed = hash64_default_seed) const noexcept {
        return hash64(&v, sizeof(float) * 3, seed);
    }
};

// template<>
// struct luisa::hash<virtual_mesh::float3> {
//     using is_avalanching = void;
//     [[nodiscard]] uint64_t operator()(const virtual_mesh::float3 &v) const noexcept {
//         const unsigned int* key = (const unsigned int*)(&v);

//         // scramble bits to make sure that integer coordinates have entropy in lower bits
// 		uint64_t x = key[0] ^ (key[0] >> 17);
// 		uint64_t y = key[1] ^ (key[1] >> 17);
// 		uint64_t z = key[2] ^ (key[2] >> 17);

// 		// Optimized Spatial Hashing for Collision Detection of Deformable Objects
// 		return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
//     }
// };