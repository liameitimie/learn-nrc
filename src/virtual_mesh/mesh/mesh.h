#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <float3.h>

namespace virtual_mesh {

struct Mesh {
    luisa::vector<float3> positions;
    luisa::vector<int> indices;
    luisa::vector<luisa::float2> texcoords;

    Mesh() = default;
    Mesh(Mesh &) = default;
    Mesh(Mesh &&) = default;
    Mesh& operator=(Mesh &rhs) = default;
    Mesh& operator=(Mesh &&rhs) = default;

    void compact();
};

Mesh load_mesh(const char* file);

void debug_out(Mesh &mesh, const char* file);
void debug_in(Mesh &mesh, const char* file);

}