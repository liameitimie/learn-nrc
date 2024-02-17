#include <mesh.h>
#include <meshoptimizer.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
// #include <fstream>

using namespace luisa;
using namespace fmt;

namespace virtual_mesh {

void assimp_load(Mesh &mesh, const char* file) {
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

    if(!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        print("failed to load scene: {}\n", importer.GetErrorString());
        exit(1);
    }
    auto ai_mesh = scene->mMeshes[0];

    if (!ai_mesh->HasPositions() || !ai_mesh->HasFaces() || !ai_mesh->HasTextureCoords(0)) {
        print("invaild mesh\n");
        exit(1);
    }

    print("convert from assimp scene: ");
    timer.tic();

    mesh.positions.resize(ai_mesh->mNumVertices);
    mesh.indices.resize(ai_mesh->mNumFaces * 3);
    mesh.texcoords.resize(ai_mesh->mNumVertices);

    for(int i = 0; i < ai_mesh->mNumVertices; i++){
        mesh.positions[i] = {
            ai_mesh->mVertices[i].x,
            ai_mesh->mVertices[i].y,
            ai_mesh->mVertices[i].z
        };
    }
    for(int i = 0; i < ai_mesh->mNumFaces; i++){
        if (ai_mesh->mFaces[i].mNumIndices != 3) {
            print("not a triangle mesh\n");
            exit(1);
        }
        mesh.indices[i * 3] = ai_mesh->mFaces[i].mIndices[0];
        mesh.indices[i * 3 + 1] = ai_mesh->mFaces[i].mIndices[1];
        mesh.indices[i * 3 + 2] = ai_mesh->mFaces[i].mIndices[2];
    }
    for(int i = 0; i < ai_mesh->mNumVertices; i++){
        mesh.texcoords[i] = {
            ai_mesh->mTextureCoords[0][i].x,
            ai_mesh->mTextureCoords[0][i].y,
        };
    }
    print("{} ms\n", timer.toc());
    print("num vertex: {}, num triangle: {}\n", mesh.positions.size(), mesh.indices.size() / 3);
}

void compact(Mesh &mesh) {
    meshopt_Stream stream[] = {
        {&mesh.positions[0], sizeof(float3), sizeof(float) * 3},
        {&mesh.texcoords[0], sizeof(float2), sizeof(float2)},
    };
    uint stream_count = sizeof(stream) / sizeof(stream[0]);
    uint index_count = mesh.indices.size();
    uint vertex_count = mesh.positions.size();
    vector<uint> remap(vertex_count);
    vertex_count = meshopt_generateVertexRemapMulti(&remap[0], &mesh.indices[0], index_count, vertex_count, stream, stream_count);

    vector<float3> new_positions(vertex_count);
    vector<float2> new_texcoords(vertex_count);

    meshopt_remapIndexBuffer(&mesh.indices[0], &mesh.indices[0], index_count, &remap[0]);
    meshopt_remapVertexBuffer(&new_positions[0], &mesh.positions[0], mesh.positions.size(), sizeof(float3), &remap[0]);
    meshopt_remapVertexBuffer(&new_texcoords[0], &mesh.texcoords[0], mesh.texcoords.size(), sizeof(float2), &remap[0]);

    mesh.positions.swap(new_positions);
    mesh.texcoords.swap(new_texcoords);
}

void Mesh::compact() {
    virtual_mesh::compact(*this);
}

Mesh load_mesh(const char* file) {
    Mesh mesh;
    assimp_load(mesh, file);

    Clock timer;
    print("compact: ");
    compact(mesh);
    print("{} ms\n", timer.toc());
    print("num vertex: {}, num triangle: {}\n", mesh.positions.size(), mesh.indices.size() / 3);

    return mesh;
}

void debug_out(Mesh &mesh, const char* file) {
    print("v_cnt:{}, i_cnt:{}\n", mesh.positions.size(), mesh.indices.size());
    print("p0: ({}, {}, {})\n", mesh.positions[0].x, mesh.positions[0].y, mesh.positions[0].z);
    print("i0: {}, i1: {}, i2: {}\n", mesh.indices[0], mesh.indices[1], mesh.indices[2]);

    int size = 1 + mesh.positions.size() * 3 + mesh.texcoords.size() * 2 + 1 + mesh.indices.size();
    vector<uint> pack_data(size);
    int offset = 0;

    pack_data[offset] = mesh.positions.size();
    offset += 1;

    memcpy(pack_data.data() + offset, mesh.positions.data(), mesh.positions.size() * 3 * sizeof(float));
    offset += mesh.positions.size() * 3;

    memcpy(pack_data.data() + offset, mesh.texcoords.data(), mesh.texcoords.size() * 2 * sizeof(float));
    offset += mesh.texcoords.size() * 2;

    pack_data[offset] = mesh.indices.size();
    offset += 1;

    memcpy(pack_data.data() + offset, mesh.indices.data(), mesh.indices.size() * sizeof(int));

    FILE* p = fopen(file, "wb");
    fwrite(pack_data.data(), pack_data.size() * sizeof(uint), 1, p);
    fclose(p);
}

void debug_in(Mesh &mesh, const char* file) {
    FILE* p = fopen(file, "rb");
    if (!p) {
        print("can't open file\n");
        exit(1);
    }

    int vert_cnt = 0;
    fread(&vert_cnt, sizeof(int), 1, p);

    mesh.positions.resize(vert_cnt);
    mesh.texcoords.resize(vert_cnt);

    fread(mesh.positions.data(), vert_cnt * 3 * sizeof(float), 1, p);
    fread(mesh.texcoords.data(), vert_cnt * 2 * sizeof(float), 1, p);

    int idx_cnt = 0;
    fread(&idx_cnt, sizeof(int), 1, p);

    mesh.indices.resize(idx_cnt);

    fread(mesh.indices.data(), idx_cnt * sizeof(int), 1, p);

    print("v_cnt:{}, i_cnt:{}\n", mesh.positions.size(), mesh.indices.size());
    print("p0: ({}, {}, {})\n", mesh.positions[0].x, mesh.positions[0].y, mesh.positions[0].z);
    print("i0: {}, i1: {}, i2: {}\n", mesh.indices[0], mesh.indices[1], mesh.indices[2]);
}

}