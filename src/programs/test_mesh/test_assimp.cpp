#include <luisa/core/basic_types.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

using namespace luisa;
using namespace fmt;

int main() {
    Clock timer;
    Assimp::Importer importer;

    timer.tic();
    print("load scene: ");
    auto load_flag = 0 
        // | aiProcess_JoinIdenticalVertices 
        | aiProcess_Triangulate
        ;
    const aiScene* scene = importer.ReadFile("assets/SM_Gate_A.FBX", load_flag);
    print("{} ms\n", timer.toc());

    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        print("failed to load scene: {}\n", importer.GetErrorString());
        exit(1);
    }
    
    print("name: {}\n", scene->mName.C_Str());

    print("meshes: [\n");
    print("\tnum: {}\n", scene->mNumMeshes);
    for (int i = 0; i < scene->mNumMeshes; i++) {
        auto mesh = scene->mMeshes[i];
        print("\t{}: [ v:{} f:{} ]\n", i, mesh->mNumVertices, mesh->mNumFaces);
    }
    print("]\n");

    auto mesh = scene->mMeshes[0];
    print("{}\n", (ulong)mesh->mVertices);
    print("{}\n", (ulong)mesh->mNormals);
    print("{}\n", (ulong)mesh->mTangents);
    print("{}\n", (ulong)mesh->mBitangents);
    print("{}\n", (ulong)mesh->mColors[0]);
    print("{}\n", (ulong)mesh->mColors[1]);
    print("{}\n", (ulong)mesh->mTextureCoords[0]);
    print("{}\n", (ulong)mesh->mTextureCoords[1]);

    auto aabb = mesh->mAABB;
    print("aabb: pmin: ({}, {}, {}), pmax: ({}, {}, {})\n", aabb.mMin.x, aabb.mMin.y, aabb.mMin.z, aabb.mMax.x, aabb.mMax.y, aabb.mMax.z);

    // print("materials: [\n");
    // print("\tnum: {}\n", scene->mNumMaterials);
    // for (int i = 0; i < scene->mNumMaterials; i++) {
    //     auto material = scene->mMaterials[i];
    //     print("\t{}: [\n", i);
    //     print("\t\tproperties: [\n");
    //     print("\t\t\tnum: {}\n", material->mNumProperties);
    //     for (int j = 0; j < material->mNumProperties; j++) {
    //         auto propertie = material->mProperties[i];
    //         print("\t\t\t{}: [\n", j);
    //         print("\t\t\t\tkey: {}\n", propertie->mKey.C_Str());
    //         print("\t\t\t\ttype: {}\n", [](aiPropertyTypeInfo typeinfo) {
    //             switch (typeinfo) {
    //                 case aiPTI_Float: return "float";
    //                 case aiPTI_Double: return "double";
    //                 case aiPTI_String: return "string";
    //                 case aiPTI_Integer: return "int";
    //                 case aiPTI_Buffer: return "buffer";
    //             }
    //         }(propertie->mType));
    //         print("\t\t\t\tdata length: {}\n", propertie->mDataLength);

    //         print("\t\t\t\t");
    //         switch (propertie->mType) {
    //             case aiPTI_Float: print("data: {}\n", *(float*)propertie->mData); break;
    //             case aiPTI_Double: print("data: {}\n", *(double*)propertie->mData); break;
    //             // case aiPTI_String: print("data: {}\n", (char*)propertie->mData); break;
    //             case aiPTI_String: {
    //                 print("data: ");
    //                 for (int t = 0; t < propertie->mDataLength; t++) {
    //                     print("{}", *(char*)propertie->mData);
    //                 }
    //                 print("\n");
    //             } break;
    //             case aiPTI_Integer: print("data: {}\n", *(long long*)propertie->mData); break;
    //             case aiPTI_Buffer: print("data: null\n"); break;
    //         }
            
    //         print("\t\t\t]\n");
    //     }
    //     print("\t\t]\n");
    //     print("\t]\n");
    // }
    // print("]\n");

    // print("num animation: {}\n", scene->mNumAnimations);
    // print("num texture: {}\n", scene->mNumTextures);
    // print("num light: {}\n", scene->mNumLights);
    // print("num camera: {}\n", scene->mNumCameras);
    // print("num skeleton: {}\n", scene->mNumSkeletons);
    return 0;
}