#include <luisa/luisa-compute.h>
#include <stb/stb_image.h>
#include <luisa/gui/window.h>
#include <meshoptimizer.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "camera.h"
#include <gpu_rands.h>
#include <nlohmann/json.hpp>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;
using json = nlohmann::json;

struct Mesh {
    vector<float4> positions;
    vector<uint> indices;
    vector<float2> texcoords;

    void load_mesh(const char* file) {
        assimp_load(file);
        compact();
    }
    void assimp_load(const char* file) {
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

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            print("failed to load scene: {}\n", importer.GetErrorString());
            exit(1);
        }
        auto mesh = scene->mMeshes[0];

        if (!mesh->HasPositions() || !mesh->HasFaces() || !mesh->HasTextureCoords(0)) {
            print("invaild mesh\n");
            exit(1);
        }

        print("convert from assimp scene: ");
        timer.tic();

        positions.resize(mesh->mNumVertices);
        indices.resize(mesh->mNumFaces * 3);
        texcoords.resize(mesh->mNumVertices);

        for(int i = 0; i < mesh->mNumVertices; i++){
            positions[i] = {
                mesh->mVertices[i].x,
                mesh->mVertices[i].y,
                mesh->mVertices[i].z,
                0.f
            };
        }
        for(int i = 0; i < mesh->mNumFaces; i++){
            if (mesh->mFaces[i].mNumIndices != 3) {
                print("not a triangle mesh\n");
                exit(1);
            }
            indices[i * 3] = mesh->mFaces[i].mIndices[0];
            indices[i * 3 + 1] = mesh->mFaces[i].mIndices[1];
            indices[i * 3 + 2] = mesh->mFaces[i].mIndices[2];
        }
        for(int i = 0; i < mesh->mNumVertices; i++){
            texcoords[i] = {
                mesh->mTextureCoords[0][i].x,
                mesh->mTextureCoords[0][i].y,
            };
        }
        print("{} ms\n", timer.toc());
        print("num vertex: {}, num triangle: {}\n", positions.size(), indices.size() / 3);
    }
    void compact() {
        Clock timer;
        print("compact: ");
        timer.tic();

        meshopt_Stream stream[] = {
            {&positions[0], sizeof(float)*4, sizeof(float)*4},
            {&texcoords[0], sizeof(float)*2, sizeof(float)*2},
        };
        uint stream_count = sizeof(stream) / sizeof(stream[0]);
        uint index_count = indices.size();
        vector<uint> remap(index_count);
        uint vertex_count = meshopt_generateVertexRemapMulti(&remap[0], &indices[0], index_count, index_count, stream, stream_count);

        vector<float4> new_positions(vertex_count);
        vector<float2> new_texcoords(vertex_count);

        meshopt_remapIndexBuffer(&indices[0], &indices[0], index_count, &remap[0]);
        meshopt_remapVertexBuffer(&new_positions[0], &positions[0], positions.size(), sizeof(float4), &remap[0]);
        meshopt_remapVertexBuffer(&new_texcoords[0], &texcoords[0], texcoords.size(), sizeof(float2), &remap[0]);

        positions.swap(new_positions);
        texcoords.swap(new_texcoords);

        print("{} ms\n", timer.toc());
        print("num vertex: {}, num triangle: {}\n", positions.size(), indices.size() / 3);
    }
};

struct MeshRenderData {
    Buffer<float3> positions;
    Buffer<uint> indices;
    float4 tt;
    Buffer<float2> texcoords;
};

LUISA_BINDING_GROUP(MeshRenderData, positions, indices, tt, texcoords) {};

struct Img {
    int width = 0;
    int height = 0;
    int channel = 0;
    unsigned char* pixels;
    
    void load(const char* file) {
        Clock timer;
        print("load texture: ");
        timer.tic();
        pixels = stbi_load(file, &width, &height, &channel, 4);
        print("{} ms\n", timer.toc());
    }
};

struct v2p {
    float4 pos;
    float2 uv;
    float3 color;
};
LUISA_STRUCT(v2p, pos, uv, color) {};

RasterStageKernel vert = [](Var<AppData> var, $buffer<float3> positions, $buffer<uint> indices, $buffer<float2> texcoords, $float4x4 vp_mat) {
// RasterStageKernel vert = [](Var<AppData> var, Var<MeshRenderData> mesh, $float4x4 vp_mat) {
    Var<v2p> out;
    $uint vid = vertex_id();

    // $uint idx = mesh.indices.read(vid);
    // $float3 pos = mesh.positions.read(idx);
    // $float2 uv = mesh.texcoords.read(idx);
    $uint idx = indices.read(vid);
    $float3 pos = positions.read(idx);
    $float2 uv = texcoords.read(idx);

    out.pos = vp_mat * make_float4(pos, 1.f);
    // out.pos = make_float4(pos, 1.f);
    out.uv = uv;

    $uint s = tea(vid / 3, 233).x;
    out.color = make_float3(
		((s >> 0) & 255) / 255.0f,
		((s >> 8) & 255) / 255.0f,
		((s >> 16) & 255) / 255.0f
	);
    
    return out;
};
RasterStageKernel pixel = [](Var<v2p> in, $image<float> texture, $uint2 texture_dim) {
    return texture.read($uint2(in.uv.x * texture_dim.x, (1 - in.uv.y) * texture_dim.y));
    // return make_float4(in.uv, 0.f, 1.f);
    // return make_float4(in.color, 1.f);
};
RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};

Kernel2D clear_kernel = []($image<float> image, Var<MeshRenderData> mesh) {
    image.write(dispatch_id().xy(), make_float4(0.1f));
};

Kernel1D fetch_inference_output_kernel = []($buffer<half4> inference_output, $image<float> inference_image) {
    $uint tid = $dispatch_x;
    $half4 out[4];
    for (int i = 0; i < 3; i++) {
        out[i] = inference_output.read(tid + i*1024/4);
    }
    $float4 c[4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) c[i][j] = out[j][i];
    }
    for (int i = 0; i < 4; i++) {
        $uint idx = tid*4 + i;
        $uint x = idx % 1920;
        $uint y = idx / 1920;
        inference_image.write($uint2{x, y}, c[i]);
    }
};

struct TypeImpl final : public Type {
    uint64_t hash{};
    Tag tag{};
    uint size{};
    uint16_t alignment{};
    uint16_t dimension{};
    uint index{};
    luisa::string description;
    luisa::vector<const Type *> members;
};

template<typename T>
json to_json(const T&) {
    print("unknown type call to_json\n");
    exit(1);
}

template<>
json to_json<Type>(const Type& type) {
    const TypeImpl* type_info = (TypeImpl*)&type;
    return {
        {"tag", luisa::to_string(type_info->tag)},
        {"size", type_info->size},
        {"alignment", type_info->alignment},
        {"dimension", type_info->dimension},
        {"description", type_info->description}
    };
}

template<>
json to_json<Variable>(const Variable& v) {
    return {
        {"type", to_json(*v.type())},
        {"uid", v.uid()},
        {"tag", luisa::to_string(v.tag())}
    };
}

template<>
json to_json<Function::Binding>(const Function::Binding& binding) {
    auto str = luisa::visit(
        [&]<typename T>(T const &v) {
            if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                return "buffer_binding";
            } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                return "texture_binding";
            } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                return "bindless_array_binding";
            } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                return "accel_binding";
            } else {
                return "null binding";
            }
        },
        binding
    );
    return str;
}
// using basic_types = std::tuple<
//     bool, float, int, uint, short, ushort, slong, ulong, half, double,
//     bool2, float2, int2, uint2, short2, ushort2, slong2, ulong2, half2, double2,
//     bool3, float3, int3, uint3, short3, ushort3, slong3, ulong3, half3, double3,
//     bool4, float4, int4, uint4, short4, ushort4, slong4, ulong4, half4, double4,
//     float2x2, float3x3, float4x4>;
template<>
json to_json<LiteralExpr::Value>(const LiteralExpr::Value& value) {
    auto str = luisa::visit(
        [&]<typename T>(T const &v) {
            if constexpr (std::is_same_v<T, bool>) {
                string s = "bool: ";
                s.append(to_string(v));
                return s;
            } else if constexpr (std::is_same_v<T, float>) {
                string s = "float: ";
                s.append(to_string(v));
                return s;
            } else if constexpr (std::is_same_v<T, int>) {
                string s = "int: ";
                s.append(to_string(v));
                return s;
            } else if constexpr (std::is_same_v<T, uint>) {
                string s = "uint: ";
                s.append(to_string(v));
                return s;
            } else {
                return string("unknwon type");
            }
        },
        value
    );
    return str;
}

// struct Arguments {
//     Image<float> image;
//     uint2 resolution;
// };

// struct ArgumentsView {
//     ImageView<float> image;
//     uint2 resolution;
// };

// struct NestedArguments {
//     ArgumentsView args;
//     Image<float> image;
// };

// LUISA_BINDING_GROUP(Arguments, image, resolution) {
//     [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
//         image->write(coord, color);
//     }
// };

// LUISA_BINDING_GROUP(ArgumentsView, image, resolution) {
//     [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
//         image->write(coord, color);
//     }
// };

// LUISA_BINDING_GROUP(NestedArguments, args, image) {
//     void blit(const UInt2 &coord) noexcept {
//         auto color = args.image.read(coord).xyz();
//         image->write(coord, make_float4(1.f - color, 1.f));
//     }
// };

// Callable color = [](UInt2 coord, Var<Arguments> args) noexcept {
//     auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
//     return make_float4(uv, .5f, 1.f);
// };

// Callable color_with_view = [](UInt2 coord, Var<ArgumentsView> args) noexcept {
//     auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
//     return make_float4(uv, .5f, 1.f);
// };

// Kernel2D kernel2 = [](Var<Arguments> args) noexcept {
//     auto coord = dispatch_id().xy();
//     args->write(coord, color(coord, args));
// };

// Kernel2D kernel_with_view = [](Var<ArgumentsView> args) noexcept {
//     auto coord = dispatch_id().xy();
//     args->write(coord, color_with_view(coord, args));
// };

// Kernel2D kernel_with_nested = [](Var<NestedArguments> args) noexcept {
//     auto coord = dispatch_id().xy();
//     args->blit(coord);
// };

class PrintASTVisitor final : public StmtVisitor, public ExprVisitor {
public:
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
    void visit(const ConstantExpr *expr) override;
    void visit(const TypeIDExpr *expr) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const StringIDExpr *expr) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const CpuCustomOpExpr *) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const GpuCustomOpExpr *) override { LUISA_NOT_IMPLEMENTED(); }

    void visit(const BreakStmt *) override;
    void visit(const ContinueStmt *) override;
    void visit(const ReturnStmt *) override;
    void visit(const ScopeStmt *) override;
    void visit(const IfStmt *) override;
    void visit(const LoopStmt *) override;
    void visit(const ExprStmt *) override;
    void visit(const SwitchStmt *) override;
    void visit(const SwitchCaseStmt *) override;
    void visit(const SwitchDefaultStmt *) override;
    void visit(const AssignStmt *) override;
    void visit(const ForStmt *) override;
    void visit(const CommentStmt *) override;
    void visit(const RayQueryStmt *) override;
    void visit(const AutoDiffStmt *stmt) override;

    json build_json;
};

template<>
json to_json<Expression>(const Expression& expr) {
    PrintASTVisitor visitor;
    expr.accept(visitor);

    json expr_json;
    expr_json["Expression"] = visitor.build_json;

    return expr_json;
}

template<>
json to_json<Statement>(const Statement& stmt) {
    PrintASTVisitor visitor;
    stmt.accept(visitor);

    json stmt_json;
    stmt_json["Statement"] = visitor.build_json;

    return stmt_json;
}

void PrintASTVisitor::visit(const UnaryExpr *expr) {
    build_json["UnaryExpr"] = {
        {"op", luisa::to_string(expr->op())},
        {"operand", to_json(expr->operand())}
    };
}
void PrintASTVisitor::visit(const BinaryExpr *expr) {
    build_json["BinaryExpr"] = {
        {"op", luisa::to_string(expr->op())},
        {"lhs", to_json(expr->lhs())},
        {"rhs", to_json(expr->rhs())}
    };
}
void PrintASTVisitor::visit(const MemberExpr *expr) {
    json tmp;
    tmp["is_swizzle"] = expr->is_swizzle();
    tmp["member_index"] = expr->member_index();

    build_json["MemberExpr"] = tmp;
}
void PrintASTVisitor::visit(const AccessExpr *expr) {
    build_json = "AccessExpr";
}
void PrintASTVisitor::visit(const LiteralExpr *expr) {
    build_json["LiteralExpr"] = {
        {"value", to_json(expr->value())}
    };
}
void PrintASTVisitor::visit(const RefExpr *expr) {
    build_json["RefExpr"] = {
        {"variable", to_json(expr->variable())}
    };
}
void PrintASTVisitor::visit(const CallExpr *expr) {
    json args;
    for (auto arg: expr->arguments()) {
        args.push_back(to_json(*arg));
    }
    build_json["CallExpr"] = {
        {"op", luisa::to_string(expr->op())},
        {"arguments", args}
    };
}
void PrintASTVisitor::visit(const CastExpr *expr) {
    build_json = "CastExpr";
}
void PrintASTVisitor::visit(const ConstantExpr *expr) {
    build_json = "ConstantExpr";
}

void PrintASTVisitor::visit(const BreakStmt *) {
    build_json = "BreakStmt";
}
void PrintASTVisitor::visit(const ContinueStmt *) {
    build_json = "ContinueStmt";
}
void PrintASTVisitor::visit(const ReturnStmt *) {
    build_json = "ReturnStmt";
}
void PrintASTVisitor::visit(const ScopeStmt *stmt) {
    json tmp;
    for (auto st: stmt->statements()) {
        tmp.push_back(to_json(*st));
    }
    build_json["ScopeStmt"] = tmp;
}
void PrintASTVisitor::visit(const IfStmt *) {
    build_json = "IfStmt";
}
void PrintASTVisitor::visit(const LoopStmt *) {
    build_json = "LoopStmt";
}
void PrintASTVisitor::visit(const ExprStmt *) {
    build_json = "ExprStmt";
}
void PrintASTVisitor::visit(const SwitchStmt *) {
    build_json = "SwitchStmt";
}
void PrintASTVisitor::visit(const SwitchCaseStmt *) {
    build_json = "SwitchCaseStmt";
}
void PrintASTVisitor::visit(const SwitchDefaultStmt *) {
    build_json = "SwitchDefaultStmt";
}
void PrintASTVisitor::visit(const AssignStmt *stmt) {
    build_json["AssignStmt"] = {
        {"lhs", to_json(*stmt->lhs())},
        {"rhs", to_json(*stmt->rhs())}
    };
}
void PrintASTVisitor::visit(const ForStmt *) {
    build_json = "ForStmt";
}
void PrintASTVisitor::visit(const CommentStmt *) {
    build_json = "CommentStmt";
}
void PrintASTVisitor::visit(const RayQueryStmt *) {
    build_json = "RayQueryStmt";
}
void PrintASTVisitor::visit(const AutoDiffStmt *stmt) {
    build_json = "AutoDiffStmt";
}

template<>
json to_json<luisa::compute::detail::FunctionBuilder>(const luisa::compute::detail::FunctionBuilder& builder) {
    json argments;
    json bound_arguments;
    json unbound_arguments;
    

    for (auto v: builder.arguments()) {
        argments.push_back(to_json(v));
    }
    for (auto b: builder.bound_arguments()) {
        bound_arguments.push_back(to_json(b));
    }
    for (auto ub: builder.unbound_arguments()) {
        unbound_arguments.push_back(to_json(ub));
    }
    // for (auto stmt: builder.)
    return {
        {"argments", argments},
        {"bound_arguments", bound_arguments},
        {"unbound_arguments", unbound_arguments},
        {"body", to_json(*((Statement*)builder.body()))}
    };
}


struct alignas(16) Node {
    float val;
    uint ls;
    uint rs;
};

LUISA_STRUCT(Node, val, ls, rs) {};

Kernel1D kernel3 = []($buffer<Node> buffer) {
    buffer.atomic(1).val.fetch_add(1.0f);
};

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    auto shader = device.compile(kernel3);

    // auto builder = kernel3.function();
    // print("{}\n", to_json(*builder).dump(4));

    // auto vs_builder = vert.function_builder();
    // auto ps_builder = pixel.function_builder();
    // auto cs_builder = clear_kernel.function();

    // auto cs2_builder = fetch_inference_output_kernel.function();
    // auto cs3_builder = kernel_with_nested.function();

    // print("vert: {}\n\n", to_json(*vs_builder).dump(4));
    // print("pixel: {}\n\n", to_json(*ps_builder).dump(4));
    // print("compute: {}\n\n", to_json(*cs_builder).dump(4));
    // print("compute2: {}\n\n", to_json(*cs2_builder).dump(4));
    // print("compute3: {}\n\n", to_json(*cs3_builder).dump(4));

    // for (auto v: vs_builder->arguments()) {
    //     print("{}\n", to_json(v).dump(4));
    //     // print("{} {}\n", v.uid(), luisa::to_string(v.tag()));
    // }
    // print("\n");
    // for (auto v: ps_builder->arguments()) {
    //     print("{} {}\n", v.uid(), luisa::to_string(v.tag()));
    // }
    // print("\n");
    // for (auto v: cs_builder->arguments()) {
    //     print("{} {}\n", v.uid(), luisa::to_string(v.tag()));
    // }

    // print("\n\n");
    // for (auto b: cs_builder->bound_arguments()) {
    //     print("{}\n", b.index());
    //     // print("{} {}\n", v.uid(), luisa::to_string(v.tag()));
    // }

    return 0;
}