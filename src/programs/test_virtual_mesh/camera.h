#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>

struct Camera {
    luisa::float3 position;
    float pitch;
    float yaw;

    float move_speed = 3;
    float rotate_speed = 0.07;

    luisa::float3 direction(){
        float rp = luisa::radians(pitch);
        float ry = luisa::radians(yaw);
        
        return luisa::float3{
            cos(rp) * cos(ry),
            sin(rp),
            cos(rp) * sin(ry)
        };
    }

    void move_front(float tick_time){
        position += direction() * tick_time * move_speed;
    }
    void move_right(float tick_time){
        luisa::float3 up = { 0, 1, 0 };
        luisa::float3 front = direction();
        luisa::float3 right = normalize(luisa::cross(front, up));
        position += right * tick_time * move_speed;
    }
    void rotate_view(float yaw_offset, float pitch_offset){
        yaw += yaw_offset * rotate_speed;
        pitch += -pitch_offset * rotate_speed;
        if (pitch > 89) pitch = 89;
        if (pitch < -89) pitch = -89;
    }

    luisa::float4x4 view_mat(){
        luisa::float3 up = { 0, 1, 0 };
        luisa::float3 front = direction();
        luisa::float3 right = normalize(cross(front, up));
        up = luisa::cross(right, front);
        
        luisa::float4x4 rotate;
        rotate[0] = luisa::make_float4(right.x, up.x, -front.x, 0);
        rotate[1] = luisa::make_float4(right.y, up.y, -front.y, 0);
        rotate[2] = luisa::make_float4(right.z, up.z, -front.z, 0);
        rotate[3] = luisa::make_float4(0, 0, 0, 1);

        luisa::float4x4 move;
        move[0] = luisa::make_float4(1, 0, 0, 0);
        move[1] = luisa::make_float4(0, 1, 0, 0);
        move[2] = luisa::make_float4(0, 0, 1, 0);
        move[3] = luisa::make_float4(-position, 1);

        return rotate * move;
    }
    luisa::float4x4 projection_mat(float fov,float aspect){
        // assuming n=-0.1, f=-inf
        float t = 1 / tan(fov * 0.5);
        luisa::float4x4 mat;
        mat[0] = luisa::make_float4(t / aspect, 0, 0, 0);
        mat[1] = luisa::make_float4(0, t, 0, 0);
        mat[2] = luisa::make_float4(0, 0, 0, -1);
        mat[3] = luisa::make_float4(0, 0, 0.1, 0);
        // mat.set_col(0,vec4(t/aspect,0,0,0));
        // mat.set_col(1,vec4(0,-t,0,0));
        // mat.set_col(2,vec4(0,0,0,-1));
        // mat.set_col(3,vec4(0,0,0.1,0));
        return mat;
    }
};