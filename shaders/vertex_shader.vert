#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (std140, binding = 0) uniform bufferValsDynamic {
    mat4 mvp;
} transform;


layout (location = 0) in vec3 displacement;
layout (location = 1) in vec2 inUv;

layout (location = 0) out vec2 outUv;

void main() {
   gl_Position = transform.mvp * vec4(displacement, 1.);
   outUv = inUv;
}
