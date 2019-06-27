#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (std140, binding = 0) uniform bufferVals {
    mat4 mvp;
    vec4 gcol;
} myBufferVals;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;
layout (location = 0) out vec4 outColor;


vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0), 
    vec3(0.0, 0.0, 1.0)
);


void main() {
   outColor = inColor;
   gl_Position = myBufferVals.mvp * pos;
   // outColor = vec4(1.0, 1.0, 1.0, 1.0);
   
   // outColor.r = inColor.a;
   // outColor = vec4(colors[gl_VertexIndex % 3], 1.0);
   // outColor.r = inColor.a;
   // outColor.g = inColor.g;
   // gl_Position = vec4(positions[gl_VertexIndex%3], 1.0, 1.0);
}
