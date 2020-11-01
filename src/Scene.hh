#pragma once

#include "VmaAllocationOwnership.hh"

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

namespace etna {

struct UniformBuffer {
    glm::mat4 mvp;
};

struct SceneObject {
    std::string name;
    glm::vec3 position = {0,0,0};
    glm::vec3 rotation = {0,0,0};
    glm::vec3 scale =    {1,1,1};

    bool visible = false;
};

class SceneObjectRendererData : public SceneObject {
    friend class Renderer;
private:
    // It is wrong to establish ownership of vertex buffers and textures at object level
    // So this is all quite temporary but it does provide some convenience while
    // other essential structural aspects are defined
    vk::UniqueBuffer vertexBuffer = {};
    UniqueVmaAllocation bufferAllocation = {};

    vk::UniqueImage textureImage = {};
    vk::UniqueImageView textureView {};
    UniqueVmaAllocation textureAllocation = {};
    vk::DescriptorSet samplerDescriptor = {};

    int bufferStart = 0;
    int numVerts = 0;

    uint32_t uniformBufferOffset = 0;
};

struct Camera {
    float fov = 110.f;
    float zNear = 1.f;

    glm::vec4 pos = {0, 0, -15, 1};
    glm::vec3 rotation = {};
};


} //::etna
