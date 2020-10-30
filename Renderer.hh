#pragma once

#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>
#include <imgui/imgui.h>

#include <GLFW/glfw3.h>

#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace etna {

    struct UniformBuffer {
        glm::mat4 mvp;
    };

    struct ColoredVertex {
        glm::vec4 position;
        glm::vec4 color;
    };

    struct SceneObject {
        std::string name;
        glm::vec3 position = {0,0,0};
        glm::vec3 rotation = {0,0,0};
        glm::vec3 scale = {1, 1, 1};

        bool visible = false;
        vk::Buffer vertexBuffer = {};
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;

        int bufferStart = 0;
        int numVerts = 0;

        uint32_t uniformBufferOffset = 0;
    };

    class Renderer {
    public:
        ~Renderer();
        Renderer() = default;

        void initialize(GLFWwindow* window);

        void initDevice();
        void initSurface(GLFWwindow* window);
        void initCommandBuffer();
        void initSwapchain();
        void initDepthBuffer();
        void initSharedUniformBuffer();

        void initPipelineLayout();
        void initDescriptorPool();
        void initSharedDescriptorSet();

        void initRenderPass();
        void initGuiRenderPass();
        void initShaders();
        void initFramebuffers();
        void initGuiFramebuffers();
        void initCubeVertexBuffers();
        void initPipeline();

        void initGui();

        void buildGui();

        void beginRecordCommandBuffer(int);
        void endRecordCommandBuffer(int);
        void submitCommandBuffer(int);
        void draw();

        void recreateSwapChain();
    protected:
        vk::SampleCountFlagBits getMaxUsableSampleCount();

        void createInstance();

        bool isNested() const;

        vk::UniqueInstance instance;
        vk::DispatchLoaderDynamic dldi;
        vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugUtilsMessenger;

        vk::UniqueDevice device;
        vk::DispatchLoaderDynamic dldid;
        std::vector<vk::PhysicalDevice> gpus;
        vk::PhysicalDevice gpu;

        void updateUniformBuffer();
        void updateUniformBuffers();

        vk::Queue queue;
        vk::UniqueSurfaceKHR wndSurface;
        vk::UniqueCommandPool commandPool;

        std::vector<vk::UniqueCommandBuffer> commandBuffers;

        vk::UniqueSwapchainKHR swapchain;
        uint32_t queueFamily = 0;

        vk::SurfaceCapabilitiesKHR surfaceCapabilities;
        vk::PresentModeKHR presentMode;

        std::vector<vk::Image> swapchainImages;
        std::vector<vk::UniqueImageView> swapchainImageViews;

        struct {
            vk::UniqueImage image;
            vk::UniqueImageView imageView;

            vk::Format format = vk::Format::eD16Unorm;
            vk::UniqueDeviceMemory memory;
        } depthBuffer;

        struct {
            vk::UniqueImage image;
            vk::UniqueImageView imageView;
            vk::UniqueDeviceMemory memory;
        } resolveBuffer;

        std::vector<SceneObject> sceneObjects;

        vk::PhysicalDeviceMemoryProperties memoryProperties;

        bool memory_type_from_properties(uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex);

        struct {
            glm::mat4x4 P;
            glm::mat4x4 V;
        } world;

        vk::UniqueBuffer sharedUniformBuffer;
        VmaAllocation sharedUniformAllocation = nullptr;
        vk::DescriptorBufferInfo sharedBufferInfo;
        vk::DescriptorSet sharedDescriptorSet;
        vk::UniqueDeviceMemory sharedUBOMemory;

        vk::UniqueDescriptorSetLayout descSetLayout;
        vk::UniquePipelineLayout pipelineLayout;
        vk::UniqueDescriptorPool descriptorPool;

        vk::UniqueShaderModule vertexShader;
        vk::UniqueShaderModule fragmentShader;
        vk::UniquePipeline cubePipeline;

        vk::UniqueRenderPass renderPass;
        VmaAllocator allocator = nullptr;

        struct {
            vk::UniqueBuffer vertexBuffer;
            vk::UniqueDeviceMemory vertexMemory;
            vk::VertexInputBindingDescription viBindings;
            std::vector<vk::VertexInputAttributeDescription> viAttribs;
            glm::vec3 rotation = {};
            glm::vec3 pos = {};
        } mesh;

        struct {
            float fov = 110.f;
            float zNear = 1.f;
            float zFar = 1000.f;

            glm::vec4 pos = glm::vec4(0, 0, -15, 1);
            glm::vec3 rotation = {};
        } camera;
        GLFWwindow *_window = nullptr;

        std::vector<vk::UniqueFramebuffer> framebuffers;
        std::vector<vk::UniqueFramebuffer> guiFramebuffers;
        vk::UniqueRenderPass guiRenderPass;
        bool done_looping = false;
    };

}

