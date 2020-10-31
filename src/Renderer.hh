#pragma once

#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>
#include <imgui/imgui.h>

#include <VulkanMemoryAllocator/vk_mem_alloc.h>
#include <GLFW/glfw3.h>

#include "Vertex.hh"
#include <filesystem>

namespace etna {

    struct UniformBuffer {
        glm::mat4 mvp;
    };

    class UniqueVmaAllocation {
    public:
        UniqueVmaAllocation() = default;
        UniqueVmaAllocation(VmaAllocator allocator, VmaAllocation allocation)
            : _allocator(allocator),
              _allocation(allocation)
        {}
        UniqueVmaAllocation(const UniqueVmaAllocation&) = delete;
        UniqueVmaAllocation(UniqueVmaAllocation&& mv)
            : _allocator(mv._allocator) {
            std::swap(_allocation, mv._allocation);
        }

        ~UniqueVmaAllocation() {
            clear();
        }

        operator const VmaAllocation& () {
            return _allocation;
        }

        void reset(VmaAllocator allocator, VmaAllocation allocation) {
            clear();
            _allocator = allocator;
            _allocation = allocation;
        }
    private:
        void clear() {
            if (_allocator && _allocation) {
                vmaFreeMemory(_allocator, _allocation);
                _allocator = nullptr;
                _allocation = nullptr;
            }
        }

        VmaAllocator _allocator = nullptr;
        VmaAllocation _allocation = nullptr;
    };

    struct SceneObject {
        std::string name;
        glm::vec3 position = {0,0,0};
        glm::vec3 rotation = {0,0,0};
        glm::vec3 scale =    {1,1,1};

        bool visible = false;
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

        glm::vec4 pos = glm::vec4(0, 0, -15, 1);
        glm::vec3 rotation = {};
    };


    struct ImageData {
        vk::UniqueImage image = {};
        vk::UniqueImageView imageView = {};

        vk::Format format = {};
        UniqueVmaAllocation vmaAlloc = {};
    };

    class Renderer {
    public:
        Renderer() = default;
        ~Renderer();

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
        void initScene();
        void initSampler();

        void loadObj(const std::filesystem::path& path);
        void loadTexture(const std::filesystem::path& path, SceneObject& obj);

        void spawnCube();

        void initPipeline();

        void initGui();

        void buildGui();

        void beginRecordCommandBuffer(int);
        void endRecordCommandBuffer(int);
        void submitCommandBuffer(int);

        void draw();

        void recreateSwapChain();
    protected:
        template<typename funcT>
        void immediateCommandBuffer(funcT&& func) {
            vk::CommandBufferAllocateInfo cmdBuffAlocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
            auto buffers = device->allocateCommandBuffersUnique(cmdBuffAlocInfo);
            vk::CommandBufferBeginInfo buffBeginInfo;
            buffBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            buffers.front()->begin(buffBeginInfo);
            func(*buffers.front());
            buffers.front()->end();

            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBuffers(std::array{*buffers.front()});
            queue.submit(std::array{submitInfo}, vk::Fence());
            queue.waitIdle();
        }

        vk::SampleCountFlagBits getMaxUsableSampleCount();

        void createInstance();

        void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);

        bool isNested() const;

        vk::UniqueInstance instance;
        vk::DispatchLoaderDynamic dldi;
        vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugUtilsMessenger;

        struct allocatorDeleter {
            void operator()(VmaAllocator alloc) {
                vmaDestroyAllocator(alloc);
            }
        };
        std::vector<vk::PhysicalDevice> gpus;
        vk::PhysicalDevice gpu;

        vk::UniqueDevice device;
        std::unique_ptr<VmaAllocator_T, allocatorDeleter> allocator;

        vk::DispatchLoaderDynamic dldid;

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

        ImageData depthBuffer = {.format=vk::Format::eD16Unorm};
        ImageData resolveBuffer = {};

        struct {
            glm::mat4x4 P;
            glm::mat4x4 V;
        } world;

        vk::UniqueBuffer sharedUniformBuffer;
        UniqueVmaAllocation sharedUniformAllocation;
        vk::DescriptorSet sharedDescriptorSet;
        std::byte* sharedUboMappedMemory = nullptr;

        vk::UniqueDescriptorSetLayout descSetLayout;

        vk::UniquePipelineLayout pipelineLayout;
        vk::UniqueDescriptorPool descriptorPool;

        vk::UniqueShaderModule vertexShader;
        vk::UniqueShaderModule fragmentShader;
        vk::UniquePipeline coloredVertexPipeline;
        vk::UniquePipeline texturedVertexPipeline;

        vk::UniqueRenderPass renderPass;

        vk::VertexInputBindingDescription viBindings;
        std::vector<vk::VertexInputAttributeDescription> viAttribs;

        vk::UniqueSampler textureSampler;

        std::vector<SceneObject> sceneObjects;
        Camera camera;

        GLFWwindow *_window = nullptr;

        int selectedSceneNode = -1;
        std::vector<vk::UniqueFramebuffer> framebuffers;
        std::vector<vk::UniqueFramebuffer> guiFramebuffers;
        vk::UniqueRenderPass guiRenderPass;
        bool done_looping = false;
        VmaStats vmaStats = {};
    };

}

