#pragma once

#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>
#include <imgui/imgui.h>

#include <VulkanMemoryAllocator/vk_mem_alloc.h>
#include <GLFW/glfw3.h>

#include "Vertex.hh"
#include "VmaAllocationOwnership.hh"
#include "Scene.hh"

#include <filesystem>

namespace etna {

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
        void loadTexture(const std::filesystem::path& path, SceneObjectRendererData& obj);

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
            auto rawBuffers = vk::uniqueToRaw(buffers);
            std::array submitInfos { vk::SubmitInfo({}, {}, rawBuffers, {})};
            queue.submit(submitInfos, vk::Fence());
            queue.waitIdle();
        }

        vk::SampleCountFlagBits getMaxUsableSampleCount();

        void createInstance();

        void transitionImageLayout(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

        bool isNested() const;

        vk::UniqueInstance instance;
        vk::DispatchLoaderDynamic dldi;
        vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugUtilsMessenger;

        struct AllocatorDeleter {
            void operator()(VmaAllocator alloc) {
                vmaDestroyAllocator(alloc);
            }
        };
        std::vector<vk::PhysicalDevice> gpus;
        vk::PhysicalDevice gpu;

        vk::UniqueDevice device;
        std::unique_ptr<VmaAllocator_T, AllocatorDeleter> allocator;

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

        vk::UniqueSampler textureSampler;

        std::vector<SceneObjectRendererData> sceneObjects;
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

