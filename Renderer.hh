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

//    struct TexturedVertex {
//        glm::vec4 position;
//    };

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
            if (_allocator and _allocation) {
                vmaFreeMemory(_allocator, _allocation);
                _allocator = nullptr;
                _allocation = nullptr;
            }
        }

        VmaAllocator _allocator = nullptr;
        VmaAllocation _allocation = nullptr;
    };

    class Renderer {
    public:
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

        struct {
            vk::UniqueImage image;
            vk::UniqueImageView imageView;

            vk::Format format = vk::Format::eD16Unorm;
            UniqueVmaAllocation vmaAlloc;
        } depthBuffer;

        struct {
            vk::UniqueImage image;
            vk::UniqueImageView imageView;
            UniqueVmaAllocation vmaAlloc;
        } resolveBuffer;

        std::vector<SceneObject> sceneObjects;

        vk::PhysicalDeviceMemoryProperties memoryProperties;

        struct {
            glm::mat4x4 P;
            glm::mat4x4 V;
        } world;

        vk::UniqueBuffer sharedUniformBuffer;
        UniqueVmaAllocation sharedUniformAllocation;
        vk::DescriptorBufferInfo sharedBufferInfo;
        vk::DescriptorSet sharedDescriptorSet;
        std::byte* sharedUboMappedMemory = nullptr;


        vk::UniqueDescriptorSetLayout descSetLayout;
        vk::UniquePipelineLayout pipelineLayout;
        vk::UniqueDescriptorPool descriptorPool;

        vk::UniqueShaderModule vertexShader;
        vk::UniqueShaderModule fragmentShader;
        vk::UniquePipeline cubePipeline;

        vk::UniqueRenderPass renderPass;

        struct {
            vk::UniqueBuffer vertexBuffer;
            UniqueVmaAllocation vmaAlloc;
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

        int selectedSceneNode = -1;
        std::vector<vk::UniqueFramebuffer> framebuffers;
        std::vector<vk::UniqueFramebuffer> guiFramebuffers;
        vk::UniqueRenderPass guiRenderPass;
        bool done_looping = false;
        VmaStats vmaStats = {};
    };

}

