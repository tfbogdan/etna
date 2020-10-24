#pragma once

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <libinput.h>
#include <libudev.h>


namespace wkt {

    struct Camera {
        glm::vec3 position;
        glm::vec3 direction;
    };

    struct Model {
        glm::vec3 position;
        glm::vec3 rotation;
        glm::vec3 scale;
    };

    class Vulkan {
    public:
        ~Vulkan();
        void initialize();

        void initInput();

        void initInstance();
        void createWindow();

        void initDevice();
        void initSurface();
        void initCommandBuffer();
        void initSwapchain();
        void initDepthBuffer();
        void initCubeUniformBuffer();
        void initGridUniformBuffer();
        void initPipelineLayout();
        void initDescriptorPool();
        void initCubeDescriptorSet();
        void intiGridDescriptorSet();
        void initRenderPass();
        void initShaders();
        void initFramebuffers();
        void initCubeVertexBuffers();
        void initGridVertexBuffers();
        void initCubePipeline();
        void initGridPipeline();

        void beginRecordCommandBuffer();
        void beginRecordCommandBuffer(int);
        void endRecordCommandBuffer();
        void submitCommandBuffer();
        void draw();

        void loop();
        void loop_input();

        void disableTTY();
        void restoreTTY();
    protected:
        static void windowResized(GLFWwindow* window, int w, int h);

        vk::SampleCountFlagBits getMaxUsableSampleCount();

        void createInstance();

        bool isNested() const;

        void cleanupSwapchainAndDependees();
        void recreateSwapChain();

        vk::UniqueInstance instance;
        vk::DispatchLoaderDynamic dldi;
        vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugUtilsMessenger;

        vk::UniqueDevice device;
        vk::DispatchLoaderDynamic dldid;
        std::vector<vk::PhysicalDevice> gpus;
        vk::PhysicalDevice gpu;

        void updateUniformBuffer();
        bool init();

        vk::Queue queue;
        vk::UniqueSurfaceKHR wndSurface;

        vk::UniqueCommandPool commandPool;
        vk::UniqueCommandBuffer commandBuffer;


        // VkCommandPool cmd_pool = VK_NULL_HANDLE;
        // VkCommandBuffer cmd_buff = VK_NULL_HANDLE;
        vk::UniqueSwapchainKHR swapchain;
        uint32_t queueFamily = 0;


        struct {
            vk::SurfaceCapabilitiesKHR capabilities;
            std::vector<vk::PresentModeKHR> presentModes;
            std::vector<vk::SurfaceFormatKHR> surfaceFormats;
            bool has_VK_FORMAT_B8G8R8A8_UNORM = false;
            vk::PresentModeKHR presentMode;
        } surfaceCharacteristics;

        // vk::SurfaceCapabilitiesKHR surface_characteristics;

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

        vk::PhysicalDeviceMemoryProperties memoryProperties;

        bool memory_type_from_properties(uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex);

        struct {
            glm::mat4x4 P;
            glm::mat4x4 V;
            glm::mat4x4 M;
            glm::mat4x4 MVP;
            glm::vec4 solid_color;
        } world;

        struct UniformInfo {
            vk::UniqueDeviceMemory memory;
            vk::DescriptorBufferInfo bufferInfo;
            vk::UniqueBuffer buffer;
        };
        UniformInfo cubeUniform;
        UniformInfo gridUniform;

        vk::UniqueDescriptorSetLayout layoutDescriptor;
        vk::UniquePipelineLayout pipelineLayout;
        vk::UniqueDescriptorPool cubeDescriptorPool;
        vk::UniqueDescriptorPool gridDescriptorPool;
        std::vector<vk::UniqueDescriptorSet> cubeDscriptorSets;
        std::vector<vk::UniqueDescriptorSet> gridDescriptorSets;
        vk::UniqueShaderModule vertexShader;
        vk::UniqueShaderModule fragmentShader;
        vk::UniquePipeline cubePipeline;

        vk::UniqueBuffer wlBuffer;
        vk::UniqueDeviceMemory wlMemory;
        vk::UniquePipeline wlPipeline;

        vk::UniqueRenderPass renderPass;

        struct {
            vk::UniqueBuffer vertexBuffer;
            vk::UniqueDeviceMemory vertexMemory;
            vk::VertexInputBindingDescription viBindings;
            std::vector<vk::VertexInputAttributeDescription> viAttribs;
            float xRot = 0., yRot = 0., zRot = 0.;
        } mesh;

        std::vector<vk::UniqueFramebuffer> framebuffers;

        // in case we're running in a graphical session
        GLFWwindow *window = nullptr;

        struct libinput *li = nullptr;
        struct libinput_event *event = nullptr;
        struct udev *udev = nullptr;

        int tty_mode = -1;

        bool done_looping = false;
    };

}

