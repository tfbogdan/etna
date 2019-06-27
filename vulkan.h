#pragma once

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>


#include <libinput.h>
#include <libudev.h>


namespace wkt {

    class Vulkan {
    public:
        ~Vulkan();
        void initialize();

        void initInstance();
        void initDevice();
        void initSurface();
        void initCommandBuffer();
        void initSwapchain();
        void initDepthBuffer();
        void initUniformBuffer();
        void initPipelineLayout();
        void initDescriptorSet();
        void initRenderPass();
        void initShaders();
        void initFramebuffers();
        void initVertexBuffers();
        void initPipeline();

        void beginRecordCommandBuffer();
        void beginRecordCommandBuffer(int);
        void endRecordCommandBuffer();
        void submitCommandBuffer();
        void draw();

        void setXRotation(float angle);
        void setYRotation(float angle);
        void setZRotation(float angle);

        void loop();
        void disableTTY();
        void restoreTTY();
    protected:

        void createInstance();
        void createDevice();

        bool isNested() const;

        /**
         * @brief initSurface
         */
        // virtual vk::SurfaceKHR createSurface() = 0;

        vk::Instance instance;
        VkDebugUtilsMessengerEXT debugUtilsMessenger;

        vk::Device device;
        std::vector<vk::PhysicalDevice> gpus;
        vk::PhysicalDevice gpu;

        vk::SurfaceKHR surface;

        void updateUniformBuffer();
        bool init();

        vk::Queue queue;
        vk::SurfaceKHR wndSurface;

        vk::CommandPool commandPool;
        vk::CommandBuffer commandBuffer;

        // VkCommandPool cmd_pool = VK_NULL_HANDLE;
        // VkCommandBuffer cmd_buff = VK_NULL_HANDLE;
        vk::SwapchainKHR swapchain;
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
        std::vector<vk::ImageView> swapchainImageViews;

        struct {
            vk::Image image;
            vk::ImageView imageView;

            vk::Format format = vk::Format::eD16Unorm;
            vk::DeviceMemory memory;
        } depthBuffer;

        vk::PhysicalDeviceMemoryProperties memoryProperties;

        bool memory_type_from_properties(uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex) {
            // Search memtypes to find first index with those properties
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
                if ((typeBits & 1) == 1) {
                    // Type is available, does it match user properties?
                    if ((memoryProperties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                        *typeIndex = i;
                        return true;
                    }
                }
                typeBits >>= 1;
            }
            // No memory types matched, return failure
            return false;
        }

        struct {
            glm::mat4x4 P;
            glm::mat4x4 V;
            glm::mat4x4 M;
            glm::mat4x4 clip;
            glm::mat4x4 MVP;
            glm::vec4 solid_color;
        } world;

//        struct {

//            struct {
//                // VkBuffer buffer = VK_NULL_HANDLE;
//                VkDeviceMemory memory = VK_NULL_HANDLE;
//                // uint32_t offset = 0;
//                // uint32_t range = 0;

//                VkDescriptorBufferInfo buffer_info;
//            } uniform;

//            std::vector<VkDescriptorSetLayout> layout_descriptors;
//            VkPipelineLayout layout = VK_NULL_HANDLE;
//            VkDescriptorPool descpool = VK_NULL_HANDLE;
//            std::vector<VkDescriptorSet> descriptor_sets;

//            VkShaderModule vert_shader = VK_NULL_HANDLE;
//            VkShaderModule frag_shader = VK_NULL_HANDLE;

//            //
//            VkPipelineShaderStageCreateInfo shaderStages[2];
//            VkPipeline pipeline = VK_NULL_HANDLE;
//        } pipeline;

        struct {
            vk::DeviceMemory memory;
            vk::DescriptorBufferInfo bufferInfo;
        } uniform;

        vk::DescriptorSetLayout layoutDescriptor;
        vk::PipelineLayout pipelineLayout;
        vk::DescriptorPool descriptorPool;
        std::vector<vk::DescriptorSet> descriptorSets;
        vk::ShaderModule vertexShader;
        vk::ShaderModule fragmentShader;
        vk::Pipeline pipeline;


//        struct {
//            VkSemaphore imgAcqSemaphore = VK_NULL_HANDLE;
//            VkRenderPass pass = VK_NULL_HANDLE;
//        } render_pass;
        vk::Semaphore imgAcquireSemaphore;
        vk::RenderPass renderPass;

        struct {
            // VkBuffer vertex_buffer = VK_NULL_HANDLE;
            vk::Buffer vertexBuffer;
            // VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
            vk::DeviceMemory vertexMemory;
            // VkVertexInputBindingDescription vi_binding;
            vk::VertexInputBindingDescription viBindings;
            std::vector<vk::VertexInputAttributeDescription> viAttribs;
            // VkVertexInputAttributeDescription vi_attribs[2];
            float xRot, yRot, zRot;
        } mesh;

        std::vector<vk::Framebuffer> framebuffers;

        // in case we're running in a graphical session
        GLFWwindow *window = nullptr;

        struct libinput *li = nullptr;
        struct libinput_event *event = nullptr;
        struct udev *udev = nullptr;

        int tty_mode = -1;
    };

}

