#include "vulkan.h"

#include <iostream>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <glm/gtx/transform.hpp>

#include <fstream>
#include <map>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/kd.h>

#include <unistd.h>
#include <fcntl.h>

#include <vertex_shader.h>
#include <fragment_shader.h>

#include <chrono>

int open_restricted(const char *path, int flags, void *user_data) {
        int fd = open(path, flags);
        spdlog::debug("Opening event device {}; Result: {}", path, fd);
        return fd < 0 ? -errno : fd;
}

void close_restricted(int fd, void *user_data) {
        close(fd);
}

const static struct libinput_interface interface = {
        open_restricted, close_restricted
};

namespace wkt {

    constexpr std::array instanceExtensions = {
        VK_KHR_DISPLAY_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    constexpr std::array deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    constexpr std::array validationLayers = {
        "VK_LAYER_LUNARG_standard_validation",
        "VK_LAYER_LUNARG_core_validation",
        "VK_LAYER_LUNARG_parameter_validation"
    };

    void Vulkan::initialize() {
        spdlog::debug("createInstance");
        createInstance();

        spdlog::debug("initSurface");
        initSurface();

        spdlog::debug("initDevice");
        initDevice();

        spdlog::debug("initCommandBuffer");
        initCommandBuffer();

        spdlog::debug("beginRecordCommandBuffer");
        beginRecordCommandBuffer();

        spdlog::debug("initSwapchain");
        initSwapchain();

        spdlog::debug("initDepthBuffer");
        initDepthBuffer();

        spdlog::debug("initUniformBuffer");
        initUniformBuffer();

        spdlog::debug("initPipelineLayout");
        initPipelineLayout();

        spdlog::debug("initDescriptorSet");
        initDescriptorSet();

        spdlog::debug("initRenderPass");
        initRenderPass();

        spdlog::debug("initShaders");
        initShaders();

        spdlog::debug("initFramebuffers");
        initFramebuffers();

        spdlog::debug("initVertexBuffers");
        initVertexBuffers();

        spdlog::debug("initPipeline");
        initPipeline();

        spdlog::debug("endRecordCommandBuffer");
        endRecordCommandBuffer();

        spdlog::debug("submitCommandBuffer");
        submitCommandBuffer();
    }



    VkBool32 vulkanDebugCallback(   VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
                                    VkDebugUtilsMessageTypeFlagsEXT                  /*messageTypes*/,
                                    const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
                                    void*                                            /*pUserData*/) {

        static const std::map messageSeverityMap = {
            std::pair(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, "error"),
            std::pair(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, "warning"),
            std::pair(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, "info"),
            std::pair(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT, "log"),
            std::pair(VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT, "")
        };

//        static const std::map messageTypeMap = {
//            std::pair(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, "general"),
//            std::pair(VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, "validation"),
//            std::pair(VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, "performance")
//        };

        try {
            switch(messageSeverity) {
                case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
                    spdlog::error("{}:{}", pCallbackData->pMessageIdName, pCallbackData->pMessage);
                break;
                case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
                    spdlog::warn("{}:{}", pCallbackData->pMessageIdName, pCallbackData->pMessage);
                break;
                case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
                    spdlog::info("{}:{}", pCallbackData->pMessageIdName, pCallbackData->pMessage);
                break;
                default:
                    spdlog::trace("{}:{}", pCallbackData->pMessageIdName, pCallbackData->pMessage);
                break;
            }
        } catch (const std::exception& e) {
            spdlog::error(e.what());
        } catch (...) {
            spdlog::error("Unrecognized exception");
        }
        return VK_FALSE;
    }



    void Vulkan::createInstance() {
        vk::ApplicationInfo appInfo("Wakota Desktop", 0, nullptr, 0, VK_API_VERSION_1_1);
        auto extensions = vk::enumerateInstanceExtensionProperties();

        std::vector requiredExtensions(instanceExtensions.begin(), instanceExtensions.end());
        if(isNested()) {
            glfwInit(); // A lot of cleaning up is needed. Like this initialization being performed twice
            uint32_t numExts;
            auto glfwRequiredExtensions = glfwGetRequiredInstanceExtensions(&numExts);
            requiredExtensions.insert(requiredExtensions.end(), glfwRequiredExtensions, glfwRequiredExtensions + numExts);
        }

        vk::InstanceCreateInfo instanceCreateInfo(
            vk::InstanceCreateFlags(),
            &appInfo,
            validationLayers.size(), validationLayers.data(),
            requiredExtensions.size(), requiredExtensions.data()
        );


        auto layers = vk::enumerateInstanceLayerProperties();
        instance = vk::createInstance(instanceCreateInfo);

//        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo(
//            vk::DebugUtilsMessengerCreateFlagsEXT(), vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose, vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,vulkanDebugCallback
//        );

        VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo;
        debugUtilsMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugUtilsMessengerCreateInfo.pNext = nullptr;
        debugUtilsMessengerCreateInfo.pUserData = nullptr;
        debugUtilsMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugUtilsMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugUtilsMessengerCreateInfo.flags = 0;
        debugUtilsMessengerCreateInfo.pfnUserCallback = vulkanDebugCallback;

        auto pfn_vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(instance.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
        (*pfn_vkCreateDebugUtilsMessengerEXT)(instance, &debugUtilsMessengerCreateInfo, nullptr, &debugUtilsMessenger);
        // instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfo);
        // vkCreateDebugUtilsMessengerEXT(instance, )
        gpus = instance.enumeratePhysicalDevices();
        memoryProperties = gpus[0].getMemoryProperties();
        // TDO: Normally one would want to pick the best candidate based on
        // some preset criteria. For now, this should cover some ground.
        gpu = gpus[0];
    }

    Vulkan::~Vulkan()
    {
        restoreTTY();
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        if (li) {
            libinput_unref(li);
        }

        if (udev) {
            udev_unref(udev);
        }

        for (auto &fb : framebuffers) {
            if (fb) {
                vkDestroyFramebuffer(device, fb, nullptr);
                fb = nullptr;
            }
        }
        framebuffers.clear();

        if (pipeline) {
            device.destroyPipeline(pipeline);
        }

        if (mesh.vertexBuffer) {
            device.destroyBuffer(mesh.vertexBuffer);
        }

        if (mesh.vertexMemory) {
            device.freeMemory(mesh.vertexMemory);
        }

        if (vertexShader) {
            device.destroyShaderModule(vertexShader);
        }

        if (fragmentShader) {
            device.destroyShaderModule(fragmentShader);
        }

        if (renderPass) {
            device.destroyRenderPass(renderPass);
        }

        if (imgAcquireSemaphore) {
            device.destroySemaphore(imgAcquireSemaphore);
        }

        if (descriptorPool) {
            device.destroyDescriptorPool(descriptorPool);
        }

        if(layoutDescriptor) {
            device.destroyDescriptorSetLayout(layoutDescriptor);
        }

        if (pipelineLayout) {
            device.destroyPipelineLayout(pipelineLayout);
        }

        if (uniform.bufferInfo.buffer) {
            device.destroyBuffer(uniform.bufferInfo.buffer);
        }

        if (uniform.memory) {
            device.freeMemory(uniform.memory);
        }

        if (depthBuffer.imageView) {
            device.destroyImageView(depthBuffer.imageView);
        }
        if (depthBuffer.image) {
            device.destroyImage(depthBuffer.image);
        }
        if (depthBuffer.memory) {
            vkFreeMemory(device, depthBuffer.memory, nullptr);
        }

        for (auto &view : swapchainImageViews) {
            if (view) {
                vkDestroyImageView(device, view, nullptr);
            }
        }
        if (swapchain) {
            vkDestroySwapchainKHR(device, swapchain, nullptr);
        }
        if (wndSurface) {
            vkDestroySurfaceKHR(instance, wndSurface, nullptr);
        }
        if (commandBuffer) {
            // vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buff);
            device.freeCommandBuffers(commandPool, {commandBuffer});
        }
        if (commandPool) {
            vkDestroyCommandPool(device, commandPool, nullptr);
            // device.free()
        }

        device.destroy();

        auto pfn_vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
        (*pfn_vkDestroyDebugUtilsMessengerEXT)(instance, debugUtilsMessenger, nullptr);

        instance.destroy();
    }

    void Vulkan::initDevice() {
        std::vector<VkBool32> supportsPresent;

        auto qFamProps = gpu.getQueueFamilyProperties();

        supportsPresent.resize(qFamProps.size());

        for (unsigned idx(0); idx < qFamProps.size(); ++idx) {
            supportsPresent[idx] = gpu.getSurfaceSupportKHR(idx, wndSurface) == VK_TRUE;
        }


        queueFamily = 0;
        for (; queueFamily < qFamProps.size(); ++queueFamily) {
            if (qFamProps[queueFamily].queueFlags & vk::QueueFlagBits::eGraphics) {
                break;
            }
        }

        if (queueFamily == qFamProps.size()) {
            throw std::runtime_error("No queue family has graphics support.");
        }

        // for simplification, we assume that we only have 1 queue family that supports all operations
        if (supportsPresent[queueFamily] != VK_TRUE) {
            throw std::runtime_error("The graphics queue family has no present support.");
        }
        float queue_priorities[1] = { .0 };

        vk::DeviceQueueCreateInfo queueInfo(
            vk::DeviceQueueCreateFlags(),
            queueFamily,
            1, queue_priorities);

        vk::DeviceCreateInfo deviceInfo(
            vk::DeviceCreateFlags(),
            1, &queueInfo,
            validationLayers.size(), validationLayers.data(),
            deviceExtensions.size(), deviceExtensions.data(),
            nullptr
        );

        device = gpu.createDevice(deviceInfo);
        queue = device.getQueue(queueFamily, 0);
    }

    void key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
    bool Vulkan::isNested() const {
        std::string_view session_type = getenv("XDG_SESSION_TYPE");
        spdlog::info("Session type {} detected", session_type);
        return session_type == "wayland" || session_type == "x11";
    }

    void Vulkan::initSurface() {
        if (isNested()) {

            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(640, 480, "wakots", nullptr, nullptr);
            glfwSetKeyCallback(window, &key_callback);
            glfwShowWindow(window);

            VkSurfaceKHR surface;
            VkResult res = glfwCreateWindowSurface(instance, window, nullptr, &surface);
            spdlog::debug("glfwCreateWindowSurface: {}", res);

            wndSurface = surface;
        } else {

            auto dev = gpu;
            auto dispProps = gpu.getDisplayPropertiesKHR();
            auto dispPlaneProps = gpu.getDisplayPlanePropertiesKHR();

            vk::DisplayKHR display;
            vk::DisplayModeKHR displayMode;

            for (const auto &disp: dispProps) {

                spdlog::trace("{}:", disp.displayName);
                spdlog::trace("\t {}*{}", disp.physicalDimensions.width, disp.physicalDimensions.height);
                spdlog::trace("\t {}*{}", disp.physicalResolution.width, disp.physicalResolution.height);
                auto modes = dev.getDisplayModePropertiesKHR(disp.display);
                spdlog::trace("\tmodes: ");
                for (const auto &mode: modes) {
                    spdlog::trace("\t\t- {}*{}@{}", mode.parameters.visibleRegion.width, mode.parameters.visibleRegion.height, mode.parameters.refreshRate);
                    if (!displayMode) {
                        displayMode = mode.displayMode;
                    }
                }

                if (!display) {
                    display = disp.display;
                }
            }

            if (!display || !displayMode) {
                spdlog::critical("Couldn't select a display or a display mode");
                throw std::runtime_error("Couldn't select a display or a display mode");
            }

            vk::DisplaySurfaceCreateInfoKHR surfaceCreateInfo(
                vk::DisplaySurfaceCreateFlagsKHR(),
                displayMode, 0, 0
            );

            wndSurface = instance.createDisplayPlaneSurfaceKHR(surfaceCreateInfo);
            if (!wndSurface) {
                spdlog::critical("Couldn't create display surface");
                throw std::runtime_error("Couldn't create display surface");
            }


        }
        udev = udev_new();
        li = libinput_udev_create_context(&interface, nullptr, udev);

        libinput_udev_assign_seat(li, "seat0");

        surfaceCharacteristics.capabilities = gpu.getSurfaceCapabilitiesKHR(wndSurface);
        surfaceCharacteristics.presentModes = gpu.getSurfacePresentModesKHR(wndSurface);
        surfaceCharacteristics.presentMode = surfaceCharacteristics.presentModes[0];
        surfaceCharacteristics.surfaceFormats = gpu.getSurfaceFormatsKHR(wndSurface);

        for (const auto &format : surfaceCharacteristics.surfaceFormats) {
            if (format.format == vk::Format::eB8G8R8A8Unorm) surfaceCharacteristics.has_VK_FORMAT_B8G8R8A8_UNORM = true;
        }
    }

    void Vulkan::disableTTY() {
        if (isatty(STDIN_FILENO) && ioctl(STDIN_FILENO, KDGKBMODE, &tty_mode) == 0) {
            spdlog::info("Disabling TTY mode");
            ioctl(STDIN_FILENO, KDSKBMODE, K_OFF);
        }
    }

    void Vulkan::restoreTTY() {
        if (tty_mode != -1) {
            spdlog::info("Restoring TTY mode");
            ioctl(STDIN_FILENO, KDSKBMODE, tty_mode);
        }
    }

    void Vulkan::loop() {
        disableTTY();
//        const auto start = std::chrono::system_clock::now();
        while(1) {
            libinput_dispatch(li);
            event = libinput_get_event(li);

//            const auto now = std::chrono::system_clock::now();
//            const std::chrono::duration<double> time_since_start = now - start;
//            if (time_since_start.count() > 5. && !window) {
//                break;
//            }
            draw();

            if (event) {
                auto evType = libinput_event_get_type(event);
                if(evType == LIBINPUT_EVENT_POINTER_MOTION) {
                    auto pointer_event = libinput_event_get_pointer_event(event);
                    auto dx = libinput_event_pointer_get_dx(pointer_event);
                    auto dy = libinput_event_pointer_get_dy(pointer_event);

                    mesh.xRot += dx / 100.f;
                    mesh.yRot += dy / 100.f;
                } else if (evType == LIBINPUT_EVENT_POINTER_BUTTON) {
//                    auto pointer_event = libinput_event_get_pointer_event(event);
                    break;
                }
                spdlog::debug("Handling event of type {}", evType);
                libinput_event_destroy(event);
            }


            if (window) {
                if (glfwWindowShouldClose(window)) {
                    break;
                }
                glfwPollEvents();
            }
        }
    }

    void Vulkan::initCommandBuffer() {
        vk::CommandPoolCreateInfo cmdPoolInfo(
            vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer),
            queueFamily
        );
        commandPool = device.createCommandPool(cmdPoolInfo);
        vk::CommandBufferAllocateInfo cmdBuffAlocInfo(
            commandPool, vk::CommandBufferLevel::ePrimary, 1
        );
        commandBuffer = device.allocateCommandBuffers(cmdBuffAlocInfo)[0];
    }

    void Vulkan::initSwapchain() {
        // if (!surface_characteristics.has_VK_FORMAT_B8G8R8A8_UNORM || surface_characteristics.presentMode == VkPresentModeKHR::VK_PRESENT_MODE_MAX_ENUM_KHR) throw std::runtime_error("");

        vk::SwapchainCreateInfoKHR swapchainInfo(
            vk::SwapchainCreateFlagsKHR(),
            wndSurface,
            surfaceCharacteristics.capabilities.minImageCount,
            vk::Format::eB8G8R8A8Unorm,
            vk::ColorSpaceKHR::eSrgbNonlinear,
            surfaceCharacteristics.capabilities.currentExtent,
            1, vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment),
            vk::SharingMode::eExclusive,
            1, &queueFamily,
            surfaceCharacteristics.capabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            surfaceCharacteristics.presentMode
        );

        swapchain = device.createSwapchainKHR(swapchainInfo);
        swapchainImages = device.getSwapchainImagesKHR(swapchain);

        for(auto &image: swapchainImages) {
            vk::ImageViewCreateInfo viewInfo(
                vk::ImageViewCreateFlags(),
                image,
                vk::ImageViewType::e2D,
                vk::Format::eB8G8R8A8Unorm,
                vk::ComponentMapping(vk::ComponentSwizzle::eIdentity),
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
            );

            swapchainImageViews.push_back(device.createImageView(viewInfo));
        }

    }

    void Vulkan::initDepthBuffer() {
        vk::ImageCreateInfo imageCreateInfo(
            vk::ImageCreateFlags(),
            vk::ImageType::e2D,
            depthBuffer.format,
            vk::Extent3D(surfaceCharacteristics.capabilities.currentExtent, 1),
            1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined
        );

        depthBuffer.image = device.createImage(imageCreateInfo);
        vk::MemoryRequirements memoryRequirements = device.getImageMemoryRequirements(depthBuffer.image);

        vk::MemoryAllocateInfo memoryAllocateInfo(memoryRequirements.size, 0);

        if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, &memoryAllocateInfo.memoryTypeIndex)) throw std::runtime_error("");
        depthBuffer.memory = device.allocateMemory(memoryAllocateInfo);
        device.bindImageMemory(depthBuffer.image, depthBuffer.memory, 0);

        vk::ImageViewCreateInfo imageViewCreateInfo(
            vk::ImageViewCreateFlags(), depthBuffer.image, vk::ImageViewType::e2D, depthBuffer.format, vk::ComponentMapping(),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)
        );
        depthBuffer.imageView = device.createImageView(imageViewCreateInfo);
    }

    void Vulkan::initUniformBuffer() {
        mesh.xRot = 0.f;
        mesh.yRot = 0.f;
        mesh.zRot = 0.f;

        /* do the math one per demo*/
        world.P = glm::perspective(glm::radians(45.f), (float)surfaceCharacteristics.capabilities.currentExtent.width / (float)surfaceCharacteristics.capabilities.currentExtent.height, .1f, 100.f);
        world.V = glm::lookAt(
            glm::vec3(-5, 3, -10),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 1, 0)
        );
        world.M = glm::mat4(1.f);
        world.M = glm::rotate(world.M, mesh.xRot, glm::vec3(1.f, 0.f, 0.f));
        world.M = glm::rotate(world.M, mesh.yRot, glm::vec3(0.f, 1.f, 0.f));
        world.M = glm::rotate(world.M, mesh.zRot, glm::vec3(0.f, 0.f, 1.f));

        world.clip = glm::mat4(
            1.f, 0.f, 0.f, 0.f,
            0.f, -1.f, 0.f, 0.f,
            0.f, 0.f, .5f, 0.f,
            0.f, 0.f, .5f, 1.f
        );

        world.MVP = world.clip * world.P * world.V * world.M;
        world.solid_color = glm::vec4(1.f, 0.f, 1.f, 1.f);

        vk::BufferCreateInfo bufferCreateInfo(
            vk::BufferCreateFlags(),
            sizeof(world.MVP) + sizeof(world.solid_color),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::SharingMode::eExclusive,
            0, nullptr
        );
        uniform.bufferInfo = device.createBuffer(bufferCreateInfo);


        vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(uniform.bufferInfo.buffer);
        vk::MemoryAllocateInfo allocInfo(
            memoryRequirements.size
        );
        if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, &allocInfo.memoryTypeIndex)) throw std::runtime_error("");
        uniform.memory = device.allocateMemory(allocInfo);

        void *pBuffData = device.mapMemory(uniform.memory, 0, memoryRequirements.size, vk::MemoryMapFlags());

        memcpy(pBuffData, &world.MVP, sizeof(world.MVP));
        memcpy(((uint8_t*)pBuffData) + sizeof(world.MVP), &world.solid_color, sizeof(world.solid_color));

        device.unmapMemory(uniform.memory);

        device.bindBufferMemory(uniform.bufferInfo.buffer, uniform.memory, 0);

        uniform.bufferInfo.offset = 0;
        uniform.bufferInfo.range = sizeof(world.MVP) + sizeof(world.solid_color);
    }

    void Vulkan::updateUniformBuffer() {
        /* do the math one per demo*/
        world.P = glm::perspective(glm::radians(45.f), surfaceCharacteristics.capabilities.currentExtent.width * 1.0f / surfaceCharacteristics.capabilities.currentExtent.height, .1f, 100.f);
        world.V = glm::lookAt(
            glm::vec3(-5, 3, -10),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 1, 0)
        );
        world.M = glm::mat4(1.f);
        world.M = glm::mat4(1.f);
        world.M = glm::rotate(world.M, mesh.xRot, glm::vec3(1.f, 0.f, 0.f));
        world.M = glm::rotate(world.M, mesh.yRot, glm::vec3(0.f, 1.f, 0.f));
        world.M = glm::rotate(world.M, mesh.zRot, glm::vec3(0.f, 0.f, 1.f));
        world.clip = glm::mat4(
            1.f, 0.f, 0.f, 0.f,
            0.f, -1.f, 0.f, 0.f,
            0.f, 0.f, .5f, 0.f,
            0.f, 0.f, .5f, 1.f
        );

        world.MVP = world.clip * world.P * world.V * world.M;
        world.solid_color = glm::vec4(1.f, 0.f, 1.f, 1.f);


        void *pBuffData = device.mapMemory(uniform.memory, 0, sizeof(world.MVP), vk::MemoryMapFlags());

        memcpy(pBuffData, &world.MVP, sizeof(world.MVP));
        device.unmapMemory(uniform.memory);
    }

    void Vulkan::initPipelineLayout() {
        vk::DescriptorSetLayoutBinding layoutBinding(
            0, vk::DescriptorType::eUniformBuffer,
            1,
            vk::ShaderStageFlags(vk::ShaderStageFlagBits::eVertex),
            nullptr
        );
        vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            1, &layoutBinding
        );

        layoutDescriptor = device.createDescriptorSetLayout(descriptorLayoutInfo);

        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            vk::PipelineLayoutCreateFlags(),
            1, &layoutDescriptor, 0, nullptr
        );

        pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
    }

    void Vulkan::initDescriptorSet() {
        vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1);
        vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlags(),
            1, 1, &descriptorPoolSize
        );

        descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);

        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(
            descriptorPool, 1, &layoutDescriptor
        );

        descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);

        vk::WriteDescriptorSet writes(
            descriptorSets[0],
            0, 0,
            1, vk::DescriptorType::eUniformBuffer,
            nullptr,
            &uniform.bufferInfo,
            nullptr
        );
        device.updateDescriptorSets(1, &writes, 0, nullptr);
    }

    void Vulkan::initRenderPass() {
        vk::SemaphoreCreateInfo semaphoreCreateInfo;

        vk::AttachmentDescription attachmentDescriptions[2] = {
            vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), vk::Format::eB8G8R8A8Unorm, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR),
            vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), depthBuffer.format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal)
        };

        vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::AttachmentReference depthReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::SubpassDescription subpassDescription(
            vk::SubpassDescriptionFlags(),
            vk::PipelineBindPoint::eGraphics,
            0, nullptr,
            1, &colorReference,
            nullptr, &depthReference,
            0, nullptr
        );

        vk::RenderPassCreateInfo renderPassCreateInfo(
            vk::RenderPassCreateFlags(),
            2, attachmentDescriptions,
            1, &subpassDescription,
            0, nullptr
        );

        renderPass = device.createRenderPass(renderPassCreateInfo);
    }

    void Vulkan::initShaders() {
        spdlog::trace("Compiling vertex shader");

        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            sizeof(vertex_shader),
            vertex_shader
        );

        vertexShader = device.createShaderModule(shaderModuleCreateInfo);

        spdlog::trace("Compiling fragment shader");
        shaderModuleCreateInfo.codeSize = sizeof(fragment_shader);
        shaderModuleCreateInfo.pCode = fragment_shader;
        fragmentShader = device.createShaderModule(shaderModuleCreateInfo);
    }

    void Vulkan::initFramebuffers() {
        vk::ImageView attachments[2];
        attachments[1] = depthBuffer.imageView;


        for (uint32_t idx(0); idx < swapchainImages.size(); ++idx) {
            spdlog::trace("Creating framebuffer {}:", idx);
            attachments[0] = swapchainImageViews[idx];

            vk::FramebufferCreateInfo framebufferCreateInfo(
                vk::FramebufferCreateFlags(),
                renderPass,
                2, attachments,
                surfaceCharacteristics.capabilities.currentExtent.width,
                surfaceCharacteristics.capabilities.currentExtent.height,
                1
            );

            VkFramebuffer framebuff;
            vkCreateFramebuffer(device, reinterpret_cast<const VkFramebufferCreateInfo*>( &framebufferCreateInfo ), nullptr, &framebuff);
            // vk::Framebuffer framebuffer;
            // framebuffer = device.createFramebuffer(framebufferCreateInfo);

            framebuffers.push_back(framebuff);
        }
    }

    struct Vertex {
        float posX, posY, posZ, posW;  // Position data
        float r, g, b, a;              // Color
    };
#define XYZ1(_x_, _y_, _z_) (_x_), (_y_), (_z_), 1.f
    static const Vertex g_vb_solid_face_colors_Data[] = {
        // red face
        { XYZ1(-1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
        { XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
        { XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
        { XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
        { XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
        { XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
        // green face
        { XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
        { XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
        { XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
        { XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
        { XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
        { XYZ1(1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
        // blue face
        { XYZ1(-1, 1, 1), XYZ1(0.f, 0.f, 1.f) },
        { XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
        { XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
        { XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
        { XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
        { XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 1.f) },
        // yellow face
        { XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 0.f) },
        { XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
        { XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
        { XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
        { XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
        { XYZ1(1, -1, -1), XYZ1(1.f, 1.f, 0.f) },
        // magenta face
        { XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
        { XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
        { XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
        { XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
        { XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
        { XYZ1(-1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
        // cyan face
        { XYZ1(1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
        { XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
        { XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
        { XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
        { XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
        { XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
    };

    void Vulkan::initVertexBuffers() {
        vk::BufferCreateInfo bufferCreateInfo(
            vk::BufferCreateFlags(),
            sizeof(g_vb_solid_face_colors_Data),
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::SharingMode::eExclusive,
            0, nullptr
        );
        mesh.vertexBuffer = device.createBuffer(bufferCreateInfo);

        vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(mesh.vertexBuffer);
        vk::MemoryAllocateInfo memoryAllocateInfo(
            memoryRequirements.size
        );

        if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, &memoryAllocateInfo.memoryTypeIndex)) throw std::runtime_error("");

        mesh.vertexMemory = device.allocateMemory(memoryAllocateInfo);

        void *vMem = device.mapMemory(mesh.vertexMemory, 0, memoryRequirements.size);
        memcpy(vMem, g_vb_solid_face_colors_Data, sizeof(g_vb_solid_face_colors_Data));
        device.unmapMemory(mesh.vertexMemory);

        device.bindBufferMemory(mesh.vertexBuffer, mesh.vertexMemory, 0);

        mesh.viBindings.binding = 0;
        mesh.viBindings.inputRate = vk::VertexInputRate::eVertex;
        mesh.viBindings.stride = sizeof(g_vb_solid_face_colors_Data[0]);

        mesh.viAttribs.emplace_back(0, 0, vk::Format::eR32G32B32A32Sfloat, 0);
        mesh.viAttribs.emplace_back(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(struct Vertex, r));
    }

    void Vulkan::initPipeline() {
        // vk::DynamicState dynamicStateEnable[VK_DYNAMIC_STATE_RANGE_SIZE] = {0};
        vk::PipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo(
            vk::PipelineDynamicStateCreateFlags(),
            0, nullptr
        );
        // memset(dynStateEnable, 0, sizeof dynStateEnable);

        vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo(
            vk::PipelineVertexInputStateCreateFlags(),
            1, &mesh.viBindings,
            mesh.viAttribs.size(), mesh.viAttribs.data()
        );
        vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo(
            vk::PipelineInputAssemblyStateCreateFlags(),
            vk::PrimitiveTopology::eTriangleList, false
        );

        vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo(
            vk::PipelineRasterizationStateCreateFlags(),
            false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise,
            false, 0, 0, 0, 1.f
        );

        vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState(
            false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlags(0xfu)
        );
        vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo(
            vk::PipelineColorBlendStateCreateFlags(),
            false, vk::LogicOp::eNoOp, 1, &pipelineColorBlendAttachmentState,
            {1.f, 1.f, 1.f, 1.f}
        );

        vk::Viewport viewport(
            0, 0,
            (float)surfaceCharacteristics.capabilities.currentExtent.width,
            (float)surfaceCharacteristics.capabilities.currentExtent.height,
            0.f, 0.f
        );
        vk::Rect2D scissor(vk::Offset2D(0, 0), surfaceCharacteristics.capabilities.currentExtent);

        vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo(
            vk::PipelineViewportStateCreateFlags(),
            1, &viewport, 1, &scissor
        );

        vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilCreateInfo(
            vk::PipelineDepthStencilStateCreateFlags(),
            true, true, vk::CompareOp::eLessOrEqual, false, false,
            vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0, 0, 0),
            vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0, 0, 0),
            0, 0
        );

        vk::PipelineMultisampleStateCreateInfo pipelineMultisampleCreateInfo(
            vk::PipelineMultisampleStateCreateFlags(),
            vk::SampleCountFlagBits::e1,
            false, 0.f, nullptr, false, false
        );


        std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eVertex, vertexShader, "main", nullptr),
            vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eFragment, fragmentShader, "main", nullptr)
        };

        vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo(
            vk::PipelineCreateFlags(),
            shaderStages.size(), shaderStages.data(),
            &pipelineVertexInputStateCreateInfo,
            &pipelineInputAssemblyStateCreateInfo,
            nullptr,
            &pipelineViewportStateCreateInfo,
            &pipelineRasterizationStateCreateInfo,
            &pipelineMultisampleCreateInfo,
            &pipelineDepthStencilCreateInfo,
            &pipelineColorBlendStateCreateInfo,
            // &pipelineDepthStencilCreateInfo,
            &pipelineDynamicStateCreateInfo,
            pipelineLayout,
            renderPass, 0,
            vk::Pipeline(), 0
        );

        pipeline = device.createGraphicsPipeline(vk::PipelineCache(), graphicsPipelineCreateInfo);
    }

    void Vulkan::beginRecordCommandBuffer() {
        vk::CommandBufferBeginInfo cmdBuffBeginInfo;
        commandBuffer.begin(cmdBuffBeginInfo);
    }

    void Vulkan::beginRecordCommandBuffer(int) {
        return  beginRecordCommandBuffer();
    }

    void Vulkan::endRecordCommandBuffer() {
        commandBuffer.end();
    }

    void Vulkan::submitCommandBuffer() {
        vk::FenceCreateInfo fenceCreateInfo;
        vk::Fence drawFence = device.createFence(fenceCreateInfo);

        vk::PipelineStageFlags pipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        vk::SubmitInfo submitInfo(
            0, nullptr, &pipelineStageFlags, 1, &commandBuffer, 0, nullptr
        );
        queue.submit({submitInfo}, drawFence);

        vk::Result res;

        do {
            res = device.waitForFences(1, &drawFence, true, 1000000000);
        } while (res == vk::Result::eTimeout);

        device.destroyFence(drawFence);
    }

    void Vulkan::setXRotation(float angle) {
        mesh.xRot = angle;
    }
    void Vulkan::setYRotation(float angle) {
        mesh.yRot = angle;
    }
    void Vulkan::setZRotation(float angle) {
        mesh.zRot = angle;
    }

    void Vulkan::draw() {
        spdlog::trace("Prepping clear values");
        std::array<vk::ClearValue, 2> clearValues = {
            vk::ClearValue(vk::ClearColorValue()),
            vk::ClearValue(vk::ClearDepthStencilValue(1.f, 0))
        };

        beginRecordCommandBuffer();
        updateUniformBuffer();

        vk::SemaphoreCreateInfo semaphoreCreateInfo;

        vk::Semaphore imageAcquiredSemaphore = device.createSemaphore(semaphoreCreateInfo);


        spdlog::trace("Acquiring the next image in the swapchain");
        uint32_t currentBuffer = device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAcquiredSemaphore, vk::Fence()).value;

        vk::RenderPassBeginInfo renderPassBeginInfo(renderPass, framebuffers[currentBuffer], vk::Rect2D(vk::Offset2D(), surfaceCharacteristics.capabilities.currentExtent), clearValues.size(), clearValues.data());

        spdlog::trace("Begin render pass");
        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        spdlog::trace("Bind pipeline");
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

        spdlog::trace("Bind descriptor set");
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.size(), descriptorSets.data(), 0, nullptr);

        const vk::DeviceSize offset(0);
        spdlog::trace("Bind vertex buffers");
        commandBuffer.bindVertexBuffers(0, 1, &mesh.vertexBuffer, &offset);

        spdlog::trace("Draw");
        commandBuffer.draw(12 * 3, 1, 0, 0);
        spdlog::trace("End render pass");
        commandBuffer.endRenderPass();
        spdlog::trace("endRecordCommandBuffer");

        endRecordCommandBuffer();


        vk::FenceCreateInfo fenceCreateInfo;
        vk::Fence drawFence = device.createFence(fenceCreateInfo);

        vk::PipelineStageFlags pipelineStageFlags = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        vk::SubmitInfo submitInfo(
            1, &imageAcquiredSemaphore, &pipelineStageFlags,  1, &commandBuffer, 0, nullptr
        );
        /* Queue the command buffer for execution */

        spdlog::trace("queue.submit");

        queue.submit(1, &submitInfo, drawFence);

        spdlog::trace("waiting for fence");
        vk::PresentInfoKHR presentInfo(
            0, nullptr, 1, &swapchain, &currentBuffer
        );

        vk::Result res;
        do {
            res = device.waitForFences(1, &drawFence, true, 1000000000);
        } while (res == vk::Result::eTimeout);

        spdlog::trace("Present");
        queue.presentKHR(&presentInfo);
        spdlog::trace("Present: {}", res);

        device.destroySemaphore(imageAcquiredSemaphore);
        device.destroyFence(drawFence);
    }
}
