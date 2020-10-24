#include "Vulkan.hh"

#include <iostream>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <glm/gtx/transform.hpp>

#include <fstream>
#include <set>
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

int open_restricted(const char *path, int flags, void */*user_data*/) {
    int fd = open(path, flags);
    spdlog::debug("Opening event device {}; Result: {}", path, fd);
    return fd < 0 ? -errno : fd;
}

void close_restricted(int fd, void */*user_data*/) {
    close(fd);
}

const static struct libinput_interface interface = {
    open_restricted, close_restricted
};

constexpr std::array instanceExtensions = {
    VK_KHR_DISPLAY_EXTENSION_NAME,
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME
};

constexpr std::array deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

constexpr std::array validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

void wkt::Vulkan::initialize() {
    spdlog::debug("createInstance");
    createInstance();

    if (isNested()) {
        createWindow();
    }

    spdlog::debug("initInput");
    initInput();

    spdlog::debug("initSurface");
    initSurface();

    spdlog::debug("initDevice");
    initDevice();


    spdlog::debug("initCommandBuffer");
    initCommandBuffer();

    spdlog::debug("beginRecordCommandBuffer");
    beginRecordCommandBuffer();

    spdlog::debug("initUniformBuffer");
    initCubeUniformBuffer();
    initGridUniformBuffer();

    spdlog::debug("initPipelineLayout");
    initPipelineLayout();

    spdlog::debug("initDescriptorSet");
    initDescriptorPool();
    initCubeDescriptorSet();
    intiGridDescriptorSet();

    spdlog::debug("initShaders");
    initShaders();

    spdlog::debug("initVertexBuffers");
    initCubeVertexBuffers();
    initGridVertexBuffers();

    recreateSwapChain();

    spdlog::debug("endRecordCommandBuffer");
    endRecordCommandBuffer();

    spdlog::debug("submitCommandBuffer");
    submitCommandBuffer();
}

void wkt::Vulkan::initInput() {
    udev = udev_new();
    li = libinput_udev_create_context(&interface, nullptr, udev);

    libinput_udev_assign_seat(li, "seat0");
}

VkBool32 vulkanDebugCallback(   VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
                                VkDebugUtilsMessageTypeFlagsEXT                  /*messageTypes*/,
                                const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
                                void*                                            /*pUserData*/) {
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
    }    return VK_FALSE;
}



void wkt::Vulkan::createInstance() {
    vk::ApplicationInfo appInfo("Wakota Desktop", 0, nullptr, 0, VK_API_VERSION_1_1);
    auto extensions = vk::enumerateInstanceExtensionProperties();

    std::vector requiredExtensions(instanceExtensions.begin(), instanceExtensions.end());
    if(isNested()) {
        glfwInit(); // A lot of cleanup is needed. Like this initialization being performed twice
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
    instance = vk::createInstanceUnique(instanceCreateInfo);
    dldi = vk::DispatchLoaderDynamic(*instance, vkGetInstanceProcAddr);

    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo(
                vk::DebugUtilsMessengerCreateFlagsEXT(),
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
                vulkanDebugCallback
                );

    debugUtilsMessenger = instance->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo, nullptr, dldi);

    gpus = instance->enumeratePhysicalDevices();
    memoryProperties = gpus[0].getMemoryProperties();
    // TDO: Normally one would want to pick the best candidate based on
    // some sensible criteria. For now, this should cover some ground.
    gpu = gpus[0];
}

void wkt::Vulkan::cleanupSwapchainAndDependees() {
    framebuffers.clear();
    swapchainImageViews.clear();
}

void wkt::Vulkan::recreateSwapChain() {
    device->waitIdle();
    cleanupSwapchainAndDependees();

    surfaceCharacteristics.capabilities = gpu.getSurfaceCapabilitiesKHR(*wndSurface);

    initSwapchain();
    initDepthBuffer();
    initRenderPass();
    initFramebuffers();
    initCubePipeline();
    initGridPipeline();
}

bool wkt::Vulkan::memory_type_from_properties(uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex) {
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

wkt::Vulkan::~Vulkan() {
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
}

void wkt::Vulkan::initDevice() {
    std::vector<VkBool32> supportsPresent;

    auto qFamProps = gpu.getQueueFamilyProperties();

    supportsPresent.resize(qFamProps.size());

    for (unsigned idx(0); idx < qFamProps.size(); ++idx) {
        supportsPresent[idx] = gpu.getSurfaceSupportKHR(idx, *wndSurface);
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

    vk::PhysicalDeviceFeatures deviceFeatures;
    deviceFeatures.sampleRateShading = true;

    vk::DeviceCreateInfo deviceInfo(
                vk::DeviceCreateFlags(),
                1, &queueInfo,
                validationLayers.size(), validationLayers.data(),
                deviceExtensions.size(), deviceExtensions.data(),
                &deviceFeatures
                );

    device = gpu.createDeviceUnique(deviceInfo);
    queue = device->getQueue(queueFamily, 0);
}

void key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

bool wkt::Vulkan::isNested() const {
    std::string_view session_type = getenv("XDG_SESSION_TYPE");
    spdlog::info("Session type {} detected", session_type);
    return session_type == "wayland" || session_type == "x11";
}

void wkt::Vulkan::windowResized(GLFWwindow* window, int /*w*/, int /*h*/) {
    auto instance = static_cast<Vulkan*>(glfwGetWindowUserPointer(window));
    instance->recreateSwapChain();
}

void wkt::Vulkan::createWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(640, 480, "wakota", nullptr, nullptr);

    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, &key_callback);
    glfwSetWindowSizeCallback(window, windowResized);
    glfwShowWindow(window);
}

void wkt::Vulkan::initSurface() {
    if (isNested()) {
        VkSurfaceKHR surface;
        VkResult res = glfwCreateWindowSurface(*instance, window, nullptr, &surface);
        spdlog::debug("glfwCreateWindowSurface: {}", res);
        wndSurface = vk::UniqueSurfaceKHR(surface, *instance);
    } else {
        auto dispProps = gpu.getDisplayPropertiesKHR();

        vk::DisplayKHR display;
        vk::DisplayModeKHR displayMode;

        for (const auto &disp: dispProps) {

            spdlog::trace("{}:", disp.displayName);
            spdlog::trace("\t {}*{}", disp.physicalDimensions.width, disp.physicalDimensions.height);
            spdlog::trace("\t {}*{}", disp.physicalResolution.width, disp.physicalResolution.height);
            auto modes = gpu.getDisplayModePropertiesKHR(disp.display);
            struct sortByParameters {
                bool operator()(const vk::DisplayModePropertiesKHR& l, const vk::DisplayModePropertiesKHR& r) const noexcept {
                    auto lproduct = 1ll * l.parameters.visibleRegion.height * l.parameters.visibleRegion.width * l.parameters.refreshRate;
                    auto rproduct = 1ll * r.parameters.visibleRegion.height * r.parameters.visibleRegion.width * r.parameters.refreshRate;
                    return lproduct < rproduct;
                }
            };
            std::set<vk::DisplayModePropertiesKHR, sortByParameters> sortedModes(modes.begin(), modes.end());

            spdlog::trace("\tmodes: ");
            for (const auto &mode: sortedModes) {
                spdlog::trace("\t\t- {}*{}@{}", mode.parameters.visibleRegion.width, mode.parameters.visibleRegion.height, mode.parameters.refreshRate);
            }
            displayMode = sortedModes.rbegin()->displayMode;
            spdlog::info("Selected mode: {}*{}@{}", sortedModes.rbegin()->parameters.visibleRegion.width, sortedModes.rbegin()->parameters.visibleRegion.height, sortedModes.rbegin()->parameters.refreshRate);

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

        wndSurface = instance->createDisplayPlaneSurfaceKHRUnique(surfaceCreateInfo);
        if (!wndSurface) {
            spdlog::critical("Couldn't create display surface");
            throw std::runtime_error("Couldn't create display surface");
        }
    }

    surfaceCharacteristics.capabilities = gpu.getSurfaceCapabilitiesKHR(*wndSurface);
    surfaceCharacteristics.presentModes = gpu.getSurfacePresentModesKHR(*wndSurface);
    bool useMailboxPresentMode = false;
    for (auto pMode: surfaceCharacteristics.presentModes) {
        if (pMode == vk::PresentModeKHR::eMailbox) {
            useMailboxPresentMode = true;
            break;
        }
    }
    spdlog::info("Mailbox presentation mode is {}", useMailboxPresentMode);
    surfaceCharacteristics.presentMode = useMailboxPresentMode ? vk::PresentModeKHR::eMailbox : surfaceCharacteristics.presentModes[0];
    surfaceCharacteristics.surfaceFormats = gpu.getSurfaceFormatsKHR(*wndSurface);

    for (const auto &format : surfaceCharacteristics.surfaceFormats) {
        if (format.format == vk::Format::eB8G8R8A8Unorm) surfaceCharacteristics.has_VK_FORMAT_B8G8R8A8_UNORM = true;
    }
}

void wkt::Vulkan::disableTTY() {
    if (isatty(STDIN_FILENO) && ioctl(STDIN_FILENO, KDGKBMODE, &tty_mode) == 0) {
        spdlog::info("Disabling TTY mode");
        ioctl(STDIN_FILENO, KDSKBMODE, K_OFF);
    }
}

void wkt::Vulkan::restoreTTY() {
    if (tty_mode != -1) {
        spdlog::info("Restoring TTY mode");
        ioctl(STDIN_FILENO, KDSKBMODE, tty_mode);
    }
}

void wkt::Vulkan::loop_input() {
    while(!done_looping) {
        libinput_dispatch(li);
        event = libinput_get_event(li);

        if (event) {
            auto evType = libinput_event_get_type(event);

            switch(evType) {
            case LIBINPUT_EVENT_POINTER_MOTION:
            {
                auto pointer_event = libinput_event_get_pointer_event(event);
                auto dx = libinput_event_pointer_get_dx(pointer_event);
                auto dy = libinput_event_pointer_get_dy(pointer_event);

                mesh.xRot += dx / 500.f;
                mesh.yRot += dy / 500.f;

            } break;
            case LIBINPUT_EVENT_POINTER_BUTTON:
                if (!isNested()) {
                    done_looping = true;
                }
                break;
            case LIBINPUT_EVENT_KEYBOARD_KEY:
            {
                auto keyEv = libinput_event_get_keyboard_event(event);
                auto key = libinput_event_keyboard_get_key(keyEv);
                auto state = libinput_event_keyboard_get_key_state(keyEv);
                spdlog::info("Got a key event with keycode: {} in state {}", key, state);
                //                    if (key == )
            } break;
            default:
                break;
            }
            spdlog::debug("Handling event of type {}", evType);
            libinput_event_destroy(event);
        }

    }
}

void wkt::Vulkan::loop() {
    disableTTY();
    const auto start = std::chrono::steady_clock::now();
    long rendered_frames = 0;
    using double_seconds = std::chrono::duration<double>;
    long seconds_threshold = 0;

    while(!done_looping) {
        if (window) {
            if (glfwWindowShouldClose(window)) {
                done_looping = true;
            }
            glfwPollEvents();
        }

        draw();
        ++rendered_frames;
        const auto elapsed_seconds = std::chrono::duration_cast<double_seconds>(std::chrono::steady_clock::now() - start).count();
        if (seconds_threshold < long(elapsed_seconds)) {
            spdlog::info("Average FPS: {}", rendered_frames / elapsed_seconds);
            seconds_threshold = long(elapsed_seconds);
        }

    }
}

void wkt::Vulkan::initCommandBuffer() {
    vk::CommandPoolCreateInfo cmdPoolInfo(
                vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer),
                queueFamily
                );
    commandPool = device->createCommandPoolUnique(cmdPoolInfo);
    vk::CommandBufferAllocateInfo cmdBuffAlocInfo(
                *commandPool, vk::CommandBufferLevel::ePrimary, 1
                );
    auto buffers = device->allocateCommandBuffersUnique(cmdBuffAlocInfo);
    commandBuffer.swap(buffers.front());
}

void wkt::Vulkan::initSwapchain() {
    // if (!surface_characteristics.has_VK_FORMAT_B8G8R8A8_UNORM || surface_characteristics.presentMode == VkPresentModeKHR::VK_PRESENT_MODE_MAX_ENUM_KHR) throw std::runtime_error("");

    vk::SwapchainCreateInfoKHR swapchainInfo(
                vk::SwapchainCreateFlagsKHR(),
                *wndSurface,
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

    swapchain = device->createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = device->getSwapchainImagesKHR(*swapchain);

    for(auto &image: swapchainImages) {
        vk::ImageViewCreateInfo viewInfo(
                    vk::ImageViewCreateFlags(),
                    image,
                    vk::ImageViewType::e2D,
                    vk::Format::eB8G8R8A8Unorm,
                    vk::ComponentMapping(vk::ComponentSwizzle::eIdentity),
                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
                    );

        swapchainImageViews.emplace_back(device->createImageViewUnique(viewInfo));
    }
}

void wkt::Vulkan::initDepthBuffer() {
    vk::ImageCreateInfo imageCreateInfo(
                vk::ImageCreateFlags(),
                vk::ImageType::e2D,
                depthBuffer.format,
                vk::Extent3D(surfaceCharacteristics.capabilities.currentExtent, 1),
                1, 1, getMaxUsableSampleCount(), vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined
                );

    depthBuffer.image = device->createImageUnique(imageCreateInfo);
    vk::MemoryRequirements memoryRequirements = device->getImageMemoryRequirements(*depthBuffer.image);

    vk::MemoryAllocateInfo memoryAllocateInfo(memoryRequirements.size, 0);

    if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, &memoryAllocateInfo.memoryTypeIndex)) throw std::runtime_error("");
    depthBuffer.memory = device->allocateMemoryUnique(memoryAllocateInfo);
    device->bindImageMemory(*depthBuffer.image, *depthBuffer.memory, 0);

    vk::ImageViewCreateInfo imageViewCreateInfo(
                vk::ImageViewCreateFlags(), *depthBuffer.image, vk::ImageViewType::e2D, depthBuffer.format, vk::ComponentMapping(),
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)
                );
    depthBuffer.imageView = device->createImageViewUnique(imageViewCreateInfo);


    vk::ImageCreateInfo resolveBufferInfo(
                vk::ImageCreateFlags{},
                vk::ImageType::e2D,
                vk::Format::eB8G8R8A8Unorm,
                vk::Extent3D(surfaceCharacteristics.capabilities.currentExtent, 1),
                1, 1, getMaxUsableSampleCount(), vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);

    resolveBuffer.image = device->createImageUnique(resolveBufferInfo);
    memoryRequirements = device->getImageMemoryRequirements(*resolveBuffer.image);
    memoryAllocateInfo = vk::MemoryAllocateInfo(memoryRequirements.size, 0);
    if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, &memoryAllocateInfo.memoryTypeIndex)) throw std::runtime_error("");
    resolveBuffer.memory = device->allocateMemoryUnique(memoryAllocateInfo);
    device->bindImageMemory(*resolveBuffer.image, *resolveBuffer.memory, 0);
    vk::ImageViewCreateInfo resolveViewCreateInfo(
                vk::ImageViewCreateFlags(), *resolveBuffer.image, vk::ImageViewType::e2D, vk::Format::eB8G8R8A8Unorm, vk::ComponentMapping(),
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
                );
    resolveBuffer.imageView = device->createImageViewUnique(resolveViewCreateInfo);
}

void wkt::Vulkan::initCubeUniformBuffer() {
    vk::BufferCreateInfo bufferCreateInfo(
                vk::BufferCreateFlags(),
                sizeof(world.MVP) + sizeof(world.solid_color),
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::SharingMode::eExclusive,
                0, nullptr
                );
    cubeUniform.buffer = device->createBufferUnique(bufferCreateInfo);
    vk::MemoryRequirements memoryRequirements = device->getBufferMemoryRequirements(*cubeUniform.buffer);
    vk::MemoryAllocateInfo allocInfo(
                memoryRequirements.size
                );
    if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, &allocInfo.memoryTypeIndex)) throw std::runtime_error("");
    cubeUniform.memory = device->allocateMemoryUnique(allocInfo);
    device->bindBufferMemory(*cubeUniform.buffer, *cubeUniform.memory, 0);

    cubeUniform.bufferInfo.buffer = *cubeUniform.buffer;
    cubeUniform.bufferInfo.offset = 0;
    cubeUniform.bufferInfo.range = sizeof(world.MVP) + sizeof(world.solid_color);
}

void wkt::Vulkan::initGridUniformBuffer() {
    vk::BufferCreateInfo bufferCreateInfo(
                vk::BufferCreateFlags(),
                sizeof(world.MVP) + sizeof(world.solid_color),
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::SharingMode::eExclusive,
                0, nullptr
                );

    gridUniform.buffer = device->createBufferUnique(bufferCreateInfo);
    vk::MemoryRequirements memoryRequirements = device->getBufferMemoryRequirements(*gridUniform.buffer);
    vk::MemoryAllocateInfo allocInfo(
                memoryRequirements.size
                );
    if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, &allocInfo.memoryTypeIndex)) throw std::runtime_error("");
    gridUniform.memory = device->allocateMemoryUnique(allocInfo);
    device->bindBufferMemory(*gridUniform.buffer, *gridUniform.memory, 0);

    gridUniform.bufferInfo.buffer = *gridUniform.buffer;
    gridUniform.bufferInfo.offset = 0;
    gridUniform.bufferInfo.range = sizeof(world.MVP) + sizeof(world.solid_color);
}

void wkt::Vulkan::updateUniformBuffer() {
    world.P = glm::perspective(glm::radians(110.f), surfaceCharacteristics.capabilities.currentExtent.width * 1.f / surfaceCharacteristics.capabilities.currentExtent.height, .01f, 10000.f);
    world.V = glm::lookAt(
                glm::vec3(0, 0, -5),
                glm::vec3(0, 0, 0),
                glm::vec3(0, 1, 0)
                );
    world.M = glm::mat4(1.f);

    world.M = glm::rotate(world.M, mesh.xRot, glm::vec3(1.f, 0.f, 0.f));
    world.M = glm::rotate(world.M, mesh.yRot, glm::vec3(0.f, 1.f, 0.f));
    world.M = glm::rotate(world.M, mesh.zRot, glm::vec3(0.f, 0.f, 1.f));

    world.MVP = world.P * world.V * world.M;
    world.solid_color = glm::vec4(1.f, 0.f, 1.f, 1.f);

    const auto gridMVP = world.P * world.V * glm::mat4(1.f);

    void *pBuffData = device->mapMemory(*cubeUniform.memory, 0, sizeof(world.MVP), vk::MemoryMapFlags());
    memcpy(pBuffData, &world.MVP, sizeof(world.MVP));
    memcpy(((uint8_t*)pBuffData) + sizeof(world.MVP), &world.solid_color, sizeof(world.solid_color));
    device->unmapMemory(*cubeUniform.memory);

    pBuffData = device->mapMemory(*gridUniform.memory, 0, sizeof(gridMVP), vk::MemoryMapFlags());
    memcpy(pBuffData, &gridMVP, sizeof(gridMVP));
    memcpy(((uint8_t*)pBuffData) + sizeof(gridMVP), &world.solid_color, sizeof(world.solid_color));
    device->unmapMemory(*gridUniform.memory);
}

void wkt::Vulkan::initPipelineLayout() {
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

    layoutDescriptor = device->createDescriptorSetLayoutUnique(descriptorLayoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
                vk::PipelineLayoutCreateFlags(),
                1, &*layoutDescriptor, 0, nullptr
                );

    pipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);
}

void wkt::Vulkan::initDescriptorPool() {
    const std::array descPoolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
    };
    ;
    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                1, descPoolSizes.size(), descPoolSizes.data()
                );

    cubeDescriptorPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);
    gridDescriptorPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);
}

void wkt::Vulkan::initCubeDescriptorSet() {
    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(
                *cubeDescriptorPool, 1, &*layoutDescriptor
                );

    cubeDscriptorSets = device->allocateDescriptorSetsUnique(descriptorSetAllocateInfo);

    vk::WriteDescriptorSet writes(
                *cubeDscriptorSets.front(),
                0, 0,
                1, vk::DescriptorType::eUniformBuffer,
                nullptr,
                &cubeUniform.bufferInfo,
                nullptr
                );
    device->updateDescriptorSets(1, &writes, 0, nullptr);
}

void wkt::Vulkan::intiGridDescriptorSet() {
    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(
                *gridDescriptorPool, 1, &*layoutDescriptor
                );

    gridDescriptorSets = device->allocateDescriptorSetsUnique(descriptorSetAllocateInfo);

    vk::WriteDescriptorSet writes(
                *gridDescriptorSets.front(),
                0, 0,
                1, vk::DescriptorType::eUniformBuffer,
                nullptr,
                &gridUniform.bufferInfo,
                nullptr
                );
    device->updateDescriptorSets(1, &writes, 0, nullptr);

}

void wkt::Vulkan::initRenderPass() {
    std::array attachmentDescriptions {
        vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), vk::Format::eB8G8R8A8Unorm, getMaxUsableSampleCount(), vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal),
                vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), depthBuffer.format, getMaxUsableSampleCount(), vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal),
                vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), vk::Format::eB8G8R8A8Unorm, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR)
    };

    vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::AttachmentReference resolvReference(2, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpassDescription(
                vk::SubpassDescriptionFlags(),
                vk::PipelineBindPoint::eGraphics,
                0, nullptr,
                1, &colorReference, &resolvReference, &depthReference,
                0, nullptr
                );

    vk::RenderPassCreateInfo renderPassCreateInfo(
                vk::RenderPassCreateFlags(),
                attachmentDescriptions.size(), attachmentDescriptions.data(),
                1, &subpassDescription,
                0, nullptr
                );

    renderPass = device->createRenderPassUnique(renderPassCreateInfo);
}

void wkt::Vulkan::initShaders() {
    spdlog::trace("Compiling vertex shader");

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
                vk::ShaderModuleCreateFlags(),
                sizeof(vertex_shader),
                vertex_shader
                );

    vertexShader = device->createShaderModuleUnique(shaderModuleCreateInfo);

    spdlog::trace("Compiling fragment shader");
    shaderModuleCreateInfo.codeSize = sizeof(fragment_shader);
    shaderModuleCreateInfo.pCode = fragment_shader;
    fragmentShader = device->createShaderModuleUnique(shaderModuleCreateInfo);
}

void wkt::Vulkan::initFramebuffers() {
    for (uint32_t idx(0); idx < swapchainImages.size(); ++idx) {
        spdlog::trace("Creating framebuffer {}:", idx);
        std::array attachments {
            *resolveBuffer.imageView,
            *depthBuffer.imageView,
            *swapchainImageViews[idx]
        };

        vk::FramebufferCreateInfo framebufferCreateInfo(
                    vk::FramebufferCreateFlags(),
                    *renderPass,
                    attachments.size(), attachments.data(),
                    surfaceCharacteristics.capabilities.currentExtent.width,
                    surfaceCharacteristics.capabilities.currentExtent.height,
                    1
                    );

        framebuffers.emplace_back(device->createFramebufferUnique(framebufferCreateInfo));
    }
}

struct Vertex {
    float posX, posY, posZ, posW;  // Position data
    float r, g, b, a;              // Color
};
#define XYZ1(_x_, _y_, _z_) (_x_), (_y_), (_z_), 1.f
constexpr std::array g_vb_solid_face_colors_Data {
    // red face
    Vertex{ XYZ1(-1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
    Vertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
    Vertex{ XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
    Vertex{ XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
    Vertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
    Vertex{ XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
    // green face
    Vertex{ XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
    Vertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
    Vertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
    // blue face
    Vertex{ XYZ1(-1, 1, 1), XYZ1(0.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 1.f) },
    // yellow face
    Vertex{ XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
    Vertex{ XYZ1(1, -1, -1), XYZ1(1.f, 1.f, 0.f) },
    // magenta face
    Vertex{ XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
    Vertex{ XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
    Vertex{ XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
    Vertex{ XYZ1(-1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
    // cyan face
    Vertex{ XYZ1(1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
    Vertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
    Vertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
    Vertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
    Vertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
    Vertex{ XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
};

constexpr std::array g_vb_grid_lines {
    // Origin X axis, red
    Vertex{ XYZ1(1000000, 0, 0), XYZ1(1.f, 0.f, 0.f) },
    Vertex{ XYZ1(-10000, 0, 0), XYZ1(1.f, 0.f, 0.f) },

    // Origin Y axis, green
    Vertex{ XYZ1(0, 1000000, 0), XYZ1(0.f, 1.f, 0.f) },
    Vertex{ XYZ1(0, -1000000, 0), XYZ1(0.f, 1.f, 0.f) },

    // Origin Y axis, blue
    Vertex{ XYZ1(0, 0, 1000000), XYZ1(0.f, 0.f, 1.f) },
    Vertex{ XYZ1(0, 0, -1000000), XYZ1(0.f, 0.f, 1.f) }

};

void wkt::Vulkan::initCubeVertexBuffers() {
    vk::BufferCreateInfo bufferCreateInfo(
                vk::BufferCreateFlags(),
                sizeof(g_vb_solid_face_colors_Data),
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::SharingMode::eExclusive,
                0, nullptr
                );
    mesh.vertexBuffer = device->createBufferUnique(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device->getBufferMemoryRequirements(*mesh.vertexBuffer);
    vk::MemoryAllocateInfo memoryAllocateInfo(
                memoryRequirements.size
                );

    if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, &memoryAllocateInfo.memoryTypeIndex)) throw std::runtime_error("");

    mesh.vertexMemory = device->allocateMemoryUnique(memoryAllocateInfo);

    void *vMem = device->mapMemory(*mesh.vertexMemory, 0, memoryRequirements.size);
    memcpy(vMem, g_vb_solid_face_colors_Data.data(), sizeof(g_vb_solid_face_colors_Data));
    device->unmapMemory(*mesh.vertexMemory);

    device->bindBufferMemory(*mesh.vertexBuffer, *mesh.vertexMemory, 0);

    mesh.viBindings.binding = 0;
    mesh.viBindings.inputRate = vk::VertexInputRate::eVertex;
    mesh.viBindings.stride = sizeof(Vertex);

    mesh.viAttribs.emplace_back(0, 0, vk::Format::eR32G32B32A32Sfloat, 0);
    mesh.viAttribs.emplace_back(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(struct Vertex, r));
}

void wkt::Vulkan::initGridVertexBuffers() {
    vk::BufferCreateInfo bufferCreateInfo(
                vk::BufferCreateFlags(),
                sizeof(g_vb_grid_lines),
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::SharingMode::eExclusive,
                0, nullptr
                );
    wlBuffer = device->createBufferUnique(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device->getBufferMemoryRequirements(*wlBuffer);
    vk::MemoryAllocateInfo memoryAllocateInfo(
                memoryRequirements.size
                );

    if (!memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, &memoryAllocateInfo.memoryTypeIndex)) throw std::runtime_error("");

    wlMemory = device->allocateMemoryUnique(memoryAllocateInfo);

    void *vMem = device->mapMemory(*wlMemory, 0, memoryRequirements.size);
    memcpy(vMem, g_vb_grid_lines.data(), sizeof(g_vb_grid_lines));
    device->unmapMemory(*wlMemory);
    device->bindBufferMemory(*wlBuffer, *wlMemory, 0);
}

vk::SampleCountFlagBits wkt::Vulkan::getMaxUsableSampleCount() {
    vk::PhysicalDeviceProperties props = gpu.getProperties();

    vk::SampleCountFlags counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64)  { return vk::SampleCountFlagBits::e64;  }
    if (counts & vk::SampleCountFlagBits::e32)  { return vk::SampleCountFlagBits::e32;  }
    if (counts & vk::SampleCountFlagBits::e16)  { return vk::SampleCountFlagBits::e16;  }
    if (counts & vk::SampleCountFlagBits::e8)   { return vk::SampleCountFlagBits::e8;   }
    if (counts & vk::SampleCountFlagBits::e4)   { return vk::SampleCountFlagBits::e4;   }
    if (counts & vk::SampleCountFlagBits::e2)   { return vk::SampleCountFlagBits::e2;   }

    return vk::SampleCountFlagBits::e1;
}

void wkt::Vulkan::initCubePipeline() {
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
                0.f, 0.f,
                surfaceCharacteristics.capabilities.currentExtent.width * 1.f,
                surfaceCharacteristics.capabilities.currentExtent.height * 1.f,
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
                getMaxUsableSampleCount(),
                true, 1.f, nullptr, false, false
                );

    std::array shaderStages = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eVertex, *vertexShader, "main", nullptr),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragmentShader, "main", nullptr)
    };

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo(
                vk::PipelineCreateFlags(),
                shaderStages.size(), shaderStages.data(),
                &pipelineVertexInputStateCreateInfo,
                &pipelineInputAssemblyStateCreateInfo,
                nullptr, // tesselation state
                &pipelineViewportStateCreateInfo,
                &pipelineRasterizationStateCreateInfo,
                &pipelineMultisampleCreateInfo,
                &pipelineDepthStencilCreateInfo,
                &pipelineColorBlendStateCreateInfo,
                nullptr, // dynamic state
                *pipelineLayout,
                *renderPass, 0,
                vk::Pipeline(), 0
                );

    cubePipeline = device->createGraphicsPipelineUnique(vk::PipelineCache(), graphicsPipelineCreateInfo).value;
}

void wkt::Vulkan::initGridPipeline() {
    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo(
                vk::PipelineVertexInputStateCreateFlags(),
                1, &mesh.viBindings,
                mesh.viAttribs.size(), mesh.viAttribs.data()
                );

    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo(
                vk::PipelineInputAssemblyStateCreateFlags(),
                vk::PrimitiveTopology::eLineList, false
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
                0.f, 0.f,
                surfaceCharacteristics.capabilities.currentExtent.width * 1.f,
                surfaceCharacteristics.capabilities.currentExtent.height * 1.f,
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
                getMaxUsableSampleCount(),
                true, 1.f, nullptr, false, false
                );

    std::array shaderStages = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eVertex, *vertexShader, "main", nullptr),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragmentShader, "main", nullptr)
    };

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo(
                vk::PipelineCreateFlags(),
                shaderStages.size(), shaderStages.data(),
                &pipelineVertexInputStateCreateInfo,
                &pipelineInputAssemblyStateCreateInfo,
                nullptr, // tesselation state
                &pipelineViewportStateCreateInfo,
                &pipelineRasterizationStateCreateInfo,
                &pipelineMultisampleCreateInfo,
                &pipelineDepthStencilCreateInfo,
                &pipelineColorBlendStateCreateInfo,
                nullptr, // dynamic state
                *pipelineLayout,
                *renderPass, 0,
                vk::Pipeline(), 0
                );

    wlPipeline = device->createGraphicsPipelineUnique(vk::PipelineCache(), graphicsPipelineCreateInfo).value;
}

void wkt::Vulkan::beginRecordCommandBuffer() {
    vk::CommandBufferBeginInfo cmdBuffBeginInfo;
    commandBuffer->begin(cmdBuffBeginInfo);
}

void wkt::Vulkan::beginRecordCommandBuffer(int) {
    return beginRecordCommandBuffer();
}

void wkt::Vulkan::endRecordCommandBuffer() {
    commandBuffer->end();
}

void wkt::Vulkan::submitCommandBuffer() {
    vk::FenceCreateInfo fenceCreateInfo;
    vk::Fence drawFence = device->createFence(fenceCreateInfo);

    vk::PipelineStageFlags pipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(
                0, nullptr, &pipelineStageFlags, 1, &*commandBuffer, 0, nullptr
                );
    queue.submit({submitInfo}, drawFence);
    vk::Result res;
    do {
        res = device->waitForFences(1, &drawFence, true, 1000000000);
    } while (res == vk::Result::eTimeout);
    device->destroyFence(drawFence);
}

void wkt::Vulkan::draw() {
    spdlog::trace("Prepping clear values");
    std::array clearValues = {
        vk::ClearValue(vk::ClearColorValue()),
        vk::ClearValue(vk::ClearDepthStencilValue(1.f, 0)),
        vk::ClearValue(vk::ClearColorValue())
    };

    updateUniformBuffer();
    beginRecordCommandBuffer();

    vk::SemaphoreCreateInfo semaphoreCreateInfo;
    vk::UniqueSemaphore imageAcquiredSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);

    spdlog::trace("Acquiring the next image in the swapchain");
    uint32_t currentBuffer;

    try {
        currentBuffer = device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore, vk::Fence()).value;
    }  catch (const vk::OutOfDateKHRError&) {
        endRecordCommandBuffer();
        recreateSwapChain();
        return;
    }

    vk::RenderPassBeginInfo renderPassBeginInfo(*renderPass, *framebuffers[currentBuffer], vk::Rect2D(vk::Offset2D(), surfaceCharacteristics.capabilities.currentExtent), clearValues.size(), clearValues.data());
    spdlog::trace("Begin render pass");
    commandBuffer->beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

    const vk::DeviceSize offset(0);
    const auto rawCubeDescSets = vk::uniqueToRaw(cubeDscriptorSets);
    const auto rawGridDescSets = vk::uniqueToRaw(gridDescriptorSets);

    // Rendering the grid lines
    commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *wlPipeline);
    commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, rawGridDescSets.size(), rawGridDescSets.data(), 0, nullptr);
    commandBuffer->bindVertexBuffers(0, 1, &*wlBuffer, &offset);
    commandBuffer->draw(g_vb_grid_lines.size(), 1, 0, 0);
    // Rendering the cube
    commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *cubePipeline);
    commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, rawCubeDescSets.size(), rawCubeDescSets.data(), 0, nullptr);
    commandBuffer->bindVertexBuffers(0, 1, &*mesh.vertexBuffer, &offset);
    commandBuffer->draw(g_vb_solid_face_colors_Data.size(), 1, 0, 0);

    commandBuffer->endRenderPass();
    endRecordCommandBuffer();

    vk::FenceCreateInfo fenceCreateInfo;
    vk::UniqueFence drawFence = device->createFenceUnique(fenceCreateInfo);

    vk::PipelineStageFlags pipelineStageFlags = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo(
                1, &*imageAcquiredSemaphore, &pipelineStageFlags,  1, &*commandBuffer, 0, nullptr
                );
    /* Queue the command buffer for execution */

    spdlog::trace("queue.submit");
    queue.submit(1, &submitInfo, *drawFence);
    spdlog::trace("waiting for fence");
    vk::PresentInfoKHR presentInfo(
                0, nullptr, 1, &*swapchain, &currentBuffer
                );

    vk::Result res;
    do {
        res = device->waitForFences(1, &*drawFence, true, 1000000000);
    } while (res == vk::Result::eTimeout);

    spdlog::trace("Present");
    queue.presentKHR(&presentInfo);
    spdlog::trace("Present: {}", res);
}
