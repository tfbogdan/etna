#include "Renderer.hh"

#include <iostream>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <fstream>
#include <set>
#include <map>

#include <vertex_shader.h>
#include <fragment_shader.h>

#include <chrono>

#include <imgui/imgui_impl_vulkan.h>
#include <imgui/imgui_impl_glfw.h>

#include <tinyobjloader/tiny_obj_loader.h>

constexpr std::array instanceExtensions = {
    VK_KHR_DISPLAY_EXTENSION_NAME,
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME
};

constexpr std::array deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME
};

constexpr std::array validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

etna::Renderer::~Renderer() {
    if (sharedUniformAllocation) {
        vmaFreeMemory(allocator, sharedUniformAllocation);
    }

    if (allocator) {
        vmaDestroyAllocator(allocator);
    }
}

void etna::Renderer::initialize(GLFWwindow* window) {
    _window = window;

    spdlog::debug("createInstance");
    createInstance();

    spdlog::debug("initSurface");
    initSurface(window);

    spdlog::debug("initDevice");
    initDevice();

    spdlog::debug("initCommandBuffer");
    initCommandBuffer();

    spdlog::debug("beginRecordCommandBuffer");
    beginRecordCommandBuffer(0);

    spdlog::debug("initPipelineLayout");
    initPipelineLayout();

    initDescriptorPool();

    spdlog::debug("initVertexBuffers");
    initCubeVertexBuffers();

    spdlog::debug("initUniformBuffer");
    initSharedUniformBuffer();


    spdlog::debug("initDescriptorSet");
    initSharedDescriptorSet();

    spdlog::debug("initShaders");
    initShaders();

    initRenderPass();
    initGuiRenderPass();
    initPipeline();

    recreateSwapChain();

    spdlog::debug("endRecordCommandBuffer");
    endRecordCommandBuffer(0);

    spdlog::debug("submitCommandBuffer");
    submitCommandBuffer(0);

    initGui();
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
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            spdlog::trace("{}:{}", pCallbackData->pMessageIdName, pCallbackData->pMessage);
            break;
        default: break;
        }
    } catch (const std::exception& e) {
        spdlog::error(e.what());
    } catch (...) {
        spdlog::error("Unrecognized exception");
    }
    return VK_FALSE;
}

void etna::Renderer::createInstance() {
    vk::ApplicationInfo appInfo("etna", 0, nullptr, 0, VK_API_VERSION_1_1);
    auto extensions = vk::enumerateInstanceExtensionProperties();

    std::vector requiredExtensions(instanceExtensions.begin(), instanceExtensions.end());

    uint32_t numExts;
    auto glfwRequiredExtensions = glfwGetRequiredInstanceExtensions(&numExts);
    requiredExtensions.insert(requiredExtensions.end(), glfwRequiredExtensions, glfwRequiredExtensions + numExts);

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
    // TDO: Pick a suitable candidate when more GPUs are available
    gpu = gpus[0];
}

void etna::Renderer::recreateSwapChain() {
    device->waitIdle();

    surfaceCapabilities = gpu.getSurfaceCapabilitiesKHR(*wndSurface);

    initSwapchain();
    initDepthBuffer();
    initFramebuffers();
    initGuiFramebuffers();
}

bool etna::Renderer::memory_type_from_properties(uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex) {
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

void etna::Renderer::initDevice() {
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
    assert(queueFamily != qFamProps.size());
    // TDO: for simplification, we assume that we only have 1 queue family that
    // supports all operations but that's clearly not good enough for real world usage
    assert (supportsPresent[queueFamily]);
    float queue_priorities[1] = { .0 };
    vk::DeviceQueueCreateInfo queueInfo(
                vk::DeviceQueueCreateFlags(),
                queueFamily,
                1, queue_priorities);

    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT dynamicStateFeatures;
    dynamicStateFeatures.extendedDynamicState = true;

    vk::PhysicalDeviceFeatures deviceFeatures;
    deviceFeatures.sampleRateShading = true;

    vk::DeviceCreateInfo deviceInfo(
                vk::DeviceCreateFlags(),
                1, &queueInfo,
                validationLayers.size(), validationLayers.data(),
                deviceExtensions.size(), deviceExtensions.data(),
                &deviceFeatures
                );

    deviceInfo.pNext = &deviceFeatures;
    vk::PhysicalDeviceDriverProperties driverProps;
    vk::PhysicalDeviceProperties2 props2;
    props2.pNext = &driverProps;
    gpu.getProperties2(&props2);
    auto props = gpu.getProperties();
    spdlog::info("Using driver {} version {}", driverProps.driverName, props.driverVersion);
    device = gpu.createDeviceUnique(deviceInfo);

    VmaAllocatorCreateInfo allocInfo = {};
    allocInfo.device = *device;
    allocInfo.instance = *instance;
    allocInfo.physicalDevice = gpu;
    vmaCreateAllocator(&allocInfo, &allocator); // TODO: check result

    queue = device->getQueue(queueFamily, 0);
}


bool etna::Renderer::isNested() const {
    std::string_view session_type = getenv("XDG_SESSION_TYPE");
    spdlog::info("Session type {} detected", session_type);
    return session_type == "wayland" || session_type == "x11";
}

void etna::Renderer::initSurface(GLFWwindow* window) {
    VkSurfaceKHR surface;
    VkResult res = glfwCreateWindowSurface(*instance, window, nullptr, &surface);
    spdlog::debug("glfwCreateWindowSurface: {}", res);
    wndSurface = vk::UniqueSurfaceKHR(surface, *instance);

    surfaceCapabilities = gpu.getSurfaceCapabilitiesKHR(*wndSurface);
    std::vector<vk::PresentModeKHR> presentModes = gpu.getSurfacePresentModesKHR(*wndSurface);
    bool useMailboxPresentMode = false;
    for (auto pMode: presentModes) {
        if (pMode == vk::PresentModeKHR::eMailbox) {
            useMailboxPresentMode = true;
            break;
        }
    }
    spdlog::info("Mailbox presentation mode is {}", useMailboxPresentMode ? "available" : "unavailable");
    presentMode = useMailboxPresentMode ? vk::PresentModeKHR::eMailbox : presentModes[0];
    std::vector<vk::SurfaceFormatKHR> surfaceFormats = gpu.getSurfaceFormatsKHR(*wndSurface);
    bool has_VK_FORMAT_B8G8R8A8_UNORM = false;
    for (const auto &format : surfaceFormats) {
        if (format.format == vk::Format::eB8G8R8A8Unorm) has_VK_FORMAT_B8G8R8A8_UNORM = true;
    }
    assert(has_VK_FORMAT_B8G8R8A8_UNORM);
}

void etna::Renderer::initCommandBuffer() {
    if (!commandPool) {
        vk::CommandPoolCreateInfo cmdPoolInfo;
        cmdPoolInfo.queueFamilyIndex = queueFamily;
        cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        commandPool = device->createCommandPoolUnique(cmdPoolInfo);
    }
    vk::CommandBufferAllocateInfo cmdBuffAlocInfo(
                *commandPool, vk::CommandBufferLevel::ePrimary, 2
                );
    commandBuffers = device->allocateCommandBuffersUnique(cmdBuffAlocInfo);
}

void etna::Renderer::initSwapchain() {
    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.surface               = *wndSurface;
    swapchainInfo.minImageCount         = surfaceCapabilities.minImageCount;
    swapchainInfo.imageFormat           = vk::Format::eB8G8R8A8Unorm;
    swapchainInfo.imageColorSpace       = vk::ColorSpaceKHR::eSrgbNonlinear;
    swapchainInfo.imageExtent           = surfaceCapabilities.currentExtent;
    swapchainInfo.imageArrayLayers      = 1;
    swapchainInfo.imageUsage            = vk::ImageUsageFlagBits::eColorAttachment;
    swapchainInfo.imageSharingMode      = vk::SharingMode::eExclusive;
    swapchainInfo.queueFamilyIndexCount = 1;
    swapchainInfo.pQueueFamilyIndices   = &queueFamily;
    swapchainInfo.preTransform          = surfaceCapabilities.currentTransform;
    swapchainInfo.compositeAlpha        = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapchainInfo.presentMode           = presentMode;

    swapchain = device->createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = device->getSwapchainImagesKHR(*swapchain);
    swapchainImageViews.clear();

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

void etna::Renderer::initDepthBuffer() {
    vk::ImageCreateInfo imageCreateInfo(
                vk::ImageCreateFlags(),
                vk::ImageType::e2D,
                depthBuffer.format,
                vk::Extent3D(surfaceCapabilities.currentExtent, 1),
                1, 1, getMaxUsableSampleCount(), vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined
                );

    depthBuffer.image = device->createImageUnique(imageCreateInfo);
    vk::MemoryRequirements memoryRequirements = device->getImageMemoryRequirements(*depthBuffer.image);

    vk::MemoryAllocateInfo memoryAllocateInfo(memoryRequirements.size, 0);

    assert(memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, &memoryAllocateInfo.memoryTypeIndex));
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
                vk::Extent3D(surfaceCapabilities.currentExtent, 1),
                1, 1, getMaxUsableSampleCount(), vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);

    resolveBuffer.image = device->createImageUnique(resolveBufferInfo);
    memoryRequirements = device->getImageMemoryRequirements(*resolveBuffer.image);
    memoryAllocateInfo = vk::MemoryAllocateInfo(memoryRequirements.size, 0);

    assert(memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, &memoryAllocateInfo.memoryTypeIndex));
    resolveBuffer.memory = device->allocateMemoryUnique(memoryAllocateInfo);
    device->bindImageMemory(*resolveBuffer.image, *resolveBuffer.memory, 0);
    vk::ImageViewCreateInfo resolveViewCreateInfo(
                vk::ImageViewCreateFlags(), *resolveBuffer.image, vk::ImageViewType::e2D, vk::Format::eB8G8R8A8Unorm, vk::ComponentMapping(),
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
                );
    resolveBuffer.imageView = device->createImageViewUnique(resolveViewCreateInfo);
}

void etna::Renderer::initSharedUniformBuffer() {
    vk::BufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    bufferCreateInfo.size = 65536;
    bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

    sharedUniformBuffer = device->createBufferUnique(bufferCreateInfo);

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocInfo.requiredFlags = VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vmaAllocateMemoryForBuffer(allocator, *sharedUniformBuffer, &allocInfo, &sharedUniformAllocation, nullptr);
    vmaBindBufferMemory(allocator, sharedUniformAllocation, *sharedUniformBuffer);
    sharedBufferInfo.buffer = *sharedUniformBuffer;
    sharedBufferInfo.offset = 0;
    sharedBufferInfo.range = VK_WHOLE_SIZE;

    // Calculate required alignment based on minimum device offset alignment
    size_t minUboAlignment = gpu.getProperties().limits.minUniformBufferOffsetAlignment;
    size_t dynamicAlignment = sizeof(UniformBuffer);
    if (minUboAlignment > 0) {
        dynamicAlignment = (dynamicAlignment + minUboAlignment - 1) & ~(minUboAlignment - 1);
    }

    for (int ix = 0; ix < std::ssize(sceneObjects); ++ix) {
        auto& obj = sceneObjects[ix];
        obj.uniformBufferOffset = ix * dynamicAlignment;
    }
}

void etna::Renderer::updateUniformBuffer() {
    world.P = glm::infinitePerspectiveLH(glm::radians(camera.fov), surfaceCapabilities.currentExtent.width * 1.f / surfaceCapabilities.currentExtent.height, camera.zNear);
    //    world.P = glm::perspectiveLH(glm::radians(camera.fov), surfaceCapabilities.currentExtent.width * 1.f / surfaceCapabilities.currentExtent.height, camera.zNear, camera.zFar);
    glm::quat qPitch    = glm::angleAxis(glm::radians(camera.rotation.x), glm::vec3(1, 0, 0));
    glm::quat qYaw      = glm::angleAxis(glm::radians(camera.rotation.y), glm::vec3(0, 1, 0));
    glm::quat qRoll     = glm::angleAxis(glm::radians(camera.rotation.z), glm::vec3(0,0,1));

    glm::quat orientation = qPitch * qYaw * qRoll;
    orientation = glm::normalize(orientation);
    glm::mat4 rotate = glm::mat4_cast(orientation);

    glm::mat4 translate = glm::mat4(1.0f);
    translate = glm::translate(translate, -glm::vec3(camera.pos));

    world.V = rotate * translate;

    updateUniformBuffers();
}

void etna::Renderer::updateUniformBuffers() {
//    void *mem = nullptr;
    void *sharedMem = nullptr;
    vmaMapMemory(allocator, sharedUniformAllocation, &sharedMem);

    for (const auto& obj: sceneObjects) {
//        mem = device->mapMemory(*obj.uboMemory, 0, VK_WHOLE_SIZE, {});
//        auto& objUniform = *reinterpret_cast<UniformBuffer*>(mem);
        auto& sharedObjUniform = *reinterpret_cast<UniformBuffer*>(static_cast<std::byte*>(sharedMem) + obj.uniformBufferOffset);
        const auto& M = glm::translate(glm::scale(glm::mat4(1.f), obj.scale), obj.position) * glm::eulerAngleXYX(
                    glm::radians(obj.rotation.x),
                    glm::radians(obj.rotation.y),
                    glm::radians(obj.rotation.z)
                    );

        sharedObjUniform.mvp /*= objUniform.mvp */= world.P * world.V * M;
//        device->unmapMemory(*obj.uboMemory);
    }
    vmaUnmapMemory(allocator, sharedUniformAllocation);
}

void etna::Renderer::initPipelineLayout() {
    // So this doesn't work
    //    vk::DescriptorSetLayoutBinding layoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex);
    //    vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo({}, 1, &layoutBinding);
    //    descSetLayout = device->createDescriptorSetLayoutUnique(descriptorLayoutInfo);
    //    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, 1, &*descSetLayout);

    // While this does?!
    vk::DescriptorSetLayoutBinding layoutBinding(
                0, vk::DescriptorType::eUniformBufferDynamic,
                1,
                vk::ShaderStageFlags(vk::ShaderStageFlagBits::eVertex),
                nullptr
                );
    vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo(
                vk::DescriptorSetLayoutCreateFlags(),
                1, &layoutBinding
                );

    descSetLayout = device->createDescriptorSetLayoutUnique(descriptorLayoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
                vk::PipelineLayoutCreateFlags(),
                1, &*descSetLayout, 0, nullptr
                );

    pipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);
}

void etna::Renderer::initDescriptorPool() {
    const std::array descPoolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformTexelBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageTexelBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, 1000),
        vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 1000)
    };

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
                {},
                1000 * descPoolSizes.size(), descPoolSizes.size(), descPoolSizes.data()
                );

    descriptorPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);
}

void etna::Renderer::initSharedDescriptorSet() {
    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(
                *descriptorPool, 1, &*descSetLayout
                );

    sharedDescriptorSet = device->allocateDescriptorSets(descriptorSetAllocateInfo).front();

    vk::WriteDescriptorSet write(
                sharedDescriptorSet,
                0, 0,
                1, vk::DescriptorType::eUniformBufferDynamic,
                nullptr,
                &sharedBufferInfo,
                nullptr
                );
    device->updateDescriptorSets(1, &write, 0, nullptr);

}

void etna::Renderer::initRenderPass() {
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

void etna::Renderer::initGuiRenderPass() {
    std::array attachmentDescriptions {vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), vk::Format::eB8G8R8A8Unorm, getMaxUsableSampleCount(), vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR)};

    vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpassDescription;
    subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;

    vk::SubpassDependency dependency;
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo renderPassCreateInfo;
    renderPassCreateInfo.attachmentCount = attachmentDescriptions.size();
    renderPassCreateInfo.pAttachments = attachmentDescriptions.data();
    renderPassCreateInfo.dependencyCount = 1;
    renderPassCreateInfo.pDependencies = &dependency;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDescription;

    guiRenderPass = device->createRenderPassUnique(renderPassCreateInfo);
}

void etna::Renderer::initShaders() {
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

void etna::Renderer::initFramebuffers() {
    framebuffers.clear();
    for (uint32_t idx(0); idx < swapchainImages.size(); ++idx) {
        spdlog::trace("Creating framebuffer {}:", idx);
        std::array attachments { *resolveBuffer.imageView, *depthBuffer.imageView,*swapchainImageViews[idx]};

        vk::FramebufferCreateInfo framebufferCreateInfo(
                    vk::FramebufferCreateFlags(),
                    *renderPass,
                    attachments.size(), attachments.data(),
                    surfaceCapabilities.currentExtent.width,
                    surfaceCapabilities.currentExtent.height,
                    1
                    );

        framebuffers.emplace_back(device->createFramebufferUnique(framebufferCreateInfo));
    }
}

void etna::Renderer::initGuiFramebuffers() {
    guiFramebuffers.clear();
    for (uint32_t idx(0); idx < swapchainImages.size(); ++idx) {
        spdlog::trace("Creating framebuffer {}:", idx);
        std::array attachments {
            *swapchainImageViews[idx]
        };

        vk::FramebufferCreateInfo framebufferCreateInfo(
                    vk::FramebufferCreateFlags(),
                    *guiRenderPass,
                    attachments.size(), attachments.data(),
                    surfaceCapabilities.currentExtent.width,
                    surfaceCapabilities.currentExtent.height,
                    1
                    );

        guiFramebuffers.emplace_back(device->createFramebufferUnique(framebufferCreateInfo));
    }

}

#define XYZ1(_x_, _y_, _z_) {(_x_), (_y_), (_z_), 1.f}
constexpr std::array g_vb_solid_face_colors_Data {
    // red face
    etna::ColoredVertex{ XYZ1(-1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 0.f) },
    // green face
    etna::ColoredVertex{ XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, 1, -1), XYZ1(0.f, 1.f, 0.f) },
    // blue face
    etna::ColoredVertex{ XYZ1(-1, 1, 1), XYZ1(0.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 1.f) },
    // yellow face
    etna::ColoredVertex{ XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(1, -1, -1), XYZ1(1.f, 1.f, 0.f) },
    // magenta face
    etna::ColoredVertex{ XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, 1, -1), XYZ1(1.f, 0.f, 1.f) },
    // cyan face
    etna::ColoredVertex{ XYZ1(1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
    etna::ColoredVertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f) },
    etna::ColoredVertex{ XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f) },
    etna::ColoredVertex{ XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 1.f) },

    // Origin X axis, red
    etna::ColoredVertex{ XYZ1(1000000, 0, 0), XYZ1(1.f, 0.f, 0.f) },
    etna::ColoredVertex{ XYZ1(-10000, 0, 0), XYZ1(1.f, 0.f, 0.f) },

    // Origin Y axis, green
    etna::ColoredVertex{ XYZ1(0, 1000000, 0), XYZ1(0.f, 1.f, 0.f) },
    etna::ColoredVertex{ XYZ1(0, -1000000, 0), XYZ1(0.f, 1.f, 0.f) },

    // Origin Y axis, blue
    etna::ColoredVertex{ XYZ1(0, 0, 1000000), XYZ1(0.f, 0.f, 1.f) },
    etna::ColoredVertex{ XYZ1(0, 0, -1000000), XYZ1(0.f, 0.f, 1.f) }
};


void etna::Renderer::initCubeVertexBuffers() {
    auto vikingRoomPath = getenv("VIKING_ROOM_OBJ");

    std::vector<etna::ColoredVertex> vertices;
    std::vector<uint32_t> indices;

    if (vikingRoomPath) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warns;
        std::string errors;

        tinyobj::LoadObj(&attrib, &shapes, &materials, &warns, &errors, vikingRoomPath);
        if (!warns.empty()) spdlog::warn(warns);
        if (!errors.empty()) spdlog::error(errors);

        constexpr float xrotation = glm::radians(-90.f);
        constexpr float yrotation = glm::radians(-90.f);
        constexpr float zrotation = glm::radians(-90.f);

        for (const auto& shape : shapes) {
            for ([[maybe_unused]]const auto& index : shape.mesh.indices) {
                ColoredVertex vertex{};

                vertex.position = {
                    attrib.vertices[3 * index.vertex_index + 2] * 5,
                    attrib.vertices[3 * index.vertex_index + 1] * 5,
                    attrib.vertices[3 * index.vertex_index + 0] * 5,
                    1.f
                };

                vertex.position = vertex.position * (glm::mat4(1.f) * glm::eulerAngleXYX(xrotation, yrotation, zrotation));

                vertex.color = {1.f, 1.f, 1.f, 1.f};
                vertices.push_back(vertex);
                indices.push_back(indices.size());
            }
        }
    }


    vk::BufferCreateInfo bufferCreateInfo(
                vk::BufferCreateFlags(),
                sizeof(g_vb_solid_face_colors_Data) + sizeof(ColoredVertex) * vertices.size(),
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::SharingMode::eExclusive,
                0, nullptr
                );
    mesh.vertexBuffer = device->createBufferUnique(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device->getBufferMemoryRequirements(*mesh.vertexBuffer);
    vk::MemoryAllocateInfo memoryAllocateInfo(memoryRequirements.size);
    assert(memory_type_from_properties(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, &memoryAllocateInfo.memoryTypeIndex));

    mesh.vertexMemory = device->allocateMemoryUnique(memoryAllocateInfo);
    std::byte *vMem = static_cast<std::byte*>(device->mapMemory(*mesh.vertexMemory, 0, memoryRequirements.size));
    memcpy(vMem, g_vb_solid_face_colors_Data.data(), sizeof(g_vb_solid_face_colors_Data));
    memcpy(&vMem[sizeof(g_vb_solid_face_colors_Data)], vertices.data(), sizeof(ColoredVertex) * vertices.size());
    device->unmapMemory(*mesh.vertexMemory);

    device->bindBufferMemory(*mesh.vertexBuffer, *mesh.vertexMemory, 0);
    mesh.viBindings.binding = 0;
    mesh.viBindings.inputRate = vk::VertexInputRate::eVertex;
    mesh.viBindings.stride = sizeof(ColoredVertex);

    mesh.viAttribs.emplace_back(0, 0, vk::Format::eR32G32B32A32Sfloat, 0);
    mesh.viAttribs.emplace_back(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(struct ColoredVertex, color));

    SceneObject cubeObj;
    cubeObj.vertexBuffer = *mesh.vertexBuffer;
    cubeObj.bufferStart = 0;
    cubeObj.numVerts = 6 * 6;
    cubeObj.name = "cube";

    SceneObject coordinatesObj;
    coordinatesObj.vertexBuffer = *mesh.vertexBuffer;
    coordinatesObj.bufferStart = cubeObj.numVerts + cubeObj.bufferStart;
    coordinatesObj.numVerts = 6;
    coordinatesObj.name = "world";
    coordinatesObj.visible = true;
    coordinatesObj.topology = vk::PrimitiveTopology::eLineList;

    sceneObjects.emplace_back(std::move(cubeObj));
    sceneObjects.emplace_back(std::move(coordinatesObj));

    if (vikingRoomPath) {
        SceneObject meshObj;
        meshObj.vertexBuffer = *mesh.vertexBuffer;
        meshObj.bufferStart = coordinatesObj.numVerts + coordinatesObj.bufferStart;
        meshObj.numVerts = vertices.size();
        meshObj.visible = true;
        meshObj.name = "viking_room";

        sceneObjects.emplace_back(std::move(meshObj));
    }
}

vk::SampleCountFlagBits etna::Renderer::getMaxUsableSampleCount() {
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

void etna::Renderer::initPipeline() {
    std::array dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
        vk::DynamicState::ePrimitiveTopologyEXT
    };
    vk::PipelineDynamicStateCreateInfo dynamicStateInfo;
    dynamicStateInfo.dynamicStateCount = dynamicStates.size();
    dynamicStateInfo.pDynamicStates = dynamicStates.data();

    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
    pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
    pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = &mesh.viBindings;
    pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = mesh.viAttribs.size();
    pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = mesh.viAttribs.data();

    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo;

    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo(
                vk::PipelineRasterizationStateCreateFlags(),
                false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise,
                false, 0, 0, 0, 1.f
                );

    vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState = {};
    pipelineColorBlendAttachmentState.blendEnable = true;
    pipelineColorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    pipelineColorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    pipelineColorBlendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
    pipelineColorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    pipelineColorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    pipelineColorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
    pipelineColorBlendAttachmentState.colorWriteMask = vk::ColorComponentFlags(0xfu);

    vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo = {};
    pipelineColorBlendStateCreateInfo.logicOpEnable = false;
    pipelineColorBlendStateCreateInfo.logicOp = vk::LogicOp::eNoOp;
    pipelineColorBlendStateCreateInfo.attachmentCount = 1;
    pipelineColorBlendStateCreateInfo.pAttachments = &pipelineColorBlendAttachmentState;
    pipelineColorBlendStateCreateInfo.blendConstants[0] = 1.f;
    pipelineColorBlendStateCreateInfo.blendConstants[1] = 1.f;
    pipelineColorBlendStateCreateInfo.blendConstants[2] = 1.f;
    pipelineColorBlendStateCreateInfo.blendConstants[3] = 1.f;

    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo;

    vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilCreateInfo;
    pipelineDepthStencilCreateInfo.depthTestEnable = true;
    pipelineDepthStencilCreateInfo.depthWriteEnable = true;
    pipelineDepthStencilCreateInfo.depthCompareOp = vk::CompareOp::eLess;
    pipelineDepthStencilCreateInfo.depthBoundsTestEnable = false;
    pipelineDepthStencilCreateInfo.stencilTestEnable = false;

    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleCreateInfo(
                vk::PipelineMultisampleStateCreateFlags(),
                getMaxUsableSampleCount(),
                true, 1.f, nullptr, false, false
                );

    std::array shaderStages = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eVertex, *vertexShader, "main", nullptr
        ),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragmentShader, "main", nullptr
        )
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
                &dynamicStateInfo,
                *pipelineLayout,
                *renderPass, 0,
                vk::Pipeline(), 0
                );
    cubePipeline = device->createGraphicsPipelineUnique(vk::PipelineCache(), graphicsPipelineCreateInfo).value;
}

void etna::Renderer::initGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    [[maybe_unused]] ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForVulkan(_window, true);

    ImGui_ImplVulkan_InitInfo imguiInitInfo = {};
    imguiInitInfo.Instance = *instance;
    imguiInitInfo.PhysicalDevice = gpu;
    imguiInitInfo.Device = *device;
    imguiInitInfo.QueueFamily = queueFamily;
    imguiInitInfo.Queue = queue;
    imguiInitInfo.PipelineCache = nullptr;
    imguiInitInfo.DescriptorPool = *descriptorPool;
    imguiInitInfo.Allocator = nullptr;

    imguiInitInfo.MinImageCount = surfaceCapabilities.minImageCount;
    imguiInitInfo.ImageCount = swapchainImageViews.size();

    imguiInitInfo.CheckVkResultFn = nullptr;
    ImGui_ImplVulkan_Init(&imguiInitInfo, *guiRenderPass);

    beginRecordCommandBuffer(1);
    ImGui_ImplVulkan_CreateFontsTexture(*commandBuffers[1]);
    endRecordCommandBuffer(1);
    submitCommandBuffer(1);
}

void etna::Renderer::buildGui() {
    int w, h;
    glfwGetFramebufferSize(_window, &w, &h);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Scene");
    for (int ix = 0; ix < std::ssize(sceneObjects); ++ix) {
        auto& obj = sceneObjects[ix];
        uint32_t flags = ImGuiTreeNodeFlags_Leaf;
        if (selectedSceneNode == ix) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }

        ImGui::TreeNodeEx(obj.name.c_str(), flags);
        if (ImGui::IsItemClicked()) {
            selectedSceneNode = ix;
        }

        ImGui::TreePop();
    }

    if (selectedSceneNode >= 0) {
        auto& obj = sceneObjects[selectedSceneNode];
        ImGui::DragFloat3("Position", &obj.position[0], .1f);
        ImGui::DragFloat3("Rotation", &obj.rotation[0], .1f, -360.f, 360.0f);
        ImGui::DragFloat3("Scale",    &obj.scale[0], .1f, 1.f, .0f);
        ImGui::Checkbox("Visible", &obj.visible);
    }

    ImGui::End();

    ImGui::Begin("Camera properties");

    ImGui::DragFloat("fov", &camera.fov, .1f, 20.f, 180.f);
    ImGui::DragFloat("nearPlane", &camera.zNear, .1f, 1.f, 100.f);
    ImGui::DragFloat("farPlane", &camera.zFar, .1f, 10.f, 1000000.f);

    ImGui::DragFloat3("Position", &camera.pos[0], .1f);
    ImGui::DragFloat3("Rotation", &camera.rotation[0], .1f, -360.f, 360.0f);
    ImGui::End();


    ImGui::Begin("System information");
    ImGui::Text("current framebuffer size: %dx%d", w, h);
    glfwGetWindowSize(_window, &w, &h);
    ImGui::Text("current window size: %dx%d", w, h);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::End();
    ImGui::ShowDemoWindow();

    ImGui::Render();
}

void etna::Renderer::beginRecordCommandBuffer(int index) {
    vk::CommandBufferBeginInfo cmdBuffBeginInfo;
    commandBuffers[index]->begin(cmdBuffBeginInfo);
}

void etna::Renderer::endRecordCommandBuffer(int index) {
    commandBuffers[index]->end();
}

void etna::Renderer::submitCommandBuffer(int index) {
    vk::FenceCreateInfo fenceCreateInfo;
    vk::UniqueFence drawFence = device->createFenceUnique(fenceCreateInfo);

    vk::PipelineStageFlags pipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(
                0, nullptr, &pipelineStageFlags, 1, &*commandBuffers[index], 0, nullptr
                );
    queue.submit({submitInfo}, *drawFence);
    vk::Result res;
    do {
        res = device->waitForFences(1, &*drawFence, true, 1000000000);
    } while (res == vk::Result::eTimeout);
}

void etna::Renderer::draw() {
    spdlog::trace("Prepping clear values");
    std::array clearValues = {
        vk::ClearValue(vk::ClearColorValue()),
        vk::ClearValue(vk::ClearDepthStencilValue(1.f, 0)),
        vk::ClearValue(vk::ClearColorValue())
    };

    vkResetCommandPool(*device, *commandPool, 0);
    updateUniformBuffer();

    vk::SemaphoreCreateInfo semaphoreCreateInfo;
    vk::UniqueSemaphore imageAcquiredSemaphore = device->createSemaphoreUnique(semaphoreCreateInfo);

    spdlog::trace("Acquiring the next image in the swapchain");
    uint32_t currentBuffer;

    try {
        currentBuffer = device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore, vk::Fence()).value;
    }  catch (const vk::OutOfDateKHRError&) {
        recreateSwapChain();
        currentBuffer = device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore, vk::Fence()).value;
    }
    beginRecordCommandBuffer(0);

    vk::RenderPassBeginInfo renderPassBeginInfo(*renderPass, *framebuffers[currentBuffer], vk::Rect2D(vk::Offset2D(), surfaceCapabilities.currentExtent), clearValues.size(), clearValues.data());
    spdlog::trace("Begin render pass");
    commandBuffers[0]->beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

    const vk::DeviceSize offset(0);
    vk::Viewport viewport;
    viewport.width = surfaceCapabilities.currentExtent.width;
    viewport.height = surfaceCapabilities.currentExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    vk::Rect2D scissor;
    scissor.extent = surfaceCapabilities.currentExtent;

    commandBuffers[0]->setViewport(0, 1, &viewport);
    commandBuffers[0]->setScissor(0, 1, &scissor);

    commandBuffers[0]->bindPipeline(vk::PipelineBindPoint::eGraphics, *cubePipeline);
    commandBuffers[0]->bindVertexBuffers( 0, 1, &*mesh.vertexBuffer, &offset);

    for (const auto& object: sceneObjects) {
        if (object.visible) {
            commandBuffers[0]->setPrimitiveTopologyEXT(object.topology, dldi);
            commandBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, 1, &sharedDescriptorSet, 1, &object.uniformBufferOffset);
            commandBuffers[0]->draw(object.numVerts, 1, object.bufferStart, 0);
        }
    }

    commandBuffers[0]->endRenderPass();
    endRecordCommandBuffer(0);
    submitCommandBuffer(0);

    beginRecordCommandBuffer(1);
    commandBuffers[1]->setViewport(0, 1, &viewport);
    commandBuffers[1]->setScissor(0, 1, &scissor);

    vk::RenderPassBeginInfo guiRenderPassBeginInfo(
                *guiRenderPass,
                *guiFramebuffers[currentBuffer],
                vk::Rect2D(vk::Offset2D(), surfaceCapabilities.currentExtent),
                0, nullptr);
    commandBuffers[1]->beginRenderPass(guiRenderPassBeginInfo, vk::SubpassContents::eInline);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffers[1]);
    commandBuffers[1]->endRenderPass();
    endRecordCommandBuffer(1);
    submitCommandBuffer(1);

    vk::PresentInfoKHR presentInfo(
                0, nullptr, 1, &*swapchain, &currentBuffer
                );

    spdlog::trace("Present");
    queue.presentKHR(&presentInfo);
}
