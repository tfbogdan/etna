#include "Renderer.hh"

#include <iostream>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <fstream>
#include <set>
#include <map>

#include <shaders/vertex_shader.h>
#include <shaders/fragment_shader.h>

#include <chrono>
#include <filesystem>

#include <imgui/imgui_impl_vulkan.h>
#include <imgui/imgui_impl_glfw.h>
#include <stb/stb_image.h>

#include <tinyobjloader/tiny_obj_loader.h>

#include "Pipeline.hh"
#include "UserInterface.hh"
#include <Scene.metadata.h>

constexpr std::array instanceExtensions = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME
};

constexpr std::array deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

constexpr std::array validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

etna::Renderer::~Renderer() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void etna::Renderer::initialize(GLFWwindow* window) {
    _window = window;

    spdlog::debug("createInstance");
    createInstance();

    spdlog::debug("initSurface");
    initSurface(window);

    spdlog::debug("initDevice");
    initDevice();

    initSampler();

    spdlog::debug("initCommandBuffer");
    initCommandBuffer();

    spdlog::debug("initPipelineLayout");
    initPipelineLayout();

    initDescriptorPool();

    spdlog::debug("initVertexBuffers");
    initScene();

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

    initGui();
}


VkBool32 vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                             VkDebugUtilsMessageTypeFlagsEXT             /*messageTypes*/,
                             const VkDebugUtilsMessengerCallbackDataEXT* cbData,
                             void*                                       /*userData*/) {
    switch(messageSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        spdlog::error("{}:{}", cbData->pMessageIdName, cbData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        spdlog::warn("{}:{}", cbData->pMessageIdName, cbData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        spdlog::info("{}:{}", cbData->pMessageIdName, cbData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        spdlog::trace("{}:{}", cbData->pMessageIdName, cbData->pMessage);
        break;
    default: break;
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

#if _DEBUG
    vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, validationLayers, requiredExtensions);
#else
    vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, {}, requiredExtensions);
#endif

    auto layers = vk::enumerateInstanceLayerProperties();
    instance = vk::createInstanceUnique(instanceCreateInfo);
    dldi = vk::DispatchLoaderDynamic(*instance, vkGetInstanceProcAddr);

    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo(
                {}, vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
                vulkanDebugCallback
                );

    debugUtilsMessenger = instance->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo, nullptr, dldi);
    gpus = instance->enumeratePhysicalDevices();
    // TDO: determine a suitable candidate when more than 1 GPU is available
    gpu = gpus[0];
}

void etna::Renderer::transitionImageLayout(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    immediateCommandBuffer([image, oldLayout, newLayout] (vk::CommandBuffer commandBuffer){
        vk::ImageMemoryBarrier barrier;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, std::array{barrier});
    });

}

void etna::Renderer::recreateSwapChain() {
    device->waitIdle();

    surfaceCapabilities = gpu.getSurfaceCapabilitiesKHR(*wndSurface);

    initSwapchain();
    initDepthBuffer();
    initFramebuffers();
    initGuiFramebuffers();
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
    std::array queuePriorities { .0f };
    std::array queueCreateInfos {
        vk::DeviceQueueCreateInfo({}, queueFamily, queuePriorities)
    };

    vk::PhysicalDeviceFeatures deviceFeatures = {};
    deviceFeatures.sampleRateShading = true;
    deviceFeatures.samplerAnisotropy = true;
#if _DEBUG
    vk::DeviceCreateInfo deviceInfo({}, queueCreateInfos, validationLayers, deviceExtensions, &deviceFeatures);
#else
    vk::DeviceCreateInfo deviceInfo({}, queueCreateInfos, {}, deviceExtensions, &deviceFeatures);
#endif
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
    VmaAllocator stagingAllocator;
    vmaCreateAllocator(&allocInfo, &stagingAllocator); // TODO: check result
    allocator.reset(stagingAllocator);

    queue = device->getQueue(queueFamily, 0);
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
    vk::CommandBufferAllocateInfo cmdBuffAlocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 2);
    commandBuffers = device->allocateCommandBuffersUnique(cmdBuffAlocInfo);
}

void etna::Renderer::initSwapchain() {
    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.surface               = *wndSurface;
    swapchainInfo.minImageCount         = std::max(2u, surfaceCapabilities.minImageCount);
    swapchainInfo.imageFormat           = vk::Format::eB8G8R8A8Unorm;
    swapchainInfo.imageColorSpace       = vk::ColorSpaceKHR::eSrgbNonlinear;
    swapchainInfo.imageExtent           = surfaceCapabilities.currentExtent;
    swapchainInfo.imageArrayLayers      = 1;
    swapchainInfo.imageUsage            = vk::ImageUsageFlagBits::eColorAttachment;
    swapchainInfo.imageSharingMode      = vk::SharingMode::eExclusive;
    swapchainInfo.setQueueFamilyIndices(std::array{queueFamily});
    swapchainInfo.preTransform          = surfaceCapabilities.currentTransform;
    swapchainInfo.compositeAlpha        = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapchainInfo.presentMode           = presentMode;

    swapchain = device->createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = device->getSwapchainImagesKHR(*swapchain);
    swapchainImageViews.clear();

    for(auto &image: swapchainImages) {
        vk::ImageViewCreateInfo viewInfo( {}, image, vk::ImageViewType::e2D,vk::Format::eB8G8R8A8Unorm,
                                          vk::ComponentMapping(vk::ComponentSwizzle::eIdentity),
                                          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        swapchainImageViews.emplace_back(device->createImageViewUnique(viewInfo));
    }
}

void etna::Renderer::initDepthBuffer() {
    vk::ImageCreateInfo imageCreateInfo(
                {}, vk::ImageType::e2D, depthBuffer.format, vk::Extent3D(surfaceCapabilities.currentExtent, 1), 1, 1,
                getMaxUsableSampleCount(), vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);

    depthBuffer.image = device->createImageUnique(imageCreateInfo);
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VmaAllocation stagingAlloc;
    vmaAllocateMemoryForImage(allocator.get(), *depthBuffer.image, &allocCreateInfo, &stagingAlloc, nullptr);
    vmaBindImageMemory(allocator.get(), stagingAlloc, *depthBuffer.image);
    depthBuffer.vmaAlloc.reset(allocator.get(), stagingAlloc);

    vk::ImageViewCreateInfo imageViewCreateInfo({}, *depthBuffer.image, vk::ImageViewType::e2D, depthBuffer.format, {},
                                                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
    depthBuffer.imageView = device->createImageViewUnique(imageViewCreateInfo);

    vk::ImageCreateInfo resolveBufferInfo(
                {}, vk::ImageType::e2D, vk::Format::eB8G8R8A8Unorm, vk::Extent3D(surfaceCapabilities.currentExtent, 1), 1, 1,
                getMaxUsableSampleCount(), vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);

    resolveBuffer.image = device->createImageUnique(resolveBufferInfo);
    vmaAllocateMemoryForImage(allocator.get(), *resolveBuffer.image, &allocCreateInfo, &stagingAlloc, nullptr);
    vmaBindImageMemory(allocator.get(), stagingAlloc, *resolveBuffer.image);
    resolveBuffer.vmaAlloc.reset(allocator.get(), stagingAlloc);

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
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo allocationInfo = {};
    VmaAllocation stagingAlloc;
    vmaAllocateMemoryForBuffer(allocator.get(), *sharedUniformBuffer, &allocInfo, &stagingAlloc, &allocationInfo);
    sharedUboMappedMemory = reinterpret_cast<std::byte*>(allocationInfo.pMappedData);
    sharedUniformAllocation.reset(allocator.get(), stagingAlloc);
    vmaBindBufferMemory(allocator.get(), sharedUniformAllocation, *sharedUniformBuffer);

    // Calculate required alignment based on minimum device offset alignment
    size_t minUboAlignment = gpu.getProperties().limits.minUniformBufferOffsetAlignment;
    size_t dynamicAlignment = sizeof(UniformBuffer);
    if (minUboAlignment > 0) {
        dynamicAlignment = (dynamicAlignment + minUboAlignment - 1) & ~(minUboAlignment - 1);
    }

    for (int ix = 0; ix < std::ssize(sceneObjects); ++ix) {
        auto& obj = sceneObjects[ix];

        vk::DescriptorImageInfo descImageInfo(*textureSampler, *obj.textureView, vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*descriptorPool, 1, &*descSetLayout);
        obj.samplerDescriptor = device->allocateDescriptorSets(descriptorSetAllocateInfo).front();

        vk::DescriptorBufferInfo sharedBufferInfo;
        sharedBufferInfo.buffer = *sharedUniformBuffer;
        sharedBufferInfo.offset = ix * dynamicAlignment;
        sharedBufferInfo.range = sizeof(UniformBuffer);
        obj.uniformBufferOffset = ix * dynamicAlignment;

        std::array writes{
            vk::WriteDescriptorSet(obj.samplerDescriptor, 0, 0, 1, vk::DescriptorType::eUniformBufferDynamic, nullptr, &sharedBufferInfo, nullptr),
            vk::WriteDescriptorSet(obj.samplerDescriptor, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descImageInfo, nullptr, nullptr)
        };
        device->updateDescriptorSets(writes, {});

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
    for (const auto& obj: sceneObjects) {

        auto& objUbo = *reinterpret_cast<UniformBuffer*>(sharedUboMappedMemory + obj.uniformBufferOffset);
        const auto& M = glm::translate(glm::scale(glm::mat4(1.f), obj.scale), obj.position) * glm::eulerAngleXYZ(
                    glm::radians(obj.rotation.x),
                    glm::radians(obj.rotation.y),
                    glm::radians(obj.rotation.z)
                    );

        objUbo.mvp = world.P * world.V * M;
    }
}

void etna::Renderer::initPipelineLayout() {
    std::array sharedLayoutBindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment)
    };
    vk::DescriptorSetLayoutCreateInfo sharedDescriptorLayoutInfo({}, sharedLayoutBindings);
    descSetLayout = device->createDescriptorSetLayoutUnique(sharedDescriptorLayoutInfo);
    std::array descSetLayouts {*descSetLayout};
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo({}, descSetLayouts);
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
    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo({},1000 * descPoolSizes.size(), descPoolSizes.size(), descPoolSizes.data());
    descriptorPool = device->createDescriptorPoolUnique(descriptorPoolCreateInfo);
}

void etna::Renderer::initSharedDescriptorSet() {
    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*descriptorPool, 1, &*descSetLayout);
    sharedDescriptorSet = device->allocateDescriptorSets(descriptorSetAllocateInfo).front();

    vk::DescriptorBufferInfo sharedBufferInfo;
    sharedBufferInfo.buffer = *sharedUniformBuffer;
    sharedBufferInfo.offset = 0;
    sharedBufferInfo.range = VK_WHOLE_SIZE;

    vk::WriteDescriptorSet write(sharedDescriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBufferDynamic, nullptr, &sharedBufferInfo, nullptr);
    device->updateDescriptorSets(1, &write, 0, nullptr);
}

void etna::Renderer::initRenderPass() {
    std::array attachmentDescriptions {
        vk::AttachmentDescription({}, vk::Format::eB8G8R8A8Unorm, getMaxUsableSampleCount(), vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal),
                vk::AttachmentDescription({}, depthBuffer.format, getMaxUsableSampleCount(), vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal),
                vk::AttachmentDescription({}, vk::Format::eB8G8R8A8Unorm, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal)
    };

    vk::AttachmentReference resolvReference(2, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);

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
    std::array attachmentDescriptions {vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), vk::Format::eB8G8R8A8Unorm, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR)};

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
                {},
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
        std::array attachments { *resolveBuffer.imageView, *depthBuffer.imageView, *swapchainImageViews[idx] };

        vk::FramebufferCreateInfo framebufferCreateInfo( {}, *renderPass, attachments,
                    surfaceCapabilities.currentExtent.width, surfaceCapabilities.currentExtent.height, 1);
        framebuffers.emplace_back(device->createFramebufferUnique(framebufferCreateInfo));
    }
}

void etna::Renderer::initGuiFramebuffers() {
    guiFramebuffers.clear();
    for (uint32_t idx(0); idx < swapchainImages.size(); ++idx) {
        std::array attachments {*swapchainImageViews[idx]};

        vk::FramebufferCreateInfo framebufferCreateInfo( {}, *guiRenderPass, attachments.size(), attachments.data(),
                    surfaceCapabilities.currentExtent.width, surfaceCapabilities.currentExtent.height, 1);
        guiFramebuffers.emplace_back(device->createFramebufferUnique(framebufferCreateInfo));
    }

}

// colors per face:
constexpr std::array faceColors {
    glm::vec4(1.f, 0.f, 0.f, 1.f),
    glm::vec4(0.f, 1.f, 0.f, 1.f),
    glm::vec4(0.f, 0.f, 1.f, 1.f),
    glm::vec4(1.f, 1.f, 0.f, 1.f),
    glm::vec4(1.f, 0.f, 1.f, 1.f),
    glm::vec4(0.f, 1.f, 1.f, 1.f)
};

void etna::Renderer::initScene() {
    auto vikingRoomPath = getenv("VIKING_ROOM_OBJ");

    if (vikingRoomPath) {
        loadObj(vikingRoomPath);
    }
}

void etna::Renderer::initSampler() {
    vk::SamplerCreateInfo samplerCreate;
    samplerCreate.magFilter = vk::Filter::eLinear;
    samplerCreate.minFilter = vk::Filter::eLinear;
    samplerCreate.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerCreate.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerCreate.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerCreate.anisotropyEnable = true;
    samplerCreate.maxAnisotropy = 12.f;
    samplerCreate.borderColor = vk::BorderColor::eIntOpaqueWhite;
    samplerCreate.unnormalizedCoordinates = false;
    samplerCreate.compareEnable = false;
    samplerCreate.compareOp = vk::CompareOp::eAlways;
    samplerCreate.mipmapMode = vk::SamplerMipmapMode::eLinear;
    textureSampler = device->createSamplerUnique(samplerCreate);
}

void etna::Renderer::loadObj(const std::filesystem::path &path) {
    assert(std::filesystem::exists(path));
    std::vector<etna::TexturedVertex> vertices;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warns;
    std::string errors;

    tinyobj::LoadObj(&attrib, &shapes, &materials, &warns, &errors, path.string().c_str());
    if (!warns.empty()) spdlog::warn(warns);
    if (!errors.empty()) spdlog::error(errors);

    for (const auto& shape : shapes) {
        for ([[maybe_unused]]const auto& index : shape.mesh.indices) {
            TexturedVertex vertex{};

            vertex.displacement = {
                attrib.vertices[3 * index.vertex_index + 2] * 5,
                attrib.vertices[3 * index.vertex_index + 1] * 5,
                attrib.vertices[3 * index.vertex_index + 0] * 5
            };

            vertex.uv = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.f - attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertices.push_back(vertex);
        }
    }

    vk::BufferCreateInfo bufferCreateInfo({}, sizeof(TexturedVertex) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer);

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vmaCreateBuffer(allocator.get(), &static_cast<VkBufferCreateInfo&>(bufferCreateInfo), &allocCreateInfo, &stagingBuffer, &stagingAlloc, nullptr);
    std::byte *vMem;

    vmaMapMemory(allocator.get(), stagingAlloc, reinterpret_cast<void**>(&vMem));
    memcpy(vMem, vertices.data(), sizeof(TexturedVertex) * vertices.size());
    vmaUnmapMemory(allocator.get(), stagingAlloc);

    SceneObjectRendererData meshObj;
    meshObj.vertexBuffer = vk::UniqueBuffer(stagingBuffer, *device);

    meshObj.bufferAllocation.reset(allocator.get(), stagingAlloc);
    meshObj.bufferStart = 0;
    meshObj.numVerts = std::ssize(vertices);
    meshObj.visible = true;
    meshObj.name = path.filename().string();

    auto imagePath = path;
    imagePath.replace_extension(".png");
    loadTexture(imagePath.c_str(), meshObj);

    sceneObjects.emplace_back(std::move(meshObj));
}

void etna::Renderer::loadTexture(const std::filesystem::path &path, etna::SceneObjectRendererData &obj) {
    assert(std::filesystem::exists(path));
    int w, h, channels;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &w, &h, &channels, STBI_rgb_alpha);

    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vk::BufferCreateInfo bufferCreate; bufferCreate
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
            .setSize(w * h * 4);
    VmaAllocationCreateInfo bufferAlloc = {};
    bufferAlloc.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    bufferAlloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo allocInfo = {};
    vmaCreateBuffer(allocator.get(), &static_cast<VkBufferCreateInfo&>(bufferCreate), &bufferAlloc, &stagingBuffer, &stagingAlloc, &allocInfo);
    memcpy(allocInfo.pMappedData, pixels, w * h * 4);
    stbi_image_free(pixels);

    VkImage stagingImage;
    VmaAllocation stagingImageAlloc;
    VmaAllocationCreateInfo imageAllocInfo = {};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vk::ImageCreateInfo imageCreate; imageCreate
            .setFormat(vk::Format::eR8G8B8A8Srgb)
            .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
            .setTiling(vk::ImageTiling::eOptimal)
            .setImageType(vk::ImageType::e2D)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setMipLevels(1)
            .setArrayLayers(1)
            .setExtent(vk::Extent3D(w, h, 1))
            .setInitialLayout(vk::ImageLayout::eUndefined);

    vmaCreateImage(allocator.get(), &static_cast<VkImageCreateInfo&>(imageCreate), &imageAllocInfo, &stagingImage, &stagingImageAlloc, nullptr);
    transitionImageLayout(stagingImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    immediateCommandBuffer([w, h, stagingBuffer, stagingImage](auto commandBuffer){
        vk::BufferImageCopy imgCopy; imgCopy
                .setImageExtent(vk::Extent3D(w, h, 1));
                imgCopy.imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setLayerCount(1);
        commandBuffer.copyBufferToImage(stagingBuffer, stagingImage, vk::ImageLayout::eTransferDstOptimal, std::array{imgCopy});
    });
    transitionImageLayout(stagingImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::ImageViewCreateInfo imgViewInfo;
    imgViewInfo
            .setFormat(vk::Format::eR8G8B8A8Srgb)
            .setViewType(vk::ImageViewType::e2D)
            .setImage(stagingImage)
            .subresourceRange
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLevelCount(1)
                .setLayerCount(1)
                .setBaseMipLevel(0)
                .setBaseArrayLayer(0);

    obj.textureView = device->createImageViewUnique(imgViewInfo);
    obj.textureImage = vk::UniqueImage(stagingImage, *device);
    obj.textureAllocation.reset(allocator.get(), stagingImageAlloc);

    vmaDestroyBuffer(allocator.get(), stagingBuffer, stagingAlloc);
}

void etna::Renderer::spawnCube() {

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
    std::array dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicStateInfo;
    dynamicStateInfo.dynamicStateCount = dynamicStates.size();
    dynamicStateInfo.pDynamicStates = dynamicStates.data();
    using iag = VertexInputAttributeGenerator<TexturedVertex>;
    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo({}, iag::vertexInputBindingDescription, iag::vertexInputAttributes);
    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList);

    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo({}, false, false, vk::PolygonMode::eFill,
                                                vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise,false, 0, 0, 0, 1.f);

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
    pipelineColorBlendStateCreateInfo.setBlendConstants({1.f, 1.f, 1.f ,1.f});

    std::array vps = { vk::Viewport{} };
    std::array scissors = { vk::Rect2D{} };
    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo({}, vps, scissors);

    vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilCreateInfo;
    pipelineDepthStencilCreateInfo.depthTestEnable = true;
    pipelineDepthStencilCreateInfo.depthWriteEnable = true;
    pipelineDepthStencilCreateInfo.depthCompareOp = vk::CompareOp::eLess;
    pipelineDepthStencilCreateInfo.depthBoundsTestEnable = false;
    pipelineDepthStencilCreateInfo.stencilTestEnable = false;

    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleCreateInfo({}, getMaxUsableSampleCount(), true, 1.f, nullptr, false, false);

    std::array shaderStages = {
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vertexShader, "main", nullptr),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fragmentShader, "main", nullptr)
    };

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo(
                {}, shaderStages, &pipelineVertexInputStateCreateInfo,
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
    coloredVertexPipeline = device->createGraphicsPipelineUnique(vk::PipelineCache(), graphicsPipelineCreateInfo).value;
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

    imguiInitInfo.MinImageCount = std::max(2u, surfaceCapabilities.minImageCount);
    imguiInitInfo.ImageCount = swapchainImageViews.size();

    imguiInitInfo.CheckVkResultFn = nullptr;
    ImGui_ImplVulkan_Init(&imguiInitInfo, *guiRenderPass);
    immediateCommandBuffer([](auto commandBuffer) {
        ImGui_ImplVulkan_CreateFontsTexture(commandBuffer);
    });
}

void etna::Renderer::buildGui() {
    int w, h;
    glfwGetFramebufferSize(_window, &w, &h);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("Scene")) {
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
            ImGui_Build(obj);
//            ImGui::DragFloat3("Position", &obj.position[0], .1f);
//            ImGui::DragFloat3("Rotation", &obj.rotation[0], .1f, -360.f, 360.0f);
//            ImGui::DragFloat3("Scale",    &obj.scale[0], .1f, 1.f, .0f);
//            ImGui::Checkbox("Visible", &obj.visible);
        }
    }

    ImGui::End();

    ImGui::Begin("Camera properties");

    ImGui::DragFloat("fov", &camera.fov, .1f, 20.f, 180.f);
    ImGui::DragFloat("near", &camera.zNear, .1f, 1.f, 100.f);

    ImGui::DragFloat3("Position", &camera.pos[0], .1f);
    ImGui::DragFloat3("Rotation", &camera.rotation[0], .1f, -360.f, 360.0f);
    ImGui::End();


    ImGui::Begin("System information");
    ImGui::Text("current framebuffer size: %dx%d", w, h);
    glfwGetWindowSize(_window, &w, &h);
    ImGui::Text("current window size: %dx%d", w, h);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    vmaCalculateStats(allocator.get(), &vmaStats);
    ImGui::Text("%ldMb of GPU memory used", vmaStats.total.usedBytes / 1024 / 1024);
    ImGui::End();

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
    vk::SubmitInfo submitInfo( 0, nullptr, &pipelineStageFlags, 1, &*commandBuffers[index], 0, nullptr);
    queue.submit({submitInfo}, *drawFence);
    vk::Result res;
    do {
        res = device->waitForFences({*drawFence}, true, std::numeric_limits<uint64_t>::max());
    } while (res == vk::Result::eTimeout);
}

void etna::Renderer::draw() {
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

    commandBuffers[0]->bindPipeline(vk::PipelineBindPoint::eGraphics, *coloredVertexPipeline);

    for (const auto& object: sceneObjects) {
        if (object.visible) {
            commandBuffers[0]->bindVertexBuffers( 0, 1, &*object.vertexBuffer, &offset);
            const std::array objectDescSets     {object.samplerDescriptor};
            const std::array dynamicUBOOffsets  {object.uniformBufferOffset};
            commandBuffers[0]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, objectDescSets, dynamicUBOOffsets);
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
