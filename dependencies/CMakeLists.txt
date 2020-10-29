

target_include_directories(etna PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(imgui)
add_subdirectory(tinyobjloader)
add_subdirectory(VulkanMemoryAllocator)
add_subdirectory(stb)

option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
add_subdirectory(glfw)

add_subdirectory(glm)

add_subdirectory(fmt)
option(SPDLOG_FMT_EXTERNAL "Use external fmt library instead of bundled" ON)
add_subdirectory(spdlog)
