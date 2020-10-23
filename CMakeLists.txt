cmake_minimum_required(VERSION 3.14)

project(wakota LANGUAGES CXX)

find_package(PkgConfig REQUIRED)

pkg_check_modules(libinput REQUIRED IMPORTED_TARGET libinput)
pkg_check_modules(libudev REQUIRED IMPORTED_TARGET libudev)
pkg_check_modules(glfw REQUIRED IMPORTED_TARGET glfw3)
pkg_check_modules(wayland-cursor REQUIRED IMPORTED_TARGET wayland-cursor)


find_program(GLSLANG_VALIDATOR NAMES glslangValidator
                         HINTS $ENV{VULKAN_SDK})

if(NOT GLSLANG_VALIDATOR)
     message(SEND_ERROR "Couldn't find glslangValidator.")
endif()

function(compile_shader target shader_file)
    get_filename_component(shaderName ${shader_file} NAME_WE)
    add_custom_command(
            COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file} --vn ${shaderName} -o ${PROJECT_BINARY_DIR}/${shaderName}.h
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}
            OUTPUT ${PROJECT_BINARY_DIR}/${shaderName}.h
            COMMENT "Compiling shader ${shaderName}"
    )
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file} ${PROJECT_BINARY_DIR}/${shaderName}.h)
endfunction()

find_package(Vulkan REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(glm REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME}
    main.cpp
    Vulkan.cpp
    Vulkan.hh
)

compile_shader(${PROJECT_NAME} vertex_shader.vert)
compile_shader(${PROJECT_NAME} fragment_shader.frag)

target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL VK_USE_PLATFORM_XLIB_KHR)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Werror>
)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan fmt::fmt PkgConfig::libinput PkgConfig::libudev spdlog::spdlog PkgConfig::glfw PkgConfig::wayland-cursor)
