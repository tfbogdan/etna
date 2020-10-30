cmake_minimum_required(VERSION 3.14)

project(etna LANGUAGES CXX)


if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(libinput REQUIRED IMPORTED_TARGET libinput)
    pkg_check_modules(libudev REQUIRED IMPORTED_TARGET libudev)
endif()


find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR NAMES glslangValidator
                         HINTS $ENV{VULKAN_SDK})

if(NOT GLSLANG_VALIDATOR)
     message(SEND_ERROR "Couldn't find glslangValidator.")
endif()

macro(compile_shader shader_file)
    get_filename_component(shaderName ${shader_file} NAME_WE)
    file(MAKE_DIRECTORY {PROJECT_CURRENT_BINARY_DIR}/shaders)

    add_custom_command(
            COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_LIST_DIR}/${shader_file} --vn ${shaderName} -o ${PROJECT_BINARY_DIR}/shaders/${shaderName}.h
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}
            OUTPUT ${PROJECT_BINARY_DIR}/${shaderName}.h
            COMMENT "Compiling shader ${shaderName}"
    )
    target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file} ${PROJECT_BINARY_DIR}/${shaderName}.h)
endmacro()


set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME}
)

add_subdirectory(src)
add_subdirectory(dependencies)
add_subdirectory(shaders)

compile_shader(shaders/vertex_shader.vert)
compile_shader(shaders/fragment_shader.frag)

target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_FORCE_LEFT_HANDED)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

#target_compile_options(${PROJECT_NAME} PRIVATE
# $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
# $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -pedantic -Werror>
#)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    Vulkan::Vulkan 
    fmt::fmt 
    spdlog::spdlog 
    glm::glm
    glfw
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} 
        PRIVATE 
        PkgConfig::libinput 
        PkgConfig::libudev 
    )
endif()
