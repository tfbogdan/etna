#pragma once
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <concepts>

namespace etna {

    template <typename T>
    concept VertexWithDisplacementConcept = requires(T& a) {
        { a.displacement } -> std::same_as<glm::vec3&>;
    };

    template <typename T>
    concept VertexWithNormalConcept = requires(T& a) {
        { a.normal } -> std::same_as<glm::vec3&>;
    };

    template <typename T>
    concept VertexWithUVConcept = requires(T& a) {
        { a.uv } -> std::same_as<glm::vec2&>;
    };

    template <typename T>
    concept Vertex = VertexWithDisplacementConcept<T> || VertexWithNormalConcept<T> || VertexWithUVConcept<T>;

    struct ColoredVertex {
        glm::vec3 displacement;
        glm::vec4 color;
    };

    struct TexturedVertex {
        glm::vec3 displacement;
        glm::vec2 uv;
    };


} // ::etna
