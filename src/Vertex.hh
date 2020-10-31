#pragma once
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace etna {

    template <typename T>
    concept Vec3Concept = std::is_same_v<glm::vec3, T>;

    template <typename T>
    concept VertexWithDisplacementConcept = requires(T& a) {
        a.displacement -> Vec3Concept;
    };

    template <typename T>
    concept VertexWithNormalConcept = requires(T& a) {
        a.normal -> Vec3Concept;
    };

    template <typename T>
    concept Vertex = VertexWithDisplacementConcept<T> || VertexWithNormalConcept<T>;

    struct ColoredVertex {
        glm::vec3 displacement;
        glm::vec4 color;
    };

    struct TexturedVertex {
        glm::vec3 displacement;
        glm::vec2 uv;
    };


} // ::etna
