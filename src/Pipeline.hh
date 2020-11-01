#pragma once

#include <vulkan/vulkan.hpp>

#include <Vertex.hh>
#include <Vertex.metadata.h>

#include <array>
#include <concepts>
#include <ranges>

namespace etna {

    template <typename fieldT>
    struct CompatibleFormatForField;

    template <>
    struct CompatibleFormatForField<glm::vec2> {
        constexpr static vk::Format format = vk::Format::eR32G32Sfloat;
    };

    template <>
    struct CompatibleFormatForField<glm::vec3> {
        constexpr static vk::Format format = vk::Format::eR32G32B32Sfloat;
    };

    template <typename T>
    concept TypeWithCompatibleVKFormat = requires {
        { CompatibleFormatForField<T>::format } -> std::same_as<const vk::Format&>;
    };

    template <Vertex vertexType>
    struct VertexInputAttributeGenerator {
        using meta_vertex_type = rosewood::meta<vertexType>;
        using attrib_type = std::array<vk::VertexInputAttributeDescription, std::tuple_size_v<decltype(meta_vertex_type::fields)>>;

        constexpr static attrib_type makeViAttribs() {
            meta_vertex_type metaVertex;
            attrib_type result;

            metaVertex.visit_fields([&result]<typename field_type, typename class_type>(rosewood::FieldDeclaration<field_type, class_type> field) {
                result[field.index] = vk::VertexInputAttributeDescription(field.index, 0, CompatibleFormatForField<field_type>::format, field.offset);
            });
            return result;
        }
        static constexpr attrib_type vertexInputAttributes = makeViAttribs();
        static constexpr std::array vertexInputBindingDescription {
            vk::VertexInputBindingDescription(0, sizeof(vertexType), vk::VertexInputRate::eVertex)
        };
    };

    template <Vertex vertexType>
    class PipelineFactory {

    };
}
