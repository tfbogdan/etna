#pragma once

#include <glm/glm.hpp>
#include <concepts>
#include <rosewood/rosewood.hpp>
#include <imgui/imgui.h>

namespace etna {

template <typename T>
concept ReflectedType = !std::same_as<T, rosewood::nil_t>;

template <ReflectedType T>
void ImGui_Build(T& obj) {
    auto meta_object = rosewood::meta<T>{};

    meta_object.visit_bases_with_metaobjects([&obj]<typename meta_base>(meta_base){
        ImGui_Build(static_cast<typename meta_base::type&>(obj));
    });

    meta_object.visit_fields([&obj]<typename fieldType>(rosewood::FieldDeclaration<fieldType, T> field) {
        auto& fieldRef = obj.*field.address;
        if constexpr(std::is_same_v<fieldType, glm::vec3>) {
            ImGui::DragFloat3(field.name.data(), &fieldRef[0], 0.1);
        } else if constexpr (std::is_same_v<fieldType, bool>) {
            ImGui::Checkbox(field.name.data(), &fieldRef);
        }
    });
}

}
