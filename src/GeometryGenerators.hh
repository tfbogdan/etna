#pragma once

#include <glm/vec3.hpp>
#include <array>
#include <span>

#include "Vertex.hh"

namespace etna::gg {
    constexpr std::array unitCubeVertices{
        glm::vec3(-.5f, -.5f,  .5f),
        glm::vec3(-.5f,  .5f,  .5f),
        glm::vec3( .5f, -.5f,  .5f),
        glm::vec3( .5f, -.5f,  .5f),
        glm::vec3(-.5f,  .5f,  .5f),
        glm::vec3( .5f,  .5f,  .5f),
        glm::vec3(-.5f, -.5f, -.5f),
        glm::vec3( .5f, -.5f, -.5f),
        glm::vec3(-.5f,  .5f, -.5f),
        glm::vec3(-.5f,  .5f, -.5f),
        glm::vec3( .5f, -.5f, -.5f),
        glm::vec3( .5f,  .5f, -.5f),
        glm::vec3(-.5f,  .5f,  .5f),
        glm::vec3(-.5f, -.5f,  .5f),
        glm::vec3(-.5f,  .5f, -.5f),
        glm::vec3(-.5f,  .5f, -.5f),
        glm::vec3(-.5f, -.5f,  .5f),
        glm::vec3(-.5f, -.5f, -.5f),
        glm::vec3( .5f,  .5f,  .5f),
        glm::vec3( .5f,  .5f, -.5f),
        glm::vec3( .5f, -.5f,  .5f),
        glm::vec3( .5f, -.5f,  .5f),
        glm::vec3( .5f,  .5f, -.5f),
        glm::vec3( .5f, -.5f, -.5f),
        glm::vec3( .5f,  .5f,  .5f),
        glm::vec3(-.5f,  .5f,  .5f),
        glm::vec3( .5f,  .5f, -.5f),
        glm::vec3( .5f,  .5f, -.5f),
        glm::vec3(-.5f,  .5f,  .5f),
        glm::vec3(-.5f,  .5f, -.5f),
        glm::vec3( .5f, -.5f,  .5f),
        glm::vec3( .5f, -.5f, -.5f),
        glm::vec3(-.5f, -.5f,  .5f),
        glm::vec3(-.5f, -.5f,  .5f),
        glm::vec3( .5f, -.5f, -.5f),
        glm::vec3(-.5f, -.5f, -.5f)
    };

    template <Vertex vertexT>
    void ggCube(std::span<vertexT> buffer, float sideLength = 1.f) {
        for (int ix = 0; ix < std::ssize(unitCubeVertices); ++ix) {
            buffer[ix].displacement = unitCubeVertices[ix] * sideLength;
        }
    }

    void ggCilinder();
} // ::etna::gg
