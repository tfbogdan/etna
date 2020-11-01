target_include_directories(etna PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(etna PRIVATE
    main.cpp
    Renderer.cpp
    Renderer.hh
    Pipeline.hh
    Pipeline.cc
    GeometryGenerators.hh
    Vertex.hh
    Scene.hh
    UserInterface.hh
    VmaAllocationOwnership.hh
)

rosewood_reflect_on_headers(etna etnaSrcReflection Vertex.hh)
rosewood_reflect_on_headers(etna etnaSrcReflection Scene.hh)
