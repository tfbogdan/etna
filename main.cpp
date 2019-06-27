// #include <QGuiApplication>
// #include <QQmlApplicationEngine>
#include <iostream>
#include <vector>

#include <vulkan.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

int main(int, char**) {
    wkt::Vulkan vulkan;

    spdlog::set_level(spdlog::level::level_enum::trace);
    spdlog::flush_on(spdlog::level::level_enum::trace);

    try {
        vulkan.initialize();
        vulkan.loop();
    } catch (const std::exception& e) {
        spdlog::critical(e.what());
    }

    return 0;
}
