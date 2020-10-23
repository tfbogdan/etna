// #include <QGuiApplication>
// #include <QQmlApplicationEngine>
#include <iostream>
#include <vector>
#include <thread>

#include <Vulkan.hh>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

int main(int, char**) {
    wkt::Vulkan vulkan;

    spdlog::set_level(spdlog::level::level_enum::info);
    spdlog::flush_on(spdlog::level::level_enum::info);

    vulkan.initialize();

    std::thread input_thread([&vulkan](){
        vulkan.loop_input();
    });
    try {
        vulkan.loop();
    } catch (const std::exception& e) {
        spdlog::critical(e.what());
    }

    input_thread.join();

    return 0;
}
