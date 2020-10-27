// #include <QGuiApplication>
// #include <QQmlApplicationEngine>
#include <iostream>
#include <vector>
#include <thread>

#include <Renderer.hh>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <libinput.h>
#include <libudev.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/kd.h>
#include <linux/input-event-codes.h>

#include <unistd.h>
#include <fcntl.h>

void glfwKeyCB(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void glfwWindowResizedCB(GLFWwindow* window, int /*w*/, int /*h*/) {
    auto instance = static_cast<etna::Renderer*>(glfwGetWindowUserPointer(window));
    instance->recreateSwapChain();
}

int open_restricted(const char *path, int flags, void */*user_data*/) {
    int fd = open(path, flags);
    spdlog::debug("Opening event device {}; Result: {}", path, fd);
    return fd < 0 ? -errno : fd;
}

void close_restricted(int fd, void */*user_data*/) {
    close(fd);
}

struct libinput *li = nullptr;
struct udev *udev = nullptr;

const static struct libinput_interface interface = {
    open_restricted, close_restricted
};

void initInput() {
    udev = udev_new();
    li = libinput_udev_create_context(&interface, nullptr, udev);

    libinput_udev_assign_seat(li, "seat0");
}

bool doneLooping = false;

void directInputLoop() {
    while(!doneLooping) {
        libinput_dispatch(li);
        struct libinput_event *event = nullptr;
        event = libinput_get_event(li);

        if (event) {
            auto evType = libinput_event_get_type(event);

            switch(evType) {
            case LIBINPUT_EVENT_POINTER_MOTION:
            {
                //                auto pointer_event = libinput_event_get_pointer_event(event);
                //                auto dx = libinput_event_pointer_get_dx(pointer_event);
                //                auto dy = libinput_event_pointer_get_dy(pointer_event);

                //                mesh.xRot += dx / 500.f;
                //                mesh.yRot += dy / 500.f;

            } break;
            case LIBINPUT_EVENT_POINTER_BUTTON:
                break;
            case LIBINPUT_EVENT_KEYBOARD_KEY:
            {
                auto keyEv = libinput_event_get_keyboard_event(event);
                auto key = libinput_event_keyboard_get_key(keyEv);
                auto state = libinput_event_keyboard_get_key_state(keyEv);
                spdlog::debug("Got a key event with keycode: {} in state {}", key, state);
            } break;
            default:
                break;
            }
            spdlog::debug("Handling event of type {}", evType);
            libinput_event_destroy(event);
        }
    }
}


int main(int, char**) {
    etna::Renderer vulkan;

    spdlog::set_level(spdlog::level::level_enum::info);
    spdlog::flush_on(spdlog::level::level_enum::info);

    GLFWwindow *window = nullptr;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(1920, 1080, "Etna", nullptr, nullptr);

    glfwSetWindowUserPointer(window, &vulkan);
    glfwSetKeyCallback(window, &glfwKeyCB);
    glfwSetWindowSizeCallback(window, &glfwWindowResizedCB);

    vulkan.initialize(window);
    initInput();

    glfwShowWindow(window);

    std::thread input_thread(directInputLoop);

    try {
        while(!doneLooping) {
            if (window) {
                if (glfwWindowShouldClose(window)) {
                    doneLooping = true;
                }
                glfwPollEvents();
            }
            vulkan.draw();
        }
    } catch (const std::exception& e) {
        spdlog::critical(e.what());
    }

    input_thread.join();

    if (window) {
        glfwDestroyWindow(window);
    }

    if (li) {
        libinput_unref(li);
    }

    if (udev) {
        udev_unref(udev);
    }

    return 0;
}
