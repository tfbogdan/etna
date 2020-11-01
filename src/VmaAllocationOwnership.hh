#pragma once
#include <VulkanMemoryAllocator/vk_mem_alloc.h>
#include <algorithm>

namespace etna {

class UniqueVmaAllocation {
public:
    UniqueVmaAllocation() = default;
    UniqueVmaAllocation(VmaAllocator allocator, VmaAllocation allocation)
        : _allocator(allocator),
          _allocation(allocation)
    {}
    UniqueVmaAllocation(const UniqueVmaAllocation&) = delete;
    UniqueVmaAllocation(UniqueVmaAllocation&& mv)
        : _allocator(mv._allocator) {
        std::swap(_allocation, mv._allocation);
    }

    ~UniqueVmaAllocation() {
        clear();
    }

    operator const VmaAllocation& () {
        return _allocation;
    }

    void reset(VmaAllocator allocator, VmaAllocation allocation) {
        clear();
        _allocator = allocator;
        _allocation = allocation;
    }
private:
    void clear() {
        if (_allocator && _allocation) {
            vmaFreeMemory(_allocator, _allocation);
            _allocator = nullptr;
            _allocation = nullptr;
        }
    }

    VmaAllocator _allocator = nullptr;
    VmaAllocation _allocation = nullptr;
};

}
