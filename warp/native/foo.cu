/** Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "foo.h"
#include "warp.h"
#include "cuda_util.h"

#include <unordered_map>

namespace
{

// host-side copy of Foo descriptors, maps GPU Foo id (address) to the CPU desc
std::unordered_map<uint64_t, wp::Foo> g_foo_descriptors;

bool foo_get_descriptor(uint64_t id, wp::Foo& foo)
{
    auto iter = g_foo_descriptors.find(id);
    if (iter != g_foo_descriptors.end())
    {
        foo = iter->second;
        return true;
    }
    return false;
}

void foo_add_descriptor(uint64_t id, const wp::Foo& foo)
{
    g_foo_descriptors[id] = foo;
}

void foo_rem_descriptor(uint64_t id)
{
    g_foo_descriptors.erase(id);
}

} // end anonymous namespace


// API definitions
extern "C"
{

WP_API uint64_t foo_create_device(void* context, float magic, int offset)
{
    // use the specified CUDA context
    ContextGuard guard(context);

    // create Foo descriptor
    wp::Foo foo(magic, offset);

    // remember the context
    foo.context = context ? context : cuda_context_get_current();

    // allocate and copy to the GPU
    void* foo_d = alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::Foo));
    memcpy_h2d(WP_CURRENT_CONTEXT, foo_d, &foo, sizeof(wp::Foo));

    // save descriptor
    uint64_t id = (uint64_t)foo_d;
    foo_add_descriptor(id, foo);

    return id;
}

WP_API void foo_destroy_device(void* context, uint64_t id)
{
    wp::Foo foo;
    if (foo_get_descriptor(id, foo))
    {
        // use the specified CUDA context
        ContextGuard guard(foo.context);

        // free GPU memory
        free_device(WP_CURRENT_CONTEXT, (void*)id);

        // remove descriptor
        foo_rem_descriptor(id);
    }
}

} // end of extern "C"
