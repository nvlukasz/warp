/** Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "foo.h"

// API definitions
extern "C"
{

WP_API uint64_t foo_create_host(float magic, int offset)
{
    wp::Foo* foo = new wp::Foo(magic, offset);
    return (uint64_t)foo;
}

WP_API void foo_destroy_host(uint64_t id)
{
    wp::Foo* foo = (wp::Foo*)id;
    delete foo;
}

// stubs for non-CUDA platforms
#if !WP_ENABLE_CUDA

WP_API uint64_t foo_create_device(void* context, float magic, int offset)
{
    return 0;
}

WP_API void foo_destroy_device(void* context, uint64_t id)
{
}

// end stubs
#endif

} // end extern "C"
