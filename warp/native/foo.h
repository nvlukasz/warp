/** Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"

namespace wp
{

struct Foo
{
    // member variables (will be exposed to kernels, e.g. foo.magic)
    float magic;
    int offset;

    // CUDA context (NULL on CPU)
    void* context;

    CUDA_CALLABLE Foo()
        : magic(0.0f), offset(0), context(nullptr)
    {
    }

    CUDA_CALLABLE Foo(float magic, int offset)
        : magic(magic), offset(offset), context(nullptr)
    {
    }
};

// implements foo = wp.get_foo(id)
CUDA_CALLABLE inline Foo foo_get(uint64_t id)
{
    return *(Foo*)(id);
}

// implements foo[i]
CUDA_CALLABLE inline float extract(const Foo& foo, int i)
{
    return 2.0f * i + foo.magic;
}

// implements foo[i, j]
CUDA_CALLABLE inline float extract(const Foo& foo, int i, int j)
{
    return (float)(i + j + foo.offset);
}

// backward foo[i]
CUDA_CALLABLE inline void adj_extract(const Foo& foo, int i, const Foo& adj_foo, int adj_i, int adj_ret)
{
}

// backward foo[i, j]
CUDA_CALLABLE inline void adj_extract(const Foo& foo, int i, int j, const Foo& adj_foo, int adj_i, int adj_j, int adj_ret)
{
}

}
