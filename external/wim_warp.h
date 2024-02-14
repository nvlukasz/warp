#pragma once

// TODO: may need to add a mechanism for include paths
#include "wim/wim.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using Color = ::wim::Color;
using Coord = ::wim::Coord;
using Image = ::wim::Image;

// Coord constructor exposed as a free function
CUDA_CALLABLE inline Coord Coord_(int x, int y)
{
    return Coord(x, y);
}

// overload operator+ for colors
CUDA_CALLABLE inline Color add(const Color& a, const Color& b)
{
    return Color(a.r + b.r, a.g + b.g, a.b + b.b);
}

// overload operator* for scaling colors
CUDA_CALLABLE inline Color mul(float s, const Color& c)
{
    return Color(s * c.r, s * c.g, s * c.b);
}

//
// TODO: Integer handles don't play well with polymorphism or explicit overloading for different Image subclasses.
// Would be better to pass specific types, pointers, or references.
//

// get image pointer from handle
CUDA_CALLABLE inline Image& img_get(uint64_t handle)
{
    return *(Image*)(handle);
}

// get image width (can't be exposed as a named var directly, because the member is private)
CUDA_CALLABLE inline int img_width(uint64_t handle)
{
    return img_get(handle).getWidth();
}

// get image height (can't be exposed as a named var directly, because the member is private)
CUDA_CALLABLE inline int img_height(uint64_t handle)
{
    return img_get(handle).getHeight();
}

// get pixel
CUDA_CALLABLE inline Color img_get_pixel(uint64_t handle, const Coord& coord)
{
    return img_get(handle).getPixel(coord);
}

// set pixel
CUDA_CALLABLE inline void img_set_pixel(uint64_t handle, const Coord& coord, const Color& color)
{
    img_get(handle).setPixel(coord, color);
}

}
