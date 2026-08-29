#pragma once

// TODO: may need to add a mechanism for include paths
#include "wim/wim.h"

// include some Warp types so we can expose the image data as a Warp array
#include "../warp/native/array.h"
#include "../warp/native/vec.h"

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

// get image width (can't be exposed as a named var directly, because the member is private)
CUDA_CALLABLE inline int img_width(const Image& img)
{
    return img.getWidth();
}

// get image height (can't be exposed as a named var directly, because the member is private)
CUDA_CALLABLE inline int img_height(const Image& img)
{
    return img.getHeight();
}

// get image data as a Warp array
CUDA_CALLABLE inline array_t<vec3f> img_data(Image& img)
{
    Color* data = img.getData();

    // TODO: can't currently use array of custom native types, so use vec3f
    return array_t<vec3f>((vec3f*)data, img.getHeight(), img.getWidth());
}

// get pixel
CUDA_CALLABLE inline Color img_get_pixel(const Image& img, const Coord& coord)
{
    return img.getPixel(coord);
}

// set pixel
CUDA_CALLABLE inline void img_set_pixel(Image& img, const Coord& coord, const Color& color)
{
    img.setPixel(coord, color);
}

}
