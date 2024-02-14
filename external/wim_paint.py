import os
import numpy as np
import warp as wp

import wim_types
from wim_types import Coord, Color, Image


@wp.kernel
def print_image_info_kernel(img_handle: wp.uint64):
    width = wp.img_width(img_handle)
    height = wp.img_height(img_handle)
    wp.printf("Image: %dx%d\n", width, height)


@wp.kernel
def fill_kernel(img_handle: wp.uint64, color: Color):
    y, x = wp.tid()
    coord = wp.Coord_(x, y)
    wp.img_set_pixel(img_handle, coord, color)


@wp.kernel
def fill_rect_kernel(img_handle: wp.uint64, half_width: int, half_height: int, pos: Coord, color: Color):
    j, i = wp.tid()
    i -= half_width
    j -= half_height
    coord = wp.Coord_(pos.x + i, pos.y + j)
    wp.img_set_pixel(img_handle, coord, color)


@wp.kernel
def fill_circle_kernel(img_handle: wp.uint64, radius: int, pos: Coord, color: Color):
    j, i = wp.tid()
    i -= radius
    j -= radius
    if i * i + j * j <= radius * radius:
        x = pos.x + i
        y = pos.y + j
        coord = wp.Coord_(x, y)
        wp.img_set_pixel(img_handle, coord, color)


@wp.kernel
def blur_kernel(img_handle: wp.uint64):
    y, x = wp.tid()

    c00 = wp.img_get_pixel(img_handle, wp.Coord_(x - 1, y - 1))
    c01 = wp.img_get_pixel(img_handle, wp.Coord_(x, y - 1))
    c02 = wp.img_get_pixel(img_handle, wp.Coord_(x + 1, y - 1))
    c10 = wp.img_get_pixel(img_handle, wp.Coord_(x - 1, y))
    c11 = wp.img_get_pixel(img_handle, wp.Coord_(x, y))
    c12 = wp.img_get_pixel(img_handle, wp.Coord_(x + 1, y))
    c20 = wp.img_get_pixel(img_handle, wp.Coord_(x - 1, y + 1))
    c21 = wp.img_get_pixel(img_handle, wp.Coord_(x, y + 1))
    c22 = wp.img_get_pixel(img_handle, wp.Coord_(x + 1, y + 1))

    c = (c00 + c02 + c20 + c22) + 2.0 * (c01 + c21 + c10 + c12) + 4.0 * c11
    c = (1.0 / 16.0) * c

    wp.img_set_pixel(img_handle, wp.Coord_(x, y), c)


def fill(img, color: Color):
    img_handle = wp.uint64(img.ptr)
    img_shape = (img.height, img.width)
    wp.launch(fill_kernel, dim=img_shape, inputs=[img_handle, color])


def draw_rect(img, width: int, height: int, pos: Coord, color: Color):
    img_handle = wp.uint64(img.ptr)
    launch_dim = (height, width)
    wp.launch(fill_rect_kernel, dim=launch_dim, inputs=[img_handle, width//2, height//2, pos, color])


def draw_circle(img, radius: int, pos: Coord, color: Color):
    img_handle = wp.uint64(img.ptr)
    launch_dim = (2 * radius, 2 * radius)
    wp.launch(fill_circle_kernel, dim=launch_dim, inputs=[img_handle, radius, pos, color])


def blur(img):
    img_handle = wp.uint64(img.ptr)
    img_shape = (img.height, img.width)
    wp.launch(blur_kernel, dim=img_shape, inputs=[img_handle])


def draw_picture(img):

    # background color
    fill(img, Color(0.3, 0.0, 0.3))

    # concentric circles in the corners
    for iter in range(10):
        g = iter / 10
        r = 20 + (10 - iter - 1) * 20
        draw_circle(img, r, Coord(0, 0), Color(0, g, 1))

        for _ in range(500):
            blur(img)

    for iter in range(10):
        # rectangle crossing the vertical edges
        if iter == 0:
            draw_rect(img, 200, 300, Coord(img.width//2, 0), Color(1, 0, 0))
        # rectangle crossing the horizontal edges
        elif iter == 9:
            draw_rect(img, 200, 50, Coord(0, img.height//2), Color(1, 1, 0))

        for _ in range(20):
            blur(img)

    # center pieces
    draw_rect(img, 100, 100, Coord(img.width//2, img.height//2), Color(0.5, 0.2, 0.5))
    draw_circle(img, 30, Coord(img.width//2, img.height//2), Color(0.9, 0.7, 0.9))


def show(img, save_path=None):
    img_shape = (img.height, img.width)
    img_array = wp.array(ptr=img.data_ptr, shape=img_shape, dtype=wp.vec3f, owner=False)
    
    img_data = img_array.numpy()

    if save_path is not None:
        import matplotlib.image as img
        img.imsave(save_path, img_data)

    import matplotlib.pyplot as plt
    plt.imshow(img_data)
    plt.show()


wp.init()

# It's a good idea to always clear the kernel when developing new native or codegen features
wp.build.clear_kernel_cache()

# !!! DO THIS BEFORE LOADING MODULES OR LAUNCHING KERNELS
wim_types.register()

with wp.ScopedDevice("cuda:0"):

    img = Image(800, 600)
    print(img)

    # run a kernel to print image info
    img_handle = wp.uint64(img.ptr)
    wp.launch(print_image_info_kernel, dim=1, inputs=[img_handle])
    wp.synchronize_device()

    # make a drawing
    draw_picture(img)

    # show and save the image
    show(img, save_path="result.png")
