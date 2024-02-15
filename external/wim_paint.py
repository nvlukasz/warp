import warp as wp

import wim_types
from wim_types import Coord, Color, Image


@wp.kernel
def print_image_info_kernel(img: Image):
    width = wp.img_width(img)
    height = wp.img_height(img)
    data = wp.img_data(img)  # this is a Warp array which wraps the image data
    wp.printf("Image: %dx%d, data array shape: (%d, %d)\n", width, height, data.shape[0], data.shape[1])


@wp.kernel
def fill_kernel(img: Image, color: Color):
    y, x = wp.tid()
    coord = wp.Coord_(x, y)
    wp.img_set_pixel(img, coord, color)


@wp.kernel
def fill_rect_kernel(img: Image, half_width: int, half_height: int, pos: Coord, color: Color):
    j, i = wp.tid()
    i -= half_width
    j -= half_height
    coord = wp.Coord_(pos.x + i, pos.y + j)
    wp.img_set_pixel(img, coord, color)


@wp.kernel
def fill_circle_kernel(img: Image, radius: int, pos: Coord, color: Color):
    j, i = wp.tid()
    i -= radius
    j -= radius
    if i * i + j * j <= radius * radius:
        x = pos.x + i
        y = pos.y + j
        coord = wp.Coord_(x, y)
        wp.img_set_pixel(img, coord, color)


@wp.kernel
def blur_kernel(img: Image):
    y, x = wp.tid()

    c00 = wp.img_get_pixel(img, wp.Coord_(x - 1, y - 1))
    c01 = wp.img_get_pixel(img, wp.Coord_(x, y - 1))
    c02 = wp.img_get_pixel(img, wp.Coord_(x + 1, y - 1))
    c10 = wp.img_get_pixel(img, wp.Coord_(x - 1, y))
    c11 = wp.img_get_pixel(img, wp.Coord_(x, y))
    c12 = wp.img_get_pixel(img, wp.Coord_(x + 1, y))
    c20 = wp.img_get_pixel(img, wp.Coord_(x - 1, y + 1))
    c21 = wp.img_get_pixel(img, wp.Coord_(x, y + 1))
    c22 = wp.img_get_pixel(img, wp.Coord_(x + 1, y + 1))

    c = (c00 + c02 + c20 + c22) + 2.0 * (c01 + c21 + c10 + c12) + 4.0 * c11
    c = (1.0 / 16.0) * c

    wp.img_set_pixel(img, wp.Coord_(x, y), c)


def fill(img: Image, color: Color):
    img_shape = (img.height, img.width)
    wp.launch(fill_kernel, dim=img_shape, inputs=[img, color])


def draw_rect(img: Image, width: int, height: int, pos: Coord, color: Color):
    launch_dim = (height, width)
    wp.launch(fill_rect_kernel, dim=launch_dim, inputs=[img, width//2, height//2, pos, color])


def draw_circle(img: Image, radius: int, pos: Coord, color: Color):
    launch_dim = (2 * radius, 2 * radius)
    wp.launch(fill_circle_kernel, dim=launch_dim, inputs=[img, radius, pos, color])


def blur(img: Image):
    img_shape = (img.height, img.width)
    wp.launch(blur_kernel, dim=img_shape, inputs=[img])


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


def show(img, title, save_path=None):
    
    img_data = img.data_array.numpy()

    if save_path is not None:
        import matplotlib.image as img
        img.imsave(save_path, img_data)

    import matplotlib.pyplot as plt
    fig = plt.figure(title)
    plt.imshow(img_data)
    plt.show()


wp.init()

# It's a good idea to always clear the kernel cache when developing new native or codegen features
wp.build.clear_kernel_cache()

# !!! DO THIS BEFORE LOADING MODULES OR LAUNCHING KERNELS
wim_types.register()

with wp.ScopedDevice("cuda:0"):

    img = Image(800, 600)

    # run a kernel to print image info
    wp.launch(print_image_info_kernel, dim=1, inputs=[img])
    wp.synchronize_device()

    # make a drawing
    draw_picture(img)

    # show and save the image
    show(img, "Result", save_path="result.png")

    try:
        import torch

        # wrap the image data as a PyTorch tensor (no copy)
        t = wp.to_torch(img.data_array, requires_grad=False)

        # invert the image in-place using PyTorch
        torch.sub(1, t, out=t)

        # show and save the image
        show(img, "Inverted using PyTorch", save_path="result_inverted.png")

    except ImportError:
        print("Torch is not installed, couldn't post-process image")