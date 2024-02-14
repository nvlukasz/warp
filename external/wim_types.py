import ctypes
import os
import warp as wp

# our external lib bindings (importing it will build the native lib and initialize the Python bindings)
import wim


class Coord:

    # define variables accessible in kernels (e.g., coord.x)
    vars = {
        "x": wp.codegen.Var("x", int),
        "y": wp.codegen.Var("y", int),
    }

    # struct that corresponds to the native Foo type
    # - used when packing arguments for kernels (pass-by-value)
    # - binary layout of fields must match native type
    class _type_(ctypes.Structure):

        _fields_ = [
            ("x", ctypes.c_int),
            ("y", ctypes.c_int),
        ]

        def __init__(self, coord):
            self.x = coord.x
            self.y = coord.y

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    # HACK: used when packing kernel argument as `arg_type._type_(value.value)` in `pack_arg()` during `wp.launch()`
    @property
    def value(self):
        return self


class Color:

    # define variables accessible in kernels
    vars = {
        "r": wp.codegen.Var("r", float),
        "g": wp.codegen.Var("g", float),
        "b": wp.codegen.Var("b", float),
    }

    # struct that corresponds to the native Foo type
    # - used when packing arguments for kernels (pass-by-value)
    # - binary layout of fields must match native type
    class _type_(ctypes.Structure):

        _fields_ = [
            ("r", ctypes.c_float),
            ("g", ctypes.c_float),
            ("b", ctypes.c_float),
        ]

        def __init__(self, color):
            self.r = color.r
            self.g = color.g
            self.b = color.b

    def __init__(self, r=0, g=0, b=0):
        self.r = r
        self.g = g
        self.b = b

    # HACK: used when packing kernel argument as `arg_type._type_(value.value)` in `pack_arg()` during `wp.launch()`
    @property
    def value(self):
        return self


class Image:
    def __init__(self, width: int, height: int, device=None):

        # image shape
        self.width = width
        self.height = height

        self.device = wp.get_device(device)

        # pointer to the native wim::Image class, either on CPU or GPU
        self.ptr = None

        # pointer to the image data (Color array), either on CPU or GPU
        self.data_ptr = None

        if self.device.is_cpu:
            data_ptr = ctypes.c_void_p()
            self.ptr = wim._core.create_image_cpu(width, height, ctypes.byref(data_ptr))
            self.data_ptr = data_ptr.value
        elif self.device.is_cuda:
            data_ptr = ctypes.c_void_p()
            self.ptr = wim._core.create_image_cuda(self.device.ordinal, width, height, ctypes.byref(data_ptr))
            self.data_ptr = data_ptr.value
        else:
            raise ValueError(f"Invalid device {device}")

    def __del__(self):
        if self.ptr:
            if self.device.is_cpu:
                wim._core.destroy_image_cpu(self.ptr)
            else:
                wim._core.destroy_image_cuda(self.device.ordinal, self.ptr)


def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive


def _register_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/wim_warp.h")


def _register_builtins():

    # Coord constructor
    wp.context.add_builtin(
        "Coord_",
        input_types={"x": int, "y": int},
        value_type=Coord,
        missing_grad=True,
    )

    # Color addition
    wp.context.add_builtin(
        "add",
        input_types={"a": Color, "b": Color},
        value_type=Color,
        missing_grad=True,
    )

    # Color scaling
    wp.context.add_builtin(
        "mul",
        input_types={"s": float, "c": Color},
        value_type=Color,
        missing_grad=True,
    )

    # get image width
    wp.context.add_builtin(
        "img_width",
        input_types={"handle": wp.uint64},
        value_type=int,
        missing_grad=True,
    )

    # get image height
    wp.context.add_builtin(
        "img_height",
        input_types={"handle": wp.uint64},
        value_type=int,
        missing_grad=True,
    )

    # get pixel
    wp.context.add_builtin(
        "img_get_pixel",
        input_types={"handle": wp.uint64, "coord": Coord},
        value_type=Color,
        missing_grad=True,
    )

    # get pixel
    wp.context.add_builtin(
        "img_set_pixel",
        input_types={"handle": wp.uint64, "coord": Coord, "color": Color},
        value_type=None,
        missing_grad=True,
    )


def register():
    _register_headers()
    _register_builtins()
