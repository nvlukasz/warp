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

    # struct that corresponds to the native Coord type
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

    # struct that corresponds to the native Color type
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

    # struct that corresponds to the native Image type
    # - used when packing arguments for kernels (pass-by-value)
    # - binary layout of fields must match native type
    class _type_(ctypes.Structure):

        _fields_ = [
            ("width", ctypes.c_int),
            ("height", ctypes.c_int),
            ("data", ctypes.c_void_p),
        ]

        def __init__(self, img):
            self.width = img.width
            self.height = img.height
            self.data = img.data

    def __init__(self, width: int, height: int, device=None):

        # image shape
        self.width = width
        self.height = height

        self.device = wp.get_device(device)

        # pointer to the native wim::Image class (on CPU)
        self.ptr = None

        if self.device.is_cpu:
            self.ptr = wim._core.create_image_cpu(width, height)
        elif self.device.is_cuda:
            self.ptr = wim._core.create_image_cuda(self.device.ordinal, width, height)
        else:
            raise ValueError(f"Invalid device {device}")

        # get pointer to the data, which could be on CPU or GPU
        img_ptr = ctypes.cast(self.ptr, ctypes.POINTER(self._type_))
        self.data = img_ptr.contents.data

    def __del__(self):
        if self.ptr:
            if self.device.is_cpu:
                wim._core.destroy_image_cpu(self.ptr)
            else:
                wim._core.destroy_image_cuda(self.device.ordinal, self.ptr)

    # HACK: used when packing kernel argument as `arg_type._type_(value.value)` in `pack_arg()` during `wp.launch()`
    @property
    def value(self):
        return self
    
    # return the data as a Warp array on the correct device
    # TODO: can't currently use arrays of custom native types, so using vec3f instead
    @property
    def data_array(self):
        shape = (self.height, self.width)
        return wp.array(ptr=self.data, shape=shape, dtype=wp.vec3f, owner=False)


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
        input_types={"img": Image},
        value_type=int,
        missing_grad=True,
    )

    # get image height
    wp.context.add_builtin(
        "img_height",
        input_types={"img": Image},
        value_type=int,
        missing_grad=True,
    )

    # get image data as a Warp array
    wp.context.add_builtin(
        "img_data",
        input_types={"img": Image},
        value_type=wp.array2d(dtype=wp.vec3f),
        missing_grad=True,
    )

    # get pixel
    wp.context.add_builtin(
        "img_get_pixel",
        input_types={"img": Image, "coord": Coord},
        value_type=Color,
        missing_grad=True,
    )

    # set pixel
    wp.context.add_builtin(
        "img_set_pixel",
        input_types={"img": Image, "coord": Coord, "color": Color},
        value_type=None,
        missing_grad=True,
    )


def register():
    _register_headers()
    _register_builtins()
