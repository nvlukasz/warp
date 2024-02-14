import ctypes
import os
import subprocess

_lib_dir = os.path.abspath(os.path.dirname(__file__))
_lib_path = os.path.join(_lib_dir, "wim.so")


def _build_lib(cuda_path="/usr/local/cuda"):
    build_cmd = [os.path.join(cuda_path, "bin", "nvcc"),
                "-shared",
                "-Xcompiler", "-fPIC",
                os.path.join(_lib_dir, "wim.cpp"),
                "-o", _lib_path]
    subprocess.run(build_cmd, check=True)


def _load_lib():
    lib_dir = os.path.abspath(os.path.dirname(__file__))
    return ctypes.CDLL(os.path.join(lib_dir, _lib_path))


# build the lib
_build_lib()

# load the lib and set up Python bindings
_core = _load_lib()
_core.create_image_cpu.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
_core.create_image_cpu.restype = ctypes.c_void_p
_core.destroy_image_cpu.argtypes = [ctypes.c_void_p]
_core.destroy_image_cpu.restype = None
_core.create_image_cuda.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
_core.create_image_cuda.restype = ctypes.c_void_p
_core.destroy_image_cuda.argtypes = [ctypes.c_int, ctypes.c_void_p]
_core.destroy_image_cuda.restype = None
