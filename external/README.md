# Overview

An example of using external native types in Warp.

The [wim](wim) subdirectory contains an independent library with [header-only types](wim/wim.h).  It also defines a [C-style public interface](wim/wim.cpp) used for the [Python bindings](wim/__init__.py).  Normally, there would be more code there, but this is a minimal viable example.  For simplicity, running `import wim` will build and load the native library and initialize the Python bindings.

The file [wim_warp.h](wim_warp.h) is a header that will be included by Warp when building kernels.  It imports the types into the `wp` namespace, which is currently necessary, but may change in the future.  It also defines some useful functions that will be exposed to Warp code generation.

The file [wim_types.py](wim_types.py) defines the Python versions of the custom types and registers the native utility functions as buitin functions that are available in kernels.

The file [wim_paint.py](wim_paint.py) is the main program for the example.  It creates an image and draws shapes using Warp kernels.

# Prerequisites

* Linux is required
* CUDA Toolkit installed in `/usr/local/cuda` (Note that `cuda_path` can be modified in [wim/__init__.py](wim/__init__.py)).  This is needed for building the "external" `wim` library.
* `pip install matplotlib` for showing and saving the generated image.
* `pip install torch` for an optional interop example!

# Running

From the repo root:

```python
$ python external/wim_paint.py
```

If PyTorch is installed, the example will also demonstrate inverting the image using PyTorch.
