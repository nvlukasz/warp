# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp

class Foo:

    from warp.codegen import Var

    # named variables accessible in kernels, e.g. foo.magic
    vars = {
        "magic": Var("magic", warp.float32),
        "offset": Var("offset", warp.int32),
    }

    def __init__(self, magic: float = 0.0, offset: int = 0, device = None):

        self.id = 0

        self.device = warp.get_device(device)
        self.runtime = warp.context.runtime

        # create on the specified device
        if self.device.is_cuda:
            self.id = self.runtime.core.foo_create_device(self.device.context, magic, offset)
        else:
            self.id = self.runtime.core.foo_create_host(magic, offset)

    def __del__(self):

        if not self.id:
            return
        
        # release on the specified device
        if self.device.is_cuda:
            self.runtime.core.foo_destroy_device(self.device.context, self.id)
        else:
            self.runtime.core.foo_destroy_host(self.id)
