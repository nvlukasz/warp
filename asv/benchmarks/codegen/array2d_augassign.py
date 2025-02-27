# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


@wp.kernel
def array2d_augassign_kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    x[i, j] += y[i, j]


class CompileModule:
    repeat = 10  # Number of samples to run
    number = 1  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()

    def teardown(self):
        array2d_augassign_kernel.module.unload()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")

    def time_cpu_codegen(self):
        wp.load_module(device="cpu")


class RunForwardKernel:
    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.x = wp.ones(N, dtype=float, device="cuda:0")
        self.y = wp.ones(N, dtype=float, device="cuda:0")

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                wp.launch(array2d_augassign_kernel, self.x.shape, inputs=[self.x, self.y], device="cuda:0")

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return average * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardKernel:
    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.x = wp.ones(N, dtype=float, device="cuda:0", requires_grad=True)
        self.y = wp.ones(N, dtype=float, device="cuda:0", requires_grad=True)

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                wp.launch(
                    array2d_augassign_kernel,
                    self.x.shape,
                    inputs=[self.x, self.y],
                    adj_inputs=[self.x.grad, self.y.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return average * 1e-3

    track_cuda.unit = "seconds"
