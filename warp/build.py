# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel build utilities, including external source support."""

# isort: skip_file

from warp._src.build import add_include_directory as add_include_directory
from warp._src.build import add_preprocessor_macro_definition as add_preprocessor_macro_definition
from warp._src.build import set_cpp_standard as set_cpp_standard
