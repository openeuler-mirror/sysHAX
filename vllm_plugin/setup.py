#!/usr/bin/env python3
"""
Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

sysHAX vLLM Plugin is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
    http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
See the Mulan PSL v2 for more details.
Created: 2026-01-09
Desc: vllm_plugin setup file
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# Read the version from __init__.py
version = '0.9.1'

# Read the requirements
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# vllm is already installed in the environment, so we don't need to specify it here
# This plugin is designed to work with vllm 0.9.1 specifically

class CMakeExtension(Extension):
    """A setuptools Extension that is built via CMake."""

    def __init__(self, name: str, sourcedir: str = "."):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"

        # 根据CMAKE_PREFIX_PATH查找TorchConfig.cmake
        try:
            import torch
            torch_cmake_prefix = getattr(torch.utils, "cmake_prefix_path", None)
        except Exception as e:
            raise RuntimeError(
                "Building sysHAX-adapter C++ extension requires PyTorch to be installed "
                "in the current Python environment (so we can locate TorchConfig.cmake). "
                "Please `pip install torch` first."
            ) from e

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if torch_cmake_prefix:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix}")

        build_args = ["--config", cfg]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["-j", str(os.cpu_count() or 2)]

        # Configure
        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=os.fspath(build_temp))
        # Build
        subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=os.fspath(build_temp))

        # cmake将构建产物放在<build_temp>/<DESTINATION>目录下（DESTINATION是"vllm_plugin"）
        built_dir = build_temp / "vllm_plugin"
        if not built_dir.exists():
            # 尝试直接在build_temp目录下查找
            built_dir = build_temp

        pattern = ext_fullpath.name  # e.g. _cpu_inference.cpython-311-aarch64-linux-gnu.so
        candidates = list(built_dir.glob(pattern))
        if not candidates:
            candidates = list(built_dir.glob(ext_fullpath.stem + "*.so"))
        if not candidates:
            raise RuntimeError(
                f"Could not find built extension for {ext.name}. "
                f"Looked for {pattern} (and fallback {ext_fullpath.stem + '*.so'}) under {built_dir}"
            )

        extdir.mkdir(parents=True, exist_ok=True)
        src_path = candidates[0]
        dst_path = ext_fullpath
        # 将构建好的动态链接库拷贝到wheel/build输出目录
        self.copy_file(os.fspath(src_path), os.fspath(dst_path))

setup(
    name="syshax-vllm-plugin",
    version=version,
    description="sysHAX vLLM Plugin for AF and PD Separation",
    long_description=open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')).read(),
    long_description_content_type="text/markdown",
    author="Huawei Technologies Co., Ltd.",
    author_email="support@huawei.com",
    url="https://gitcode.com/openeuler/sysHAX/",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mulan Public License v2 (MulanPL-2.0)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "vllm.general_plugins": [
            "syshax_plugin = vllm_plugin:setup",
        ],
    },
    ext_modules=[
        # Python imports: `import vllm_plugin._cpu_inference as cpu_inference`
        CMakeExtension("vllm_plugin._cpu_inference", sourcedir=os.path.dirname(os.path.dirname(__file__))),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
