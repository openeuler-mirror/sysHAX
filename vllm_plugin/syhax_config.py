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
Desc: vllm_plugin syshax config
"""

import os
import logging
from dataclasses import dataclass

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class SyshaxConfig:
    ENABLE_AUTO_PD_OFFLOAD: bool = False
    USE_GREDDY: bool = True
    MODEL_LOADING_SCHEME: int = 0

    def __post_init__(self):
        enable_auto_pd_offload = os.getenv("ENABLE_AUTO_PD_OFFLOAD")
        if enable_auto_pd_offload is not None:
            self.ENABLE_AUTO_PD_OFFLOAD = self._parse_bool(enable_auto_pd_offload)
        if self.ENABLE_AUTO_PD_OFFLOAD:
            logger.info("开启了 ENABLE_AUTO_PD_OFFLOAD 选项")
        else:
            logger.info("未启动 ENABLE_AUTO_PD_OFFLOAD 选项")

        use_greddy = os.getenv("USE_GREDDY")
        if use_greddy is not None:
            self.USE_GREDDY = self._parse_bool(use_greddy)
        if self.USE_GREDDY:
            logger.info("开启了 USE_GREDDY 选项")
        else:
            logger.info("未启动 USE_GREDDY 选项")

        model_loading_scheme = os.getenv("MODEL_LOADING_SCHEME")
        if model_loading_scheme is not None:
            self.MODEL_LOADING_SCHEME = int(model_loading_scheme)
        if self.MODEL_LOADING_SCHEME == 0:
            logger.info("采用默认模型加载方案")
        else:
            logger.info(f"采用自定义模型加载方案，MODEL_LOADING_SCHEME = {self.MODEL_LOADING_SCHEME}")

    @staticmethod
    def _parse_bool(value: str) -> bool:
        return value.lower() in ("1", "yes", "true", "on")

    @classmethod
    def instance(cls) -> "SyshaxConfig":
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance
    
    def enable_auto_pd_offload(self) -> bool:
        return self.ENABLE_AUTO_PD_OFFLOAD
    
    def use_greddy(self) -> bool:
        return self.USE_GREDDY

    def model_loading_scheme(self) -> int:
        return self.MODEL_LOADING_SCHEME
