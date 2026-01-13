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
Desc: vllm_plugin cpu loader tools for AF separation
"""

import logging
import os
from typing import Dict, Optional, Tuple, List
import torch

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Import C++ extension
import vllm_plugin._cpu_inference as cpu_inference

# torch.dtype -> C++ 支持的 dtype 字符串（须与 csrc/cpu/weight.h 的 dtype_nbytes_map 匹配）
_TORCH_DTYPE_TO_CPP_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
}

# Model format list
_MODEL_FORMAT_LIST = ["safetensors", "pytorch_model"]

def _dtype_to_cpp_str(dtype: torch.dtype) -> str:
    # bfloat16 在 CPU 上兼容性差，这里按现有逻辑通常会先转换为 float16
    if dtype == torch.bfloat16:
        return "float16"
    return _TORCH_DTYPE_TO_CPP_STR.get(dtype, "float32")

def _convert_dtype_for_cpu(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    将tensor转换为目标dtype
    """
    if tensor.dtype == target_dtype:
        return tensor   # 已经是目标dtype，直接返回
    else:
        convert_tensor = tensor.to(target_dtype)
        del tensor
        return convert_tensor

def init_weight_storage():
    """Initialize the CPU weight storage."""
    cpu_inference.clear_weights()
    logger.info("Initialized C++ weight storage backend")

def load_weight(name: str, tensor: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> Tuple[bool, torch.dtype]:
    """
    Load a weight tensor into CPU storage.
    
    Args:
        name: Weight name
        tensor: Weight tensor
        target_dtype: Optional target dtype for the tensor
        
    Returns:
        Tuple of (success, actual_dtype)
    """
    if target_dtype is None:
        target_dtype = tensor.dtype
    
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    
    # 转换为CPU支持的dtype
    tensor = _convert_dtype_for_cpu(tensor, target_dtype)
    tensor = tensor.contiguous()
    current_dtype = _dtype_to_cpp_str(tensor.dtype)
    
    success = cpu_inference.copy_weight(name, _dtype_to_cpp_str(tensor.dtype), tensor)
    del tensor  # 立即释放转换后的tensor引用，减少峰值内存
    
    if not success:
        logger.warning(f"Failed to load weight: {name}")
    
    return success, tensor.dtype

def get_weight(name: str) -> Optional[torch.Tensor]:
    """
    Get a weight tensor from CPU storage.
    
    Args:
        name: Weight name
        
    Returns:
        Weight tensor or None if not found
    """
    if has_weight(name):
        return cpu_inference.get_weight(name)
    return None

def has_weight(name: str) -> bool:
    """
    Check if a weight is in CPU storage.
    
    Args:
        name: Weight name
        
    Returns:
        True if the weight exists, False otherwise
    """
    return bool(cpu_inference.has_weight(name))

def clear_weight_storage():
    """Clear all weights from CPU storage."""
    cpu_inference.clear_weights()
    logger.info("Cleared CPU weight storage")

def get_all_weight_names() -> List[str]:
    """
    Get all weight names in CPU storage.
    
    Returns:
        List of weight names
    """
    return list(cpu_inference.get_all_weight_names())

def _detect_format_from_file(file_path: str) -> str:
    """
    Detect the format of a weight file.
    
    Args:
        file_path: Path to the weight file
        
    Returns:
        Format name ("safetensors" or "pytorch_model")
    """
    if file_path.endswith(".safetensors"):
        return "safetensors"
    return "pytorch_model"
