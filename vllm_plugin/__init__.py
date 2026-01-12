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
Desc: vllm_plugin initialization
"""

__version__ = '0.1.0'

# Import necessary components
from .shared_memory_manager import SharedMemoryManager
from .syshax_config import SyshaxConfig
from .scheduler_patch import patch_scheduler

# Global references to track patches and resources
_patches_applied = False
_shared_memory_manager = None
_syshax_config = None

from vllm.model_executor.model_loader import register_model_loader
from .model_loader.moe_af_separated_loader import SyshaxMoEAFSeparatedModelLoader

# Register the custom model loader
register_model_loader("syshax_moe_af_separated")(SyshaxMoEAFSeparatedModelLoader)

def setup():
    """
    Setup the sysHAX vLLM plugin.
    This function initializes the plugin and applies all necessary patches to vLLM.
    """
    global _patches_applied, _shared_memory_manager, _syshax_config
    
    if _patches_applied:
        return
    
    # Initialize configuration
    _syshax_config = SyshaxConfig.instance()
    
    # Initialize shared memory manager
    _shared_memory_manager = SharedMemoryManager.instance()
    
    # Apply scheduler patches
    patch_scheduler()
    
    # Apply engine patches 假设修改了原版vllm的engine模块，可以这样应用patch
    from .syshax_engine import patch_engine
    patch_engine()
    
    _patches_applied = True

def teardown():
    """
    Teardown the sysHAX vLLM plugin.
    This function cleans up all resources and removes patches.
    """
    global _patches_applied, _shared_memory_manager, _syshax_config
    
    if not _patches_applied:
        return
    
    # Clean up shared memory manager
    if _shared_memory_manager is not None:
        # Currently, SharedMemoryManager doesn't have a teardown method
        # We'll just set the reference to None
        _shared_memory_manager = None
    
    # Reset configuration
    if _syshax_config is not None:
        # Currently, SyshaxConfig doesn't have a teardown method
        # We'll just set the reference to None
        _syshax_config = None
    
    _patches_applied = False
