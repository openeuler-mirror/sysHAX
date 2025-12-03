"""
Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

sysHAX is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
    http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
See the Mulan PSL v2 for more details.
Created: 2025-05-23
Desc:sysHAX 配置管理模块
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
import httpx

from src.utils.logger import Logger

@dataclass
class SyshaxConfig:
    # 服务地址与端口
    gpu_host: str
    gpu_port: int
    cpu_host: str
    cpu_port: int
    syshax_host: str
    syshax_port: int

    # 系统参数
    request_timeout: int

    # 模型参数
    model_name: str

    # 调度决策器
    auto_pd_offload: bool
    cpu_max_batch_size: int

    @property
    def gpu_url(self) -> str:
        return f"http://{self.gpu_host}:{self.gpu_port}"

    @property
    def cpu_url(self) -> str:
        return f"http://{self.cpu_host}:{self.cpu_port}"

    @classmethod
    def from_dict(cls, data: dict) -> "SyshaxConfig":
        try:
            return cls(
                gpu_host=data["services"]["gpu"]["host"],
                gpu_port=data["services"]["gpu"]["port"],
                cpu_host=data["services"]["cpu"]["host"],
                cpu_port=data["services"]["cpu"]["port"],
                syshax_host=data["services"]["conductor"]["host"],
                syshax_port=data["services"]["conductor"]["port"],
                request_timeout=data["system"]["request_timeout"],
                auto_pd_offload=data["decider"]["auto_pd_offload"],
                cpu_max_batch_size=data["decider"]["cpu_max_batch_size"],
            )
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(f"配置缺失必要字段: {missing}") from e

def load_raw_config() -> dict[str, Any]:
    """加载原始 YAML 配置（不含 model_name）"""
    base = Path(__file__).parent.parent.parent / "config"
    primary = base / "config.yaml"
    try:
        with primary.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except (OSError, FileNotFoundError):
        Logger.warning("请配置 config/config.yaml 文件，使用示例配置启动")
        fallback = base / "config.example.yaml"
        if not fallback.exists():
            raise FileNotFoundError("未检测到 config/config.example.yaml 文件") from None
        with fallback.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"配置文件解析失败: {e}") from e
    
def fetch_model_name(gpu_url: str, cpu_url: str, timeout: int) -> str:
    """
    从 GPU 和 CPU 服务分别获取模型名称，并校验一致性。
    """
    def get_model_id_from_service(url: str) -> str:
        try:
            resp = httpx.get(f"{url}/v1/models", timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            models = data.get("data")
            if not models or not isinstance(models, list) or len(models) == 0:
                raise RuntimeError("返回的模型列表为空")
            model_id = models[0].get("id")
            if not model_id or not isinstance(model_id, str):
                raise RuntimeError("模型 id 缺失或无效")
            return model_id
        except Exception as e:
            raise RuntimeError(f"从 {url} 获取模型失败: {e}") from e

    gpu_model = get_model_id_from_service(gpu_url)
    cpu_model = get_model_id_from_service(cpu_url)

    if gpu_model != cpu_model:
        raise RuntimeError(
            f"GPU 与 CPU 服务的模型不一致！\n"
            f"GPU: {gpu_model}\n"
            f"CPU: {cpu_model}"
        )

    Logger.info(f"模型一致性校验通过，使用模型: {gpu_model}")
    return gpu_model

def load_syshax_config() -> SyshaxConfig:
    raw = load_raw_config()

    # 从配置中提取必要字段（除 model_name）
    try:
        base_config = {
            "gpu_host": raw["services"]["gpu"]["host"],
            "gpu_port": raw["services"]["gpu"]["port"],
            "cpu_host": raw["services"]["cpu"]["host"],
            "cpu_port": raw["services"]["cpu"]["port"],
            "syshax_host": raw["services"]["conductor"]["host"],
            "syshax_port": raw["services"]["conductor"]["port"],
            "request_timeout": raw["system"]["request_timeout"],
            "auto_pd_offload": raw["decider"]["auto_pd_offload"],
            "cpu_max_batch_size": raw["decider"]["cpu_max_batch_size"],
        }
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"配置缺失必要字段: {missing}") from e
    temp_config = SyshaxConfig(**base_config, model_name="placeholder")

    # 动态获取模型名
    model_name = fetch_model_name(
        gpu_url=temp_config.gpu_url,
        cpu_url=temp_config.cpu_url,
        timeout=temp_config.request_timeout
    )

    return SyshaxConfig(**base_config, model_name=model_name)

