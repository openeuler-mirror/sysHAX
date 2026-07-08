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
class Worker:
    """
    表示一个 GPU 或 CPU 推理服务实例。

    使用 frozen=False，但手动实现 __hash__/__eq__ 以支持作为字典键，
    同时保持实例可变（如需扩展运行时状态）。
    """
    host: str
    port: int
    device: str  # "GPU" 或 "CPU"

    def __hash__(self) -> int:
        return hash((self.host, self.port, self.device))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Worker):
            return NotImplemented
        return (self.host, self.port, self.device) == (other.host, other.port, other.device)

    @property
    def label(self) -> str:
        """人类可读的实例标识，用于日志输出"""
        return f"{self.device}({self.host}:{self.port})"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def metrics_url(self) -> str:
        return f"{self.base_url}/metrics"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


@dataclass
class SyshaxConfig:
    """sysHAX 全局配置，支持 GPU/CPU 多实例池"""

    # 服务实例池
    gpu_workers: list[Worker]
    cpu_workers: list[Worker]

    # conductor 服务地址
    syshax_host: str
    syshax_port: int

    # 系统参数
    request_timeout: int

    # 模型名称（启动时从服务动态获取）
    model_name: str

    # 调度决策器
    auto_pd_offload: bool
    cpu_max_batch_size: int

    # 心跳探活配置
    heartbeat_interval: int = 10       # 探活间隔（秒）
    heartbeat_timeout: float = 3.0     # 单次探活超时（秒）
    heartbeat_fail_threshold: int = 3  # 连续失败多少次后摘除节点

    @property
    def gpu_url(self) -> str:
        """向后兼容：返回第一个 GPU Worker 的 base URL"""
        return self.gpu_workers[0].base_url if self.gpu_workers else ""

    @property
    def cpu_url(self) -> str:
        """向后兼容：返回第一个 CPU Worker 的 base URL"""
        return self.cpu_workers[0].base_url if self.cpu_workers else ""


def _parse_workers(service_cfg: Any, device: str) -> list[Worker]:
    """
    解析服务配置，支持两种格式：
      - 列表格式（多实例池）：[{host: ..., port: ...}, ...]
      - 字典格式（单实例，向后兼容）：{host: ..., port: ...}
    """
    if isinstance(service_cfg, list):
        workers = []
        for item in service_cfg:
            workers.append(Worker(host=item["host"], port=item["port"], device=device))
        if not workers:
            raise ValueError(f"{device} 服务实例池为空，请至少配置一个实例")
        return workers
    elif isinstance(service_cfg, dict):
        # 向后兼容：单实例写法
        return [Worker(host=service_cfg["host"], port=service_cfg["port"], device=device)]
    else:
        raise ValueError(f"{device} 服务配置格式无效，应为列表或字典")


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

    try:
        gpu_workers = _parse_workers(raw["services"]["gpu"], "GPU")
        cpu_workers = _parse_workers(raw["services"]["cpu"], "CPU")
        syshax_host = raw["services"]["conductor"]["host"]
        syshax_port = raw["services"]["conductor"]["port"]
        request_timeout = raw["system"]["request_timeout"]
        auto_pd_offload = raw["decider"]["auto_pd_offload"]
        cpu_max_batch_size = raw["decider"]["cpu_max_batch_size"]
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"配置缺失必要字段: {missing}") from e

    # 心跳配置（全部有默认值，缺失时不报错）
    hb_cfg = raw.get("heartbeat", {})
    heartbeat_interval = int(hb_cfg.get("interval", 10))
    heartbeat_timeout = float(hb_cfg.get("timeout", 3.0))
    heartbeat_fail_threshold = int(hb_cfg.get("fail_threshold", 3))

    # 构建临时配置以获取 URL（用于动态获取模型名）
    temp_gpu_url = gpu_workers[0].base_url
    temp_cpu_url = cpu_workers[0].base_url

    model_name = fetch_model_name(
        gpu_url=temp_gpu_url,
        cpu_url=temp_cpu_url,
        timeout=request_timeout
    )

    return SyshaxConfig(
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        syshax_host=syshax_host,
        syshax_port=syshax_port,
        request_timeout=request_timeout,
        model_name=model_name,
        auto_pd_offload=auto_pd_offload,
        cpu_max_batch_size=cpu_max_batch_size,
        heartbeat_interval=heartbeat_interval,
        heartbeat_timeout=heartbeat_timeout,
        heartbeat_fail_threshold=heartbeat_fail_threshold,
    )
