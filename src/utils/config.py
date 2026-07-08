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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.utils.logger import Logger


@dataclass(frozen=True)
class Worker:
    """单个 vLLM 服务实例（GPU 或 CPU 池中的一个成员）"""

    host: str
    port: int
    device: str  # "GPU" | "CPU"
    idx: int      # 池内序号，用于日志与指标标签

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
    def models_url(self) -> str:
        return f"{self.base_url}/v1/models"

    @property
    def label(self) -> str:
        return f"{self.device}#{self.idx}"


def _parse_workers(node: Any, device: str) -> list[Worker]:
    """将 services.{gpu,cpu} 配置节归一化为 Worker 列表。

    兼容两种写法：
    - 单值 dict：{host, port} -> 单元素池（向后兼容旧配置）
    - 列表 list：[{host, port}, ...] -> 多实例池
    """
    key = device.lower()
    if node is None:
        raise ValueError(f"配置缺失必要字段: services.{key}")
    if isinstance(node, dict):
        entries = [node]
    elif isinstance(node, list):
        entries = node
    else:
        raise ValueError(f"services.{key} 必须是映射(单实例)或列表(多实例)")
    if not entries:
        raise ValueError(f"services.{key} 至少需要配置一个实例")

    workers: list[Worker] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"services.{key}[{idx}] 必须包含 host 与 port")
        try:
            host = entry["host"]
            port = entry["port"]
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(f"services.{key}[{idx}] 缺少字段: {missing}") from e
        try:
            port_int = int(port)
        except (TypeError, ValueError) as e:
            raise ValueError(f"services.{key}[{idx}] 的 port 必须为整数: {port!r}") from e
        workers.append(Worker(host=str(host), port=port_int, device=device, idx=idx))
    return workers


@dataclass
class SyshaxConfig:
    # 服务实例池
    gpu_workers: list[Worker]
    cpu_workers: list[Worker]

    # conductor(sysHAX 自身)服务地址与端口
    syshax_host: str
    syshax_port: int

    # 系统参数
    request_timeout: int

    # 模型参数
    model_name: str

    # 调度决策器
    auto_pd_offload: bool
    cpu_max_batch_size: int

    # ---- 向后兼容：单值访问指向池内首个实例 ----
    # 现有 runner/monitor/routes 仍以单实例语义读取这些属性，后续 PR 再迁移到池。
    @property
    def gpu_host(self) -> str:
        return self.gpu_workers[0].host

    @property
    def gpu_port(self) -> int:
        return self.gpu_workers[0].port

    @property
    def cpu_host(self) -> str:
        return self.cpu_workers[0].host

    @property
    def cpu_port(self) -> int:
        return self.cpu_workers[0].port

    @property
    def gpu_url(self) -> str:
        return self.gpu_workers[0].base_url

    @property
    def cpu_url(self) -> str:
        return self.cpu_workers[0].base_url

    @classmethod
    def from_dict(cls, data: dict, model_name: str = "placeholder") -> "SyshaxConfig":
        try:
            return cls(
                gpu_workers=_parse_workers(data["services"]["gpu"], "GPU"),
                cpu_workers=_parse_workers(data["services"]["cpu"], "CPU"),
                syshax_host=data["services"]["conductor"]["host"],
                syshax_port=data["services"]["conductor"]["port"],
                request_timeout=data["system"]["request_timeout"],
                model_name=model_name,
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


def fetch_model_name(workers: list[Worker], timeout: int) -> str:
    """
    从所有服务实例分别获取模型名称，并校验全体一致性。
    """
    def get_model_id_from_service(worker: Worker) -> str:
        try:
            resp = httpx.get(worker.models_url, timeout=timeout)
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
            raise RuntimeError(f"从 {worker.label} ({worker.base_url}) 获取模型失败: {e}") from e

    if not workers:
        raise RuntimeError("未配置任何服务实例，无法探测模型")

    model_ids: dict[str, str] = {}
    for worker in workers:
        model_ids[worker.label] = get_model_id_from_service(worker)

    unique_models = set(model_ids.values())
    if len(unique_models) != 1:
        detail = "\n".join(f"  {label}: {mid}" for label, mid in model_ids.items())
        raise RuntimeError(f"各服务实例的模型不一致！\n{detail}")

    model_name = next(iter(unique_models))
    Logger.info(f"模型一致性校验通过（共 {len(workers)} 个实例），使用模型: {model_name}")
    return model_name


def load_syshax_config() -> SyshaxConfig:
    raw = load_raw_config()

    # 先构造(model_name 为占位符)，用于按池探测模型
    temp_config = SyshaxConfig.from_dict(raw)

    # 动态获取模型名，并校验所有实例一致
    model_name = fetch_model_name(
        workers=[*temp_config.gpu_workers, *temp_config.cpu_workers],
        timeout=temp_config.request_timeout,
    )

    return SyshaxConfig.from_dict(raw, model_name=model_name)
