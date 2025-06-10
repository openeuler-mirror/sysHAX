#!/usr/bin/env python3

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
Desc:sysHAX 命令行工具
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import (
    CPU_HOST,
    CPU_PORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    GPU_HOST,
    GPU_PORT,
    SYSHAX_HOST,
    SYSHAX_PORT,
    load_config,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 确保项目根目录在 sys.path 以导入模块
sys.path.insert(0, str(Path(__file__).parent))

# 在文件顶部 imports 之后，添加 BASE_DIR
BASE_DIR = Path(sys.path[0]).resolve()

# 用于比较 keys 列表长度的常量
KEY_PARTS_TWO = 2


def get_version() -> str:
    """从 sysHAX.spec 中读取版本号"""
    spec_path = Path(__file__).resolve().parent / "sysHAX.spec"
    if spec_path.exists():
        try:
            for line in spec_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
        except (OSError, UnicodeDecodeError):
            logger.exception("读取版本文件失败")
    return "unknown"


def cmd_run() -> None:
    """启动服务"""
    from main import run as main_run

    main_run()


def cmd_version() -> None:
    """返回版本号"""
    logger.info(get_version())


def cmd_init() -> None:
    """生成 config/config.yaml"""
    example = BASE_DIR / "config" / "config.example.yaml"
    target = BASE_DIR / "config" / "config.yaml"
    try:
        shutil.copy(example, target)
        logger.info("已生成配置文件：%s", target)
    except (OSError, shutil.Error):
        logger.exception("初始化配置失败")
        sys.exit(1)


def cmd_check_config() -> None:
    """检查 config.yaml 是否存在及合法"""
    try:
        _ = load_config()
        logger.info("配置文件合法")
    except FileNotFoundError:
        logger.exception("配置文件不存在")
        sys.exit(1)
    except yaml.YAMLError:
        logger.exception("配置文件解析失败")
        sys.exit(1)
    except Exception:
        logger.exception("配置文件不合法")
        sys.exit(1)


def cmd_interfaces() -> None:
    """返回 gpu, cpu, conductor 三个 URL"""
    logger.info("gpu: http://%s:%s", GPU_HOST, GPU_PORT)
    logger.info("cpu: http://%s:%s", CPU_HOST, CPU_PORT)
    logger.info("conductor: http://%s:%s", SYSHAX_HOST, SYSHAX_PORT)


def cmd_model() -> None:
    """返回 model name, max_tokens, temperature"""
    logger.info("model name: %s", DEFAULT_MODEL)
    logger.info("max_tokens: %d", DEFAULT_MAX_TOKENS)
    logger.info("temperature: %s", DEFAULT_TEMPERATURE)


# ---------- cmd_config 辅助函数 ----------
def _load_cfg(path: Path) -> dict[str, Any]:
    """加载配置文件"""
    if not path.exists():
        logger.error("配置文件不存在，请先运行 syshax init")
        sys.exit(1)
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.exception("加载配置文件失败")
        sys.exit(1)


def _write_cfg(path: Path, cfg: dict[str, Any], key: str, value: str) -> None:
    """写入配置文件"""
    try:
        path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("已设置 %s 为 %s", key, value)
    except OSError:
        logger.exception("写入配置失败")
        sys.exit(1)


def _update_service(cfg: dict[str, Any], service: str, key: str, value: str) -> None:
    """更新服务（gpu/cpu/conductor）配置"""
    if service not in cfg["services"] or key not in cfg["services"][service]:
        logger.error("不支持的键: services.%s.%s", service, key)
        sys.exit(1)
    old = cfg["services"][service][key]
    if isinstance(old, int):
        cfg["services"][service][key] = int(value)
    elif isinstance(old, float):
        cfg["services"][service][key] = float(value)
    else:
        cfg["services"][service][key] = value


def _update_section(cfg: dict[str, Any], section: str, key: str, value: str) -> None:
    """更新通用配置节"""
    if section not in cfg or key not in cfg[section]:
        logger.error("不支持的键: %s.%s", section, key)
        sys.exit(1)
    old = cfg[section][key]
    if isinstance(old, int):
        cfg[section][key] = int(value)
    elif isinstance(old, float):
        cfg[section][key] = float(value)
    else:
        cfg[section][key] = value


def _update_model(cfg: dict[str, Any], key: str, value: str) -> None:
    """更新模型配置"""
    if key == "model":
        cfg["models"]["default"] = value
    elif key == "max_tokens":
        cfg["models"]["params"]["max_tokens"] = int(value)
    elif key == "temperature":
        cfg["models"]["params"]["temperature"] = float(value)
    elif key == "test_prompt":
        cfg["models"]["params"]["test_prompt"] = value
    elif key == "test_tokens":
        cfg["models"]["params"]["test_tokens"] = int(value)
    else:
        logger.error("不支持的键: %s", key)
        sys.exit(1)


def _handle_services(cfg: dict[str, Any], keys: list[str], value: str) -> None:
    """处理 services.* 修改"""
    if len(keys) == 1 and keys[0] == "host":
        for svc in cfg["services"]:
            _update_service(cfg, svc, "host", value)
    elif len(keys) == KEY_PARTS_TWO:
        svc, sub = keys
        _update_service(cfg, svc, sub, value)
    else:
        logger.error("不支持的键: services.%s", ".".join(keys))
        sys.exit(1)


def _handle_models(cfg: dict[str, Any], keys: list[str], value: str) -> None:
    """处理 models.* 修改"""
    if len(keys) == 1 and keys[0] == "default":
        _update_section(cfg, "models", "default", value)
    elif len(keys) == KEY_PARTS_TWO and keys[0] == "params":
        _update_model(cfg, keys[1], value)
    else:
        logger.error("不支持的键: models.%s", ".".join(keys))
        sys.exit(1)


def _handle_system(cfg: dict[str, Any], keys: list[str], value: str) -> None:
    """处理 system.* 修改"""
    if len(keys) != 1:
        logger.error("不支持的键: system.%s", ".".join(keys))
        sys.exit(1)
    _update_section(cfg, "system", keys[0], value)


def _handle_decider(cfg: dict[str, Any], keys: list[str], value: str) -> None:
    """处理 decider.* 修改"""
    if len(keys) != 1:
        logger.error("不支持的键: decider.%s", ".".join(keys))
        sys.exit(1)
    _update_section(cfg, "decider", keys[0], value)


def _handle_monitor(cfg: dict[str, Any], keys: list[str], value: str) -> None:
    """处理 monitor.* 修改"""
    if len(keys) != 1:
        logger.error("不支持的键: monitor.%s", ".".join(keys))
        sys.exit(1)
    _update_section(cfg, "monitor", keys[0], value)


def cmd_config(args: argparse.Namespace) -> None:
    """设置配置项"""
    key, value = args.key, args.value
    cfg_path = BASE_DIR / "config" / "config.yaml"
    if not cfg_path.exists():
        logger.info("配置文件不存在，自动初始化配置文件")
        cmd_init()
    cfg = _load_cfg(cfg_path)

    parts = key.split(".")
    handlers = {
        "services": _handle_services,
        "models": _handle_models,
        "system": _handle_system,
        "decider": _handle_decider,
        "monitor": _handle_monitor,
    }
    head, *rest = parts
    if head not in handlers:
        logger.error("不支持的键: %s", key)
        sys.exit(1)
    handlers[head](cfg, rest, value)
    _write_cfg(cfg_path, cfg, key, value)


def main() -> None:
    """SysHAX 命令行工具"""
    parser = argparse.ArgumentParser(
        prog="syshax",
        usage="syshax [OPTIONS] COMMAND [ARGS]...",
        description="SysHAX 命令行工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "可用命令:\n"
            "  run               启动 sysHAX 服务\n"
            "  init              生成 config/config.yaml（从示例文件复制）\n"
            "  check-config      检查 config.yaml 是否存在且合法\n"
            "  interfaces        打印 GPU/CPU/Conductor 三个服务的 URL\n"
            "  model             打印当前模型名称、max_tokens、temperature\n"
            '  config            设置配置项；使用 "syshax config --help" 查看详细'
        ),
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("--version", action="store_true", help="打印当前版本号并退出")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", help=argparse.SUPPRESS)

    subparsers.add_parser("run", help="启动服务")
    subparsers.add_parser("init", help="生成 config/config.yaml")
    subparsers.add_parser("check-config", help="检查 config.yaml 是否存在及合法")
    subparsers.add_parser("interfaces", help="返回 gpu, cpu, conductor 三个 URL")
    subparsers.add_parser("model", help="返回 model name, max_tokens, temperature")

    parser_config = subparsers.add_parser(
        "config",
        help='设置配置项；使用 "syshax config --help" 查看详细',
        description="设置或修改 config/config.yaml 中的某个配置项",
        usage="syshax config <key> <value>",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""可用 <key> 列表:
  services.host                        同时更新所有服务（gpu/cpu/conductor）的 host
  services.gpu.host                    GPU 服务 host
  services.gpu.port                    GPU 服务 port
  services.cpu.host                    CPU 服务 host
  services.cpu.port                    CPU 服务 port
  services.conductor.host              Conductor 服务 host
  services.conductor.port              Conductor 服务 port

  models.default                       默认模型名称
  models.params.max_tokens             生成的最大 token 数量
  models.params.temperature            随机性参数 (0.0–1.0)
  models.params.test_prompt            性能测试使用的提示语
  models.params.test_tokens            性能测试生成的 token 数量

  system.request_timeout               请求超时时间（秒）

  decider.max_num_seqs                 最大并发序列数
  decider.gpu_cache_threshold          GPU 缓存使用率阈值（%）
  decider.cpu_throughput_threshold     CPU 吞吐量阈值
  decider.token_limit_multiplier       任务转移到 CPU 的 token 限制倍数
  decider.token_limit_min              任务转移到 CPU 的 token 限制最小值
  decider.token_limit_max              任务转移到 CPU 的 token 限制最大值

  monitor.interval                     监控间隔（秒）
""",
    )
    parser_config.add_argument("key", help="配置键，例如 services.gpu.port 或 models.params.temperature")
    parser_config.add_argument("value", help="配置值")

    args = parser.parse_args()
    if args.version:
        cmd_version()
    elif args.command == "run":
        cmd_run()
    elif args.command == "init":
        cmd_init()
    elif args.command == "check-config":
        cmd_check_config()
    elif args.command == "interfaces":
        cmd_interfaces()
    elif args.command == "model":
        cmd_model()
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
