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

from src.utils.config import load_syshax_config
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 确保项目根目录在 sys.path 以导入模块
sys.path.insert(0, str(Path(__file__).parent))

# 在文件顶部 imports 之后，添加 BASE_DIR
BASE_DIR = Path(sys.path[0]).resolve()

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
        if target.exists():
            target.unlink()
        shutil.copy(example, target)
        logger.info("已生成配置文件：%s", target)
    except (OSError, shutil.Error):
        logger.exception("初始化配置失败")
        sys.exit(1)


def cmd_check_config() -> None:
    """检查 config.yaml 是否存在及合法"""
    try:
        _ = load_syshax_config()
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

def _set_gpu_host(cfg: dict[str, Any], value: str) -> None:
    cfg["services"]["gpu"]["host"] = str(value)

def _set_gpu_port(cfg: dict[str, Any], value: str) -> None:
    try:
        cfg["services"]["gpu"]["port"] = int(value)
    except ValueError:
        logger.error("GPU 服务端口必须为整数")
        sys.exit(1)

def _set_cpu_host(cfg: dict[str, Any], value: str) -> None:
    cfg["services"]["cpu"]["host"] = str(value)

def _set_cpu_port(cfg: dict[str, Any], value: str) -> None:
    try:
        cfg["services"]["cpu"]["port"] = int(value)
    except ValueError:
        logger.error("CPU 服务端口必须为整数")
        sys.exit(1)

def _set_conductor_host(cfg: dict[str, Any], value: str) -> None:
    cfg["services"]["conductor"]["host"] = int(value)

def _set_conductor_port(cfg: dict[str, Any], value: str) -> None:
    try:
        cfg["services"]["conductor"]["port"] = int(value)
    except ValueError:
        logger.error("sysHAX 服务端口必须为整数")
        sys.exit(1)

def _set_host(cfg: dict[str, Any], value: str) -> None:
    _set_gpu_host(cfg, value)
    _set_cpu_host(cfg, value)
    _set_conductor_host(cfg, value)

def _set_auto_pd_offload(cfg: dict[str, Any], value: str) -> None:
    val_lower = value.lower()
    if val_lower in ("true", "false"):
        cfg["decider"]["auto_pd_offload"] = val_lower == "true"
    else:
        logger.error("auto_pd_offload 必须为 true 或 false")
        sys.exit(1)

def _set_cpu_max_batch_size(cfg: dict[str, Any], value: str) -> None:
    try:
        cfg["decider"]["cpu_max_batch_size"] = int(value)
    except ValueError:
        logger.error("CPU 侧最大并发量必须为整数")
        sys.exit(1)

def _set_request_timeout(cfg: dict[str, Any], value: str) -> None:
    try:
        cfg["system"]["request_timeout"] = int(value)
    except ValueError:
        logger.error("请求超时时间必须为整数")
        sys.exit(1)

HANDLERS = {
    "host": _set_host,
    "gpu.host": _set_gpu_host,
    "gpu.port": _set_gpu_port,
    "cpu.host": _set_cpu_host,
    "cpu.port": _set_cpu_port,
    "conductor.host": _set_conductor_host,
    "conductor.port": _set_conductor_port,
    "cpu_max_batch_size": _set_cpu_max_batch_size,
    "auto_pd_offload": _set_auto_pd_offload,
    "request_timeout": _set_request_timeout,
}

def _handle(cfg: dict[str, Any], key: str, value: str) -> bool:
    """处理config"""
    handler = HANDLERS.get(key)
    if handler:
        handler(cfg, value)
        return True
    else:
        logger.error("不支持的键：%s", key)
        return False

def cmd_config(args: argparse.Namespace) -> None:
    """设置配置项"""
    key, value = args.key, args.value
    cfg_path = BASE_DIR / "config" / "config.yaml"
    if not cfg_path.exists():
        logger.info("配置文件不存在，自动初始化配置文件")
        cmd_init()
    cfg = _load_cfg(cfg_path)

    if _handle(cfg, key, value):
        _write_cfg(cfg_path, cfg, key, value)
    else:
        sys.exit(1)


def main() -> None:
    """使用 sysHAX 命令行工具"""
    parser = argparse.ArgumentParser(
        prog="syshax",
        usage="syshax [OPTIONS] COMMAND [ARGS]...",
        description="欢迎使用 sysHAX 命令行工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "可用命令:\n"
            "  run               启动 sysHAX 服务\n"
            "  init              生成 config/config.yaml（从示例文件复制）\n"
            "  check-config      检查 config.yaml 是否存在且合法\n"
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

    parser_config = subparsers.add_parser(
        "config",
        help='设置配置项；使用 "syshax config --help" 查看详细',
        description="设置或修改 config/config.yaml 中的某个配置项",
        usage="syshax config <key> <value>",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""可用 <key> 列表:
            host                                 同时更新所有服务（gpu/cpu/conductor）的 host
            gpu.host                             GPU 服务 host
            gpu.port                             GPU 服务 port
            cpu.host                             CPU 服务 host
            cpu.port                             CPU 服务 port
            conductor.host                       sysHAX 服务 host
            conductor.port                       sysHAX 服务 port
            auto_pd_offload                      是否开启自动 PD offload（true/false）
            cpu_max_batch_size                   CPU 侧最大并发量
            request_timeout                      请求超时时间（秒）
            """,
    )
    parser_config.add_argument("key", help="配置键，例如 gpu.port")
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
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
