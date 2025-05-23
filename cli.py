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
from urllib.parse import urlparse, urlunparse

import yaml

from src.utils.config import (
    CPU_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    GPU_URL,
    SYSHAX_HOST,
    SYSHAX_PORT,
    load_config,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 确保项目根目录在 sys.path 以导入模块
sys.path.insert(0, str(Path(__file__).parent))

# 导入主程序和配置模块常量
main_run = None

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
    except AssertionError:
        logger.exception("配置文件不存在")
        sys.exit(1)
    except Exception:
        logger.exception("配置文件不合法")
        sys.exit(1)


def cmd_interfaces() -> None:
    """返回 gpu, cpu, conductor 三个 URL"""
    logger.info("gpu: %s", GPU_URL)
    logger.info("cpu: %s", CPU_URL)
    logger.info("conductor: http://%s:%s", SYSHAX_HOST, SYSHAX_PORT)


def cmd_model() -> None:
    """返回 model name, max_tokens, temperature"""
    logger.info("model name: %s", DEFAULT_MODEL)
    logger.info("max_tokens: %d", DEFAULT_MAX_TOKENS)
    logger.info("temperature: %s", DEFAULT_TEMPERATURE)


def update_url_host_port(url: str, host: str | None = None, port: int | None = None) -> str:
    """更新 URL 中的 host 或 port，并保留原始占位符"""
    p = urlparse(url)
    orig_netloc = p.netloc
    if ":" in orig_netloc:
        orig_hostname, orig_port = orig_netloc.split(":", 1)
    else:
        orig_hostname, orig_port = orig_netloc, None
    new_hostname = host or orig_hostname
    new_port = str(port) if port is not None else orig_port
    new_netloc = f"{new_hostname}:{new_port}" if new_port else new_hostname
    return urlunparse(p._replace(netloc=new_netloc))


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
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        logger.info("已设置 %s 为 %s", key, value)
    except OSError:
        logger.exception("写入配置失败")
        sys.exit(1)


def _update_gpu(cfg: dict[str, Any], sub: str, value: str) -> None:
    """更新 GPU 配置"""
    url = cfg["services"]["gpu"]["url"]
    murl = cfg["services"]["gpu"]["metrics_url"]
    if sub == "host":
        cfg["services"]["gpu"]["url"] = update_url_host_port(url, host=value)
        cfg["services"]["gpu"]["metrics_url"] = update_url_host_port(murl, host=value)
    elif sub == "port":
        p = int(value)
        cfg["services"]["gpu"]["url"] = update_url_host_port(url, port=p)
        cfg["services"]["gpu"]["metrics_url"] = update_url_host_port(murl, port=p)
    else:
        logger.error("不支持的键: gpu.%s", sub)
        sys.exit(1)


def _update_cpu(cfg: dict[str, Any], sub: str, value: str) -> None:
    """更新 CPU 配置"""
    url = cfg["services"]["cpu"]["url"]
    murl = cfg["services"]["cpu"]["metrics_url"]
    if sub == "host":
        cfg["services"]["cpu"]["url"] = update_url_host_port(url, host=value)
        cfg["services"]["cpu"]["metrics_url"] = update_url_host_port(murl, host=value)
    elif sub == "port":
        p = int(value)
        cfg["services"]["cpu"]["url"] = update_url_host_port(url, port=p)
        cfg["services"]["cpu"]["metrics_url"] = update_url_host_port(murl, port=p)
    else:
        logger.error("不支持的键: cpu.%s", sub)
        sys.exit(1)


def _update_conductor(cfg: dict[str, Any], sub: str, value: str) -> None:
    """更新 conductor 配置"""
    if sub == "host":
        cfg["services"]["conductor"]["host"] = value
    elif sub == "port":
        cfg["services"]["conductor"]["port"] = int(value)
    else:
        logger.error("不支持的键: conductor.%s", sub)
        sys.exit(1)


def _update_model(cfg: dict[str, Any], key: str, value: str) -> None:
    """更新模型配置"""
    if key == "model":
        cfg["models"]["default"] = value
    elif key == "maxtokens":
        cfg["models"]["params"]["max_tokens"] = int(value)
    elif key == "temperature":
        cfg["models"]["params"]["temperature"] = float(value)
    else:
        logger.error("不支持的键: %s", key)
        sys.exit(1)


def cmd_config(args: argparse.Namespace) -> None:
    """设置配置项"""
    key, value = args.key, args.value
    cfg_path = BASE_DIR / "config" / "config.yaml"
    # 如果 config.yaml 不存在，先自动初始化
    if not cfg_path.exists():
        logger.info("配置文件不存在，自动初始化配置文件")
        cmd_init()
    cfg = _load_cfg(cfg_path)
    # 一次更新所有服务的 host
    if key == "host":
        _update_gpu(cfg, "host", value)
        _update_cpu(cfg, "host", value)
        _update_conductor(cfg, "host", value)
    elif "." in key:
        service, sub = key.split(".", 1)
        if service == "gpu":
            _update_gpu(cfg, sub, value)
        elif service == "cpu":
            _update_cpu(cfg, sub, value)
        elif service == "conductor":
            _update_conductor(cfg, sub, value)
        else:
            logger.error("不支持的键: %s", key)
            sys.exit(1)
    else:
        _update_model(cfg, key, value)
    _write_cfg(cfg_path, cfg, key, value)


def main() -> None:
    """SysHAX 命令行工具"""
    parser = argparse.ArgumentParser(prog="syshax")
    parser.add_argument("--version", action="store_true", help="返回版本号")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="启动服务")
    subparsers.add_parser("init", help="生成 config/config.yaml")
    subparsers.add_parser("check-config", help="检查 config.yaml 是否存在及合法")
    subparsers.add_parser("interfaces", help="返回 gpu, cpu, conductor 三个 URL")
    subparsers.add_parser("model", help="返回 model name, max_tokens, temperature")

    parser_config = subparsers.add_parser("config", help="设置配置项")
    parser_config.add_argument("key", help="配置键，例如 gpu.host")
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
