"""
Copyright (c) KylinSoft Co., Ltd. [2026].All rights reserved.

sysHAX is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
    http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
See the Mulan PSL v2 for more details.
Created: 2026-07-08
Desc:sysHAX 心跳探活模块

设计要点：
  - 每个 Worker 拥有独立的探活协程，互不干扰
  - 通过回调（on_down / on_up）解耦探活逻辑与调度逻辑
  - 连续失败 fail_threshold 次后触发 on_down；恢复后触发 on_up
  - 探活端点：{worker.base_url}/health（vLLM 默认暴露）
"""

import asyncio
from collections.abc import Callable, Awaitable
import httpx

from src.utils.config import SyshaxConfig, Worker
from src.utils.logger import Logger

# 回调类型别名
WorkerCallback = Callable[[Worker], Awaitable[None] | None]


class HealthChecker:
    """
    后台心跳探活器。

    为每个 Worker 启动独立的无限探活协程，根据探活结果调用回调：
      - on_worker_down(worker)：连续失败达到阈值时调用（节点摘除）
      - on_worker_up(worker)  ：节点从故障状态恢复时调用（节点重入）
    """

    def __init__(
        self,
        workers: list[Worker],
        config: SyshaxConfig,
        on_worker_down: WorkerCallback,
        on_worker_up: WorkerCallback,
    ) -> None:
        self._workers = workers
        self._interval = config.heartbeat_interval
        self._timeout = config.heartbeat_timeout
        self._fail_threshold = config.heartbeat_fail_threshold
        self._on_worker_down = on_worker_down
        self._on_worker_up = on_worker_up

        # 每个 worker 的连续失败次数
        self._fail_count: dict[Worker, int] = {w: 0 for w in workers}
        # 每个 worker 当前是否处于健康状态（初始假定健康）
        self._is_healthy: dict[Worker, bool] = {w: True for w in workers}

        self._tasks: list[asyncio.Task] = []
        self._client: httpx.AsyncClient | None = None

    def start(self) -> None:
        """启动所有 Worker 的探活协程（须在事件循环运行时调用）"""
        if self._tasks:
            Logger.warning("HealthChecker 已经在运行中，忽略重复启动")
            return
        self._client = httpx.AsyncClient()
        for worker in self._workers:
            task = asyncio.create_task(
                self._probe_loop(worker),
                name=f"heartbeat-{worker.label}"
            )
            self._tasks.append(task)
        Logger.info(
            f"HealthChecker 已启动，监控 {len(self._workers)} 个节点，"
            f"间隔={self._interval}s，超时={self._timeout}s，"
            f"摘除阈值={self._fail_threshold}次"
        )

    async def stop(self) -> None:
        """停止所有探活协程并释放 HTTP 客户端"""
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._client:
            await self._client.aclose()
            self._client = None
        Logger.info("HealthChecker 已停止")

    async def _probe_loop(self, worker: Worker) -> None:
        """单 Worker 探活无限循环"""
        while True:
            try:
                await self._probe_once(worker)
            except asyncio.CancelledError:
                Logger.debug(f"探活协程已取消: {worker.label}")
                break
            except Exception as e:
                # 探活函数内部已捕获常见异常；此处兜底防止协程意外退出
                Logger.error(f"探活协程发生未预期异常 [{worker.label}]: {e}", exc_info=True)
            await asyncio.sleep(self._interval)

    async def _probe_once(self, worker: Worker) -> None:
        """
        执行一次探活请求并更新状态。

        成功：fail_count 归零；若之前为故障状态，触发 on_worker_up。
        失败：fail_count 递增；达到阈值且当前健康，触发 on_worker_down。
        """
        success = False
        try:
            assert self._client is not None
            resp = await self._client.get(worker.health_url, timeout=self._timeout)
            success = resp.status_code == 200
            if not success:
                Logger.debug(f"探活失败 [{worker.label}]：HTTP {resp.status_code}")
        except httpx.TimeoutException:
            Logger.debug(f"探活超时 [{worker.label}]（>{self._timeout}s）")
        except httpx.ConnectError:
            Logger.debug(f"探活连接失败 [{worker.label}]：无法连接 {worker.health_url}")
        except Exception as e:
            Logger.debug(f"探活异常 [{worker.label}]：{e}")

        if success:
            self._fail_count[worker] = 0
            if not self._is_healthy[worker]:
                # 从故障中恢复
                self._is_healthy[worker] = True
                Logger.info(
                    f"\033[1;32m[心跳] 节点恢复健康，重新加入调度池: {worker.label}\033[0m"
                )
                result = self._on_worker_up(worker)
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    await result
        else:
            self._fail_count[worker] += 1
            current = self._fail_count[worker]
            if current >= self._fail_threshold and self._is_healthy[worker]:
                # 连续失败达到阈值，触发摘除
                self._is_healthy[worker] = False
                Logger.warning(
                    f"\033[1;33m[心跳] 节点连续失败 {current} 次，已从调度池摘除: "
                    f"{worker.label}\033[0m"
                )
                result = self._on_worker_down(worker)
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    await result
            elif not self._is_healthy[worker]:
                # 节点已在故障状态，继续等待恢复，每次失败静默记录
                Logger.debug(
                    f"节点仍处于故障状态 [{worker.label}]，累计失败 {current} 次"
                )
            else:
                # 尚未达到阈值，仅打 debug
                Logger.debug(
                    f"探活失败 [{worker.label}]，连续失败 {current}/{self._fail_threshold} 次"
                )
