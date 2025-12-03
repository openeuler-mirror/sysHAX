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
Created: 2025-09-22
Desc: sysHAX核心调度引擎
"""

import asyncio
from src.core.metrics import MetricsService
from src.core.scheduler import Scheduler
from src.core.monitor import SystemMonitor
from src.utils.logger import Logger

class Engine:
    def __init__(self,
                 scheduler: Scheduler,
                 metrics_service: MetricsService) -> None:
        self._task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None
        self._running = False

        self.scheduler = scheduler
        self.metrics_service = metrics_service

    def start(self):
        if self._task and not self._task.done():
            Logger.warning("Engine is already running.")
            return

        self._running = True
        self._task = asyncio.create_task(self._engine_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        Logger.info("Engine started.")

    async def stop(self):
        self._running = False
        await self.scheduler.cancel_all_tasks()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        Logger.info("Engine stopped.")

    async def _engine_loop(self):
        while self._running:
            try:
                if self.scheduler.has_unfinshed_tasks():
                    await self.scheduler.scheduler()
                else:
                    await asyncio.sleep(1)  # 避免忙等
            except asyncio.CancelledError:
                Logger.info("Engine 调度循环被取消")
                break
            except Exception as e:
                Logger.error(f"Engine 调度循环异常: {e}", exc_info=True)

    async def _metrics_loop(self):
        last_state = False
        while self._running:
            try:
                has_running = self.scheduler.has_running_tasks()
                if has_running and not last_state:
                    await self.metrics_service.start()
                    last_state = True
                elif not has_running and last_state:
                    await asyncio.sleep(5)
                    if not self.scheduler.has_running_tasks():
                        await self.metrics_service.stop()
                        last_state = False
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                Logger.info("Engine 指标循环被取消")
                break
            except Exception as e:
                Logger.error(f"Engine 指标循环异常: {e}", exc_info=True)

