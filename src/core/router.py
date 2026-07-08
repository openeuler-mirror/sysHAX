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
Desc:sysHAX 请求字段路由规则引擎

职责：
  - 持有一组 RoutingRule，按 priority 降序排列
  - match(request) 返回第一个命中规则的目标设备（"GPU"/"CPU"），未命中返回 None
  - 条件评估支持：数值比较、字符串匹配、布尔判断、列表元素数、字段存在性

与调度器的关系：
  RequestRouter 仅做"建议"，scheduler 决定是否采纳（软强制）。
  当建议设备无容量时，scheduler 回退到现有的指标驱动决策，不阻塞队列。
"""

from typing import Any

from src.utils.config import RoutingCondition, RoutingRule
from src.utils.logger import Logger

# 支持的操作符集合（用于启动时校验）
_NUMERIC_OPS = {"gte", "lte", "gt", "lt", "eq"}
_STRING_OPS = {"eq", "in"}
_LIST_OPS = {"min_count", "max_count"}
_EXIST_OPS = {"exists"}
_ALL_OPS = _NUMERIC_OPS | _STRING_OPS | _LIST_OPS | _EXIST_OPS


class RequestRouter:
    """
    请求字段路由规则引擎。

    使用方式：
        router = RequestRouter(config.routing_rules)
        hint = router.match(request_dict)   # "GPU" | "CPU" | None
    """

    def __init__(self, rules: list[RoutingRule]) -> None:
        # 按 priority 降序排序，优先级高的规则先匹配
        self._rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self._warn_unknown_ops()

    def match(self, request: dict[str, Any]) -> str | None:
        """
        遍历规则，返回第一个命中规则的目标设备。
        无规则命中时返回 None（调用方继续走指标决策）。
        """
        for rule in self._rules:
            if self._evaluate(request, rule.conditions):
                Logger.debug(
                    f"\033[1;36m[路由规则] 命中规则: 「{rule.name}」"
                    f"(priority={rule.priority})，建议设备: {rule.action_device}\033[0m"
                )
                return rule.action_device
        return None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _evaluate(self, request: dict[str, Any], conditions: list[RoutingCondition]) -> bool:
        """所有条件 AND 关系：任意一个不满足即返回 False"""
        return all(self._eval_one(request, cond) for cond in conditions)

    def _eval_one(self, request: dict[str, Any], cond: RoutingCondition) -> bool:
        """评估单个条件"""
        field = cond.field
        op = cond.operator
        expected = cond.value

        # ── 列表类型的特殊操作符（作用于列表元素数） ──────────────────
        if op in _LIST_OPS:
            val = request.get(field)
            if not isinstance(val, list):
                return False
            count = len(val)
            if op == "min_count":
                return count >= int(expected)
            if op == "max_count":
                return count <= int(expected)

        # ── 字段存在性 ────────────────────────────────────────────────
        if op == "exists":
            present = field in request and request[field] is not None
            return present == bool(expected)

        # ── 通用字段取值 ──────────────────────────────────────────────
        actual = request.get(field)
        if actual is None:
            return False   # 字段缺失时所有比较操作均不命中

        try:
            if op == "eq":
                return actual == expected
            if op == "in":
                return actual in expected
            if op == "gte":
                return actual >= expected
            if op == "lte":
                return actual <= expected
            if op == "gt":
                return actual > expected
            if op == "lt":
                return actual < expected
        except TypeError as e:
            Logger.debug(f"[路由规则] 条件比较类型错误 field={field} op={op}: {e}")
            return False

        Logger.debug(f"[路由规则] 未知操作符 op={op}，条件视为不满足")
        return False

    def _warn_unknown_ops(self) -> None:
        """启动时对未知操作符打 warning，尽早发现配置错误"""
        for rule in self._rules:
            for cond in rule.conditions:
                if cond.operator not in _ALL_OPS:
                    Logger.warning(
                        f"[路由规则] 规则「{rule.name}」包含未知操作符: "
                        f"field={cond.field}, op={cond.operator}，该条件将永远不命中"
                    )
