"""
自主探索模块：实现自主房间探索和避障
- 不需要目标点，自动探索房间
- 实时避障处理
- 与语义地图集成
- 持续探索直到按键终止
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


class AutonomousExplorer:
    """自主探索器 - 自动探索房间"""

    def __init__(self, grid_size: float = 0.15):
        self.grid_size = grid_size
        self.is_enabled = False
        self.is_exploring = False

        # 探索状态
        self.exploration_state = {
            "visited_positions": set(),  # 已访问的位置
            "current_path": [],  # 当前路径
            "path_idx": 0,
            "stuck_count": 0,
            "last_position": None,
            "exploration_start_time": None,
            "total_distance": 0.0,
        }

        # 避障状态
        self.obstacle_state = {
            "blocked_count": 0,
            "avoidance_attempts": 0,
            "last_avoidance_time": None,
        }

    def enable(self):
        """启用自主探索"""
        self.is_enabled = True
        self.is_exploring = True
        self.exploration_state["exploration_start_time"] = time.time()
        print("✅ 自主探索已启用 - 开始探索房间")

    def disable(self):
        """禁用自主探索"""
        self.is_enabled = False
        self.is_exploring = False

        # 输出探索统计
        elapsed = time.time() - self.exploration_state.get("exploration_start_time", time.time())
        visited = len(self.exploration_state["visited_positions"])
        distance = self.exploration_state["total_distance"]

        print(f"❌ 自主探索已停止")
        print(f"   📊 探索统计:")
        print(f"      - 探索时间: {elapsed:.1f}秒")
        print(f"      - 访问位置: {visited}个")
        print(f"      - 总距离: {distance:.2f}米")

    def execute_exploration_step(self, controller, event) -> Any:
        """执行探索的一步 - 持续探索直到禁用"""
        if not self.is_enabled or not self.is_exploring:
            return event

        try:
            current_pos = self._get_agent_pos(event)
            current_pos_rounded = (self._round2(current_pos[0]), self._round2(current_pos[1]))

            # 记录访问的位置
            self.exploration_state["visited_positions"].add(current_pos_rounded)

            # 计算移动距离
            last_pos = self.exploration_state["last_position"]
            if last_pos:
                dist = ((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)**0.5
                self.exploration_state["total_distance"] += dist
            self.exploration_state["last_position"] = current_pos

            # 获取可达点
            reachable_event = controller.step(action="GetReachablePositions")
            reachable_points = reachable_event.metadata.get("actionReturn", []) or []

            if not reachable_points:
                print("⚠ 无可达点，停止探索")
                self.is_exploring = False
                return event

            # 构建可达点集合
            reach_set = {(self._round2(p.get("x", 0.0)), self._round2(p.get("z", 0.0)))
                        for p in reachable_points}

            # 找到未访问的可达点
            unvisited = [p for p in reach_set if p not in self.exploration_state["visited_positions"]]

            if unvisited:
                # 选择最近的未访问点
                target = min(unvisited,
                           key=lambda p: (p[0] - current_pos[0])**2 + (p[1] - current_pos[1])**2)
            else:
                # 所有点都访问过，随机选择一个
                target = list(reach_set)[np.random.randint(0, len(reach_set))]

            # 规划到目标的路径
            path = self._plan_exploration_path(current_pos_rounded, target, reach_set)

            if path and len(path) > 1:
                # 执行路径的第一步
                next_pos = path[1]
                event = self._move_to_position(controller, event, next_pos)

                # 检查是否卡住
                if not event.metadata.get("lastActionSuccess", False):
                    self.obstacle_state["blocked_count"] += 1
                    if self.obstacle_state["blocked_count"] > 3:
                        # 尝试避障
                        event = self._try_obstacle_avoidance(controller, event)
                        self.obstacle_state["blocked_count"] = 0
                else:
                    self.obstacle_state["blocked_count"] = 0
            else:
                # 无法规划路径，尝试随机移动
                event = self._random_move(controller, event)

            return event

        except Exception as e:
            print(f"⚠ 探索执行失败: {e}")
            return event

    def _plan_exploration_path(self, start: Tuple[float, float],
                               target: Tuple[float, float],
                               reach_set: set) -> Optional[List[Tuple[float, float]]]:
        """规划探索路径"""
        try:
            if start == target:
                return [start]

            queue = deque([[start]])
            visited = {start}

            while queue:
                path = queue.popleft()
                current = path[-1]

                if current == target:
                    return path

                # 探索4个方向
                for dx, dz in [(0, self.grid_size), (0, -self.grid_size),
                              (self.grid_size, 0), (-self.grid_size, 0)]:
                    next_pos = (self._round2(current[0] + dx),
                               self._round2(current[1] + dz))

                    if next_pos in reach_set and next_pos not in visited:
                        visited.add(next_pos)
                        queue.append(path + [next_pos])

            return None
        except Exception as e:
            print(f"⚠ 探索路径规划失败: {e}")
            return None

    def _move_to_position(self, controller, event, target: Tuple[float, float]) -> Any:
        """移动到指定位置"""
        current_pos = self._get_agent_pos(event)

        # 计算方向
        dx = target[0] - current_pos[0]
        dz = target[1] - current_pos[1]

        # 确定目标方向
        if abs(dz) > abs(dx):
            target_yaw = 0 if dz > 0 else 180
        else:
            target_yaw = 90 if dx > 0 else 270

        current_yaw = round(event.metadata["agent"]["rotation"]["y"]) % 360
        rot_diff = self._normalize_deg(target_yaw - current_yaw)

        if abs(rot_diff) > 1:
            # 旋转
            rotate_deg = min(abs(rot_diff), 90)
            return controller.step(action=("RotateRight" if rot_diff > 0 else "RotateLeft"),
                                 degrees=rotate_deg)
        else:
            # 前进
            return controller.step(action="MoveAhead", forceAction=True)

    def _try_obstacle_avoidance(self, controller, event) -> Any:
        """尝试避障"""
        avoidance_attempts = self.obstacle_state["avoidance_attempts"]

        # 尝试不同的避障策略
        strategies = [
            ("RotateLeft", 15),
            ("RotateRight", 15),
            ("RotateLeft", 45),
            ("RotateRight", 45),
            ("MoveBack", 0),
        ]

        if avoidance_attempts < len(strategies):
            action, degrees = strategies[avoidance_attempts]
            if degrees > 0:
                event = controller.step(action=action, degrees=degrees)
            else:
                event = controller.step(action=action)
            self.obstacle_state["avoidance_attempts"] += 1
        else:
            # 重置避障计数
            self.obstacle_state["avoidance_attempts"] = 0

        return event

    def _random_move(self, controller, event) -> Any:
        """随机移动"""
        actions = ["MoveAhead", "RotateLeft", "RotateRight", "MoveBack"]
        action = actions[np.random.randint(0, len(actions))]
        return controller.step(action=action)

    @staticmethod
    def _get_agent_pos(event) -> Tuple[float, float]:
        """获取Agent位置"""
        agent = event.metadata.get("agent", {})
        pos = agent.get("position", {"x": 0.0, "z": 0.0})
        return (pos.get("x", 0.0), pos.get("z", 0.0))

    @staticmethod
    def _get_agent_yaw(event) -> float:
        """获取Agent方向"""
        return event.metadata.get("agent", {}).get("rotation", {}).get("y", 0.0)

    @staticmethod
    def _normalize_deg(angle: float) -> float:
        """归一化角度到[-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    @staticmethod
    def _round2(val: float) -> float:
        """四舍五入到2位小数"""
        return round(val, 2)

    def get_status(self) -> Dict[str, Any]:
        """获取探索状态"""
        elapsed = 0
        if self.exploration_state.get("exploration_start_time"):
            elapsed = time.time() - self.exploration_state["exploration_start_time"]

        return {
            "enabled": self.is_enabled,
            "exploring": self.is_exploring,
            "visited_positions": len(self.exploration_state["visited_positions"]),
            "total_distance": f"{self.exploration_state['total_distance']:.2f}m",
            "elapsed_time": f"{elapsed:.1f}s",
            "blocked_count": self.obstacle_state["blocked_count"],
        }

