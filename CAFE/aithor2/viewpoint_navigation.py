"""
视角变化导航模块：基于 Frontier 采样的多视角探索
- Frontier = "已知 free 区域与 unknown 区域的边界"
- 在 Frontier 附近采样候选位置
- 每个候选位置尝试多个视角（仅 45° 或 90°）
- 选择能覆盖最多未知区域的(位置, 视角)组合

用于对比/补充自主探索的单视角策略。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque
import time


class ViewpointNavigator:
    """多视角导航器 - 考虑视角变化的探索"""

    # AI2-THOR 允许的旋转角度（度数）
    ALLOWED_ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # 限制为仅 45° 和 90° 增量的旋转
    ALLOWED_ROTATIONS_CONSTRAINED = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, grid_size: float = 0.15, cluster_radius: float = 0.5):
        """
        初始化视角导航器
        
        Args:
            grid_size: 栅格大小（米）
            cluster_radius: Frontier 聚类半径（米）- 避免采样太近
        """
        self.grid_size = grid_size
        self.cluster_radius = cluster_radius
        self.is_enabled = False
        self.is_exploring = False

        # 探索状态
        self.exploration_state = {
            "visited_positions": set(),  # 已访问的位置
            "visited_views": set(),  # 已访问的(位置, 旋转)对
            "frontier_history": [],  # Frontier 点聚类记录
            "current_target": None,  # 当前目标 (pos, rotation)
            "current_path": [],
            "path_idx": 0,
            "stuck_count": 0,
            "last_position": None,
            "last_rotation": None,
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
        """启用多视角探索"""
        self.is_enabled = True
        self.is_exploring = True
        self.exploration_state["exploration_start_time"] = time.time()
        print("✅ 多视角导航已启用 - 开始探索房间（考虑视角变化）")

    def disable(self):
        """禁用多视角探索"""
        self.is_enabled = False
        self.is_exploring = False

        elapsed = time.time() - self.exploration_state.get("exploration_start_time", time.time())
        visited_pos = len(self.exploration_state["visited_positions"])
        visited_views = len(self.exploration_state["visited_views"])
        distance = self.exploration_state["total_distance"]

        print(f"❌ 多视角导航已停止")
        print(f"   📊 探索统计:")
        print(f"      - 探索时间: {elapsed:.1f}秒")
        print(f"      - 访问位置: {visited_pos}个")
        print(f"      - 访问视角数: {visited_views}个")
        print(f"      - 总距离: {distance:.2f}米")

    def execute_exploration_step(self, controller, event) -> Any:
        """执行多视角探索的一步"""
        if not self.is_enabled or not self.is_exploring:
            return event

        try:
            current_pos = self._get_agent_pos(event)
            current_pos_rounded = (self._round2(current_pos[0]), self._round2(current_pos[1]))
            current_rot = self._get_agent_rotation(event)

            # 记录访问的位置和视角
            self.exploration_state["visited_positions"].add(current_pos_rounded)
            self.exploration_state["visited_views"].add((current_pos_rounded, current_rot))

            # 计算移动距离
            last_pos = self.exploration_state["last_position"]
            if last_pos:
                dist = ((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)**0.5
                self.exploration_state["total_distance"] += dist
            self.exploration_state["last_position"] = current_pos
            self.exploration_state["last_rotation"] = current_rot

            # 获取可达点
            reachable_event = controller.step(action="GetReachablePositions")
            reachable_points = reachable_event.metadata.get("actionReturn", []) or []

            if not reachable_points:
                print("⚠ 无可达点，停止多视角探索")
                self.is_exploring = False
                return event

            reach_set = {(self._round2(p.get("x", 0.0)), self._round2(p.get("z", 0.0)))
                        for p in reachable_points}

            # 【新增】检测 Frontier：未访问过的邻接点
            unvisited = [p for p in reach_set if p not in self.exploration_state["visited_positions"]]

            # 【新增】如果有未访问点，聚类后生成候选 (位置, 视角) 对
            if unvisited:
                candidates = self._generate_viewpoint_candidates(current_pos_rounded, unvisited)
                if candidates:
                    # 选择最有潜力的候选视角
                    target_pos, target_rot = self._select_best_viewpoint(candidates, current_pos, current_rot)
                    self.exploration_state["current_target"] = (target_pos, target_rot)
                    print(f"🎯 目标视角: 位置={target_pos}, 旋转={target_rot}°")
                else:
                    # 无候选，回退到最近的未访问点
                    target = min(unvisited,
                               key=lambda p: (p[0] - current_pos[0])**2 + (p[1] - current_pos[1])**2)
                    self.exploration_state["current_target"] = (target, current_rot)
                    print(f"🔄 回退到最近未访问点: {target}")
            else:
                # 所有点都访问过，尝试旋转以改变视角
                next_rot = self._get_next_viewpoint_rotation(current_rot)
                if next_rot != current_rot:
                    print(f"🔁 尝试改变视角到 {next_rot}°")
                    return self._rotate_to(controller, event, next_rot)
                else:
                    # 所有位置和视角都访问过，随机选择
                    target = list(reach_set)[np.random.randint(0, len(reach_set))]
                    self.exploration_state["current_target"] = (target, self._get_next_viewpoint_rotation(current_rot))

            # 执行当前目标
            target_pos, target_rot = self.exploration_state["current_target"]
            
            # 首先移动到位置
            if target_pos != current_pos_rounded:
                event = self._move_to_position(controller, event, target_pos)
                if not event.metadata.get("lastActionSuccess", False):
                    self.obstacle_state["blocked_count"] += 1
                    if self.obstacle_state["blocked_count"] > 3:
                        event = self._try_obstacle_avoidance(controller, event)
                        self.obstacle_state["blocked_count"] = 0
                else:
                    self.obstacle_state["blocked_count"] = 0
            else:
                # 已在目标位置，调整视角
                current_rot = self._get_agent_rotation(event)
                if target_rot != current_rot:
                    event = self._rotate_to(controller, event, target_rot)
                else:
                    # 位置和视角都到达，继续探索
                    pass

            return event

        except Exception as e:
            print(f"⚠ 多视角探索执行失败: {e}")
            return event

    def _generate_viewpoint_candidates(self, 
                                      center_pos: Tuple[float, float],
                                      unvisited_neighbors: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], int]]:
        """
        从未访问的邻接点（Frontier）生成候选视角对
        
        Args:
            center_pos: 当前位置
            unvisited_neighbors: 未访问的邻接点列表
            
        Returns:
            [(位置, 旋转角度), ...]
        """
        if not unvisited_neighbors:
            return []

        # 【第1步】聚类：避免采样太近的点
        clusters = self._cluster_frontier_points(unvisited_neighbors)

        candidates = []
        
        # 【第2步】为每个聚类代表生成视角候选
        for cluster_center in clusters:
            # 计算指向该点的方向
            dx = cluster_center[0] - center_pos[0]
            dz = cluster_center[1] - center_pos[1]
            
            # 计算目标方向（0-360）
            target_yaw = self._pos_to_yaw(dx, dz)
            
            # 【第3步】为该方向附近生成 45° 或 90° 朝向的候选
            for rot in self.ALLOWED_ROTATIONS:
                # 只选择面向 Frontier 方向的朝向（范围内）
                angle_diff = abs(self._normalize_deg(rot - target_yaw))
                if angle_diff <= 90:  # 宽松范围：±90°
                    candidates.append((cluster_center, rot))

        # 按角度差升序排序（优先选择最接近 Frontier 方向的）
        candidates.sort(key=lambda c: abs(self._normalize_deg(c[1] - target_yaw)))

        return candidates

    def _cluster_frontier_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        聚类 Frontier 点，避免采样太近的点
        
        使用简单的均值平移聚类
        """
        if not points:
            return []
        
        if len(points) <= 3:
            return points

        clusters = []
        remaining = set(points)

        while remaining:
            # 选择任意一点作为种子
            seed = remaining.pop()
            cluster = [seed]

            # 扩展聚类
            for point in list(remaining):
                dist_to_seed = ((point[0] - seed[0])**2 + (point[1] - seed[1])**2)**0.5
                if dist_to_seed <= self.cluster_radius:
                    cluster.append(point)
                    remaining.discard(point)

            # 计算聚类中心
            cx = np.mean([p[0] for p in cluster])
            cz = np.mean([p[1] for p in cluster])
            clusters.append((self._round2(cx), self._round2(cz)))

        return clusters

    def _select_best_viewpoint(self,
                              candidates: List[Tuple[Tuple[float, float], int]],
                              current_pos: Tuple[float, float],
                              current_rot: int) -> Tuple[Tuple[float, float], int]:
        """
        从候选视角中选择最有潜力的一个
        
        优先级：
        1. 未访问过的视角
        2. 距离最近
        3. 角度转移最小
        """
        # 过滤掉已访问的视角
        visited = self.exploration_state["visited_views"]
        unvisited_candidates = [c for c in candidates if c not in visited]

        if unvisited_candidates:
            candidates = unvisited_candidates

        # 按距离和角度转移排序
        def score(candidate):
            pos, rot = candidate
            # 距离：越小越好
            dist = ((pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2)**0.5
            # 角度转移：越小越好
            rot_delta = abs(self._normalize_deg(rot - current_rot))
            # 综合评分（距离权重更高）
            return (dist, rot_delta)

        best = min(candidates, key=score)
        return best

    def _get_next_viewpoint_rotation(self, current_rot: int) -> int:
        """
        获取下一个视角（如果已访问所有位置）
        依次尝试 45°, 90°, 135°, ... 的增量旋转
        """
        preferred_rotations = [45, 90, 135, 180]

        for delta in preferred_rotations:
            next_rot = (current_rot + delta) % 360
            # 检查这个视角是否已访问过
            if (self.exploration_state["last_position"] is not None):
                pos_rounded = (self._round2(self.exploration_state["last_position"][0]),
                              self._round2(self.exploration_state["last_position"][1]))
                if (pos_rounded, next_rot) not in self.exploration_state["visited_views"]:
                    return next_rot

        # 所有视角都访问过，返回当前旋转
        return current_rot

    def _move_to_position(self, controller, event, target: Tuple[float, float]) -> Any:
        """移动到指定位置"""
        current_pos = self._get_agent_pos(event)
        current_rot = self._get_agent_rotation(event)

        dx = target[0] - current_pos[0]
        dz = target[1] - current_pos[1]

        # 计算目标方向
        target_yaw = self._pos_to_yaw(dx, dz)

        # 旋转到目标方向（选择 45° 或 90° 增量）
        rot_diff = self._normalize_deg(target_yaw - current_rot)
        
        if abs(rot_diff) > 5:
            # 需要旋转：选择最接近的 45°/90° 增量
            rotate_deg = self._snap_to_allowed_rotation(rot_diff)
            return controller.step(action=("RotateRight" if rot_diff > 0 else "RotateLeft"),
                                 degrees=rotate_deg)
        else:
            # 方向对齐，前进
            return controller.step(action="MoveAhead", forceAction=True)

    def _rotate_to(self, controller, event, target_rotation: int) -> Any:
        """旋转到指定角度"""
        current_rot = self._get_agent_rotation(event)
        rot_diff = self._normalize_deg(target_rotation - current_rot)

        if abs(rot_diff) <= 5:
            return event

        # 选择最接近目标的 45°/90° 增量旋转
        rotate_deg = self._snap_to_allowed_rotation(rot_diff)
        return controller.step(action=("RotateRight" if rot_diff > 0 else "RotateLeft"),
                             degrees=rotate_deg)

    def _try_obstacle_avoidance(self, controller, event) -> Any:
        """尝试避障"""
        strategies = [
            ("RotateLeft", 45),
            ("RotateRight", 45),
            ("RotateLeft", 90),
            ("RotateRight", 90),
            ("MoveBack", 0),
        ]

        idx = self.obstacle_state["avoidance_attempts"] % len(strategies)
        action, degrees = strategies[idx]

        if degrees > 0:
            event = controller.step(action=action, degrees=degrees)
        else:
            event = controller.step(action=action)

        self.obstacle_state["avoidance_attempts"] += 1
        return event

    # ——— 辅助方法 ———

    @staticmethod
    def _snap_to_allowed_rotation(angle_diff: float) -> int:
        """
        将旋转角度对齐到最接近的 45° 或 90°
        AI2-THOR 允许任意角度，我们限制为 45°/90° 增量
        """
        # 允许的增量：45, 90, 135（或 -45, -90, -135）
        allowed_increments = [45, 90, 135]
        
        abs_diff = abs(angle_diff)
        best_increment = min(allowed_increments, key=lambda x: abs(x - abs_diff))
        
        return best_increment if angle_diff > 0 else -best_increment

    @staticmethod
    def _pos_to_yaw(dx: float, dz: float) -> float:
        """
        从位置差计算 yaw 角度（0-360）
        yaw = 0: +Z 方向
        yaw = 90: +X 方向
        """
        angle_rad = np.arctan2(dx, dz)  # 注意顺序：tan2(dx, dz)
        angle_deg = np.degrees(angle_rad)
        return angle_deg % 360

    @staticmethod
    def _get_agent_pos(event) -> Tuple[float, float]:
        """获取 Agent 位置"""
        agent = event.metadata.get("agent", {})
        pos = agent.get("position", {"x": 0.0, "z": 0.0})
        return (pos.get("x", 0.0), pos.get("z", 0.0))

    @staticmethod
    def _get_agent_rotation(event) -> int:
        """获取 Agent 旋转角度（0-360）"""
        rot = event.metadata.get("agent", {}).get("rotation", {}).get("y", 0.0)
        return round(rot) % 360

    @staticmethod
    def _normalize_deg(angle: float) -> float:
        """归一化角度到 [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    @staticmethod
    def _round2(val: float) -> float:
        """四舍五入到 2 位小数"""
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
            "visited_views": len(self.exploration_state["visited_views"]),
            "total_distance": f"{self.exploration_state['total_distance']:.2f}m",
            "elapsed_time": f"{elapsed:.1f}s",
            "blocked_count": self.obstacle_state["blocked_count"],
        }
