"""
全图自主导航（Frontier-based Exploration）
- 使用深度图 + 相机内参近似投影可见区域到2D网格
- 维护已探索掩码（explored/visited）
- 以 frontier（已探索自由区与未探索自由区边界）为目标
- 在离散网格上使用 A* 规划
- 在 THOR 中执行 MoveAhead / RotateLeft / RotateRight
"""

from __future__ import annotations

import math
import time
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple


GridCell = Tuple[int, int]


class FrontierFullMapNavigator:
    def __init__(self, grid_size: float = 0.25):
        self.grid_size = float(grid_size)
        # NOTE: In AI2-THOR, default MoveAhead magnitude is typically 0.25m.
        # Keeping grid_size aligned with moveMagnitude avoids "path cell never reached" stalls.
        self.is_enabled = False

        self.free_cells: Set[GridCell] = set()       # 全图可达自由网格（由GetReachablePositions构建）
        self.explored_cells: Set[GridCell] = set()   # 已探索网格（由深度投影+轨迹更新）
        self.visited_cells: Set[GridCell] = set()    # 机器人经过的网格
        self.obstacle_cells: Set[GridCell] = set()   # 障碍网格（从深度图主动检测）
        self.visit_counts: Dict[GridCell, int] = {}

        self.current_path: List[GridCell] = []
        self.path_index: int = 0
        self.current_frontier: Optional[GridCell] = None
        self.current_goal_started_at: float = 0.0

        self.last_plan_time = 0.0
        self.plan_interval_sec = 1.0
        self.last_reachable_refresh = 0.0
        self.reachable_refresh_sec = 1.0

        self.stuck_count = 0
        self.action_fail_count = 0
        self.last_pose: Optional[Tuple[float, float]] = None
        self.goal_blacklist_until: Dict[GridCell, float] = {}

        self.start_time: Optional[float] = None
        self.total_steps = 0
        self.recovery_step = 0

        # 长期探索监控
        self.coverage_history: List[Tuple[float, float]] = []
        self.last_coverage_update = 0.0
        self.last_coverage_progress_time = 0.0
        self.best_coverage = 0.0

        # 目标保持与切换策略
        self.goal_timeout_sec = 12.0
        self.frontier_cooldown_sec = 10.0
        self.force_diversify_after_sec = 15.0

    def enable(self):
        self.is_enabled = True
        self.start_time = time.time()
        self.last_coverage_progress_time = self.start_time
        self.best_coverage = 0.0
        print("✅ 全图Frontier自主导航已启用")

    def disable(self):
        self.is_enabled = False
        self.current_path = []
        self.path_index = 0
        self.current_frontier = None
        elapsed = 0.0 if self.start_time is None else (time.time() - self.start_time)
        print("❌ 全图Frontier自主导航已停止")
        print(f"   - 运行时长: {elapsed:.1f}s")
        print(f"   - 自由网格: {len(self.free_cells)}")
        print(f"   - 已探索网格: {len(self.explored_cells)}")
        print(f"   - 已访问网格: {len(self.visited_cells)}")

    def get_status(self) -> Dict[str, Any]:
        elapsed = 0.0 if self.start_time is None else (time.time() - self.start_time)
        coverage = 0.0
        if self.free_cells:
            coverage = 100.0 * len(self.explored_cells & self.free_cells) / max(1, len(self.free_cells))

        goal_age = 0.0
        if self.current_frontier is not None and self.current_goal_started_at > 0:
            goal_age = time.time() - self.current_goal_started_at

        return {
            "enabled": self.is_enabled,
            "free_cells": len(self.free_cells),
            "explored_cells": len(self.explored_cells),
            "visited_cells": len(self.visited_cells),
            "coverage_pct": f"{coverage:.1f}%",
            "path_len": len(self.current_path),
            "target_frontier": self.current_frontier,
            "goal_age_sec": round(goal_age, 1),
            "blacklisted_goals": len(self.goal_blacklist_until),
            "steps": self.total_steps,
            "elapsed_sec": round(elapsed, 1),
        }

    def step(self, controller, event):
        if not self.is_enabled:
            return event

        try:
            self._refresh_free_cells(controller)
            self._update_obstacles_from_depth(event)
            self._update_explored_from_depth(event)

            cur_world = self._agent_pos(event)
            cur_cell = self._world_to_cell(cur_world[0], cur_world[1])
            self.visited_cells.add(cur_cell)
            self.explored_cells.add(cur_cell)
            self.visit_counts[cur_cell] = self.visit_counts.get(cur_cell, 0) + 1

            self._update_coverage_progress()

            if not self.free_cells:
                return event

            now = time.time()
            goal_timed_out = self.current_frontier is not None and (now - self.current_goal_started_at > self.goal_timeout_sec)
            if goal_timed_out:
                self._blacklist_goal(self.current_frontier)
                self._clear_plan()

            need_replan = (
                not self.current_path
                or (self.path_index >= len(self.current_path) - 1)
                or (now - self.last_plan_time >= self.plan_interval_sec)
                or goal_timed_out
            )
            if need_replan:
                self._replan(cur_cell)

            if not self.current_path or self.path_index >= len(self.current_path) - 1:
                return self._fallback_explore_action(controller, event)

            next_cell = self.current_path[self.path_index + 1]
            ev2 = self._execute_move_towards_cell(controller, event, next_cell)
            self.total_steps += 1

            if ev2.metadata.get("lastActionSuccess", False):
                self.stuck_count = 0
                self.action_fail_count = 0
                new_pos = self._agent_pos(ev2)
                if self.last_pose is not None:
                    moved = math.hypot(new_pos[0] - self.last_pose[0], new_pos[1] - self.last_pose[1])
                    if moved < self.grid_size * 0.25:
                        self.stuck_count += 1
                self.last_pose = new_pos

                new_cell = self._world_to_cell(new_pos[0], new_pos[1])
                self.visited_cells.add(new_cell)
                self.explored_cells.add(new_cell)
                self.visit_counts[new_cell] = self.visit_counts.get(new_cell, 0) + 1

                if new_cell == next_cell:
                    self.path_index += 1

                if self.current_frontier is not None and self._euclidean(new_cell, self.current_frontier) <= 1.0:
                    self._clear_plan()
            else:
                self.stuck_count += 1
                self.action_fail_count += 1
                if self.action_fail_count >= 2 and self.current_frontier is not None:
                    self._blacklist_goal(self.current_frontier)
                    self._clear_plan()
                elif self.stuck_count >= 3:
                    self._clear_plan()

            return ev2
        except Exception as e:
            print(f"⚠ 全图Frontier导航一步执行失败: {e}")
            return event

    def _refresh_free_cells(self, controller):
        now = time.time()
        if now - self.last_reachable_refresh < self.reachable_refresh_sec and self.free_cells:
            return
        ev = controller.step(action="GetReachablePositions")
        points = ev.metadata.get("actionReturn", []) or []
        if not points:
            return
        free = set()
        for p in points:
            free.add(self._world_to_cell(float(p.get("x", 0.0)), float(p.get("z", 0.0))))
        self.free_cells = free
        self.last_reachable_refresh = now

    def _update_obstacles_from_depth(self, event):
        """从深度图主动检测近距离障碍物，更新 obstacle_cells"""
        depth = getattr(event, "depth_frame", None)
        if depth is None:
            return

        h, w = depth.shape[:2]
        if h <= 0 or w <= 0:
            return

        self.obstacle_cells.clear()

        fov = float(event.metadata.get("fov", event.metadata.get("cameraFieldOfView", 90.0)))
        fx = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
        fy = fx
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        agent = event.metadata.get("agent", {})
        apos = agent.get("position", {})
        arot = agent.get("rotation", {})
        ax = float(apos.get("x", 0.0))
        az = float(apos.get("z", 0.0))
        yaw = math.radians(float(arot.get("y", 0.0)))
        sin_y = math.sin(yaw)
        cos_y = math.cos(yaw)

        obstacle_depth_threshold = 0.5
        stride = 4

        for v in range(0, h, stride):
            for u in range(0, w, stride):
                z = float(depth[v, u])
                if 0.05 < z < obstacle_depth_threshold:
                    x_cam = (u - cx) * z / fx
                    world_x = ax + x_cam * cos_y + z * sin_y
                    world_z = az + z * cos_y - x_cam * sin_y

                    cell = self._world_to_cell(world_x, world_z)
                    self.obstacle_cells.add(cell)
                    if cell in self.free_cells:
                        self.free_cells.discard(cell)

    def _update_explored_from_depth(self, event):
        depth = getattr(event, "depth_frame", None)
        if depth is None:
            return

        h, w = depth.shape[:2]
        if h <= 0 or w <= 0:
            return

        fov = float(event.metadata.get("fov", event.metadata.get("cameraFieldOfView", 90.0)))
        fx = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
        fy = fx
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        agent = event.metadata.get("agent", {})
        apos = agent.get("position", {})
        arot = agent.get("rotation", {})
        ax = float(apos.get("x", 0.0))
        az = float(apos.get("z", 0.0))
        yaw = math.radians(float(arot.get("y", 0.0)))
        sin_y = math.sin(yaw)
        cos_y = math.cos(yaw)

        stride = 8
        max_depth = 4.0

        for v in range(0, h, stride):
            for u in range(0, w, stride):
                z = float(depth[v, u])
                if not (0.05 < z < max_depth):
                    continue

                x_cam = (u - cx) * z / fx
                # THOR中以地平面导航，使用前向深度和水平偏移构造2D点
                world_x = ax + x_cam * cos_y + z * sin_y
                world_z = az + z * cos_y - x_cam * sin_y

                cell = self._world_to_cell(world_x, world_z)
                snapped = self._snap_to_nearest_free(cell)
                if snapped is not None:
                    self.explored_cells.add(snapped)
                else:
                    self.explored_cells.add(cell)

    def _replan(self, cur_cell: GridCell):
        start_cell = self._snap_to_nearest_free(cur_cell)
        if start_cell is None:
            self._clear_plan()
            return

        now = time.time()
        target = self.current_frontier

        # 若当前目标无效或仍在冷却中，则重选
        if target is None or target not in self.free_cells or now < self.goal_blacklist_until.get(target, 0.0):
            target = None

        if target is None:
            target = self._select_long_horizon_target(start_cell)

        if target is None:
            self._clear_plan()
            print("🧭 [Frontier] 无可用目标（frontier/未探索点为空或被冷却）")
            return

        if target == start_cell:
            alt = self._pick_alternative_target(start_cell)
            if alt is not None:
                target = alt

        path = self._a_star(start_cell, target)

        if path and len(path) >= 2:
            if self.current_frontier != target:
                self.current_goal_started_at = now
            self.current_frontier = target
            self.current_path = path
            self.path_index = 0
            self.last_plan_time = now
            print(f"🧭 [Frontier] 重规划成功 start={start_cell} goal={target} path_len={len(path)}")
        else:
            self._blacklist_goal(target)
            self._clear_plan()
            print(f"🧭 [Frontier] 重规划失败 start={start_cell} goal={target}")

    def _select_long_horizon_target(self, start_cell: GridCell) -> Optional[GridCell]:
        now = time.time()
        frontier_cells = self._find_frontiers()
        frontier_clusters = self._cluster_cells(frontier_cells)

        candidates: List[GridCell] = []
        if frontier_clusters:
            for cluster in frontier_clusters:
                center = self._cluster_center(cluster)
                snapped = self._snap_to_nearest_free(center)
                if snapped is not None:
                    candidates.append(snapped)

        # 停滞时强制多样化：优先最远的未探索可达点
        if self._is_coverage_stagnated():
            unexplored = [c for c in (self.free_cells - (self.explored_cells & self.free_cells)) if c not in self._active_blacklist(now)]
            if unexplored:
                return max(unexplored, key=lambda c: self._euclidean(start_cell, c))

        if not candidates:
            candidates = [c for c in (self.free_cells - (self.explored_cells & self.free_cells))]

        candidates = [c for c in candidates if c not in self._active_blacklist(now)]
        if not candidates:
            return None

        # score越高越好：未知增益高、距离适中、重复访问惩罚
        def score(cell: GridCell) -> float:
            gain = self._unknown_gain(cell, radius=4)
            dist = self._euclidean(start_cell, cell)
            revisit = float(self.visit_counts.get(cell, 0))
            return gain * 3.0 - dist * 0.6 - revisit * 1.2

        return max(candidates, key=score)

    def _pick_alternative_target(self, start_cell: GridCell) -> Optional[GridCell]:
        # 优先未探索可达点，排除障碍物
        unexplored = [c for c in (self.free_cells - (self.explored_cells & self.free_cells)) 
                      if c != start_cell and c not in self.obstacle_cells]
        if unexplored:
            return min(unexplored, key=lambda c: self._euclidean(c, start_cell))

        # 其次用邻域可达点，排除障碍物
        nbs = [c for c in self._neighbors_free(start_cell, max_radius=2) 
               if c != start_cell and c not in self.obstacle_cells]
        if nbs:
            return nbs[0]

        return None

    def _fallback_explore_action(self, controller, event):
        # 当暂无有效路径时，执行轻量兜底动作以打破停滞
        try:
            phase = self.recovery_step % 4
            if phase == 0:
                ev = controller.step(action="RotateRight", degrees=90)
            elif phase == 1:
                ev = controller.step(action="MoveAhead", moveMagnitude=self.grid_size, forceAction=True)
            elif phase == 2:
                ev = controller.step(action="RotateLeft", degrees=90)
            else:
                ev = controller.step(action="MoveAhead", moveMagnitude=self.grid_size, forceAction=True)
            self.recovery_step += 1
            self.total_steps += 1
            return ev
        except Exception:
            return event

    def _find_frontiers(self) -> List[GridCell]:
        if not self.free_cells:
            return []

        known_free = self.free_cells
        explored_free = self.explored_cells & known_free
        unexplored_free = (known_free - explored_free) - self.obstacle_cells
        if not unexplored_free:
            return []

        frontiers = []
        for cell in explored_free:
            if cell in self.obstacle_cells:
                continue
            for nb in self._neighbors8(cell):
                if nb in unexplored_free:
                    frontiers.append(cell)
                    break

        # 稀疏网格兜底：若直接边界为空，允许“距离1~2格”作为近似frontier
        if not frontiers:
            unexplored_list = list(unexplored_free)
            for cell in explored_free:
                if cell in self.obstacle_cells:
                    continue
                for u in unexplored_list:
                    if max(abs(cell[0] - u[0]), abs(cell[1] - u[1])) <= 6:
                        frontiers.append(cell)
                        break
                if frontiers:
                    break

        return frontiers

    def _a_star(self, start: GridCell, goal: GridCell) -> Optional[List[GridCell]]:
        if start not in self.free_cells or goal not in self.free_cells:
            return None

        open_heap = []
        heappush(open_heap, (0.0, start))

        came_from: Dict[GridCell, GridCell] = {}
        g_score: Dict[GridCell, float] = {start: 0.0}

        while open_heap:
            _, current = heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nb in self._neighbors_free(current, max_radius=1):
                step_cost = self._euclidean(current, nb)
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(nb, 1e18):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f = tentative + self._euclidean(nb, goal)
                    heappush(open_heap, (f, nb))

        return None

    def _execute_move_towards_cell(self, controller, event, next_cell: GridCell):
        cur_x, cur_z = self._agent_pos(event)
        target_x, target_z = self._cell_to_world(next_cell)

        dx = target_x - cur_x
        dz = target_z - cur_z

        desired_yaw = math.degrees(math.atan2(dx, dz))
        current_yaw = float(event.metadata.get("agent", {}).get("rotation", {}).get("y", 0.0))
        diff = self._normalize_angle(desired_yaw - current_yaw)

        if abs(diff) > 12.0:
            action = "RotateRight" if diff > 0 else "RotateLeft"
            # Use a finer turn when we're roughly aligned to reduce left-right oscillation.
            deg = 45 if abs(diff) < 60.0 else 90
            return controller.step(action=action, degrees=deg)


        return controller.step(action="MoveAhead", moveMagnitude=self.grid_size, forceAction=True)

    def _neighbors4(self, cell: GridCell):
        x, z = cell
        return ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1))

    def _neighbors8(self, cell: GridCell):
        x, z = cell
        return (
            (x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1),
            (x + 1, z + 1), (x + 1, z - 1), (x - 1, z + 1), (x - 1, z - 1),
        )

    def _neighbors_free(self, cell: GridCell, max_radius: int = 1) -> List[GridCell]:
        """Return reachable neighbor cells for planning.

        IMPORTANT: Planning must match the executor's primitive actions.
        If A* is allowed to jump multiple cells (radius>1), the executor (MoveAhead)
        may never land exactly on `next_cell`, causing the agent to appear "stuck".
        We therefore plan on 4-neighborhood with step=1 cell.
        """
        out: List[GridCell] = []
        for nb in self._neighbors4(cell):
            if nb in self.free_cells and nb not in self.obstacle_cells:
                out.append(nb)
        return out


    def _snap_to_nearest_free(self, cell: GridCell, max_radius: int = 8) -> Optional[GridCell]:
        if cell in self.free_cells:
            return cell
        cx, cz = cell
        best = None
        best_d = 1e18
        for fx, fz in self.free_cells:
            d = max(abs(fx - cx), abs(fz - cz))
            if d <= max_radius and d < best_d:
                best = (fx, fz)
                best_d = d
        # 局部半径内找不到时，退化为全局最近可达点，避免“探索点悬空”
        if best is None and self.free_cells:
            for fx, fz in self.free_cells:
                d = max(abs(fx - cx), abs(fz - cz))
                if d < best_d:
                    best = (fx, fz)
                    best_d = d
        return best

    def _cluster_cells(self, cells: List[GridCell], radius: int = 2) -> List[List[GridCell]]:
        if not cells:
            return []
        cell_set = set(cells)
        clusters: List[List[GridCell]] = []
        seen: Set[GridCell] = set()
        for c in cells:
            if c in seen:
                continue
            q = [c]
            seen.add(c)
            group: List[GridCell] = []
            while q:
                cur = q.pop()
                group.append(cur)
                for nb in self._neighbors_within(cur, radius):
                    if nb in cell_set and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            clusters.append(group)
        return clusters

    @staticmethod
    def _neighbors_within(cell: GridCell, radius: int) -> List[GridCell]:
        x, z = cell
        out: List[GridCell] = []
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dz == 0:
                    continue
                out.append((x + dx, z + dz))
        return out

    def _cluster_center(self, cluster: List[GridCell]) -> GridCell:
        if not cluster:
            return (0, 0)
        sx = sum(c[0] for c in cluster)
        sz = sum(c[1] for c in cluster)
        return (int(round(sx / len(cluster))), int(round(sz / len(cluster))))

    def _unknown_gain(self, cell: GridCell, radius: int = 4) -> int:
        x, z = cell
        gain = 0
        explored_free = self.explored_cells & self.free_cells
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                c = (x + dx, z + dz)
                if c in self.free_cells and c not in explored_free:
                    gain += 1
        return gain

    def _active_blacklist(self, now: Optional[float] = None) -> Set[GridCell]:
        if now is None:
            now = time.time()
        active = {c for c, t in self.goal_blacklist_until.items() if t > now}
        # 顺带清理过期条目
        expired = [c for c, t in self.goal_blacklist_until.items() if t <= now]
        for c in expired:
            self.goal_blacklist_until.pop(c, None)
        return active

    def _blacklist_goal(self, goal: Optional[GridCell]):
        if goal is None:
            return
        self.goal_blacklist_until[goal] = time.time() + self.frontier_cooldown_sec

    def _clear_plan(self):
        self.current_path = []
        self.path_index = 0
        self.current_frontier = None
        self.current_goal_started_at = 0.0

    def _update_coverage_progress(self):
        if not self.free_cells:
            return
        now = time.time()
        if now - self.last_coverage_update < 1.0:
            return
        covered = len(self.explored_cells & self.free_cells) / max(1, len(self.free_cells))
        self.coverage_history.append((now, covered))
        self.coverage_history = self.coverage_history[-60:]
        self.last_coverage_update = now
        if covered > self.best_coverage + 0.002:
            self.best_coverage = covered
            self.last_coverage_progress_time = now

    def _is_coverage_stagnated(self) -> bool:
        if self.last_coverage_progress_time <= 0:
            return False
        return (time.time() - self.last_coverage_progress_time) >= self.force_diversify_after_sec

    @staticmethod
    def _manhattan(a: GridCell, b: GridCell) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _euclidean(a: GridCell, b: GridCell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _normalize_angle(deg: float) -> float:
        while deg > 180.0:
            deg -= 360.0
        while deg < -180.0:
            deg += 360.0
        return deg

    def _world_to_cell(self, x: float, z: float) -> GridCell:
        return (int(round(x / self.grid_size)), int(round(z / self.grid_size)))

    def _cell_to_world(self, cell: GridCell) -> Tuple[float, float]:
        return (cell[0] * self.grid_size, cell[1] * self.grid_size)

    @staticmethod
    def _agent_pos(event) -> Tuple[float, float]:
        agent = event.metadata.get("agent", {})
        pos = agent.get("position", {})
        return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))

    @staticmethod
    def _reconstruct_path(came_from: Dict[GridCell, GridCell], current: GridCell) -> List[GridCell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
