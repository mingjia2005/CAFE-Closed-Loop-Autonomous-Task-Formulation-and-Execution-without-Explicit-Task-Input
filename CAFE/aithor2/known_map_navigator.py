"""
基于已知地图的全环境导航 (Known Map Navigation)
- 在初始化时获取 AI2-THOR 的完整可达地图
- 从深度图检测实时障碍物
- 在已知障碍地图上使用 A* 规划
- 支持"访问所有点"或"导航到指定位置"的模式
- ✨ 异步规划：路径规划在后台线程执行，不阻塞主循环
"""

from __future__ import annotations

import math
import time
import threading
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple


GridCell = Tuple[int, int]


class KnownMapNavigator:
    """基于已知 AI2-THOR 地图的导航器"""
    
    def __init__(self, grid_size: float = 0.15):
        self.grid_size = float(grid_size)
        self.is_enabled = False
        
        # 已知的全局信息
        self.known_points: Set[GridCell] = set()        # AI2-THOR 返回的所有可达点（网格化）
        self.obstacle_cells: Set[GridCell] = set()      # 实时检测的障碍（深度图投影）
        
        # 访问跟踪
        self.visited_points: Set[GridCell] = set()      # 已访问过的点
        self.visit_counts: Dict[GridCell, int] = {}     # 每点的访问次数
        
        # 路径规划
        self.current_path: List[GridCell] = []
        self.path_index: int = 0
        self.current_goal: Optional[GridCell] = None
        self.goal_started_at: float = 0.0
        
        # 运动跟踪
        self.last_pose: Optional[Tuple[float, float]] = None
        self.stuck_count: int = 0
        self.action_fail_count: int = 0
        
        # 计时
        self.last_plan_time = 0.0
        self.plan_interval_sec = 1.0
        self.last_reachable_refresh = 0.0
        self.reachable_refresh_sec = 2.0  # 更新频率较低，因为地图固定
        
        # 黑名单（无法到达的点）
        self.goal_blacklist_until: Dict[GridCell, float] = {}
        
        # 统计
        self.start_time: Optional[float] = None
        self.total_steps = 0
        self.goal_timeout_sec = 15.0  # 目标超时
        
        # ✨ 异步规划相关
        self.planning_thread: Optional[threading.Thread] = None
        self.planning_lock = threading.Lock()
        self.pending_plan_request: Optional[Tuple[GridCell, GridCell]] = None  # (current_cell, goal)
        self.planned_path: Optional[List[GridCell]] = None  # 规划结果
        self.is_planning = False  # 是否正在规划中
        
    def enable(self):
        """启用导航器"""
        self.is_enabled = True
        self.start_time = time.time()
        print("✅ 已知地图导航已启用")
        print(f"   - 目标点数: {len(self.known_points)}")
        
    def disable(self):
        """禁用导航器"""
        self.is_enabled = False
        self.current_path = []
        self.path_index = 0
        self.current_goal = None
        elapsed = 0.0 if self.start_time is None else (time.time() - self.start_time)
        coverage = 100.0 * len(self.visited_points) / max(1, len(self.known_points)) if self.known_points else 0
        print("❌ 已知地图导航已停止")
        print(f"   - 运行时长: {elapsed:.1f}s")
        print(f"   - 总目标点: {len(self.known_points)}")
        print(f"   - 已访问点: {len(self.visited_points)}")
        print(f"   - 覆盖率: {coverage:.1f}%")
        
    def get_status(self) -> Dict[str, Any]:
        """获取导航状态"""
        elapsed = 0.0 if self.start_time is None else (time.time() - self.start_time)
        coverage = 100.0 * len(self.visited_points) / max(1, len(self.known_points)) if self.known_points else 0
        
        goal_age = 0.0
        if self.current_goal is not None and self.goal_started_at > 0:
            goal_age = time.time() - self.goal_started_at
        
        return {
            "enabled": self.is_enabled,
            "total_points": len(self.known_points),
            "visited_points": len(self.visited_points),
            "coverage_pct": f"{coverage:.1f}%",
            "path_len": len(self.current_path),
            "target": self.current_goal,
            "obstacles": len(self.obstacle_cells),
            "elapsed_sec": f"{elapsed:.1f}",
            "goal_age": f"{goal_age:.1f}",
            "stuck_count": self.stuck_count,
        }
        
    def initialize_map(self, controller) -> bool:
        """初始化地图：从 AI2-THOR 获取所有可达点"""
        try:
            ev = controller.step(action="GetReachablePositions")
            points = ev.metadata.get("actionReturn", []) or []
            
            if not points:
                print("⚠️  获取可达点失败")
                return False
            
            self.known_points = set()
            for p in points:
                x = float(p.get("x", 0.0))
                z = float(p.get("z", 0.0))
                cell = self._world_to_cell(x, z)
                self.known_points.add(cell)
            
            print(f"✅ 已加载地图: {len(self.known_points)} 个可达点")
            self.last_reachable_refresh = time.time()
            return True
        except Exception as e:
            print(f"❌ 初始化地图失败: {e}")
            return False
    
    def step(self, controller, event) -> Any:
        """执行一步导航"""
        if not self.is_enabled:
            return event
        
        try:
            self.total_steps += 1
            
            # 1. 轻量级深度图采样检测（不全面扫描，减少耗时）
            self._update_obstacles_from_depth_fast(event)
            
            # 2. 更新当前位置
            agent = event.metadata.get("agent", {})
            pos = agent.get("position", {})
            new_pos = (float(pos.get("x", 0.0)), float(pos.get("z", 0.0)))
            new_cell = self._world_to_cell(new_pos[0], new_pos[1])
            self.visited_points.add(new_cell)
            self.visit_counts[new_cell] = self.visit_counts.get(new_cell, 0) + 1
            
            # 3. 检测运动情况
            if self.last_pose is not None:
                moved = math.hypot(new_pos[0] - self.last_pose[0], new_pos[1] - self.last_pose[1])
                if moved < self.grid_size * 0.1:
                    self.stuck_count += 1
                else:
                    self.stuck_count = max(0, self.stuck_count - 1)
            self.last_pose = new_pos
            
            # 4. 检查是否到达当前目标
            if self.current_goal is not None and new_cell == self.current_goal:
                print(f"✓ 到达目标: {self.current_goal}")
                self._clear_plan()
            
            # 5. 检查目标超时
            if self.current_goal is not None and self.goal_started_at > 0:
                goal_age = time.time() - self.goal_started_at
                if goal_age > self.goal_timeout_sec:
                    print(f"⏱ 目标 {self.current_goal} 超时，切换")
                    self.goal_blacklist_until[self.current_goal] = time.time() + 10.0
                    self._clear_plan()
            
            # 6. 需要规划新路径时 - 异步触发规划
            now = time.time()
            if len(self.current_path) - self.path_index <= 1 or now - self.last_plan_time > self.plan_interval_sec:
                self._async_replan(new_cell)
                self.last_plan_time = now
            
            # 7. 若有路径则执行 - 直接 Teleport 到下一个路径点
            if self.path_index < len(self.current_path):
                next_cell = self.current_path[self.path_index]
                world_pos = self._cell_to_world(next_cell[0], next_cell[1])
                
                # 使用 Teleport 直接到达下一个点（需确保该点在可达范围内）
                ev2 = controller.step(
                    action="Teleport",
                    x=world_pos[0],
                    y=agent.get("position", {}).get("y", 0.9),
                    z=world_pos[1],
                    forceAction=True
                )
                
                if ev2.metadata.get("lastActionSuccess", False):
                    self.action_fail_count = 0
                    self.path_index += 1
                    self.stuck_count = 0
                else:
                    self.action_fail_count += 1
                    print(f"✗ Teleport 到 {next_cell} 失败: {ev2.metadata.get('errorMessage', 'Unknown')}")
                    
                    if self.action_fail_count >= 3 and self.current_goal is not None:
                        print(f"✗ 无法到达 {self.current_goal}，加入黑名单")
                        self.goal_blacklist_until[self.current_goal] = time.time() + 15.0
                        self._clear_plan()
                    elif self.stuck_count >= 5:
                        self._clear_plan()
                
                return ev2
            else:
                # 无路径时的兜底动作
                self._fallback_action(controller)
            
            return event
        except Exception as e:
            print(f"⚠️  导航步骤失败: {e}")
            return event
    
    def _replan(self, current_cell: GridCell):
        """重新规划路径"""
        # 清理过期黑名单
        now = time.time()
        expired = [g for g, t in self.goal_blacklist_until.items() if t < now]
        for g in expired:
            del self.goal_blacklist_until[g]
        
        # 选择目标：未访问的点优先
        unvisited = [p for p in self.known_points if p not in self.visited_points and p not in self.goal_blacklist_until]
        
        if not unvisited:
            # 若所有点都访问过，选择访问次数最少的
            unvisited = [p for p in self.known_points if p not in self.goal_blacklist_until]
            if unvisited:
                unvisited.sort(key=lambda p: self.visit_counts.get(p, 0))
        
        if not unvisited:
            print("⚠️  无可用目标（全部黑名单或已访问）")
            self.current_goal = None
            self.current_path = []
            self.path_index = 0
            return
        
        # 选择距离最近的未访问点
        goal = min(unvisited, key=lambda p: self._euclidean(p, current_cell))
        self.current_goal = goal
        self.goal_started_at = time.time()
        
        # A* 规划路径
        path = self._a_star(current_cell, goal)
        
        if path and len(path) > 1:
            self.current_path = path
            self.path_index = 1  # 跳过起点
            print(f"📍 规划成功: 当前 {current_cell} → 目标 {goal}，路径长度 {len(path)}")
        else:
            print(f"✗ 规划失败到 {goal}")
            self.goal_blacklist_until[goal] = time.time() + 15.0
            self.current_goal = None
            self.current_path = []
            self.path_index = 0
    
    def _a_star(self, start: GridCell, goal: GridCell) -> Optional[List[GridCell]]:
        """A* 路径规划"""
        if start not in self.known_points or goal not in self.known_points:
            return None
        
        if start == goal:
            return [start]
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        in_open = {start}
        
        while open_set:
            _, current = heappop(open_set)
            in_open.discard(current)
            
            if current == goal:
                # 重建路径
                path = [goal]
                while goal in came_from:
                    goal = came_from[goal]
                    path.append(goal)
                return path[::-1]
            
            for neighbor in self._neighbors4(current):
                if neighbor not in self.known_points or neighbor in self.obstacle_cells:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._euclidean(neighbor, goal)
                    
                    if neighbor not in in_open:
                        heappush(open_set, (f_score, neighbor))
                        in_open.add(neighbor)
        
        return None
    
    def _update_obstacles_from_depth_fast(self, event):
        """快速采样版本的深度图检测（不全面扫描，每10个像素采样一次）"""
        depth = getattr(event, "depth_frame", None)
        if depth is None:
            return
        
        h, w = depth.shape[:2]
        if h <= 0 or w <= 0:
            return
        
        # 仅清空部分障碍，保留之前的信息（减少计算量）
        if self.total_steps % 10 == 0:
            self.obstacle_cells.clear()
        
        try:
            fov = float(event.metadata.get("fov", 90.0))
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
            
            obstacle_depth_threshold = 0.4
            stride = 10  # 更大的步长，减少计算
            
            for v in range(0, h, stride):
                for u in range(0, w, stride):
                    z = float(depth[v, u])
                    if 0.05 < z < obstacle_depth_threshold:
                        x_cam = (u - cx) * z / fx
                        world_x = ax + x_cam * cos_y + z * sin_y
                        world_z = az + z * cos_y - x_cam * sin_y
                        
                        cell = self._world_to_cell(world_x, world_z)
                        self.obstacle_cells.add(cell)
        except Exception:
            pass  # 静默处理异常，避免输出干扰
    
    def _async_replan(self, current_cell: GridCell):
        """异步规划：为主循环减负"""
        # 如果已有规划结果，应用它
        if self.planned_path is not None and self.planned_path:
            with self.planning_lock:
                self.current_path = self.planned_path
                self.path_index = 1
                self.planned_path = None
                self.is_planning = False
                print(f"📍 应用后台规划结果: 路径长度 {len(self.current_path)}")
        
        # 如果已在规划中，跳过此次请求
        if self.is_planning:
            return
        
        # 触发新的规划请求
        with self.planning_lock:
            self.pending_plan_request = (current_cell, None)
            self.is_planning = True
        
        # 启动规划线程（如果还没有）
        if self.planning_thread is None or not self.planning_thread.is_alive():
            self.planning_thread = threading.Thread(target=self._planning_worker, daemon=True)
            self.planning_thread.start()
    
    def _planning_worker(self):
        """后台规划线程"""
        while self.is_enabled:
            try:
                # 获取待规划请求
                with self.planning_lock:
                    if self.pending_plan_request is None:
                        self.is_planning = False
                        break
                    
                    current_cell, _ = self.pending_plan_request
                
                # 在锁外执行规划（避免长时间持锁）
                self._do_replan(current_cell)
                
                with self.planning_lock:
                    self.pending_plan_request = None
                    self.is_planning = False
                    
            except Exception as e:
                print(f"⚠️  后台规划异常: {e}")
                with self.planning_lock:
                    self.is_planning = False
                break
            
            time.sleep(0.01)  # 避免CPU占用
    
    def _do_replan(self, current_cell: GridCell):
        """实际的规划逻辑（可在后台执行）"""
        # 清理过期黑名单
        now = time.time()
        expired = [g for g, t in self.goal_blacklist_until.items() if t < now]
        for g in expired:
            del self.goal_blacklist_until[g]
        
        # 选择目标：未访问的点优先
        unvisited = [p for p in self.known_points if p not in self.visited_points and p not in self.goal_blacklist_until]
        
        if not unvisited:
            # 若所有点都访问过，选择访问次数最少的
            unvisited = [p for p in self.known_points if p not in self.goal_blacklist_until]
            if unvisited:
                unvisited.sort(key=lambda p: self.visit_counts.get(p, 0))
        
        if not unvisited:
            print("⚠️  无可用目标（全部黑名单或已访问）")
            self.current_goal = None
            with self.planning_lock:
                self.planned_path = None
            return
        
        # 选择距离最近的未访问点
        goal = min(unvisited, key=lambda p: self._euclidean(p, current_cell))
        self.current_goal = goal
        self.goal_started_at = time.time()
        
        # A* 规划路径（在后台执行，不阻塞主循环）
        path = self._a_star(current_cell, goal)
        
        if path and len(path) > 1:
            with self.planning_lock:
                self.planned_path = path
        else:
            print(f"✗ 后台规划失败到 {goal}")
            self.goal_blacklist_until[goal] = time.time() + 15.0
            self.current_goal = None
            with self.planning_lock:
                self.planned_path = None
    
    def _update_obstacles_from_depth(self, event):
        """从深度图检测障碍"""
        depth = getattr(event, "depth_frame", None)
        if depth is None:
            return
        
        h, w = depth.shape[:2]
        if h <= 0 or w <= 0:
            return
        
        self.obstacle_cells.clear()
        
        try:
            fov = float(event.metadata.get("fov", 90.0))
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
            
            obstacle_depth_threshold = 0.4
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
        except Exception as e:
            print(f"⚠️  障碍检测失败: {e}")
    
    def _fallback_action(self, controller):
        """无路径时的兜底动作"""
        try:
            phase = (self.total_steps // 4) % 4
            if phase == 0:
                controller.step(action="RotateRight", degrees=90)
            elif phase == 1:
                controller.step(action="MoveAhead", moveMagnitude=self.grid_size)
            elif phase == 2:
                controller.step(action="RotateLeft", degrees=90)
            else:
                controller.step(action="MoveAhead", moveMagnitude=self.grid_size)
        except Exception:
            pass
    
    def _clear_plan(self):
        """清空当前规划"""
        self.current_path = []
        self.path_index = 0
        self.goal_started_at = 0.0
    
    def _world_to_cell(self, x: float, z: float) -> GridCell:
        """世界坐标转网格坐标"""
        return (int(round(x / self.grid_size)), int(round(z / self.grid_size)))
    
    def _cell_to_world(self, grid_x: int, grid_z: int) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        return (grid_x * self.grid_size, grid_z * self.grid_size)
    
    def _euclidean(self, cell1: GridCell, cell2: GridCell) -> float:
        """欧氏距离"""
        return math.hypot(cell1[0] - cell2[0], cell1[1] - cell2[1])
    
    def _neighbors4(self, cell: GridCell) -> List[GridCell]:
        """4邻接"""
        x, z = cell
        return [(x+1, z), (x-1, z), (x, z+1), (x, z-1)]
    
    def _neighbors8(self, cell: GridCell) -> List[GridCell]:
        """8邻接"""
        x, z = cell
        return [
            (x+1, z), (x-1, z), (x, z+1), (x, z-1),
            (x+1, z+1), (x+1, z-1), (x-1, z+1), (x-1, z-1),
        ]
