"""
基于 AI2-THOR PointNav 的全环境导航 (PointNav-based Navigation)
- 使用 AI2-THOR 提供的 Teleport 动作
- 直接导航到指定坐标，无需手动路径规划
- 快速覆盖所有可达点
- ✨ 异步目标选择：避免阻塞主循环
"""

from __future__ import annotations

import math
import time
import threading
from typing import Any, Dict, List, Optional, Set, Tuple


GridCell = Tuple[int, int]


class PointNavNavigator:
    """基于 PointNav 的全环境导航器"""
    
    def __init__(self, grid_size: float = 0.15):
        self.grid_size = float(grid_size)
        self.is_enabled = False
        
        # 已知的全局信息
        self.known_points: Set[GridCell] = set()        # 网格坐标
        self.known_points_world: List[Tuple[float, float]] = []  # 世界坐标 (x, z)
        
        # 访问跟踪
        self.visited_points: Set[GridCell] = set()      # 已访问的网格坐标
        self.visit_counts: Dict[GridCell, int] = {}     # 每点访问次数
        
        # 导航状态
        self.current_goal_world: Optional[Tuple[float, float]] = None  # 当前目标(世界坐标)
        self.current_goal_grid: Optional[GridCell] = None              # 当前目标(网格坐标)
        self.goal_started_at: float = 0.0
        self.is_navigating: bool = False  # PointNav 是否正在执行
        
        # 运动跟踪
        self.last_pose: Optional[Tuple[float, float]] = None
        self.stuck_count: int = 0
        self.nav_fail_count: int = 0
        
        # 计时
        self.last_plan_time = 0.0
        self.plan_interval_sec = 2.0
        self.last_reachable_refresh = 0.0
        self.reachable_refresh_sec = 2.0
        
        # 黑名单（无法到达的点）
        self.goal_blacklist_until: Dict[GridCell, float] = {}
        
        # 统计
        self.start_time: Optional[float] = None
        self.total_steps = 0
        self.goal_timeout_sec = 60.0  # 导航可能较慢，需要较长超时时间
        
        # ✨ 异步目标选择相关
        self.target_selection_lock = threading.Lock()
        self.pending_target: Optional[Tuple[float, float]] = None  # 待选目标
        
        # 统计
        self.start_time: Optional[float] = None
        self.total_steps = 0
        self.goal_timeout_sec = 60.0  # GetShortestPath-follow 可能较慢，需要较长超时时间
        
    def enable(self):
        """启用导航器"""
        self.is_enabled = True
        self.start_time = time.time()
        print("✅ PointNav 导航已启用 (使用 AI2-THOR 原生导航)")
        print(f"   - 目标点数: {len(self.known_points)}")
        
    def disable(self):
        """禁用导航器"""
        self.is_enabled = False
        self.current_goal_world = None
        self.current_goal_grid = None
        self.is_navigating = False
        elapsed = 0.0 if self.start_time is None else (time.time() - self.start_time)
        coverage = 100.0 * len(self.visited_points) / max(1, len(self.known_points)) if self.known_points else 0
        print("❌ PointNav 导航已停止")
        print(f"   - 运行时长: {elapsed:.1f}s")
        print(f"   - 总目标点: {len(self.known_points)}")
        print(f"   - 已访问点: {len(self.visited_points)}")
        print(f"   - 覆盖率: {coverage:.1f}%")
        
    def get_status(self) -> Dict[str, Any]:
        """获取导航状态"""
        elapsed = 0.0 if self.start_time is None else (time.time() - self.start_time)
        coverage = 100.0 * len(self.visited_points) / max(1, len(self.known_points)) if self.known_points else 0
        
        goal_age = 0.0
        if self.goal_started_at > 0:
            goal_age = time.time() - self.goal_started_at
        
        return {
            "enabled": self.is_enabled,
            "total_points": len(self.known_points),
            "visited_points": len(self.visited_points),
            "coverage_pct": f"{coverage:.1f}%",
            "target": self.current_goal_grid,
            "target_world": self.current_goal_world,
            "navigating": self.is_navigating,
            "elapsed_sec": f"{elapsed:.1f}",
            "goal_age": f"{goal_age:.1f}",
            "stuck_count": self.stuck_count,
            "nav_fail_count": self.nav_fail_count,
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
            self.known_points_world = []
            
            for p in points:
                x = float(p.get("x", 0.0))
                z = float(p.get("z", 0.0))
                cell = self._world_to_cell(x, z)
                
                if cell not in self.known_points:  # 去重
                    self.known_points.add(cell)
                    self.known_points_world.append((x, z))
            
            print(f"✅ 已加载地图: {len(self.known_points)} 个可达点 ({len(self.known_points_world)} 个世界坐标)")
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
            
            # 1. 获取当前位置
            agent = event.metadata.get("agent", {})
            pos = agent.get("position", {})
            new_pos = (float(pos.get("x", 0.0)), float(pos.get("z", 0.0)))
            new_cell = self._world_to_cell(new_pos[0], new_pos[1])
            
            # 2. 更新访问记录
            self.visited_points.add(new_cell)
            self.visit_counts[new_cell] = self.visit_counts.get(new_cell, 0) + 1
            
            # 3. 检测运动情况（卡顿检测）
            if self.last_pose is not None:
                moved = math.hypot(new_pos[0] - self.last_pose[0], new_pos[1] - self.last_pose[1])
                if moved < 0.01:  # 极少移动 = 卡顿
                    self.stuck_count += 1
                else:
                    self.stuck_count = max(0, self.stuck_count - 1)
            self.last_pose = new_pos
            
            # 4. 检查导航是否完成
            if self.is_navigating and self.current_goal_world is not None:
                dist_to_goal = math.hypot(
                    new_pos[0] - self.current_goal_world[0],
                    new_pos[1] - self.current_goal_world[1]
                )
                
                # 到达目标（距离 < 0.2m）
                if dist_to_goal < 0.2:
                    self._clear_goal()
                else:
                    # 检查超时
                    goal_age = time.time() - self.goal_started_at
                    if goal_age > self.goal_timeout_sec:
                        print(f"⏱ 导航超时 {self.current_goal_grid} ({goal_age:.1f}s)，切换目标")
                        self.goal_blacklist_until[self.current_goal_grid] = time.time() + 10.0
                        self._clear_goal()
                    # 检查卡顿
                    elif self.stuck_count >= 15:
                        print(f"🔄 导航卡顿 {self.current_goal_grid}，加入黑名单")
                        self.goal_blacklist_until[self.current_goal_grid] = time.time() + 15.0
                        self._clear_goal()
            
            # 5. 需要选择新目标时（仅当未在导航且无目标时）
            now = time.time()
            if not self.is_navigating and self.current_goal_world is None:
                self._select_next_target()
                self.last_plan_time = now
            
            # 6. 有目标时执行导航（使用 Teleport 直接传送）
            if self.current_goal_world is not None and not self.is_navigating:
                # 使用 Teleport 直接传送到目标点
                ev2 = controller.step(
                    action="Teleport",
                    x=self.current_goal_world[0],
                    y=agent.get("position", {}).get("y", 0.9),
                    z=self.current_goal_world[1],
                    forceAction=True
                )
                
                if ev2.metadata.get("lastActionSuccess", False):
                    self.is_navigating = True
                    self.goal_started_at = time.time()
                    self.nav_fail_count = 0
                    return ev2
                else:
                    self.nav_fail_count += 1
                    error_msg = ev2.metadata.get('errorMessage', 'Unknown')
                    print(f"✗ 传送失败 (尝试 {self.nav_fail_count}/3): {error_msg}")
                    
                    if self.nav_fail_count >= 3:
                        print(f"✗ 连续失败 3 次，加入黑名单")
                        self.goal_blacklist_until[self.current_goal_grid] = time.time() + 15.0
                        self._clear_goal()
                    return event
            
            return event
        except Exception as e:
            print(f"⚠️  导航步骤失败: {e}")
            return event
    
    def _select_next_target(self):
        """选择下一个目标点"""
        # 清理过期黑名单
        now = time.time()
        expired = [g for g, t in self.goal_blacklist_until.items() if t < now]
        for g in expired:
            del self.goal_blacklist_until[g]
        
        # 优先选择未访问的点
        unvisited = [p for p in self.known_points 
                     if p not in self.visited_points and p not in self.goal_blacklist_until]
        
        if not unvisited:
            # 次优选择访问最少的点
            unvisited = [p for p in self.known_points if p not in self.goal_blacklist_until]
            if unvisited:
                unvisited.sort(key=lambda p: self.visit_counts.get(p, 0))
        
        if not unvisited:
            print("⚠️  无可用目标（全部黑名单或完全覆盖）")
            self._clear_goal()
            return
        
        # 选择最近的点
        if self.last_pose is None:
            target_grid = unvisited[0]
        else:
            current_cell = self._world_to_cell(self.last_pose[0], self.last_pose[1])
            target_grid = min(unvisited, key=lambda p: self._euclidean_grid(p, current_cell))
        
        # 从网格坐标转换为世界坐标
        # 直接使用已知的世界坐标列表中最接近的点
        target_world = None
        
        if self.known_points_world:
            # 找到已知世界坐标中最接近的一个
            target_grid_world = self._cell_to_world(target_grid[0], target_grid[1])
            best_world = None
            best_dist = float('inf')
            
            for world_pos in self.known_points_world:
                dist = math.hypot(world_pos[0] - target_grid_world[0], 
                                world_pos[1] - target_grid_world[1])
                if dist < best_dist:
                    best_world = world_pos
                    best_dist = dist
            
            # 如果距离很近（< 2 倍网格大小），使用已知坐标
            if best_world and best_dist < self.grid_size * 2:
                target_world = best_world
            else:
                # 否则使用计算的坐标
                target_world = target_grid_world
        else:
            # 备选：直接计算世界坐标
            target_world = self._cell_to_world(target_grid[0], target_grid[1])
        
        self.current_goal_grid = target_grid
        self.current_goal_world = target_world
        self.goal_started_at = time.time()
    
    def _clear_goal(self):
        """清空当前目标"""
        self.current_goal_grid = None
        self.current_goal_world = None
        self.is_navigating = False
        self.goal_started_at = 0.0
    
    def _find_world_coords_by_grid(self, grid_cell: GridCell) -> Optional[Tuple[float, float]]:
        """从已知的世界坐标列表中查找最接近的世界坐标"""
        if not self.known_points_world:
            return None
        
        grid_world = self._cell_to_world(grid_cell[0], grid_cell[1])
        
        # 找最接近的已知世界坐标
        best = None
        best_dist = float('inf')
        
        for world_pos in self.known_points_world:
            dist = math.hypot(
                world_pos[0] - grid_world[0],
                world_pos[1] - grid_world[1]
            )
            if dist < best_dist:
                best = world_pos
                best_dist = dist
        
        return best if best_dist < self.grid_size else None
    
    def _world_to_cell(self, x: float, z: float) -> GridCell:
        """世界坐标转网格坐标"""
        return (int(round(x / self.grid_size)), int(round(z / self.grid_size)))
    
    def _cell_to_world(self, grid_x: int, grid_z: int) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        return (grid_x * self.grid_size, grid_z * self.grid_size)
    
    def _euclidean_grid(self, cell1: GridCell, cell2: GridCell) -> float:
        """网格坐标的欧氏距离"""
        return math.hypot(cell1[0] - cell2[0], cell1[1] - cell2[1])
