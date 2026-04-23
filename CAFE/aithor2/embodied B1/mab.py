#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多臂老虎机（MAB）平衡模块
用于平衡探索（追问新信息）和利用（基于现有信息生成任务）
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MABBalancer:
    """
    多臂老虎机平衡器
    实现UCB1算法来平衡探索和利用
    """
    
    def __init__(self, 
                 exploration_factor: float = 2.0,
                 max_questions_per_subtask: int = 8,
                 min_reward_threshold: float = 0.6,
                 min_questions_before_stop: int = 5):
        """
        初始化MAB平衡器
        
        Args:
            exploration_factor (float): UCB1探索因子
            max_questions_per_subtask (int): 每个子任务最大追问次数
            min_reward_threshold (float): 最小回报阈值
        """
        self.exploration_factor = exploration_factor
        self.max_questions_per_subtask = max_questions_per_subtask
        self.min_reward_threshold = min_reward_threshold
        self.min_questions_before_stop = min_questions_before_stop
        
        # 子任务状态跟踪
        self.subtask_stats = {}  # {subtask_id: SubtaskStats}
        
        # 决策日志
        self.decision_log = []
        
        # 全局统计
        self.total_questions = 0
        self.total_rewards = 0.0
        
        logger.info("🎰 MAB平衡器初始化完成")
    
    def should_continue_questioning(self, 
                                  subtask_id: str, 
                                  current_question_count: int) -> bool:
        """
        决定是否继续追问某个子任务
        
        Args:
            subtask_id (str): 子任务ID
            current_question_count (int): 当前已追问次数
            
        Returns:
            bool: 是否继续追问
        """
        # 检查是否达到最大追问次数
        if current_question_count >= self.max_questions_per_subtask:
            logger.info(f"⏹️ 子任务 {subtask_id} 达到最大追问次数限制")
            self._log_decision(subtask_id, "stop", "max_questions_reached", 0.0)
            return False
        
        # 获取或初始化子任务统计
        if subtask_id not in self.subtask_stats:
            self.subtask_stats[subtask_id] = SubtaskStats(subtask_id)
        
        stats = self.subtask_stats[subtask_id]
        
        # 在达到最小追问阈值前，强制继续（除非达到最大追问次数）
        if current_question_count < self.min_questions_before_stop:
            logger.info(f"🔁 子任务 {subtask_id} 未达到最小追问阈值({self.min_questions_before_stop})，继续追问")
            self._log_decision(subtask_id, "continue", "min_rounds", stats.average_reward)
            return True
        
        # 若累计追问数达到下限且平均回报达到阈值，停止追问
        if stats.question_count >= self.min_questions_before_stop and stats.average_reward >= self.min_reward_threshold:
            logger.info(f"✅ 子任务 {subtask_id} 已获得足够信息，停止追问")
            self._log_decision(subtask_id, "stop", "sufficient_info", stats.average_reward)
            return False
        
        # 使用UCB1算法计算是否继续
        should_continue = self._ucb1_decision(stats, current_question_count)
        
        decision_reason = "ucb1_algorithm"
        if should_continue:
            logger.info(f"🔍 MAB决定继续追问子任务 {subtask_id}")
        else:
            logger.info(f"⏹️ MAB决定停止追问子任务 {subtask_id}")
        
        self._log_decision(subtask_id, "continue" if should_continue else "stop", 
                          decision_reason, stats.average_reward)
        
        return should_continue
    
    def update_reward(self, subtask_id: str, reward: float):
        """
        更新子任务的回报
        
        Args:
            subtask_id (str): 子任务ID
            reward (float): 回报值 (0.0-1.0)
        """
        if subtask_id not in self.subtask_stats:
            self.subtask_stats[subtask_id] = SubtaskStats(subtask_id)
        
        stats = self.subtask_stats[subtask_id]
        stats.update_reward(reward)
        
        # 更新全局统计
        self.total_questions += 1
        self.total_rewards += reward
        
        logger.info(f"💰 子任务 {subtask_id} 获得回报: {reward:.2f}")
    
    def _ucb1_decision(self, stats: 'SubtaskStats', current_question_count: int) -> bool:
        """
        使用UCB1算法决定是否继续追问
        
        Args:
            stats (SubtaskStats): 子任务统计信息
            current_question_count (int): 当前追问次数
            
        Returns:
            bool: 是否继续追问
        """
        if stats.question_count == 0:
            # 如果还没有追问过，优先探索
            return True
        
        # 计算UCB1值
        exploitation_term = stats.average_reward
        exploration_term = self.exploration_factor * np.sqrt(
            np.log1p(self.total_questions) / stats.question_count
        )
        ucb1_value = exploitation_term + exploration_term
        
        # 动态调整阈值：如果平均回报很低（说明一直在问"无法确定"的问题），
        # 降低阈值，让系统更倾向于继续追问
        if stats.average_reward < 0.2:  # 如果平均回报很低
            threshold = self.min_reward_threshold * 0.5  # 降低阈值
        else:
            threshold = self.min_reward_threshold
        
        # 如果UCB1值高于阈值，继续追问
        return ucb1_value > threshold
    
    def _log_decision(self, 
                     subtask_id: str, 
                     decision: str, 
                     reason: str, 
                     current_reward: float):
        """
        记录决策日志
        
        Args:
            subtask_id (str): 子任务ID
            decision (str): 决策结果
            reason (str): 决策原因
            current_reward (float): 当前回报
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "subtask_id": subtask_id,
            "decision": decision,
            "reason": reason,
            "current_reward": current_reward,
            "total_questions": self.total_questions,
            "global_avg_reward": self.total_rewards / max(1, self.total_questions)
        }
        
        self.decision_log.append(log_entry)
        logger.debug(f"📝 MAB决策日志: {log_entry}")
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """
        获取决策日志
        
        Returns:
            List[Dict[str, Any]]: 决策日志列表
        """
        return self.decision_log.copy()
    
    def get_subtask_stats(self, subtask_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定子任务的统计信息
        
        Args:
            subtask_id (str): 子任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 统计信息
        """
        if subtask_id in self.subtask_stats:
            stats = self.subtask_stats[subtask_id]
            return {
                "subtask_id": stats.subtask_id,
                "question_count": stats.question_count,
                "total_reward": stats.total_reward,
                "average_reward": stats.average_reward,
                "reward_history": stats.reward_history.copy()
            }
        return None
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        获取全局统计信息
        
        Returns:
            Dict[str, Any]: 全局统计
        """
        return {
            "total_questions": self.total_questions,
            "total_rewards": self.total_rewards,
            "global_average_reward": self.total_rewards / max(1, self.total_questions),
            "active_subtasks": len(self.subtask_stats),
            "exploration_factor": self.exploration_factor,
            "max_questions_per_subtask": self.max_questions_per_subtask,
            "min_reward_threshold": self.min_reward_threshold,
            "min_questions_before_stop": self.min_questions_before_stop
        }
    
    def reset_subtask(self, subtask_id: str):
        """
        重置特定子任务的状态
        
        Args:
            subtask_id (str): 子任务ID
        """
        if subtask_id in self.subtask_stats:
            del self.subtask_stats[subtask_id]
            logger.info(f"🔄 子任务 {subtask_id} 状态已重置")
    
    def reset_all(self):
        """重置所有状态"""
        self.subtask_stats.clear()
        self.decision_log.clear()
        self.total_questions = 0
        self.total_rewards = 0.0
        logger.info("🔄 MAB平衡器状态已重置")
    
    def save_state(self, file_path: str) -> bool:
        """
        保存状态到文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            state = {
                "exploration_factor": self.exploration_factor,
                "max_questions_per_subtask": self.max_questions_per_subtask,
                "min_reward_threshold": self.min_reward_threshold,
                "subtask_stats": {
                    subtask_id: stats.to_dict() 
                    for subtask_id, stats in self.subtask_stats.items()
                },
                "decision_log": self.decision_log,
                "total_questions": self.total_questions,
                "total_rewards": self.total_rewards,
                "min_questions_before_stop": self.min_questions_before_stop
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 MAB状态已保存到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存MAB状态失败: {str(e)}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        从文件加载状态
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.exploration_factor = state.get("exploration_factor", 2.0)
            self.max_questions_per_subtask = state.get("max_questions_per_subtask", 5)
            self.min_reward_threshold = state.get("min_reward_threshold", 0.3)
            self.min_questions_before_stop = state.get("min_questions_before_stop", 3)
            
            # 恢复子任务统计
            self.subtask_stats.clear()
            for subtask_id, stats_data in state.get("subtask_stats", {}).items():
                stats = SubtaskStats(subtask_id)
                stats.from_dict(stats_data)
                self.subtask_stats[subtask_id] = stats
            
            self.decision_log = state.get("decision_log", [])
            self.total_questions = state.get("total_questions", 0)
            self.total_rewards = state.get("total_rewards", 0.0)
            
            logger.info(f"📂 MAB状态已从文件加载: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载MAB状态失败: {str(e)}")
            return False


class SubtaskStats:
    """子任务统计信息类"""
    
    def __init__(self, subtask_id: str):
        """
        初始化子任务统计
        
        Args:
            subtask_id (str): 子任务ID
        """
        self.subtask_id = subtask_id
        self.question_count = 0
        self.total_reward = 0.0
        self.average_reward = 0.0
        self.reward_history = []
    
    def update_reward(self, reward: float):
        """
        更新回报
        
        Args:
            reward (float): 新的回报值
        """
        self.question_count += 1
        self.total_reward += reward
        self.average_reward = self.total_reward / self.question_count
        self.reward_history.append(reward)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "subtask_id": self.subtask_id,
            "question_count": self.question_count,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "reward_history": self.reward_history.copy()
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """从字典恢复"""
        self.subtask_id = data.get("subtask_id", self.subtask_id)
        self.question_count = data.get("question_count", 0)
        self.total_reward = data.get("total_reward", 0.0)
        self.average_reward = data.get("average_reward", 0.0)
        self.reward_history = data.get("reward_history", [])


def main():
    """测试MAB平衡器"""
    # 创建MAB平衡器
    mab = MABBalancer(exploration_factor=2.0, max_questions_per_subtask=3)
    
    # 模拟一些决策
    subtask_ids = ["subtask_001", "subtask_002", "subtask_003"]
    
    print("🎰 MAB平衡器测试")
    print("=" * 40)
    
    for i in range(5):
        print(f"\n--- 第 {i+1} 轮 ---")
        
        for subtask_id in subtask_ids:
            # 模拟决策
            should_continue = mab.should_continue_questioning(subtask_id, i)
            print(f"子任务 {subtask_id}: {'继续' if should_continue else '停止'}")
            
            if should_continue:
                # 模拟回报（随机）
                reward = np.random.choice([0.0, 0.5, 1.0], p=[0.3, 0.4, 0.3])
                mab.update_reward(subtask_id, reward)
                print(f"  获得回报: {reward:.1f}")
    
    # 显示统计信息
    print("\n" + "=" * 40)
    print("📊 统计信息:")
    
    global_stats = mab.get_global_stats()
    for key, value in global_stats.items():
        print(f"  {key}: {value}")
    
    print("\n📝 决策日志:")
    for entry in mab.decision_log[-5:]:  # 显示最后5条
        print(f"  {entry['timestamp']} - {entry['subtask_id']}: {entry['decision']} ({entry['reason']})")


if __name__ == "__main__":
    main()

