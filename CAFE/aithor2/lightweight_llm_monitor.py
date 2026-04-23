#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化LLM监控系统
每15秒对比JSON文件变化，决定是否需要触发完整的三LLM理解
使用轻量化模型，10秒内获得答案
"""

import json
import time
import os
import threading
from typing import Dict, Any, Optional, Tuple
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# 尝试导入DashScope API
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

class LightweightLLMMonitor:
    """轻量化LLM监控系统"""
    
    def __init__(self, api_key: str, check_interval: int = 15, trigger_callback=None):
        self.api_key = api_key
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        self.trigger_callback = trigger_callback  # 回调函数，用于触发完整理解
        self.three_llm_running = False  # 标记三LLM是否正在运行
        self.pending_changes = None  # 存储待处理的变化信息
        
        # 历史状态跟踪（双文件机制不再需要这些）
        self.last_check_time = 0
        
        # 轻量化模型配置
        self.model_name = "qwen-turbo"  # 使用轻量化的通义千问模型
        
        # 系统提示词
        self.system_prompt = """你是一个轻量化的场景变化监控助手。你的任务是对比两个时间点的场景JSON文件，快速判断是否需要触发完整的三LLM场景理解。

评分标准（必须严格按照以下规则计算分数）：
1. **新增对象**: 发现了新的物体（每个+2分）
2. **对象状态变化**: 物体的重要状态改变，如isDirty/isOpen/isToggledOn（每个+1分）
3. **位置显著变化**: 物体位置移动超过0.5米（每个+1分）
4. **对象消失**: 之前存在的物体不见了（每个+2分）
5. **容器内容变化**: 柜子、抽屉等容器的内容发生变化（每个+1分）

计算步骤：
1. 对比两个场景的对象列表
2. 统计新增对象数量 × 2分
3. 统计消失对象数量 × 2分
4. 检查状态变化数量 × 1分
5. 检查位置变化数量 × 1分
6. 计算总分

阈值规则：
- 总分 >= 20分：建议触发完整理解
- 总分 < 20分：无需触发

请以JSON格式回复：
{
  "should_trigger": boolean,
  "score": number,
  "reason": "简短说明变化原因和计算过程",
  "changes": ["变化1", "变化2", ...]
}

要求：
- 必须按照评分标准严格计算分数
- 对象消失也是重要变化，必须计分
- 响应时间控制在10秒内
- 如果JSON格式有问题或无法解析，返回should_trigger=false"""

    def _get_api_key(self) -> Optional[str]:
        """获取API密钥"""
        if self.api_key:
            return self.api_key
            
        # 尝试从环境变量获取
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if api_key:
            return api_key
            
        # 尝试从配置文件获取
        try:
            config_path = os.path.join('embodied B1', 'config.py')
            if os.path.exists(config_path):
                import sys
                sys.path.insert(0, os.path.dirname(config_path))
                import config
                return getattr(config, 'DASHSCOPE_API_KEY', None)
        except Exception:
            pass
            
        return None

    def _call_lightweight_llm(self, prompt: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """调用轻量化LLM，带超时控制"""
        if not DASHSCOPE_AVAILABLE:
            print("⚠️ DashScope SDK不可用，跳过轻量化监控")
            return None

        api_key = self._get_api_key()
        if not api_key:
            print("⚠️ 未找到API密钥，跳过轻量化监控")
            return None

        try:
            dashscope.api_key = api_key

            full_prompt = f"{self.system_prompt}\n\n{prompt}"

            print(f"📡 调用{self.model_name}模型 (超时: {timeout}秒)...")
            start_time = time.time()

            # 使用线程池执行API调用，添加超时保护
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    Generation.call,
                    model=self.model_name,
                    prompt=full_prompt,
                    result_format='message',
                    max_tokens=500,
                    temperature=0.3,
                    top_p=0.8,
                )
                
                try:
                    response = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    print(f"⚠️ LLM API 调用超时 ({timeout}秒)，监控跳过此周期")
                    return None

            elapsed_time = time.time() - start_time
            print(f"⏱️ API调用耗时: {elapsed_time:.2f}秒")

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                print(f"📝 LLM原始回复: {content[:200]}...")

                # 尝试解析JSON
                try:
                    result = json.loads(content)
                    print(f"✅ JSON解析成功")
                    return result
                except json.JSONDecodeError:
                    # 如果不是标准JSON，尝试提取JSON部分
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                            print(f"✅ 从文本中提取JSON成功")
                            return result
                        except json.JSONDecodeError:
                            print(f"⚠️ 提取的JSON格式仍然错误")
                            return None
                    else:
                        print(f"⚠️ LLM返回格式错误，无法找到JSON: {content}")
                        return None
            else:
                print(f"⚠️ LLM调用失败 (状态码: {response.status_code}): {response.message}")
                return None

        except Exception as e:
            print(f"⚠️ LLM调用异常: {e}")
            return None

    def _get_json_files(self) -> Tuple[Optional[str], Optional[str]]:
        """获取实时JSON文件和基线JSON文件路径"""
        # 固定的实时状态文件 - 持续更新的当前场景状态
        current_path = "semantic_maps/current_scene_state.json"

        # 15秒前的基线文件 - 用于比较的快照
        baseline_path = "semantic_maps/baseline_15s_ago.json"

        return current_path, baseline_path

    def _load_json_safely(self, file_path: str) -> Optional[Dict[str, Any]]:
        """安全加载JSON文件"""
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                return None
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ 加载JSON文件失败 {file_path}: {e}")
            return None

    def _calculate_json_hash(self, json_data: Dict[str, Any]) -> str:
        """计算JSON内容的哈希值"""
        # 只关注objects部分，忽略时间戳等
        relevant_data = {}
        if "objects" in json_data:
            relevant_data["objects"] = json_data["objects"]
        elif "nodes" in json_data:
            relevant_data["nodes"] = json_data["nodes"]
            
        json_str = json.dumps(relevant_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(json_str.encode()).hexdigest()

    def _save_baseline_snapshot(self, data: Dict[str, Any], baseline_path: str):
        """保存基线快照"""
        try:
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            with open(baseline_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 基线快照已保存: {baseline_path}")
        except Exception as e:
            print(f"⚠️ 保存基线快照失败: {e}")

    def _check_for_changes(self) -> Optional[Dict[str, Any]]:
        """检查场景变化 - 使用固定状态文件机制"""
        # 导入状态管理器
        try:
            import scene_state_manager as SSM
            state_manager = SSM.scene_state_manager
        except ImportError:
            print("❌ 无法导入场景状态管理器")
            return None

        # 检查当前状态文件是否存在
        if not state_manager.has_current_state():
            print("📊 实时状态文件不存在，等待场景数据...")
            return None

        # 获取当前状态
        current_data = state_manager.get_current_state()
        if not current_data:
            print("❌ 当前状态文件加载失败或为空")
            return None

        # 检查基线状态文件是否存在
        if not state_manager.has_baseline_state():
            # 首次运行 - 创建基线快照
            objects_count = len(current_data.get("objects", []))
            print(f"🔍 轻量化监控首次启动，创建基线快照 (当前{objects_count}个对象)")
            state_manager.save_current_as_baseline()
            return None

        # 获取基线状态（15秒前）
        baseline_data = state_manager.get_baseline_state()
        if not baseline_data:
            print("❌ 基线状态文件加载失败")
            return None

        # 检查对象数量变化
        current_objects = current_data.get("objects", [])
        baseline_objects = baseline_data.get("objects", [])

        current_count = len(current_objects)
        baseline_count = len(baseline_objects)

        # 注意：不再每次都更新基线，只在触发三LLM时更新
        # state_manager.save_current_as_baseline()  # 已移除

        if current_count == baseline_count:
            print("📊 场景内容无变化")
            return None

        # 有变化，进行轻量化分析
        print(f"🔍 检测到场景变化，启动轻量化分析... (对象数量: {baseline_count} -> {current_count})")

        # 提取关键信息进行对比
        baseline_objects = baseline_data.get("objects", []) or baseline_data.get("nodes", [])
        current_objects = current_data.get("objects", []) or current_data.get("nodes", [])

        # 生成对象摘要（包含状态和位置信息）
        def extract_object_info(obj):
            # 处理中文字段名的JSON结构
            if "名称" in obj:
                name_info = obj["名称"]
                obj_type = name_info.get("type", "unknown")
                obj_id = name_info.get("id", "unknown")

                # 提取状态信息
                states = obj.get("状态", {})
                position = obj.get("位置", {})

                return {
                    "id": obj_id,
                    "type": obj_type,
                    "isDirty": states.get("isDirty", False),
                    "isOpen": states.get("isOpen", False),
                    "isToggledOn": states.get("isToggledOn", False),
                    "position": {
                        "x": round(position.get("x", 0), 2),
                        "y": round(position.get("y", 0), 2),
                        "z": round(position.get("z", 0), 2)
                    }
                }
            else:
                # 兼容其他格式
                obj_type = obj.get("objectType", obj.get("type", "unknown"))
                obj_id = obj.get("name", obj.get("id", "unknown"))
                return {"id": obj_id, "type": obj_type}

        # 基于位置匹配对象（解决"不可见≠消失"的问题）
        def match_objects_by_position(baseline_objs, current_objs, threshold=0.15):
            """
            基于位置和类型匹配对象
            threshold: 位置匹配阈值（米），默认15cm
            返回: (真正新增的对象, 真正消失的对象, 位置变化的对象)
            """
            baseline_summaries = [extract_object_info(obj) for obj in baseline_objs]
            current_summaries = [extract_object_info(obj) for obj in current_objs]

            # 构建位置索引
            baseline_by_pos = {}  # (type, x, y, z) -> obj_info
            for obj in baseline_summaries:
                pos = obj.get("position", {})
                key = (obj["type"],
                       round(pos.get("x", 0) / threshold) * threshold,
                       round(pos.get("y", 0) / threshold) * threshold,
                       round(pos.get("z", 0) / threshold) * threshold)
                baseline_by_pos[key] = obj

            current_by_pos = {}
            for obj in current_summaries:
                pos = obj.get("position", {})
                key = (obj["type"],
                       round(pos.get("x", 0) / threshold) * threshold,
                       round(pos.get("y", 0) / threshold) * threshold,
                       round(pos.get("z", 0) / threshold) * threshold)
                current_by_pos[key] = obj

            # 找出真正新增和消失的对象
            baseline_keys = set(baseline_by_pos.keys())
            current_keys = set(current_by_pos.keys())

            truly_new = current_keys - baseline_keys
            truly_missing = baseline_keys - current_keys

            # 检查位置变化（同一对象但位置改变）
            position_changed = []
            for key in baseline_keys & current_keys:
                baseline_obj = baseline_by_pos[key]
                current_obj = current_by_pos[key]

                # 精确比较位置
                b_pos = baseline_obj.get("position", {})
                c_pos = current_obj.get("position", {})
                distance = ((b_pos.get("x", 0) - c_pos.get("x", 0)) ** 2 +
                           (b_pos.get("y", 0) - c_pos.get("y", 0)) ** 2 +
                           (b_pos.get("z", 0) - c_pos.get("z", 0)) ** 2) ** 0.5

                if distance > 0.5:  # 移动超过50cm
                    position_changed.append({
                        "id": current_obj["id"],
                        "type": current_obj["type"],
                        "distance": round(distance, 2)
                    })

            return (
                [current_by_pos[k] for k in truly_new],
                [baseline_by_pos[k] for k in truly_missing],
                position_changed
            )

        # 使用基于位置的匹配来识别真正的变化
        truly_new, truly_missing, position_changed = match_objects_by_position(
            baseline_objects, current_objects, threshold=0.15
        )

        # 调试信息
        print(f"🔍 [DEBUG] 基线总数: {baseline_count}, 当前总数: {current_count}")
        print(f"🔍 [DEBUG] 基于位置匹配结果:")
        print(f"   - 真正新增: {len(truly_new)} 个对象")
        print(f"   - 真正消失: {len(truly_missing)} 个对象")
        print(f"   - 位置变化: {len(position_changed)} 个对象")

        if truly_new:
            print(f"🔍 [DEBUG] 新增对象: {[obj['id'] for obj in truly_new[:5]]}...")
        if truly_missing:
            print(f"🔍 [DEBUG] 消失对象: {[obj['id'] for obj in truly_missing[:5]]}...")
        if position_changed:
            moved_info = [f"{obj['id']}(移动{obj['distance']}m)" for obj in position_changed[:5]]
            print(f"🔍 [DEBUG] 位置变化: {moved_info}...")

        # 准备给LLM的摘要（采样以减少token消耗）
        sample_size = 25
        baseline_sample = baseline_objects[:sample_size]
        current_sample = current_objects[:sample_size]
        baseline_summary = [extract_object_info(obj) for obj in baseline_sample]
        current_summary = [extract_object_info(obj) for obj in current_sample]

        prompt = f"""请对比以下两个时间点的场景数据，判断是否需要触发完整的三LLM理解：

**基线场景 ({baseline_count}个对象，显示前30个)：**
```json
{json.dumps(baseline_summary, ensure_ascii=False, indent=2)}
```

**当前场景 ({current_count}个对象，显示前30个)：**
```json
{json.dumps(current_summary, ensure_ascii=False, indent=2)}
```

**⚠️ 重要：基于位置匹配的真实变化统计（已排除视角变化导致的误判）：**
- 真正新增的对象：{len(truly_new)} 个（这些是之前不存在的新对象）
- 真正消失的对象：{len(truly_missing)} 个（这些对象从原位置消失了，不是因为视角变化）
- 位置显著变化：{len(position_changed)} 个对象移动超过50cm

**评分标准：**
- 新增对象：每个 +2分
- 消失对象：每个 +2分
- 位置变化：每个 +1分
- 状态变化（isDirty/isOpen/isToggledOn）：每个 +1分
- 阈值：≥20分触发完整理解

请基于真实变化统计进行评分和分析。"""

        print("🤖 正在调用轻量化LLM分析变化...")
        result = self._call_lightweight_llm(prompt)

        if result:
            print(f"✅ LLM分析完成: {result}")
        else:
            print("❌ LLM分析失败或返回空结果")

        return result

    def _monitor_loop(self):
        """监控循环"""
        print(f"🤖 轻量化LLM监控启动，每{self.check_interval}秒检查一次")

        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_check_time >= self.check_interval:
                    print(f"⏰ [{time.strftime('%H:%M:%S')}] 开始检查场景变化...")

                    result = self._check_for_changes()

                    if result:
                        score = result.get("score", 0)
                        reason = result.get("reason", "未知原因")
                        changes = result.get("changes", [])
                        should_trigger = result.get("should_trigger", False)

                        # 修复：基于评分判断，而不是should_trigger字段（LLM可能输出不一致）
                        if score >= 20:
                            print(f"🚨 轻量化监控检测到重要变化！")
                            print(f"   评分: {score}分 (阈值: 20分)")
                            print(f"   原因: {reason}")
                            print(f"   变化: {', '.join(changes)}")
                            print(f"   LLM建议: {'触发' if should_trigger else '不触发'} (已忽略，以评分为准)")

                            # 触发完整的三LLM理解，传递变化信息
                            self._trigger_full_understanding(result)
                        else:
                            print(f"ℹ️ 场景变化不显著 (评分: {score}分 < 20分) - {reason}")
                    else:
                        print(f"✅ 无场景变化检测到")

                    self.last_check_time = current_time
                else:
                    # 显示倒计时
                    remaining = int(self.check_interval - (current_time - self.last_check_time))
                    if remaining % 5 == 0 and remaining > 0:  # 每5秒显示一次倒计时
                        print(f"⏳ 下次检查还有 {remaining} 秒...")

                time.sleep(1)  # 避免CPU占用过高

            except Exception as e:
                print(f"⚠️ 监控循环异常: {e}")
                time.sleep(5)

    def _trigger_full_understanding(self, changes_info: Dict[str, Any] = None):
        """触发完整的三LLM理解"""
        try:
            # 检查三LLM是否正在运行
            if self.three_llm_running:
                print("⏸️ 三LLM正在运行中，暂不触发新的理解")
                print("📝 变化信息已记录，等待当前理解完成")
                self.pending_changes = changes_info
                return

            print("🚨 [DEBUG] 准备触发完整理解...")
            if self.trigger_callback:
                print("🚨 [DEBUG] 调用回调函数...")

                # 标记三LLM开始运行
                self.three_llm_running = True

                # 保存变化信息到文件，供三LLM使用
                if changes_info:
                    self._save_changes_for_three_llm(changes_info)
                # 在触发前，如果存在候选的 realtime 文件，则将其推送为正式 realtime 文件
                try:
                    import exploration_io as EIO
                    promoted = EIO.promote_candidate_to_realtime()
                    if promoted:
                        print("🔁 候选 realtime 已推广为正式文件，三LLM 将使用最新地图")
                except Exception:
                    pass

                self.trigger_callback()

                # 仅在“真正触发了三LLM”后更新基线快照
                try:
                    import scene_state_manager as SSM
                    if SSM.scene_state_manager.save_current_as_baseline():
                        print("💾 基线快照已保存（仅在触发三LLM后更新）")
                    else:
                        print("⚠️ 基线快照更新失败（触发后）")
                except Exception as e:
                    print(f"⚠️ 触发后保存基线快照失败: {e}")

                print("✅ 已触发完整的三LLM场景理解")
                print("🚨 [DEBUG] 回调函数调用完成")
            else:
                print("⚠️ 未设置触发回调函数")
        except Exception as e:
            print(f"⚠️ 触发完整理解失败: {e}")
            import traceback
            traceback.print_exc()
            self.three_llm_running = False  # 出错时重置状态

    def _save_changes_for_three_llm(self, changes_info: Dict[str, Any]):
        """保存变化信息到文件，供三LLM使用"""
        try:
            # 添加时间戳
            changes_info["detected_at"] = time.time()

            changes_file = "semantic_maps/detected_changes.json"
            with open(changes_file, 'w', encoding='utf-8') as f:
                json.dump(changes_info, f, ensure_ascii=False, indent=2)
            print(f"💾 变化信息已保存到: {changes_file}")
        except Exception as e:
            print(f"⚠️ 保存变化信息失败: {e}")

    def mark_three_llm_completed(self):
        """标记三LLM理解完成"""
        self.three_llm_running = False
        print("✅ 三LLM理解已完成，监控系统恢复正常")

        # 检查是否有待处理的变化
        if self.pending_changes:
            print("📝 检测到待处理的变化，准备触发新的理解...")
            self._trigger_full_understanding(self.pending_changes)
            self.pending_changes = None

    def start(self):
        """启动监控"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("🤖 轻量化LLM监控已停止")

# 全局实例
_monitor_instance = None

def get_monitor_instance():
    """获取监控实例"""
    global _monitor_instance
    return _monitor_instance

def start_lightweight_monitor(api_key: str = "", check_interval: int = 15, trigger_callback=None):
    """启动轻量化监控"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = LightweightLLMMonitor(api_key, check_interval, trigger_callback)
        _monitor_instance.start()
    return _monitor_instance

def stop_lightweight_monitor():
    """停止轻量化监控"""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop()
        _monitor_instance = None
