#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三个LLM协作的具身任务规划系统 V2（保留就绪度 + 支持跨房间）
LLMA: 世界场景描述LLM
LLMB: 任务思考LLM（含就绪度抽取）
LLMC: 任务序列优化LLM
TaskExpander: 步骤展开LLM
"""

import argparse
import os
import json
import time
import re
import math
from typing import Dict, List, Any, Optional, Tuple
import dashscope
from dashscope import Generation
from config import DASHSCOPE_API_KEY
from prompts import (
    LLMA_SYSTEM_PROMPT,
    LLMB_SYSTEM_PROMPT,
    LLMC_SYSTEM_PROMPT,
    CONVERSATION_FLOW_PROMPTS,
    TASK_EXPANDER_SYSTEM_PROMPT,
    TASK_EXPANDER_PROMPT,
)

# =========================================================
# DashScope API 配置（Qwen3-Max）
# =========================================================
QWEN_MODEL = "qwen3-max-2026-01-23"
# 保留旧常量名，避免外部调用（如 batch_compare）报错
GOOGLE_API_KEY = DASHSCOPE_API_KEY
GOOGLE_MODEL = QWEN_MODEL

# -------------------------
# 轻量就绪度抽取提示（不限制回答风格，只要求在抽取时给出JSON）
# -------------------------
READINESS_EXTRACTOR_INSTRUCTION = """
你是一个仅做信息抽取的助手。请阅读“任务描述”和“对话历史”，
仅输出一个 JSON（不要多余文字），字段为：
{
  "object_id_known": true/false,
  "tool_or_consumable_known": true/false,
  "location_or_access_known": true/false,
  "key_state_known": true/false,
  "readiness_score": 0.0~1.0 的小数（四项均等加权平均）
}
- 你的判断必须只基于对话出现或可以严格推出的证据；不允许臆测。
- readiness_score = (四个布尔项的平均值)。
- 只输出 JSON。
"""

# =========================================================
# 基础 LLM 调用类
# =========================================================
class LLMBase:
    """LLM基础类"""

    def __init__(self, api_key: str, system_prompt: str, name: str):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.name = name
        self.setup_api()

    def setup_api(self):
        dashscope.api_key = self.api_key
        print(f"✅ {self.name} DashScope API密钥已设置（模型: {QWEN_MODEL}）")

    def call_llm(self, prompt: str, max_tokens: int = 1500, max_retries: int = 3, temperature: float = 0.7) -> str:
        """调用LLM，带重试机制"""
        for attempt in range(max_retries):
            try:
                response = Generation.call(
                    api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"),
                    model=QWEN_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    result_format="message",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.8,
                    enable_thinking=False,
                )

                text = ""
                if getattr(response, "status_code", None) == 200:
                    try:
                        text = (response.output.choices[0].message.content or "").strip()
                    except Exception:
                        text = ""
                if text:
                    return text
                print(f"⚠️ {self.name} 第{attempt + 1}次尝试失败: 响应为空")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
            except Exception as e:
                print(f"⚠️ {self.name} 第{attempt + 1}次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        return f"❌ {self.name} API调用失败"

    def call_llm_with_temp_system(self, temp_system_prompt: str, prompt: str, **kwargs) -> str:
        """临时替换 system prompt 进行一次调用"""
        original = self.system_prompt
        try:
            self.system_prompt = temp_system_prompt
            return self.call_llm(prompt, **kwargs)
        finally:
            self.system_prompt = original

# =========================================================
# LLMA: 世界场景描述
# =========================================================
class LLMA(LLMBase):
    """世界场景描述LLM"""

    def __init__(self, api_key: str):
        super().__init__(api_key, LLMA_SYSTEM_PROMPT, "LLMA")
        self.world_model: Optional[Dict[str, Any]] = None
        self.extra_rooms: List[Dict[str, Any]] = []

    def load_world_model(self, world_model_path: str):
        try:
            with open(world_model_path, 'r', encoding='utf-8') as f:
                self.world_model = json.load(f)
            print(f"✅ {self.name} 世界模型已加载: {world_model_path}")
        except Exception as e:
            print(f"❌ {self.name} 世界模型加载失败: {e}")

    def add_extra_room(self, room_path: str):
        """可选：加载额外房间（跨房间检索时使用）"""
        try:
            with open(room_path, "r", encoding="utf-8") as f:
                room = json.load(f)
            self.extra_rooms.append(room)
            print(f"✅ {self.name} 已加载额外房间: {room_path}")
        except Exception as e:
            print(f"⚠️ {self.name} 额外房间加载失败 {room_path}: {e}")

    def _json_str(self, data: Dict[str, Any]) -> str:
        """将字典转换为JSON字符串，并进行长度控制"""
        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        # 如果JSON太长，进行压缩
        if len(json_str) > 50000:  # 50KB限制
            compressed_data = self._compress_world_model(data)
            json_str = json.dumps(compressed_data, ensure_ascii=False, indent=2)
            print(f"🗜️ 世界模型已压缩: {len(json_str)} 字符")

        return json_str

    def _compress_world_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """压缩世界模型数据，保留最重要的信息"""
        compressed = {
            "scene_id": data.get("scene_id", ""),
            "exploration_stats": data.get("exploration_stats", {}),
            "nodes": [],
            "edges": data.get("edges", [])[:20]  # 只保留前20个关系
        }

        # 按类型分组并限制数量
        nodes_by_type = {}
        for node in data.get("nodes", []):
            node_type = node.get("label", "Unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)

        # 每种类型最多保留5个对象，总共不超过50个
        total_nodes = 0
        for node_type, nodes in sorted(nodes_by_type.items()):
            if total_nodes >= 50:
                break

            # 优先保留有位置信息的节点
            nodes_with_pos = [n for n in nodes if "position_3d" in n.get("attributes", {})]
            nodes_without_pos = [n for n in nodes if "position_3d" not in n.get("attributes", {})]

            selected_nodes = (nodes_with_pos[:4] + nodes_without_pos[:1])[:5]

            # 简化节点信息
            for node in selected_nodes:
                if total_nodes >= 50:
                    break

                simplified_node = {
                    "id": node.get("id", ""),
                    "label": node.get("label", ""),
                    "category": node.get("category", "")
                }

                # 只保留关键属性
                attrs = node.get("attributes", {})
                if "position_3d" in attrs:
                    pos = attrs["position_3d"]
                    simplified_node["position"] = f"({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}, {pos.get('z', 0):.1f})"

                compressed["nodes"].append(simplified_node)
                total_nodes += 1

        return compressed

    def describe_scene(self) -> str:
        if not self.world_model:
            return "❌ 世界模型未加载"
        # 先根据实际场景对象过滤，再过滤探索状态，避免上帝视角
        actual_filtered_data = self._filter_by_actual_objects(self.world_model)
        explored_data = self._filter_explored_objects(actual_filtered_data)
        wm = self._json_str(explored_data)

        # 尝试读取轻量化LLM检测到的变化信息
        changes_info = ""
        try:
            import os, json
            changes_file = "semantic_maps/detected_changes.json"
            if os.path.exists(changes_file):
                with open(changes_file, 'r', encoding='utf-8') as f:
                    changes_data = json.load(f)
                    if changes_data:
                        changes_info = f"""

**⚠️ 轻量化监控检测到的场景变化（评分: {changes_data.get('score', 0)}分）：**
原因：{changes_data.get('reason', '未知')}
变化详情：
{chr(10).join('- ' + str(c) for c in changes_data.get('changes', []))}

请特别关注这些变化，它们可能影响任务规划。
"""
        except Exception:
            pass

        prompt = CONVERSATION_FLOW_PROMPTS["llma_initial_description"].format(world_model=wm) + changes_info
        return self.call_llm(prompt)

    def propose_macro_tasks(self) -> str:
        """由A基于已观察世界模型直接给出【高层候选任务清单】（非原子动作）。"""
        if not self.world_model:
            return ""
        # 仍然只用已观察的过滤模型，防止上帝视角
        actual_filtered = self._filter_by_actual_objects(self.world_model)
        explored = self._filter_explored_objects(actual_filtered)
        wm = self._json_str(explored)
        prompt = CONVERSATION_FLOW_PROMPTS["llma_task_discovery"].format(world_model=wm)
        # 增加 max_tokens 以支持复杂场景的大量任务输出
        return self.call_llm(prompt, max_tokens=4000, temperature=0.3)

    def _filter_explored_objects(self, world_model: Dict[str, Any]) -> Dict[str, Any]:
        """过滤出已探索的对象，避免给LLM提供上帝视角"""
        if not world_model:
            return {}

        # 创建过滤后的世界模型副本
        filtered_model = world_model.copy()

        # 如果有nodes字段（语义地图格式）
        if "nodes" in world_model:
            # 只保留已被观察到的对象
            # 这里可以根据实际的探索状态来过滤
            # 暂时保持原有逻辑，但添加标记表明这是基于当前观察
            filtered_nodes = []
            for node in world_model.get("nodes", []):
                # 添加探索状态标记
                node_copy = node.copy()
                node_copy["exploration_status"] = "observed"  # 标记为已观察
                filtered_nodes.append(node_copy)
            filtered_model["nodes"] = filtered_nodes

        return filtered_model

    def set_actual_scene_objects(self, actual_objects: List[str]):
        """设置实际场景中存在的对象列表，用于过滤世界模型"""
        self.actual_scene_objects = set(actual_objects)
        print(f"✅ {self.name} 已更新实际场景对象列表: {len(actual_objects)} 个对象")

    def _filter_by_actual_objects(self, world_model: Dict[str, Any]) -> Dict[str, Any]:
        """根据实际场景中存在的对象过滤世界模型。
        兼容两种输入格式：
        1) 语义图格式：{ "nodes": [{ id, label, attributes: { ... } }] }
        2) 实时结构化格式：{ "objects": [{ "名称": {id,type}, "位置": {x,y,z}, ... }] }
        """
        if not world_model or not hasattr(self, 'actual_scene_objects'):
            return world_model

        filtered_model = world_model.copy()

        # 语义图格式
        if "nodes" in world_model:
            filtered_nodes = []
            for node in world_model.get("nodes", []):
                node_id = node.get("id", "")
                # 只保留实际存在的对象
                if node_id in self.actual_scene_objects or node.get("label") == "Floor":
                    filtered_nodes.append(node)
                else:
                    print(f"🔍 过滤掉不存在的对象: {node_id}")
            filtered_model["nodes"] = filtered_nodes

        # 实时结构化格式
        if "objects" in world_model and isinstance(world_model.get("objects"), list):
            filtered_objs = []
            for obj in world_model.get("objects", []):
                name = obj.get("名称") if isinstance(obj, dict) else None
                oid = (name or {}).get("id") if isinstance(name, dict) else None
                if not oid:
                    # 兜底：有些源可能直接给 id/objectId 字段
                    oid = obj.get("id") or obj.get("objectId")
                if oid and oid in self.actual_scene_objects:
                    filtered_objs.append(obj)
            filtered_model["objects"] = filtered_objs

        return filtered_model

    def answer_question(self, question: str) -> str:
        if not self.world_model:
            return "❌ 世界模型未加载"
        wm = self._json_str(self.world_model)
        prompt = CONVERSATION_FLOW_PROMPTS["llma_question"].format(world_model=wm, question=question)
        return self.call_llm(prompt)

    def answer_questions_batch(self, questions_text: str) -> str:
        if not self.world_model:
            return "❌ 世界模型未加载"
        wm = self._json_str(self.world_model)
        prompt = CONVERSATION_FLOW_PROMPTS["llma_batch_answer"].format(world_model=wm, questions=questions_text)
        # 增加 max_tokens 以支持复杂场景的大量问题回答
        return self.call_llm(prompt, max_tokens=4000, temperature=0.2)

# =========================================================
# LLMB: 任务思考 + 就绪度抽取
# =========================================================
class LLMB(LLMBase):
    """任务思考LLM（分层追问机制 + 就绪度抽取）"""

    def __init__(self, api_key: str):
        super().__init__(api_key, LLMB_SYSTEM_PROMPT, "LLMB")
        # 追问状态跟踪
        self.inquiry_state = {
            "current_layer": 1,  # 1=分类, 2=对象, 3=解决方案
            "categories_identified": False,
            "objects_identified": False,
            "solutions_identified": False,
            "main_category": "",
            "identified_objects": [],
            "available_containers": []
        }

    def propose_tasks(self, scene_description: str, available_objects_text: Optional[str] = None) -> str:
        """让LLMB输出候选任务（参考已知对象列表），以减少臆测。"""
        obj_block = f"\n\n仅使用以下已知对象ID（如果需要引用具体ID）：\n{available_objects_text}" if available_objects_text else ""
        prompt = (
            "以下是世界场景的描述：\n"
            f"{scene_description}{obj_block}\n\n"
            "请给出你认为需要执行的候选任务清单（建议每行一个任务、短语级别即可；不要使用未在上方对象列表中的ID）。"
        )
        return self.call_llm(prompt)

    def ask_batch_questions(self, current_task: str, available_objects_text: str, conversation_history: str = "") -> str:
        """
        生成一组针对执行步骤的具体问题（一次3~6个），每个问题都必须直指可执行信息：
        - 缺失的 objectId（从已知对象列表中选择）
        - 必要设备/容器的具体ID
        - 关键状态/可达性（能否打开/是否在范围内/是否有水等）
        - 必要的前置/后置操作
        若信息已充分，则输出：NO_QUESTION: READY
        """
        prompt = (
            "【任务】\n" + current_task + "\n\n"
            "【已知对象（仅允许引用这里的ID）】\n" + (available_objects_text or "（无）") + "\n\n"
            "【对话历史】\n" + (conversation_history or "（无）") + "\n\n"
            "请一次性给出3~6个‘对执行有直接作用’的问题（编号列出），避免冗余；"
            "每个问题都要指向具体缺口（ID/位置/状态/开关/可达性/顺序），并尽量引用上面的ID。"
            "如果信息已充分，不要提问，直接输出：NO_QUESTION: READY"
            "但只有当你确信以下四项均为已知时，才允许输出：NO_QUESTION: READY ——"
            "object_id_known=true, tool_or_consumable_known=true, location_or_access_known=true, key_state_known=true；"
            "否则必须提出上述成组问题。"

        )
        return self.call_llm(prompt, temperature=0.3)




    def ask_questions_for_all(self, macro_tasks_text: str, allowed_ids_text: str) -> str:
        """针对A给出的所有高层任务，一次性生成按任务分组的成组问题。"""
        prompt = CONVERSATION_FLOW_PROMPTS["llmb_batch_questions_for_all"].format(
            tasks=macro_tasks_text,
            allowed_ids=allowed_ids_text or "（无）",
        )
        # 增加 max_tokens 以支持复杂场景的大量任务提问
        return self.call_llm(prompt, max_tokens=4000, temperature=0.3)

    def synthesize_from_answers(
        self,
        macro_tasks_text: str,
        questions_text: str,
        answers_text: str,
        allowed_ids_text: str,
    ) -> str:
        """根据A的批量回答，把已就绪的高层任务转成原子任务清单。"""
        prompt = CONVERSATION_FLOW_PROMPTS["llmb_synthesize_from_answers"].format(
            tasks=macro_tasks_text,
            questions=questions_text,
            answers=answers_text,
            allowed_ids=allowed_ids_text or "（无）",
        )
        # 增加 max_tokens 以支持复杂场景的大量任务输出
        return self.call_llm(prompt, max_tokens=4000, temperature=0.4)

    def ask_one_question(self, current_task: str, conversation_history: str) -> str:
        """实现分层追问机制：分类→对象→解决方案"""

        # 分析对话历史，更新状态
        self._update_inquiry_state(conversation_history)

        if self.inquiry_state["current_layer"] == 1:
            # 第一层：问题分类
            return "这个场景中主要存在哪几类问题？比如：物品散乱、设备状态异常、清洁问题等。请按重要性排序。"

        elif self.inquiry_state["current_layer"] == 2:
            # 第二层：具体对象识别
            if self.inquiry_state["main_category"]:
                return f"针对{self.inquiry_state['main_category']}这个问题，具体涉及哪些物品？请列出它们的准确ID和当前位置。"
            else:
                return "针对最重要的问题类别，具体涉及哪些物品？请列出它们的准确ID和当前位置。"

        elif self.inquiry_state["current_layer"] == 3:
            # 第三层：解决方案匹配
            if self.inquiry_state["identified_objects"]:
                objects_str = "、".join(self.inquiry_state["identified_objects"][:3])  # 限制长度
                return f"要解决这些物品的问题：{objects_str}等，场景中有哪些合适的目标位置或容器？请提供具体的ID和位置信息。"
            else:
                return "对于需要整理的这些物品，场景中有哪些合适的目标位置或容器？请提供具体的ID和位置信息。"

        else:
            # 兜底问题
            return f"关于任务'{current_task}'，还需要了解哪些关键信息才能制定具体的执行计划？"

    def _update_inquiry_state(self, conversation_history: str):
        """根据对话历史更新追问状态"""
        if not conversation_history:
            return

        # 检查是否已识别问题分类
        if not self.inquiry_state["categories_identified"]:
            # 更宽泛的关键词匹配
            category_keywords = ["物品散乱", "设备状态", "清洁", "危险", "整理", "散落", "掉落", "位置不当", "异常", "混乱"]
            if any(keyword in conversation_history for keyword in category_keywords):
                self.inquiry_state["categories_identified"] = True
                self.inquiry_state["current_layer"] = 2
                # 提取主要问题类别
                if any(keyword in conversation_history for keyword in ["物品散乱", "散落", "掉落", "位置不当"]):
                    self.inquiry_state["main_category"] = "物品散乱"
                elif any(keyword in conversation_history for keyword in ["设备状态", "异常"]):
                    self.inquiry_state["main_category"] = "设备状态异常"
                elif "危险" in conversation_history:
                    self.inquiry_state["main_category"] = "危险物品"
                else:
                    self.inquiry_state["main_category"] = "物品整理"

        # 检查是否已识别具体对象
        if not self.inquiry_state["objects_identified"] and self.inquiry_state["categories_identified"]:
            # 查找对象ID模式
            import re
            object_patterns = re.findall(r'([A-Za-z]+)\|[+\-]?\d+\.\d+\|[+\-]?\d+\.\d+\|[+\-]?\d+\.\d+', conversation_history)
            if object_patterns:
                self.inquiry_state["identified_objects"] = list(set(object_patterns))
                self.inquiry_state["objects_identified"] = True
                self.inquiry_state["current_layer"] = 3

        # 检查是否已识别解决方案
        if not self.inquiry_state["solutions_identified"] and self.inquiry_state["objects_identified"]:
            # 查找容器相关词汇
            if any(keyword in conversation_history for keyword in ["桌子", "柜子", "架子", "碗", "盘子", "容器"]):
                self.inquiry_state["solutions_identified"] = True
                self.inquiry_state["current_layer"] = 4  # 完成所有层次

    def extract_readiness(self, current_task: str, conversation_history: str) -> Dict[str, Any]:
        """
        让LLMB基于对话历史抽取就绪度四元组与分数（仅输出 JSON）。
        采用临时 system prompt（READINESS_EXTRACTOR_INSTRUCTION）。
        """
        prompt = (
            f"任务描述：{current_task}\n\n"
            f"对话历史：\n{conversation_history}\n\n"
            "请按要求输出 JSON。"
        )
        raw = self.call_llm_with_temp_system(READINESS_EXTRACTOR_INSTRUCTION, prompt, temperature=0.0)
        # 容错解析
        try:
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            text = m.group(0) if m else raw
            data = json.loads(text)
        except Exception:
            data = {
                "object_id_known": False,
                "tool_or_consumable_known": False,
                "location_or_access_known": False,
                "key_state_known": False,
                "readiness_score": 0.0,
            }
        # 兜底与一致性校正
        for k in [
            "object_id_known",
            "tool_or_consumable_known",
            "location_or_access_known",
            "key_state_known",
            "readiness_score",
        ]:
            data.setdefault(k, False if k.endswith("_known") else 0.0)
        bools = [
            bool(data.get("object_id_known")),
            bool(data.get("tool_or_consumable_known")),
            bool(data.get("location_or_access_known")),
            bool(data.get("key_state_known")),
        ]
        data["readiness_score"] = sum(1.0 if b else 0.0 for b in bools) / 4.0
        return data

# =========================================================
# LLMC: 任务序列优化
# =========================================================
class LLMC(LLMBase):
    """任务序列优化LLM"""

    def __init__(self, api_key: str):
        super().__init__(api_key, LLMC_SYSTEM_PROMPT, "LLMC")

    def optimize_sequence(self, task_list_text: str, conversation_history: str) -> str:
        """将任务列表转换为排序的可执行序列（不给定死格式）"""
        prompt = CONVERSATION_FLOW_PROMPTS["llmc_task_optimization"].format(
            task_sequence=task_list_text
        ) + "\n\n（可参考对话上下文，但不必受其格式限制）\n" + conversation_history
        # 增加 max_tokens 以支持复杂场景的大量任务优化
        return self.call_llm(prompt, max_tokens=4000, temperature=0.5)

# =========================================================
# TaskExpander: 步骤扩展
# =========================================================
class TaskExpander(LLMBase):
    """任务扩展LLM"""

    def __init__(self, api_key: str):
        super().__init__(api_key, TASK_EXPANDER_SYSTEM_PROMPT, "TaskExpander")

    def expand(self, task_sequence: str, world_model_data: dict = None, actual_objects: List[str] = None) -> str:
        # 先根据实际对象过滤世界模型数据
        if actual_objects and world_model_data:
            filtered_data = self._filter_by_actual_objects(world_model_data, actual_objects)
        else:
            filtered_data = world_model_data

        # 提取场景中存在的对象列表
        available_objects = self._extract_available_objects(filtered_data)

        prompt = TASK_EXPANDER_PROMPT.format(
            task_sequence=task_sequence,
            available_objects=available_objects
        )
        # 增加 max_tokens 以支持复杂场景的大量任务扩展
        return self.call_llm(prompt, max_tokens=6000)

    def _filter_by_actual_objects(self, world_model_data: dict, actual_objects: List[str]) -> dict:
        """根据实际场景中存在的对象过滤世界模型数据"""
        if not world_model_data or not actual_objects:
            return world_model_data

        actual_objects_set = set(actual_objects)
        filtered_data = world_model_data.copy()

        # 1) 
        if "nodes" in world_model_data:
            filtered_nodes = []
            for node in world_model_data.get("nodes", []):
                node_id = node.get("id", "")
                # 只保留实际存在的对象
                if node_id in actual_objects_set or node.get("label") == "Floor":
                    filtered_nodes.append(node)
            filtered_data["nodes"] = filtered_nodes

        # 2) 
        if "objects" in world_model_data and isinstance(world_model_data.get("objects"), list):
            filtered_objs = []
            for obj in world_model_data.get("objects", []):
                name = obj.get("名称") if isinstance(obj, dict) else None
                oid = name.get("id") if isinstance(name, dict) else (obj.get("id") or obj.get("objectId"))
                if oid in actual_objects_set:
                    filtered_objs.append(obj)
            filtered_data["objects"] = filtered_objs

        return filtered_data

    def _extract_available_objects(self, world_model_data: dict) -> str:
        """从语义地图世界模型中提取可用对象列表（只使用观察到的对象）"""
        if world_model_data is None:
            return "（无世界模型数据）"

        # 检查是否是字典类型
        if not isinstance(world_model_data, dict):
            return "（无世界模型数据）"

        # 优先使用语义地图节点
        nodes = world_model_data.get("nodes", [])
        available_objects: List[str] = []
        if nodes:
            for node in nodes:
                node_id = node.get("id", "")
                node_label = node.get("label", "Unknown")
                if node_label == "Floor":
                    continue
                attributes = node.get("attributes", {})
                position = attributes.get("position_3d")
                if position:
                    pos_str = f"({position.get('x', 0):.2f}, {position.get('y', 0):.2f}, {position.get('z', 0):.2f})"
                else:
                    bbox = attributes.get("bounding_box_3d", {})
                    center = bbox.get("center", {"x": 0.0, "y": 0.0, "z": 0.0})
                    pos_str = f"({center.get('x', 0):.2f}, {center.get('y', 0):.2f}, {center.get('z', 0):.2f})"
                available_objects.append(f"- {node_id} [{node_label}] at {pos_str}")
        else:
            # 兼容实时结构化格式 { "objects": [ {"名称":{id,type}, "位置":{x,y,z}} ] }
            objs = world_model_data.get("objects", [])
            if objs:
                for o in objs:
                    name = o.get("名称") if isinstance(o, dict) else None
                    oid = name.get("id") if isinstance(name, dict) else (o.get("id") or o.get("objectId"))
                    typ = name.get("type") if isinstance(name, dict) else (o.get("objectType") or "Object")
                    posd = o.get("位置") if isinstance(o, dict) else {}
                    x = float(posd.get("x", 0.0)) if isinstance(posd, dict) else 0.0
                    y = float(posd.get("y", 0.0)) if isinstance(posd, dict) else 0.0
                    z = float(posd.get("z", 0.0)) if isinstance(posd, dict) else 0.0
                    if oid:
                        available_objects.append(f"- {oid} [{typ}] at ({x:.2f}, {y:.2f}, {z:.2f})")
            else:
                return "（场景中无对象）"

        if not available_objects:
            return "（场景中无可用对象）"

        result = "**当前场景中存在的对象（仅使用这些ID）：**\n" + "\n".join(available_objects)
        return result

# =========================================================
# 主流程
# =========================================================
class ThreeLLMSystemV2:
    """三个LLM协作系统 V2（含就绪度，弱约束问答循环 + 跨房间加载）"""

    def __init__(self, api_key: str, world_model_path: str,
                 extra_room_paths: Optional[List[str]] = None,
                 readiness_threshold: float = 0.75,
                 max_rounds_per_task: int = 5):
        self.api_key = api_key
        self.llma = LLMA(api_key)
        self.llmb = LLMB(api_key)
        self.llmc = LLMC(api_key)
        self.expander = TaskExpander(api_key)

        # 加载主房间
        self.llma.load_world_model(world_model_path)
        # 将“已知对象ID”注入A，用于后续严格基于已探索对象进行描述/回答
        try:
            self.llma.set_actual_scene_objects(self._available_object_ids())
        except Exception:
            pass

        # 可选：加载额外房间（如不需要请勿提供，以免超出已探索事实）
        if extra_room_paths:
            for p in extra_room_paths:
                self.llma.add_extra_room(p)

        self.readiness_threshold = readiness_threshold
        self.max_rounds_per_task = max_rounds_per_task
        print("✅ 三个LLM协作系统 V2 初始化完成")

    def _available_object_ids(self) -> List[str]:
        ids: List[str] = []
        wm = self.llma.world_model or {}
        if isinstance(wm, dict):
            if wm.get("nodes"):
                for n in wm.get("nodes", []):
                    nid = n.get("id")
                    if nid and n.get("label") != "Floor":
                        ids.append(nid)
            if wm.get("objects"):
                for o in wm.get("objects", []):
                    name = o.get("名称") if isinstance(o, dict) else None
                    oid = (name.get("id") if isinstance(name, dict) else (o.get("id") or o.get("objectId")))
                    if oid:
                        ids.append(oid)

        return ids
    def _available_objects_text(self) -> str:
        wm = self.llma.world_model or {}
        if not isinstance(wm, dict):
            return "（场景中无对象）"
        lines: List[str] = []
        # 1) 语义图格式
        if wm.get("nodes"):
            for n in wm.get("nodes", []):
                label = n.get("label", "Unknown")
                if label == "Floor":
                    continue
                nid = n.get("id", "")
                attrs = n.get("attributes", {})
                pos = attrs.get("position_3d") or {}
                pos_str = f"({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})" if pos else "(?, ?, ?)"
                lines.append(f"- {nid} [{label}] at {pos_str}")
        # 2) 实时结构化格式
        if not lines and wm.get("objects"):
            for o in wm.get("objects", []):
                name = o.get("名称") if isinstance(o, dict) else None
                oid = name.get("id") if isinstance(name, dict) else (o.get("id") or o.get("objectId"))
                typ = name.get("type") if isinstance(name, dict) else (o.get("objectType") or "Object")
                posd = o.get("位置") if isinstance(o, dict) else {}
                x = float(posd.get("x", 0.0)) if isinstance(posd, dict) else 0.0
                y = float(posd.get("y", 0.0)) if isinstance(posd, dict) else 0.0
                z = float(posd.get("z", 0.0)) if isinstance(posd, dict) else 0.0
                if oid:
                    lines.append(f"- {oid} [{typ}] at ({x:.2f}, {y:.2f}, {z:.2f})")
        return "\n".join(lines) if lines else "（场景中无对象）"


    @staticmethod
    def _split_tasks(tasks_text: str, max_tasks: int = 100) -> List[str]:
        """
        将LLM输出解析为任务行列表，鲁棒支持：
        - Markdown 代码块（```json ... ```）
        - 纯 JSON（数组/对象）
        - 普通换行/句号/分号分隔的自然语言
        始终返回字符串列表。
        """
        import json, re
        text = (tasks_text or "").strip()
        if not text:
            return []

        # 1) 去除 Markdown 代码块围栏
        fenced = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", text)
        candidate = fenced.group(1).strip() if fenced else text

        # 2) 若内容看起来像 JSON，先尝试解析
        json_candidate = candidate.strip()
        if json_candidate.startswith("[") or json_candidate.startswith("{"):
            try:
                data = json.loads(json_candidate)
                lines: List[str] = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            lines.append(item.strip())
                        elif isinstance(item, dict):
                            if item.get("issue_description"):
                                lines.append(str(item.get("issue_description")).strip())
                            elif item.get("implied_action"):
                                ia = item.get("implied_action")
                                src = item.get("primary_object_id") or item.get("object_id") or "?"
                                dst = item.get("target_receptacle_id") or item.get("receptacleObjectId") or "?"
                                lines.append(f"{ia} objectId={src} -> {dst}")
                            else:
                                lines.append(json.dumps(item, ensure_ascii=False))
                elif isinstance(data, dict):
                    # 处理包含 "初步诊断" 或 "tasks" 的特殊格式
                    if "初步诊断" in data and isinstance(data["初步诊断"], list):
                        for item in data["初步诊断"]:
                            if isinstance(item, dict):
                                if item.get("issue_description"):
                                    lines.append(str(item.get("issue_description")).strip())
                                elif item.get("implied_action"):
                                    ia = item.get("implied_action")
                                    src = item.get("primary_object_id") or item.get("object_id") or "?"
                                    dst = item.get("target_receptacle_id") or item.get("receptacleObjectId") or "?"
                                    lines.append(f"{ia} objectId={src} -> {dst}")
                            elif isinstance(item, str):
                                lines.append(item.strip())
                    elif "tasks" in data and isinstance(data["tasks"], list):
                        for t in data["tasks"]:
                            if isinstance(t, str):
                                lines.append(t.strip())
                            elif isinstance(t, dict):
                                if t.get("issue_description"):
                                    lines.append(str(t.get("issue_description")).strip())
                                else:
                                    lines.append(json.dumps(t, ensure_ascii=False))
                    else:
                        for k, v in data.items():
                            if isinstance(v, (str, int, float)):
                                lines.append(f"{k}: {v}")
                lines = [l for l in (l.strip() for l in lines) if l]
                seen, res = set(), []
                for l in lines:
                    if l not in seen:
                        seen.add(l)
                        res.append(l)
                    if len(res) >= max_tasks:
                        break
                if res:
                    return res
            except Exception as e:
                # JSON 解析失败，继续走普通文本分割
                pass

        # 3) 普通文本切分 - 只当不是JSON时才用
        # 关键修复：检查是否是格式化JSON失败的情况
        # 如果文本包含很多 { } [ ] : 等JSON字符，说明可能是JSON分行了，不应该按行分割
        json_char_count = text.count('{') + text.count('}') + text.count('[') + text.count(']') + text.count(':')
        lines = candidate.splitlines()
        
        # 如果太多JSON字符且行数过多，说明是JSON被分行了，需要特殊处理
        if json_char_count > 5 and len(lines) > 5:
            # 尝试对每一行进行JSON修复和合并
            fixed_text = ThreeLLMSystemV2._reconstruct_json_from_lines(candidate)
            if fixed_text != candidate:
                # 重新尝试JSON解析
                try:
                    data = json.loads(fixed_text)
                    result_lines: List[str] = []
                    if isinstance(data, dict) and "初步诊断" in data:
                        for item in data["初步诊断"]:
                            if isinstance(item, dict) and item.get("issue_description"):
                                result_lines.append(str(item.get("issue_description")).strip())
                    if result_lines:
                        return result_lines[:max_tasks]
                except:
                    pass
        
        # 标准文本分割
        lines = [l.strip("- •\t ").strip() for l in candidate.splitlines() if l.strip()]
        if len(lines) <= 1:
            lines = re.split(r"[；;。\n]+", candidate)
            lines = [l.strip() for l in lines if l.strip()]
        # 去重并截断
        seen, res = set(), []
        for l in lines:
            if l not in seen:
                seen.add(l)
                res.append(l)
            if len(res) >= max_tasks:
                break
        return res

    @staticmethod
    def _reconstruct_json_from_lines(text: str) -> str:
        """
        尝试从格式化的JSON中恢复完整的JSON字符串
        """
        import json
        lines = text.splitlines()
        
        # 找到开始和结束的括号
        start_idx = -1
        end_idx = -1
        for i, line in enumerate(lines):
            if '{' in line or '[' in line:
                if start_idx == -1:
                    start_idx = i
            if '}' in line or ']' in line:
                end_idx = i
        
        if start_idx >= 0 and end_idx >= start_idx:
            # 合并从start到end的所有行
            merged = "".join(lines[start_idx:end_idx+1])
            return merged
        
        return text


    def _positions_map(self) -> Dict[str, Tuple[float, float]]:
        wm = self.llma.world_model or {}
        pos_map: Dict[str, Tuple[float, float]] = {}
        try:
            for n in wm.get("nodes", []) if isinstance(wm, dict) else []:
                nid = n.get("id", "")
                attrs = n.get("attributes", {}) or {}
                p = attrs.get("position_3d")
                if not p:
                    bb = attrs.get("bounding_box_3d", {}) or {}
                    p = bb.get("center") if isinstance(bb, dict) else None
                if nid and p:
                    x = float(p.get("x", 0.0))
                    z = float(p.get("z", 0.0))
                    pos_map[nid] = (x, z)
        except Exception:
            pass
        return pos_map

    def _agent_pos_approx(self) -> Tuple[float, float]:
        # 若世界模型中含有 Agent 节点，可取其位置；否则返回 (0,0)
        wm = self.llma.world_model or {}
        try:
            for n in wm.get("nodes", []) if isinstance(wm, dict) else []:
                if (n.get("label") or "").lower() == "agent":
                    attrs = n.get("attributes", {}) or {}
                    p = attrs.get("position_3d")
                    if p:
                        return float(p.get("x", 0.0)), float(p.get("z", 0.0))
        except Exception:
            pass
        return 0.0, 0.0

    @staticmethod
    def _primary_object_id_from_task(line: str) -> Optional[str]:
        # 优先匹配 objectId=XXX
        try:
            m = re.search(r"objectId\s*=\s*([A-Za-z0-9_\|.+\-]+)", line)
            if m:
                return m.group(1)
            # 其次匹配标准ID模式: Type|+x|+y|+z
            m2 = re.search(r"[A-Za-z]+\|[+\-]?\d+\.\d+\|[+\-]?\d+\.\d+\|[+\-]?\d+\.\d+", line)
            if m2:
                return m2.group(0)
        except Exception:
            pass
        return None

    @staticmethod
    def _urgency_score(line: str) -> float:
        s = (line or "").lower()
        critical_kw = ["toggleobjectoff", "turnoff", "closeobject", "faucet", "stove", "microwave", "oven", "toaster", "heater", "burner", "leak", "fire"]
        cleaning_kw = ["cleanobject", "dirty", "wash", "wipe"]
        if any(k in s for k in critical_kw):
            return 1.0
        if any(k in s for k in cleaning_kw):
            return 0.7
        if "putobject" in s and any(k in s for k in ["cabinet", "drawer", "shelf", "countertop", "table"]):
            return 0.4
        return 0.2

    def _rerank_with_distance(self, optimized_sequence: str) -> str:
        tasks = self._split_tasks(optimized_sequence, max_tasks=50)
        if not tasks:
            return optimized_sequence
        pos_map = self._positions_map()
        ax, az = self._agent_pos_approx()

        enriched = []
        for idx, t in enumerate(tasks):
            oid = self._primary_object_id_from_task(t)
            px, pz = pos_map.get(oid, (None, None)) if oid else (None, None)
            if px is None:
                dist = 9999.0
            else:
                dx, dz = float(px) - ax, float(pz) - az
                dist = math.hypot(dx, dz)
            urg = self._urgency_score(t)
            enriched.append({"idx": idx, "text": t, "urg": urg, "dist": dist})

        # 规则：若任务处于LLMC排序前2且紧急度>=0.9，则保持其原序靠前（不因路程改变）
        locked = [e for e in enriched if e["idx"] < 2 and e["urg"] >= 0.9]
        locked_idxs = {e["idx"] for e in locked}
        rest = [e for e in enriched if e["idx"] not in locked_idxs]

        # 将其余任务按原 rank 分桶（每3个为一组），在桶内按距离升序；并以原索引作为稳定次序
        def bucket(i: int) -> int:
            return i // 3
        buckets: Dict[int, List[dict]] = {}
        for e in rest:
            b = bucket(e["idx"])
            buckets.setdefault(b, []).append(e)
        sorted_rest: List[dict] = []
        for b in sorted(buckets.keys()):
            arr = buckets[b]
            arr.sort(key=lambda x: (x["dist"], x["idx"]))
            sorted_rest.extend(arr)

        final_list: List[str] = [e["text"] for e in sorted(locked, key=lambda x: x["idx"]) + sorted_rest]
        return "\n".join(final_list)

    def _qa_loop_for_task(self, task: str) -> Tuple[str, Dict[str, Any], str]:
        """
        针对单个任务执行一轮“成组、针对性的问答”：
        - LLMB 一次性提出 3~6 个“对执行有直接作用”的问题（基于已知对象ID）
        - LLMA 基于“已观察数据”逐一回答
        - LLMB 抽取就绪度
        返回：conversation_history, readiness_dict, last_llma_answer
        """
        # 重置/保留状态（当前流程不依赖分层开关）
        self.llmb.inquiry_state = {
            "current_layer": 1,
            "categories_identified": False,
            "objects_identified": False,
            "solutions_identified": False,
            "main_category": "",
            "identified_objects": [],
            "available_containers": []
        }

        conversation_history = ""
        last_answer = ""
        readiness = {
            "object_id_known": False,
            "tool_or_consumable_known": False,
            "location_or_access_known": False,
            "key_state_known": False,
            "readiness_score": 0.0,
        }

        # —— 成组提问 ——
        available_text = self._available_objects_text()
        question_batch = self.llmb.ask_batch_questions(task, available_text, conversation_history)
        print(f"B (Q-batch):\n{question_batch}")

        answer = self.llma.answer_question(question_batch)
        print(f"A: {answer}")
        conversation_history += f"B: {question_batch}\nA: {answer}\n"
        last_answer = answer

        # —— 就绪度抽取 ——
        readiness = self.llmb.extract_readiness(task, conversation_history)
        score = readiness.get("readiness_score", 0.0)
        print(
            f"📈 就绪度: {score:.2f}  |  细项: "
            f"obj={readiness.get('object_id_known')}, "
            f"tool={readiness.get('tool_or_consumable_known')}, "
            f"loc={readiness.get('location_or_access_known')}, "
            f"state={readiness.get('key_state_known')}"
        )
        return conversation_history, readiness, last_answer

    def execute_planning(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # 步骤1: LLMA 描述场景
        print("\n📊 步骤1: 场景描述")
        scene_description = self.llma.describe_scene()
        results["scene_description"] = scene_description
        print(f"A (scene): {scene_description}")

        # 步骤2: 由A（LLMA）提取【高层候选任务】（不是原子动作）
        print("\n📋 步骤2: A 提取高层候选任务")
        tasks_text = self.llma.propose_macro_tasks()
        results["llma_macro_tasks_raw"] = tasks_text
        task_list = self._split_tasks(tasks_text, max_tasks=100)
        results["candidate_macro_tasks"] = task_list
        for i, t in enumerate(task_list, 1):
            print(f"  {i}. {t}")

        # 步骤3: B 批量分析并提问（按任务分组，一次性提出所有问题）
        print("\n💬 步骤3: B 批量分析并提问（逐任务分组）")
        macro_tasks_text_joined = "\n".join(task_list)
        allowed_ids_text = self._available_objects_text()
        questions_text = self.llmb.ask_questions_for_all(macro_tasks_text_joined, allowed_ids_text)
        results["llmb_questions_batch"] = questions_text
        print(f"B (questions): {questions_text}")

        # 步骤3.1: A 批量逐条解答
        print("\n✍️ 步骤3.1: A 批量回答")
        answers_text = self.llma.answer_questions_batch(questions_text)
        results["llma_answers_batch"] = answers_text
        print(f"A (answers): {answers_text}")

        # 步骤4: B 汇总为最小化可执行任务清单（仅原子动作）
        print("\n📝 步骤4: B 汇总为可执行任务清单")
        final_task_list = self.llmb.synthesize_from_answers(
            macro_tasks_text_joined, questions_text, answers_text, allowed_ids_text
        )
        results["final_task_list_from_llmb"] = final_task_list
        print(final_task_list)

        # 供排序器参考的对话历史
        merged_history = (
            "## MacroTasks\n" + macro_tasks_text_joined +
            "\n\n## Questions\n" + questions_text +
            "\n\n## Answers\n" + answers_text
        )

        # 步骤5: LLMC 优化任务序列（自由排序，不限定输出样式）
        print("\n🔧 步骤5: 任务序列优化（LLMC）")
        optimized_sequence = self.llmc.optimize_sequence(final_task_list, merged_history)
        results["optimized_sequence_llmc"] = optimized_sequence
        print(optimized_sequence)

        # 步骤5.1: 结合路径距离进行二次重排（保持高优先+高紧急任务不被距离改动）
        print("\n🚦 步骤5.1: 结合路程远近进行二次重排（优先级/紧急度优先）")
        reranked_sequence = self._rerank_with_distance(optimized_sequence)
        results["optimized_sequence_reranked"] = reranked_sequence
        print(reranked_sequence)

        # 步骤6: TaskExpander 进一步展开（基于二次重排后的序列）
        print("\n🧩 步骤6: 详细步骤扩展")
        if not (reranked_sequence or "").strip():
            results["detailed_steps"] = "（无可展开的任务序列）"
        else:
            # 传递世界模型数据和实际对象信息给TaskExpander
            world_model_data = self.llma.world_model if self.llma.world_model else None
            actual_objects = getattr(self.llma, 'actual_scene_objects', None)
            actual_objects_list = list(actual_objects) if actual_objects else None
            results["detailed_steps"] = self.expander.expand(
                reranked_sequence,
                world_model_data,
                actual_objects_list
            )
        print(results["detailed_steps"])

        return results

    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None):
        if not output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"three_llm_v2_results_{timestamp}.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

# =========================================================
# 单一LLM对照组：统一代理一次性完成全部步骤
# =========================================================
# =========================================================
# 入口：仅保留原始三个LLM协作版本
# =========================================================
def run_multi() -> Dict[str, Any]:
    """原始三个LLM协作版本（保持行为不变，返回结果字典）"""
    print("🎯 三个LLM协作的具身任务规划系统 V2（含就绪度，弱约束 + 跨房间）")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    world_model_path = os.path.join(base_dir, "maps", "rooms", "living_room_simple.json")
    # 如需跨房间，可在此配置 extra_room_paths；当前版本仅使用单一主房间
    extra_room_paths = None

    system = ThreeLLMSystemV2(
        GOOGLE_API_KEY,
        world_model_path=world_model_path,
        extra_room_paths=extra_room_paths,   # 当前为 None，仅使用主房间
        readiness_threshold=0.75,
        max_rounds_per_task=5,
    )
    results = system.execute_planning()
    # 仍然各自保存一份原始结果，方便单独分析
    system.save_results(results)
    return results


def main():
    """命令行入口：只运行三个LLM协作版本"""
    run_multi()


if __name__ == "__main__":
    main()
