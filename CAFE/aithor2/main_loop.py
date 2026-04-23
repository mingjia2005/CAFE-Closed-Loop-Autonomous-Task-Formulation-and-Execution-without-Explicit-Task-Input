
"""
主循环控制模块：从 main_with_depth.py 抽离
- 封装主事件循环逻辑
- 协调各个模块的交互
- 保持代码结构清晰
"""
from __future__ import annotations
import threading
import os
import time
from pynput import keyboard
from typing import Any, Dict, Optional, Callable

import input_handler as IH
import display_manager as DM
import pointcloud_utils as PCU
import exploration_io as EIO


def run_main_loop(controller, semantic_map: Dict[str, Any], 
                 detection_mode: str = 'gt',
                 save_captures: bool = False,
                 **kwargs):
    """
    运行主事件循环
    
    Args:
        controller: AI2-THOR Controller
        semantic_map: 语义地图数据
        detection_mode: 检测模式 ('gt' 或 'yolo')
        save_captures: 是否启用保存功能
        **kwargs: 其他需要的函数和变量
    """
    # 从kwargs中提取需要的函数和变量
    get_current_key = kwargs.get('get_current_key')
    clear_current_key = kwargs.get('clear_current_key')
    user_command_mode = kwargs.get('user_command_mode', False)
    executing_plan = kwargs.get('executing_plan', False)
    planned_actions = kwargs.get('planned_actions', [])
    _plan_lock = kwargs.get('_plan_lock')
    
    # 提取需要的函数
    start_llm_scene_understanding = kwargs.get('start_llm_scene_understanding')
    _start_user_command_window_async = kwargs.get('_start_user_command_window_async')
    execute_next_planned_action = kwargs.get('execute_next_planned_action')
    _sleep_if_slow = kwargs.get('_sleep_if_slow')
    update_semantic_map = kwargs.get('update_semantic_map')
    chaos_drop_apple_on_floor = kwargs.get('chaos_drop_apple_on_floor')
    chaos_tip_chair = kwargs.get('chaos_tip_chair')
    detect_objects = kwargs.get('detect_objects')
    detect_objects_from_segmentation = kwargs.get('detect_objects_from_segmentation')
    
    # 其他需要的变量
    REALTIME_EXPLORATION_JSON = kwargs.get('REALTIME_EXPLORATION_JSON')
    THIRD_PERSON_ENABLED = kwargs.get('THIRD_PERSON_ENABLED', False)
    THIRD_PERSON_CAMERA_ID = kwargs.get('THIRD_PERSON_CAMERA_ID', 0)
    SLOW_EXECUTION = kwargs.get('SLOW_EXECUTION', False)
    _nav_state = kwargs.get('_nav_state', {})
    _execute_navmesh_navigation = kwargs.get('_execute_navmesh_navigation')
    _start_navmesh_navigation = kwargs.get('_start_navmesh_navigation')
    detect_and_enqueue_tidy_tasks = kwargs.get('detect_and_enqueue_tidy_tasks')
    detect_scene_tasks = kwargs.get('detect_scene_tasks')
    print_task_overview = kwargs.get('print_task_overview')
    enqueue_task_plan = kwargs.get('enqueue_task_plan')
    _save_preference_learning = kwargs.get('_save_preference_learning')
    
    frame_idx = 0
    image_counter = 0
    
    try:
        while True:
            frame_idx += 1
            
            # 保持场景更新
            event = controller.step(action="Pass")
            # 记录最新事件供命令LLM解析使用
            globals()['latest_event'] = event
            
            # 快照并清空按键，避免线程间竞争导致丢键
            current_key = get_current_key() if get_current_key else None
            if clear_current_key:
                clear_current_key()
            
            # 处理特殊按键（M键、计划执行等）
            should_break = _handle_special_keys(
                current_key, event, controller, kwargs
            )
            if should_break:
                break
            
            # 处理常规按键输入
            if not kwargs.get('executing_plan', False):
                event, should_break = _handle_regular_keys(
                    current_key, event, controller, kwargs
                )
                if should_break:
                    break
            
            # 更新语义地图
            frame_count = kwargs.get('frame_count', 0) + 1
            kwargs['frame_count'] = frame_count
            try:
                if update_semantic_map:
                    update_semantic_map(event, frame_count)
            except Exception as e:
                print(f"⚠ 语义地图更新失败: {e}")
            
            # 实时更新点云可视化
            if event.depth_frame is not None and kwargs.get('vis_running', False):
                try:
                    pcd = PCU.generate_point_cloud(event.frame, event.depth_frame)
                    PCU.update_point_cloud_visualizer(pcd)
                except Exception:
                    pass
            
            # 更新第三视角相机
            if THIRD_PERSON_ENABLED:
                _update_third_person_camera(event, controller, THIRD_PERSON_CAMERA_ID)
            
            # 处理导航状态
            if _nav_state and _nav_state.get('active', False) and _execute_navmesh_navigation:
                event, reached = _execute_navmesh_navigation(controller, event)
                if reached:
                    _nav_state['active'] = False
            
            # 慢速执行模式
            if SLOW_EXECUTION and (current_key is not None or kwargs.get('executing_plan', False)):
                if _sleep_if_slow:
                    _sleep_if_slow()
            
            # 实时显示图像
            save_image = (current_key == 'p')
            image_counter = DM.display_and_save_images(
                event, save_image, detection_mode, save_captures, image_counter,
                detect_objects, detect_objects_from_segmentation
            )
            
            # 检查窗口是否被关闭
            if DM.check_window_closed():
                break
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理资源
        print("正在清理资源...")
        
        # 保存学习数据
        try:
            if _save_preference_learning:
                _save_preference_learning()
                print("💾 偏好学习数据已保存")
        except Exception as e:
            print(f"⚠ 保存偏好学习数据失败: {e}")
        
        DM.cleanup_display()
        PCU.close_point_cloud_visualizer()
        controller.stop()
        if 'listener' in kwargs:
            kwargs['listener'].stop()
        print("程序已退出")


def _handle_special_keys(current_key, event, controller, kwargs) -> bool:
    """处理特殊按键（M键、计划执行等）"""
    executing_plan = kwargs.get('executing_plan', False)
    
    # 处理 m/M：触发场景理解
    if current_key in ('m', 'M'):
        try:
            if not kwargs.get('user_command_mode', False):
                _start_user_command_window_async = kwargs.get('_start_user_command_window_async')
                if _start_user_command_window_async:
                    _start_user_command_window_async()
            print('🤖 启动LLM场景理解...')
            try:
                start_llm_scene_understanding = kwargs.get('start_llm_scene_understanding')
                if start_llm_scene_understanding:
                    start_llm_scene_understanding(event=event)
            except Exception as _e:
                print(f"⚠ 场景理解启动失败: {_e}")
        except Exception as _e:
            print(f"⚠ 处理 m 键失败: {_e}")
    
    # 自动执行计划优先；执行期间仅支持按 x 取消，其他按键忽略
    if executing_plan:
        if current_key == 'x':
            _plan_lock = kwargs.get('_plan_lock')
            planned_actions = kwargs.get('planned_actions', [])
            if _plan_lock:
                with _plan_lock:
                    planned_actions.clear()
                    kwargs['executing_plan'] = False
            print('⏹ 已取消当前计划执行')
        else:
            execute_next_planned_action = kwargs.get('execute_next_planned_action')
            _sleep_if_slow = kwargs.get('_sleep_if_slow')
            if execute_next_planned_action:
                event = execute_next_planned_action(controller, event)
            if _sleep_if_slow:
                _sleep_if_slow()
            return False  # continue主循环
    
    return False


def _handle_regular_keys(current_key, event, controller, kwargs):
    """处理常规按键输入"""
    # 使用input_handler处理基础按键
    event, should_break = IH.process_keyboard_input(
        controller, event, current_key,
        chaos_drop_apple_on_floor=kwargs.get('chaos_drop_apple_on_floor'),
        chaos_tip_chair=kwargs.get('chaos_tip_chair')
    )
    
    if should_break:
        return event, True
    
    # 处理其他复杂按键（这些需要访问更多全局状态）
    if current_key == 'n':  # 测试导航
        _handle_navigation_test(event, controller, kwargs)
    elif current_key == 'z':  # 整理任务
        _handle_tidy_tasks(event, kwargs)
    elif current_key == 'T':  # 任务清单
        _handle_task_overview(event, kwargs)
    elif current_key == 'H':  # 安全任务
        _handle_safety_tasks(event, kwargs)
    elif current_key in ('L',):  # 重载配置
        _handle_config_reload(event, kwargs)
    
    return event, False


def _handle_navigation_test(event, controller, kwargs):
    """处理导航测试"""
    try:
        all_objects = event.metadata.get("objects", [])
        visible_objects = [obj for obj in all_objects if obj.get("visible")]
        pickupable_objects = [obj for obj in visible_objects if obj.get("pickupable")]
        
        print(f"📊 场景统计: 总物体{len(all_objects)}个, 可见{len(visible_objects)}个, 可拾取{len(pickupable_objects)}个")
        
        if visible_objects:
            print("👁️ 可见物体:")
            for i, obj in enumerate(visible_objects[:5]):
                print(f"  {i+1}. {obj['objectId']} ({obj['objectType']}) - 可拾取: {obj.get('pickupable', False)}")
        
        target_objects = pickupable_objects if pickupable_objects else visible_objects
        
        if target_objects:
            target_obj = target_objects[0]
            print(f"🧭 测试导航到: {target_obj['objectId']} ({target_obj['objectType']})")
            _start_navmesh_navigation = kwargs.get('_start_navmesh_navigation')
            if _start_navmesh_navigation:
                success = _start_navmesh_navigation(controller, event, target_obj['objectId'])
                print("✅ 导航启动成功" if success else "❌ 导航启动失败")
        else:
            print("❌ 没有找到可导航的物体")
    except Exception as e:
        print(f"⚠ 测试导航失败: {e}")


def _handle_tidy_tasks(event, kwargs):
    """处理整理任务"""
    detect_and_enqueue_tidy_tasks = kwargs.get('detect_and_enqueue_tidy_tasks')
    if detect_and_enqueue_tidy_tasks:
        ok = detect_and_enqueue_tidy_tasks(event)
        if not ok:
            print('ℹ️ 暂无需要整理的轻度混乱')


def _handle_task_overview(event, kwargs):
    """处理任务清单"""
    detect_scene_tasks = kwargs.get('detect_scene_tasks')
    print_task_overview = kwargs.get('print_task_overview')
    if detect_scene_tasks and print_task_overview:
        tasks = detect_scene_tasks(event)
        print_task_overview(tasks, limit=12)


def _handle_safety_tasks(event, kwargs):
    """处理安全任务"""
    detect_scene_tasks = kwargs.get('detect_scene_tasks')
    enqueue_task_plan = kwargs.get('enqueue_task_plan')
    if detect_scene_tasks and enqueue_task_plan:
        tasks = detect_scene_tasks(event)
        top_safety = next((t for t in tasks if t.get('category') == 'Safety'), None)
        if top_safety:
            ok = enqueue_task_plan(event, top_safety)
            if not ok:
                print('⚠ 入列安全任务失败')
        else:
            print('ℹ️ 未发现可自动修复的安全任务')


def _handle_config_reload(event, kwargs):
    """处理配置重载"""
    try:
        semantic_map = kwargs.get('semantic_map')
        if semantic_map:
            EIO.load_container_labels(semantic_map)
            EIO.load_replan_triggers(semantic_map)
            print('🔄 已重载: container_labels.json 与 replan_triggers.json')
            _maybe_trigger_replan = kwargs.get('_maybe_trigger_replan')
            if _maybe_trigger_replan:
                _maybe_trigger_replan(event, reason='配置重载')
    except Exception as e:
        print(f"⚠ 配置重载失败: {e}")


def _update_third_person_camera(event, controller, camera_id):
    """更新第三视角相机"""
    try:
        apos = event.metadata.get('agent', {}).get('position', {})
        arot = event.metadata.get('agent', {}).get('rotation', {})
        yaw = float(arot.get('y', 0.0))
        import math
        dist, height = 1.6, 1.7
        rad = math.radians(yaw)
        cx = apos.get('x', 0.0) - math.sin(rad) * dist
        cy = apos.get('y', 0.0) + height
        cz = apos.get('z', 0.0) - math.cos(rad) * dist
        controller.step(action="UpdateThirdPartyCamera",
                       thirdPartyCameraId=camera_id,
                       position={"x": cx, "y": cy, "z": cz},
                       rotation={"x": 35.0, "y": yaw, "z": 0.0})
    except Exception:
        pass
