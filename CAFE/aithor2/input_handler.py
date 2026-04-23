"""
键盘输入处理模块：从 main_with_depth.py 抽离
- 处理所有键盘输入逻辑
- 执行对应的AI2-THOR动作
- 返回更新后的事件
"""
from __future__ import annotations
import threading
import os
from pynput import keyboard
from typing import Any, Dict, Optional


def handle_movement_keys(controller, event, key: str):
    """处理基础移动按键 (WASD, QE)"""
    if key == 'w':  # 前进
        return controller.step(action="MoveAhead")
    elif key == 's':  # 后退
        return controller.step(action="MoveBack")
    elif key == 'a':  # 左移
        return controller.step(action="MoveLeft")
    elif key == 'd':  # 右移
        return controller.step(action="MoveRight")
    elif key == 'q':  # 左转
        return controller.step(action="RotateLeft")
    elif key == 'e':  # 右转
        return controller.step(action="RotateRight")
    return event


def handle_object_interaction_keys(controller, event, key: str):
    """处理物体交互按键 (F, G, K, L, T, Y等)"""
    def _agent_pos(event):
        """获取Agent位置"""
        pos = event.metadata.get('agent', {}).get('position', {})
        return pos.get('x', 0.0), pos.get('z', 0.0)
    
    def _find_closest_object(event, filter_func):
        """找到最近的满足条件的物体"""
        objs = [o for o in event.metadata.get('objects', []) if filter_func(o)]
        if not objs:
            return None
        ax, az = _agent_pos(event)
        return min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)

    if key == 'f':  # 抓取最近的可见且可拾取物体
        all_objects = event.metadata.get('objects', [])
        visible = [o for o in all_objects if o.get('visible')]
        pickupable = [o for o in visible if o.get('pickupable')]
        target = None
        if pickupable:
            ax, az = _agent_pos(event)
            def _dist2(o):
                p = o.get('position', {})
                return (p.get('x', 0) - ax) ** 2 + (p.get('z', 0) - az) ** 2
            target = min(pickupable, key=_dist2)
        elif visible:
            target = visible[0]
        if target:
            pick_event = controller.step(action='PickupObject', objectId=target['objectId'], forceAction=True)
            if pick_event.metadata.get('lastActionSuccess', False):
                print(f"✓ 已抓取: {target['objectId']}")
                return pick_event
            else:
                print(f"⚠ 抓取失败: {pick_event.metadata.get('errorMessage', 'Unknown error')}")
        else:
            print('❌ 没有可抓取的物体')
            
    elif key == 'g':  # 放下手中物体
        inv = event.metadata.get('inventoryObjects', [])
        if inv:
            drop_event = controller.step(action='DropHandObject', forceAction=True)
            if drop_event.metadata.get('lastActionSuccess', False):
                print('✓ 已放下手中物体')
                return drop_event
            else:
                print(f"⚠ 放下失败: {drop_event.metadata.get('errorMessage', 'Unknown error')}")
        else:
            print('❌ 手中没有物体可放下')
            
    elif key == 'k':  # 打开最近的可开对象
        target = _find_closest_object(event, lambda o: o.get('visible') and o.get('openable'))
        if target:
            ev2 = controller.step(action='OpenObject', objectId=target.get('objectId'), forceAction=True)
            print('✅ 打开' if ev2.metadata.get('lastActionSuccess') else f"⚠ 打开失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('❌ 附近没有可打开的对象')
            
    elif key == 'l':  # 关闭最近的可开对象
        target = _find_closest_object(event, lambda o: o.get('visible') and o.get('openable'))
        if target:
            ev2 = controller.step(action='CloseObject', objectId=target.get('objectId'), forceAction=True)
            print('✅ 关闭' if ev2.metadata.get('lastActionSuccess') else f"⚠ 关闭失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('❌ 附近没有可关闭的对象')
            
    elif key == 't':  # 打开最近的可切换对象（如水龙头、灯）
        target = _find_closest_object(event, lambda o: o.get('visible') and (o.get('toggleable') or o.get('canToggle')))
        if target:
            ev2 = controller.step(action='ToggleObjectOn', objectId=target.get('objectId'), forceAction=True)
            print('✅ Toggle On' if ev2.metadata.get('lastActionSuccess') else f"⚠ 操作失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('❌ 附近没有可切换的对象')
            
    elif key == 'y':  # 关闭最近的可切换对象
        target = _find_closest_object(event, lambda o: o.get('visible') and (o.get('toggleable') or o.get('canToggle')))
        if target:
            ev2 = controller.step(action='ToggleObjectOff', objectId=target.get('objectId'), forceAction=True)
            print('✅ Toggle Off' if ev2.metadata.get('lastActionSuccess') else f"⚠ 操作失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('❌ 附近没有可切换的对象')
            
    return event


def handle_liquid_and_cleaning_keys(controller, event, key: str):
    """处理液体和清洁相关按键 (J, U, R, B, I)"""
    def _agent_pos(event):
        pos = event.metadata.get('agent', {}).get('position', {})
        return pos.get('x', 0.0), pos.get('z', 0.0)
    
    def _find_closest_object(event, filter_func):
        objs = [o for o in event.metadata.get('objects', []) if filter_func(o)]
        if not objs:
            return None
        ax, az = _agent_pos(event)
        return min(objs, key=lambda o: (o.get('position',{}).get('x',0)-ax)**2 + (o.get('position',{}).get('z',0)-az)**2)

    if key == 'j':  # 为手中容器注水
        inv = event.metadata.get('inventoryObjects', []) or []
        if inv:
            held = inv[0].get('objectId')
            ev2 = controller.step(action='FillObjectWithLiquid', objectId=held, fillLiquid='water', forceAction=True)
            print('✅ 注水成功' if ev2.metadata.get('lastActionSuccess') else f"⚠ 注水失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('❌ 手中没有容器')
            
    elif key == 'u':  # 倒空手中液体
        inv = event.metadata.get('inventoryObjects', []) or []
        if inv:
            held = inv[0].get('objectId')
            ev2 = controller.step(action='EmptyLiquidFromObject', objectId=held, forceAction=True)
            print('✅ 倒空成功' if ev2.metadata.get('lastActionSuccess') else f"⚠ 倒空失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('❌ 手中没有容器')
            
    elif key == 'r':  # 清洁最近的可清洁对象
        target = _find_closest_object(event, lambda o: o.get('visible') and o.get('dirtyable') and o.get('isDirty'))
        if target:
            ev2 = controller.step(action='CleanObject', objectId=target.get('objectId'), forceAction=True)
            print('✅ 清洁完成' if ev2.metadata.get('lastActionSuccess') else f"⚠ 清洁失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('ℹ️ 附近没有需要清洁的对象')
            
    elif key == 'b':  # 弄脏最近的可清洁对象（演示）
        target = _find_closest_object(event, lambda o: o.get('visible') and o.get('dirtyable') and not o.get('isDirty'))
        if target:
            ev2 = controller.step(action='DirtyObject', objectId=target.get('objectId'), forceAction=True)
            print('✅ 已弄脏' if ev2.metadata.get('lastActionSuccess') else f"⚠ 操作失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('ℹ️ 附近没有可弄脏的对象')
            
    elif key == 'i':  # 切片最近的可切片对象（如 Apple/Bread）
        target = _find_closest_object(event, lambda o: o.get('visible') and (o.get('sliceable') or o.get('canBeSliced')))
        if target:
            ev2 = controller.step(action='SliceObject', objectId=target.get('objectId'), forceAction=True)
            print('✅ 已切片' if ev2.metadata.get('lastActionSuccess') else f"⚠ 切片失败: {ev2.metadata.get('errorMessage','')}")
            return ev2
        else:
            print('ℹ️ 附近没有可切片的对象')
            
    return event


def handle_special_action_keys(controller, event, key: str, chaos_drop_apple_on_floor, chaos_tip_chair):
    """处理特殊动作按键 (C - 制造混乱等)"""
    if key == 'c':  # 制造混乱：苹果落地 + 椅子放倒
        try:
            event = chaos_drop_apple_on_floor(controller, event)
        except Exception as e:
            print(f"⚠ drop apple 失败: {e}")
        try:
            event = chaos_tip_chair(controller, event)
        except Exception as e:
            print(f"⚠ tip chair 失败: {e}")
    return event


def handle_help_key():
    """处理帮助按键 (?)"""
    print('— 按键帮助 —')
    print('  w/s/a/d/q/e: 移动/旋转')
    print('  f: 抓取最近可拾取  g: 放下')
    print('  o: 打开最近的柜子  k: 打开最近可开  l: 关闭最近可开')
    print('  t/y: 打开/关闭 最近的可切换对象（如水龙头、灯）')
    print('  j/u: 为手中容器注水/倒空液体')
    print('  r/b: 清洁/弄脏 最近的可清洁对象')
    print('  i: 切片最近可切片对象')
    print('  z: 检测并入列一次整理任务')
    print('  v: 启用/禁用自主探索  V: 显示探索状态')
    print('  m: 启动LLM场景理解  n: 测试导航  p: 截图  x: 取消当前计划  ESC: 退出')


def handle_autonomous_exploration_keys(explorer, event, key: str):
    """处理自主探索按键"""
    if key == 'v':  # 启用/禁用自主探索
        if explorer.is_enabled:
            explorer.disable()
        else:
            explorer.enable()
        return event, False

    elif key == 'V':  # 显示探索状态
        status = explorer.get_status()
        print(f"🤖 自主探索状态:")
        for k, v in status.items():
            print(f"   {k}: {v}")
        return event, False

    return event, False


def process_keyboard_input(controller, event, key, **kwargs):
    """
    统一的键盘输入处理入口

    Args:
        controller: AI2-THOR Controller
        event: 当前事件
        key: 按键字符
        **kwargs: 其他需要的函数和变量

    Returns:
        tuple: (updated_event, should_break)
    """
    if key == keyboard.Key.esc:
        return event, True

    # 基础移动
    if key in ['w', 's', 'a', 'd', 'q', 'e']:
        return handle_movement_keys(controller, event, key), False

    # 物体交互
    if key in ['f', 'g', 'k', 'l', 't', 'y']:
        return handle_object_interaction_keys(controller, event, key), False

    # 液体和清洁
    if key in ['j', 'u', 'r', 'b', 'i']:
        return handle_liquid_and_cleaning_keys(controller, event, key), False

    # 自主探索控制
    if key in ['v', 'V']:
        explorer = kwargs.get('explorer')
        if explorer:
            return handle_autonomous_exploration_keys(explorer, event, key)

    # 特殊动作
    if key == 'c':
        chaos_drop_apple_on_floor = kwargs.get('chaos_drop_apple_on_floor')
        chaos_tip_chair = kwargs.get('chaos_tip_chair')
        if chaos_drop_apple_on_floor and chaos_tip_chair:
            return handle_special_action_keys(controller, event, key, chaos_drop_apple_on_floor, chaos_tip_chair), False

    # 帮助
    if key == '?':
        handle_help_key()
        return event, False

    return event, False
