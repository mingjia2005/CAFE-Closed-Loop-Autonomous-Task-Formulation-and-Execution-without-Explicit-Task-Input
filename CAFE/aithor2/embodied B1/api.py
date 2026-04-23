#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DashScope API配置文件
用于管理API密钥和配置参数
"""

import os
from typing import Optional

class DashScopeConfig:
    """DashScope API配置管理类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化配置
        
        Args:
            api_key: API密钥，如果不提供则从环境变量读取
        """
        self.api_key = api_key or self._get_env_api_key()
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = "qwen-max"  # 默认使用通义千问Max模型
        
        # 验证配置
        if not self.api_key:
            raise ValueError(
                "DashScope API密钥未设置！\n"
                "请通过以下方式之一设置：\n"
                "1. 设置环境变量 DASHSCOPE_API_KEY\n"
                "2. 在代码中传入 api_key 参数\n"
                "3. 在 .env 文件中设置 DASHSCOPE_API_KEY"
            )
    
    def _get_env_api_key(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        # 尝试从多个环境变量获取
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            api_key = os.getenv('DASH_SCOPE_API_KEY')
        if not api_key:
            api_key = os.getenv('ALIBABA_API_KEY')
        
        return api_key
    
    def get_headers(self) -> dict:
        """获取API请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_api_url(self) -> str:
        """获取API端点URL"""
        return self.api_url
    
    def get_model(self) -> str:
        """获取模型名称"""
        return self.model
    
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        if not self.api_key:
            return False
        if not self.api_key.startswith('sk-'):
            return False
        if len(self.api_key) < 20:
            return False
        return True
    
    def print_config_info(self):
        """打印配置信息（隐藏敏感信息）"""
        print("🔧 DashScope API 配置信息:")
        print(f"   • API端点: {self.api_url}")
        print(f"   • 模型: {self.model}")
        print(f"   • API密钥: {self.api_key[:8]}...{self.api_key[-4:]}")
        print(f"   • 配置有效: {'✅' if self.validate_config() else '❌'}")


def load_config_from_env() -> DashScopeConfig:
    """从环境变量加载配置"""
    return DashScopeConfig()


def load_config_from_file(config_file: str = ".env") -> DashScopeConfig:
    """从配置文件加载配置"""
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        
        return DashScopeConfig()
    except Exception as e:
        print(f"⚠️ 配置文件加载失败: {str(e)}")
        return DashScopeConfig()


def create_env_template():
    """创建环境变量模板文件"""
    template = """# DashScope API 配置文件
# 请将您的API密钥填入下面的变量中

# DashScope API密钥 (必需)
DASHSCOPE_API_KEY=sk-your-api-key-here

# 可选配置
# DASH_SCOPE_API_KEY=sk-your-api-key-here
# ALIBABA_API_KEY=sk-your-api-key-here

# 使用说明:
# 1. 将 sk-your-api-key-here 替换为您的真实API密钥
# 2. 保存文件名为 .env
# 3. 确保 .env 文件在项目根目录下
# 4. 不要将 .env 文件提交到版本控制系统
"""
    
    try:
        with open('.env.template', 'w', encoding='utf-8') as f:
            f.write(template)
        print("✅ 环境变量模板文件已创建: .env.template")
        print("💡 请复制 .env.template 为 .env 并填入您的API密钥")
    except Exception as e:
        print(f"❌ 创建模板文件失败: {str(e)}")


if __name__ == "__main__":
    print("🔧 DashScope API 配置管理工具")
    print("="*50)
    
    try:
        # 尝试加载配置
        config = load_config_from_env()
        config.print_config_info()
        
        if config.validate_config():
            print("\n✅ 配置验证通过，可以正常使用API")
        else:
            print("\n❌ 配置验证失败，请检查API密钥")
            create_env_template()
            
    except ValueError as e:
        print(f"\n❌ 配置错误: {str(e)}")
        create_env_template()
    except Exception as e:
        print(f"\n❌ 未知错误: {str(e)}")
        create_env_template()



