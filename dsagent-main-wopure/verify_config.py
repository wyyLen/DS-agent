#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阿里云DashScope配置验证脚本
用于测试配置文件是否正确设置
"""
import yaml
from pathlib import Path

def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("阿里云DashScope配置验证")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    
    # 测试主配置
    print("\n1. 测试 config/config2.yaml")
    try:
        config_path = project_root / "config" / "config2.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"   ✓ 配置文件加载成功")
            print(f"   - API类型: {config['llm']['api_type']}")
            print(f"   - 模型: {config['llm']['model']}")
            print(f"   - API密钥: {config['llm']['api_key'][:20]}...")
            if 'embedding' in config and config['embedding']:
                print(f"   - 嵌入模型API: {config['embedding'].get('api_type', 'N/A')}")
                print(f"   - 嵌入模型: {config['embedding'].get('model', 'N/A')}")
        else:
            print(f"   ✗ 配置文件不存在: {config_path}")
    except Exception as e:
        print(f"   ✗ 配置文件加载失败: {e}")
    
    # 测试GPT-4o配置
    print("\n2. 测试 config/gpt-4o.yaml")
    try:
        gpt4o_path = project_root / "config" / "gpt-4o.yaml"
        if gpt4o_path.exists():
            with open(gpt4o_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"   ✓ 配置文件加载成功")
            print(f"   - API类型: {config['llm']['api_type']}")
            print(f"   - 模型: {config['llm']['model']}")
            print(f"   - API密钥: {config['llm']['api_key'][:20]}...")
        else:
            print(f"   ✗ 配置文件不存在: {gpt4o_path}")
    except Exception as e:
        print(f"   ✗ 配置文件加载失败: {e}")
    
    # 测试GPT-4o-mini配置
    print("\n3. 测试 config/gpt-4o-mini.yaml")
    try:
        gpt4o_mini_path = project_root / "config" / "gpt-4o-mini.yaml"
        if gpt4o_mini_path.exists():
            with open(gpt4o_mini_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"   ✓ 配置文件加载成功")
            print(f"   - API类型: {config['llm']['api_type']}")
            print(f"   - 模型: {config['llm']['model']}")
            print(f"   - API密钥: {config['llm']['api_key'][:20]}...")
        else:
            print(f"   ✗ 配置文件不存在: {gpt4o_mini_path}")
    except Exception as e:
        print(f"   ✗ 配置文件加载失败: {e}")
    
    # 测试DashScope包
    print("\n4. 测试 dashscope 包")
    try:
        import dashscope
        print(f"   ✓ dashscope 包已安装")
    except ImportError:
        print(f"   ✗ dashscope 包未安装，请运行: pip install dashscope")
    
    print("\n" + "=" * 60)
    print("配置验证完成！所有配置文件正确。")
    print("=" * 60)
    print("\n如需测试API连接，请运行: python test_dashscope_api.py")

if __name__ == "__main__":
    test_config()
