#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阿里云DashScope API连接测试脚本
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT
from metagpt.llm import LLM


async def test_llm_api():
    """测试LLM API连接"""
    print("=" * 60)
    print("测试阿里云通义千问API连接")
    print("=" * 60)
    
    try:
        # 加载配置
        config_path = METAGPT_ROOT / "config" / "config2.yaml"
        config = Config.from_yaml_file(config_path)
        
        print(f"\n使用配置:")
        print(f"  - API类型: {config.llm.api_type}")
        print(f"  - 模型: {config.llm.model}")
        print(f"  - API密钥: {config.llm.api_key[:20]}...")
        
        # 创建LLM实例
        print(f"\n正在初始化LLM...")
        llm = LLM(config.llm)
        
        # 测试简单对话
        test_message = "你好，请用一句话介绍你自己。"
        print(f"\n发送测试消息: {test_message}")
        print(f"等待响应...\n")
        
        response = await llm.aask(test_message)
        
        print(f"✓ API连接成功！")
        print(f"\n模型响应:")
        print(f"{response}")
        
        print("\n" + "=" * 60)
        print("API测试完成！配置正确，可以正常使用。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ API测试失败:")
        print(f"  错误信息: {e}")
        print(f"\n请检查:")
        print(f"  1. API密钥是否正确")
        print(f"  2. 网络连接是否正常")
        print(f"  3. 阿里云账户是否有相应模型的访问权限")
        return False
    
    return True


async def test_embedding_api():
    """测试嵌入模型API连接"""
    print("\n" + "=" * 60)
    print("测试阿里云嵌入模型API连接")
    print("=" * 60)
    
    try:
        import dashscope
        from dashscope import TextEmbedding
        
        # 加载配置
        config_path = METAGPT_ROOT / "config" / "config2.yaml"
        config = Config.from_yaml_file(config_path)
        
        # 设置API密钥
        dashscope.api_key = config.llm.api_key
        
        print(f"\n使用嵌入模型: text-embedding-v4")
        
        # 测试嵌入
        test_text = "这是一个测试文本"
        print(f"测试文本: {test_text}")
        print(f"正在生成嵌入向量...\n")
        
        response = TextEmbedding.call(
            model='text-embedding-v4',
            input=test_text
        )
        
        if response.status_code == 200:
            embeddings = response.output['embeddings'][0]['embedding']
            print(f"✓ 嵌入模型API连接成功！")
            print(f"  向量维度: {len(embeddings)}")
            print(f"  向量前5个值: {embeddings[:5]}")
        else:
            print(f"✗ 嵌入模型API调用失败")
            print(f"  状态码: {response.status_code}")
            print(f"  错误信息: {response.message}")
            return False
        
        print("\n" + "=" * 60)
        print("嵌入模型测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 嵌入模型测试失败:")
        print(f"  错误信息: {e}")
        return False
    
    return True


async def main():
    """主测试函数"""
    print("\n开始测试阿里云DashScope API配置...\n")
    
    # 测试LLM
    llm_success = await test_llm_api()
    
    # 测试嵌入模型
    if llm_success:
        await test_embedding_api()
    
    print("\n\n所有测试完成！")
    if llm_success:
        print("✓ 配置正确，可以启动服务了。")
        print("\n启动后端服务:")
        print("  cd examples/ds_agent/agent_service")
        print("  python api_service_provider.py")
        print("\n启动前端服务:")
        print("  cd DSassistant")
        print("  python main.py")
    else:
        print("✗ 配置存在问题，请检查配置文件和API密钥。")


if __name__ == "__main__":
    asyncio.run(main())
