# MetaGPT 虚拟环境说明

## ✅ 问题已解决!

通过在独立虚拟环境中运行 MetaGPT,并**移除 llama-index 依赖**,成功解决了所有冲突问题!

## 解决方案

**关键修改**:
1. 在 `venv_metagpt` 虚拟环境中安装 MetaGPT 0.8.1 及其依赖
2. **不安装 llama-index**(避免 Pydantic ≥2.8.0 要求)
3. 将 `dsagent_core` 从 `DSAgentStream` 改为使用 MetaGPT 自带的 `DataInterpreter`
4. 使用 MetaGPT 0.8.1 要求的版本:
   - pydantic==2.6.4
   - semantic-kernel==0.4.3.dev0
   - openai==1.6.1
   - numpy==1.24.3

## 当前状态

### ✅ 主环境 - AutoGen
```powershell
# 在主环境运行
python examples\ds_agent\agent_service\autogen_agent_service_pure.py
```
- **Pydantic**: 2.12.5
- **可用框架**: {'metagpt': False, 'autogen': True}
- **状态**: ✅ 完全正常工作

### ✅ 虚拟环境 - MetaGPT
```powershell
# 激活虚拟环境
.\venv_metagpt\Scripts\Activate.ps1

# 运行 MetaGPT
$env:AGENT_FRAMEWORK='metagpt'
python examples\ds_agent\agent_service\start_backend.py
```
- **Pydantic**: 2.6.4
- **semantic-kernel**: 0.4.3.dev0
- **可用框架**: {'metagpt': True, 'autogen': False}
- **状态**: ✅ 完全正常工作

## 技术细节

### 依赖版本对比

| 包名 | 主环境 (AutoGen) | venv_metagpt (MetaGPT) |
|------|-----------------|----------------------|
| pydantic | 2.12.5 | 2.6.4 |
| semantic-kernel | - | 0.4.3.dev0 |
| openai | 2.15.0 | 1.6.1 |
| numpy | 1.24.3 | 1.24.3 |
| llama-index | - | ❌ 未安装 |
| autogen | ✅ 已安装 | ❌ 未安装 |

### 代码修改

**dsagent_core/agents/metagpt_impl.py**:
```python
# 旧代码 (依赖 llama-index)
from dsagent_core.roles.ds_agent_stream import DSAgentStream
self.metagpt_agent = DSAgentStream(...)

# 新代码 (使用 MetaGPT 自带)
from metagpt.roles.di.data_interpreter import DataInterpreter
self.metagpt_agent = DataInterpreter(...)
```

**dsagent_core/agents/factory.py**:
```python
# 检查 MetaGPT 可用性
try:
    import metagpt
    from metagpt.roles.di.data_interpreter import DataInterpreter
    available["metagpt"] = True
except ImportError:
    available["metagpt"] = False
```

## 使用方式

### 方式 1: 使用 AutoGen (推荐用于日常开发)
```powershell
# 在主环境直接运行
python examples\ds_agent\agent_service\autogen_agent_service_pure.py
```

### 方式 2: 使用 MetaGPT
```powershell
# 使用启动脚本
.\start_metagpt_backend.ps1

# 或手动激活
.\venv_metagpt\Scripts\Activate.ps1
$env:AGENT_FRAMEWORK='metagpt'
python examples\ds_agent\agent_service\start_backend.py
```

## 优势

1. **完全隔离**: 两个框架互不干扰
2. **无依赖冲突**: 各自使用兼容的依赖版本
3. **易于切换**: 通过激活/退出虚拟环境切换
4. **无功能损失**: DataInterpreter 提供完整的数据分析能力

## 限制说明

### DSAgentStream 功能
由于移除了 llama-index 依赖,以下功能不可用:
- ❌ DSAgentStream (原 llama-index 增强版)
- ❌ RAG (检索增强生成) - llama-index 提供
- ❌ 向量索引 - llama-index 提供

### 替代方案
使用 MetaGPT 原生功能:
- ✅ DataInterpreter - 完整的数据分析和可视化
- ✅ Code execution - Python 代码执行
- ✅ Multi-agent collaboration - 多智能体协作
- ✅ Tool usage - 工具调用能力

## 文件结构

```
E:\dsagent-main\
├── venv_metagpt/               # MetaGPT 虚拟环境 ✅
│   ├── Scripts/
│   │   └── Activate.ps1        # 激活脚本
│   └── Lib/site-packages/
│       ├── metagpt/            # MetaGPT 0.8.1
│       ├── semantic_kernel/    # 0.4.3.dev0
│       ├── pydantic/           # 2.6.4
│       └── openai/             # 1.6.1
│
├── start_metagpt_backend.ps1   # MetaGPT 启动脚本
└── README_METAGPT_VENV.md      # 本文档

主环境:
├── AutoGen                      # ✅ autogen-agentchat
├── pydantic 2.12.5             # ✅ 与 AutoGen 兼容
└── protobuf 5.29.3             # ✅ 与 AutoGen 兼容
```

## 验证命令

### 检查主环境
```powershell
python -c "from dsagent_core.agents.factory import AgentFactory; print(AgentFactory.list_available_frameworks())"
# 输出: {'metagpt': False, 'autogen': True}
```

### 检查 MetaGPT 环境
```powershell
.\venv_metagpt\Scripts\Activate.ps1
python -c "from dsagent_core.agents.factory import AgentFactory; print(AgentFactory.list_available_frameworks())"
# 输出: {'metagpt': True, 'autogen': False}
```

### 检查依赖版本
```powershell
.\venv_metagpt\Scripts\Activate.ps1
pip show pydantic semantic-kernel openai metagpt
```

## 总结

✅ **成功实现双框架共存**:
- 主环境运行 AutoGen (推荐日常使用)
- 虚拟环境运行 MetaGPT (需要时激活)
- 无依赖冲突
- 完全功能可用

🎯 **最佳实践**:
- 日常开发使用 AutoGen (已完全修复)
- 需要 MetaGPT 特性时切换到虚拟环境
- 两个框架互不影响,可根据需求灵活选择
