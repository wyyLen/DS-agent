## 🎯 DSAgent与MetaGPT解耦架构说明

### **解耦目标**

将DSAgent与MetaGPT完全解耦，使得：
1. ✅ 本地不需要修改MetaGPT源码
2. ✅ DSAgent可以独立于MetaGPT存在
3. ✅ 通过适配器模式使用MetaGPT作为底层框架
4. ✅ 可以轻松切换到其他框架（AutoGen、LangChain等）

---

### **新架构概览**

```
DSAgent项目
├── dsagent_core/                    # 核心功能（框架无关）
│   ├── retrieval/                   # RAG检索
│   ├── search/                      # 树搜索
│   ├── agents/                      # 【新增】Agent抽象层
│   │   ├── base_agent.py           # 抽象基类
│   │   ├── metagpt_impl.py         # MetaGPT实现（适配器）
│   │   ├── autogen_impl.py         # AutoGen实现（适配器）
│   │   └── factory.py              # Agent工厂
│   └── adapters/                    # 框架适配器
│       ├── metagpt_adapter.py
│       └── autogen_adapter.py
│
└── examples/ds_agent/agent_service/
    ├── agent_service_refactored.py  # 【新】使用工厂的服务
    └── agent_service.py             # 【旧】直接导入MetaGPT
```

---

### **核心设计模式**

#### **1. 抽象基类（BaseAgent）**

所有Agent实现都继承自 `BaseAgent`，提供统一接口：

```python
from dsagent_core.agents import BaseAgent

class MyAgent(BaseAgent):
    async def acquire(self) -> bool:
        """获取Agent使用权"""
        
    def release(self):
        """释放Agent"""
        
    async def process_stream(self, query, **kwargs):
        """流式处理查询"""
```

#### **2. 工厂模式（AgentFactory）**

使用工厂创建Agent，无需直接导入具体框架：

```python
from dsagent_core.agents import AgentFactory

# 创建MetaGPT agent
agent = AgentFactory.create_agent(
    agent_id="test-001",
    framework="metagpt",
    agent_type="ds"
)

# 创建AutoGen agent
agent = AgentFactory.create_agent(
    agent_id="test-002",
    framework="autogen",
    agent_type="ds"
)
```

#### **3. 适配器模式**

每个框架有自己的适配器实现：

```python
# metagpt_impl.py
class MetaGPTDSAgent(BaseAgent):
    def __init__(self, config):
        # 内部使用MetaGPT
        from metagpt.roles.ds_agent.ds_agent_stream import DSAgentStream
        self.metagpt_agent = DSAgentStream(...)

# autogen_impl.py  
class AutoGenDSAgent(BaseAgent):
    def __init__(self, config):
        # 内部使用AutoGen
        from autogen_agent_service_pure import PureAutoGenDSAgent
        self.autogen_agent = PureAutoGenDSAgent(...)
```

---

### **使用方式对比**

#### **旧方式（直接耦合MetaGPT）**

```python
# agent_service.py - 需要直接导入MetaGPT
from metagpt.roles.ds_agent.ds_agent_stream import DSAgentStream
from metagpt.llm import LLM

agent = DSAgentStream(
    name="DSAgent_001",
    llm=LLM(),
    use_rag=True
)
```

**问题：**
- ❌ 必须修改MetaGPT源码（添加ds_agent目录）
- ❌ 直接依赖MetaGPT内部实现
- ❌ 难以切换框架

#### **新方式（解耦架构）**

```python
# agent_service_refactored.py - 使用工厂
from dsagent_core.agents import AgentFactory

agent = AgentFactory.create_agent(
    agent_id="DSAgent_001",
    framework="metagpt",  # 或 "autogen"
    agent_type="ds",
    use_rag=True
)
```

**优势：**
- ✅ 不需要修改MetaGPT源码
- ✅ 框架切换只需改变参数
- ✅ 统一的接口，易于扩展

---

### **迁移步骤**

#### **步骤1：使用新的服务提供者**

修改 `api_service_provider.py`：

```python
# 旧
from agent_service import AgentServiceProvider

# 新
from agent_service_refactored import AgentServiceProvider
```

#### **步骤2：设置环境变量**

```bash
# 使用MetaGPT
export AGENT_FRAMEWORK=metagpt

# 使用AutoGen
export AGENT_FRAMEWORK=autogen
```

#### **步骤3：启动后端**

```bash
python examples/ds_agent/agent_service/start_backend.py
```

工厂会自动检测可用框架并创建相应的Agent。

---

### **框架可用性检查**

```python
from dsagent_core.agents import AgentFactory

available = AgentFactory.list_available_frameworks()
print(available)
# {'metagpt': True, 'autogen': False}
```

如果MetaGPT不可用，系统会自动fallback到可用的框架。

---

### **扩展新框架**

添加新框架（如LangChain）只需3步：

**1. 创建实现类**

```python
# dsagent_core/agents/langchain_impl.py
class LangChainDSAgent(BaseAgent):
    def __init__(self, config):
        from langchain.agents import AgentExecutor
        self.langchain_agent = AgentExecutor(...)
    
    async def process_stream(self, query, **kwargs):
        # 实现LangChain的流式处理
        ...
```

**2. 更新工厂**

```python
# dsagent_core/agents/factory.py
def _create_langchain_agent(agent_type, config):
    from dsagent_core.agents.langchain_impl import LangChainDSAgent
    return LangChainDSAgent(config)
```

**3. 使用**

```python
agent = AgentFactory.create_agent(
    agent_id="test",
    framework="langchain",
    agent_type="ds"
)
```

---

### **依赖关系**

#### **解耦前**
```
agent_service.py
    ↓ 直接依赖
metagpt/roles/ds_agent/ds_agent_stream.py
    ↓ 必须存在
MetaGPT源码必须被修改
```

#### **解耦后**
```
agent_service_refactored.py
    ↓ 使用
dsagent_core/agents/factory.py
    ↓ 动态导入
dsagent_core/agents/metagpt_impl.py
    ↓ 适配器模式
metagpt（原生pip包，不修改）
```

---

### **测试解耦方案**

```python
# test_decoupled_agents.py
from dsagent_core.agents import create_agent

async def test_metagpt():
    agent = create_agent("test-001", framework="metagpt")
    async for chunk in agent.process_stream("分析数据"):
        print(chunk)

async def test_autogen():
    agent = create_agent("test-002", framework="autogen")
    async for chunk in agent.process_stream("分析数据"):
        print(chunk)
```

---

### **FAQ**

**Q: 还需要metagpt/roles/ds_agent/目录吗？**
A: 需要，但不需要修改。解耦方案是在DSAgent侧添加适配层，MetaGPT侧保持原样。如果MetaGPT官方版本有ds_agent，就用官方的；如果没有，可以作为独立包安装。

**Q: 如何完全移除MetaGPT依赖？**
A: 将 `metagpt/roles/ds_agent/` 和 `metagpt/actions/di/` 移到独立包，然后修改 `metagpt_impl.py` 导入路径。

**Q: 性能有影响吗？**
A: 几乎没有。适配器只是薄薄的一层封装，实际执行仍由底层框架完成。

**Q: 可以同时使用多个框架吗？**
A: 可以！工厂可以创建不同框架的Agent放入同一个池中。

---

### **下一步计划**

1. ✅ 创建独立的code executor（不依赖MetaGPT的ExecuteNbCode）
2. ⬜ 将ds_agent和di目录打包为独立的pip包
3. ⬜ 添加LangChain适配器
4. ⬜ 完善文档和示例
