# 如何为 DSAgent 添加新框架支持

本文档说明如何为 DSAgent 添加新的 Agent 框架支持（如 LangChain、CrewAI、Semantic Kernel 等）。

---

## 📋 目录

1. [架构概览](#架构概览)
2. [需要实现的组件](#需要实现的组件)
3. [详细步骤](#详细步骤)
4. [示例：添加 LangChain 支持](#示例添加-langchain-支持)
5. [集成到服务](#集成到服务)

---

## 🏗️ 架构概览

DSAgent 采用**核心 + 适配器**模式：

```
dsagent_core/               # 框架无关的核心
├── retrieval/              # 文本/工作流检索（已完成）
├── search/
│   ├── tree_search.py      # 通用树搜索（已完成）
│   └── lats_core.py        # LATS 核心算法（已完成）
└── adapters/               # 各框架适配器
    ├── autogen_adapter.py          # AutoGen 基础功能
    ├── autogen_lats_adapter.py     # AutoGen LATS
    ├── metagpt_lats_adapter.py     # MetaGPT LATS
    └── [新框架]_adapter.py         # 👈 在这里添加
```

---

## 🎯 需要实现的组件

为支持新框架，需要实现 **2 类适配器**：

### 1. 基础适配器（可选，用于 RAG）
- 位置：`dsagent_core/adapters/[框架名]_adapter.py`
- 功能：文本检索、工作流检索

### 2. LATS 适配器（核心）
- 位置：`dsagent_core/adapters/[框架名]_lats_adapter.py`
- 功能：实现 4 个接口，连接 LATS 核心

---

## 📝 详细步骤

### 步骤 1：创建基础适配器（可选）

**文件位置**：`dsagent_core/adapters/[框架名]_adapter.py`

**需要实现的功能**：
1. 初始化文本和工作流检索器
2. 提供检索接口
3. 与框架原生功能集成（如 LangChain 的 Tool、CrewAI 的 Tool 等）

**参考模板**：
```python
# dsagent_core/adapters/langchain_adapter.py

from dsagent_core.retrieval import TextExperienceRetriever, WorkflowExperienceRetriever

class LangChainAdapter:
    """LangChain 框架基础适配器"""
    
    def __init__(self, text_exp_path=None, workflow_exp_path=None):
        # 初始化检索器
        self.text_retriever = TextExperienceRetriever(text_exp_path) if text_exp_path else None
        self.workflow_retriever = WorkflowExperienceRetriever(workflow_exp_path) if workflow_exp_path else None
    
    def retrieve_text_experience(self, query: str, top_k: int = 5):
        """检索文本经验"""
        return self.text_retriever.retrieve(query, top_k)
    
    def as_langchain_tool(self):
        """转换为 LangChain Tool"""
        from langchain.tools import Tool
        
        def retrieve(query: str) -> str:
            result = self.retrieve_text_experience(query)
            return self._format_result(result)
        
        return Tool(
            name="experience_retriever",
            func=retrieve,
            description="检索相似经验"
        )
```

---

### 步骤 2：创建 LATS 适配器（必需）

**文件位置**：`dsagent_core/adapters/[框架名]_lats_adapter.py`

**必须实现 4 个接口**：

#### 接口 1：`CodeExecutor` - 代码执行
```python
from dsagent_core.search.lats_core import CodeExecutor

class LangChainCodeExecutor(CodeExecutor):
    """使用 Jupyter 执行代码"""
    
    def __init__(self):
        from metagpt.actions import ExecuteNbCode
        self.executor = ExecuteNbCode()
    
    async def execute(self, code: str, context: dict) -> tuple[bool, str]:
        """执行代码，返回 (成功与否, 输出)"""
        result = await self.executor.run(code=code)
        return result['is_success'], result['output']
    
    async def terminate(self):
        """清理资源"""
        if self.executor:
            await self.executor.terminate()
```

#### 接口 2：`ThoughtGenerator` - 思维生成
```python
from dsagent_core.search.lats_core import ThoughtGenerator, LATSNode

class LangChainThoughtGenerator(ThoughtGenerator):
    """使用 LLM 生成下一步思维"""
    
    def __init__(self, llm):
        self.llm = llm  # LangChain 的 LLM 实例
    
    async def generate(self, node: LATSNode, context: dict, n_samples: int = 1) -> list[dict]:
        """生成思维，返回 [{'thought': '...', 'task_type': '...'}]"""
        prompt = self._build_prompt(node, context)
        
        thoughts = []
        for _ in range(n_samples):
            response = await self.llm.agenerate([prompt])
            thought = self._parse_response(response)
            thoughts.append(thought)
        
        return thoughts
```

#### 接口 3：`ActionGenerator` - 动作生成
```python
from dsagent_core.search.lats_core import ActionGenerator

class LangChainActionGenerator(ActionGenerator):
    """将思维转换为可执行代码"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def generate(self, thought: dict, context: dict) -> str:
        """生成代码，返回 Python 代码字符串"""
        prompt = f"根据思维生成代码：{thought['thought']}"
        response = await self.llm.agenerate([prompt])
        code = self._extract_code(response)
        return code
```

#### 接口 4：`StateEvaluator` - 状态评估
```python
from dsagent_core.search.lats_core import StateEvaluator

class LangChainStateEvaluator(StateEvaluator):
    """评估方案质量"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def evaluate(self, trajectory: str, is_terminal: bool, context: dict) -> float:
        """评估轨迹，返回 0-10 分"""
        prompt = f"评估这个解决方案（0-10分）：\n{trajectory}"
        response = await self.llm.agenerate([prompt])
        score = self._parse_score(response)
        return float(score)
```

#### 组合为完整适配器
```python
from dsagent_core.search.lats_core import LATSCore

class LangChainLATSAdapter:
    """LangChain LATS 完整适配器"""
    
    def __init__(self, llm, max_depth=10):
        self.llm = llm
        
        # 创建四个组件
        self.code_executor = LangChainCodeExecutor()
        self.thought_generator = LangChainThoughtGenerator(llm)
        self.action_generator = LangChainActionGenerator(llm)
        self.state_evaluator = LangChainStateEvaluator(llm)
        
        # 初始化核心
        self.lats_core = LATSCore(
            thought_generator=self.thought_generator,
            action_generator=self.action_generator,
            code_executor=self.code_executor,
            state_evaluator=self.state_evaluator,
            max_depth=max_depth
        )
    
    async def run(self, goal: str, iterations=10):
        """运行 LATS 搜索"""
        best_node, all_nodes = await self.lats_core.search(
            goal=goal,
            iterations=iterations
        )
        return best_node, all_nodes
```

---

### 步骤 3：更新导出文件

**文件位置**：`dsagent_core/adapters/__init__.py`

**添加内容**：
```python
# 添加到文件末尾

try:
    from dsagent_core.adapters.langchain_adapter import LangChainAdapter
    from dsagent_core.adapters.langchain_lats_adapter import LangChainLATSAdapter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LangChainAdapter = None
    LangChainLATSAdapter = None
    LANGCHAIN_AVAILABLE = False

__all__ = [
    # ... 现有的导出
    "LangChainAdapter",
    "LangChainLATSAdapter",
    "LANGCHAIN_AVAILABLE",
]
```

---

### 步骤 4：集成到服务（可选）

**文件位置**：`examples/ds_agent/agent_service/agent_service.py`

**在 `AgentServiceProvider._init_agent_pool()` 中添加**：

```python
def _init_agent_pool(self, initial_agent_counts: dict):
    for mode, count in initial_agent_counts.items():
        for _ in range(count):
            agent_id = f"{mode.upper()}-{len(self.agents_pool[mode]) + 1}-{int(time.time())}"
            
            # ... 现有的 autogen 和 metagpt 代码 ...
            
            elif self.framework == 'langchain' and LANGCHAIN_ENABLED:
                # 添加 LangChain 支持
                if mode == "ds":
                    from langchain.llms import OpenAI
                    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    
                    # 创建基础 Agent（使用 RAG）
                    adapter = LangChainAdapter(
                        text_exp_path=Path("examples/data/exp_bank/plan_exp.json")
                    )
                    tool = adapter.as_langchain_tool()
                    # ... 创建 LangChain Agent
                    
                elif mode == "lats":
                    # 创建 LATS Agent
                    from dsagent_core.adapters import LangChainLATSAdapter
                    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    
                    lats_adapter = LangChainLATSAdapter(llm=llm)
                    # 包装成统一的 Agent 接口
                    self.agents_pool[mode][agent_id] = LangChainLATSAgentWrapper(
                        agent_id, lats_adapter
                    )
```

---

## 📚 示例：添加 LangChain 支持

### 完整文件清单

1. **基础适配器**
   - 📄 `dsagent_core/adapters/langchain_adapter.py`
   - 功能：RAG、Tool 创建

2. **LATS 适配器**
   - 📄 `dsagent_core/adapters/langchain_lats_adapter.py`
   - 包含：
     - `LangChainCodeExecutor`
     - `LangChainThoughtGenerator`
     - `LangChainActionGenerator`
     - `LangChainStateEvaluator`
     - `LangChainLATSAdapter`（组合类）

3. **导出更新**
   - 📄 `dsagent_core/adapters/__init__.py`
   - 添加 import 和 `__all__`

4. **服务集成**（可选）
   - 📄 `examples/ds_agent/agent_service/agent_service.py`
   - 在 `_init_agent_pool()` 中添加分支

---

## 🔍 关键点总结

### 必须实现的 4 个接口

| 接口 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `CodeExecutor` | 执行代码 | 代码字符串 | (成功, 输出) |
| `ThoughtGenerator` | 生成思维 | 当前节点 | 思维列表 |
| `ActionGenerator` | 生成代码 | 思维 | 代码字符串 |
| `StateEvaluator` | 评估状态 | 轨迹 | 分数(0-10) |

### 核心优势

1. ✅ **只需实现 4 个接口**，核心算法已完成
2. ✅ **检索功能开箱即用**，无需重新实现
3. ✅ **统一的 LATS 算法**，质量有保障
4. ✅ **易于测试**，每个组件可单独测试

### 参考现有实现

- **AutoGen 实现**：`dsagent_core/adapters/autogen_lats_adapter.py`（最完整）
- **MetaGPT 实现**：`dsagent_core/adapters/metagpt_lats_adapter.py`
- **核心接口定义**：`dsagent_core/search/lats_core.py`

---

## 🚀 快速开始

1. 复制 `autogen_lats_adapter.py` 作为模板
2. 替换 LLM 调用为新框架的 API
3. 替换代码执行器（如果需要）
4. 测试 4 个接口是否正常工作
5. 集成到服务中

---

## ❓ 常见问题

### Q: 是否必须实现基础适配器？
A: 不是。基础适配器只提供 RAG 功能，如果只需要 LATS，可以跳过。

### Q: 可以复用 ExecuteNbCode 吗？
A: 可以！所有框架都可以使用 MetaGPT 的 ExecuteNbCode 执行代码。

### Q: 如何测试新适配器？
A: 参考 `test_lats_core.py`，创建简单的测试脚本。

### Q: 性能如何优化？
A: 
- 使用缓存减少 LLM 调用
- 并行生成多个 thoughts
- 提前终止低质量分支

---

## 📖 相关文档

- [LATS 核心实现](./dsagent_core/LATS_README.md)
- [升级指南](./LATS_UPGRADE_GUIDE.md)
- [使用示例](./examples/lats_usage_examples.py)
