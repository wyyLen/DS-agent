# 如何切换到 AutoGen 框架

DSAgent 现在支持两种 Agent 框架：
- **MetaGPT** (默认)
- **AutoGen** (新增)

## 方法 1: 使用环境变量 (推荐)

### Windows PowerShell:
```powershell
# 设置为 AutoGen
$env:AGENT_FRAMEWORK="autogen"
cd DSassistant
python main.py

# 设置为 MetaGPT (默认)
$env:AGENT_FRAMEWORK="metagpt"
cd DSassistant
python main.py
```

### Linux/Mac:
```bash
# 设置为 AutoGen
export AGENT_FRAMEWORK=autogen
cd DSassistant
python main.py

# 设置为 MetaGPT (默认)
export AGENT_FRAMEWORK=metagpt
cd DSassistant
python main.py

#然后调用启动后端
$env:PYTHONPATH="e:\dsagent-main"; 
python examples/ds_agent/agent_service/start_backend.py
```

## 方法 2: 修改代码

在 `examples/ds_agent/agent_service/agent_service.py` 中修改：

```python
# 第30行左右，修改这一行：
AGENT_FRAMEWORK = os.getenv('AGENT_FRAMEWORK', 'metagpt').lower()

# 改为：
AGENT_FRAMEWORK = os.getenv('AGENT_FRAMEWORK', 'autogen').lower()
```

## 方法 3: 程序中动态指定

在创建 `AgentServiceProvider` 时指定框架：

```python
# 使用 AutoGen
service_provider = AgentServiceProvider(
    initial_agent_counts={"ds": 1, "lats": 1},
    framework='autogen'
)

# 使用 MetaGPT
service_provider = AgentServiceProvider(
    initial_agent_counts={"ds": 1, "lats": 1},
    framework='metagpt'
)
```

## 验证当前使用的框架

启动服务后，在日志中查看：

```
============================================================
Agent Framework: AUTOGEN
============================================================
Initialized AutoGen ds agent DS-1-1736573983
```

或

```
============================================================
Agent Framework: METAGPT
============================================================
Initialized MetaGPT ds agent DS-1-1736573983
```

## 框架对比

| 特性 | MetaGPT | AutoGen |
|------|---------|---------|
| 经验检索 | ✅ 内置 | ✅ 通过适配器 |
| 流式输出 | ✅ | ✅ |
| RAG支持 | ✅ | ✅ |
| 树搜索 | ✅ LATS | ✅ 简化版 |
| API密钥 | DashScope | 可配置 |
| 成熟度 | 高 | 新增 |

## 配置 AutoGen LLM

AutoGen 默认使用 DashScope API，与 MetaGPT 相同。

如需修改，编辑 `autogen_agent_service.py` 中的 `_get_default_llm_config()` 方法：

```python
def _get_default_llm_config(self) -> Dict:
    return {
        "model": "your-model",
        "api_key": "your-api-key",
        "base_url": "your-api-base-url",
        "temperature": 0.7
    }
```

## 注意事项

1. **首次使用 AutoGen**: 确保已安装 `pip install pyautogen`
2. **经验文件**: AutoGen 使用相同的经验库文件
3. **API 兼容**: 两种框架的 API 接口完全相同，前端无需修改
4. **性能**: MetaGPT 更成熟，AutoGen 正在完善中

## 故障排除

### 如果看到 "AutoGen requested but not available"
```bash
pip install pyautogen
```

### 如果看到导入错误
```bash
# 设置 PYTHONPATH
export PYTHONPATH=/path/to/dsagent-main:$PYTHONPATH
```

### 查看详细日志
修改 `agent_service.py` 第17行：
```python
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
```
