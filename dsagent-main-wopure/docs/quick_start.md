## 快速开始

### step1 安装准备

首先参照`requirements-dev.txt`配置python环境（>=3.9）。

### step2 配置LLM

dsagent的配置项保存在`/config/config2.yaml`文件中，统一配置项目中所需的LLM API（也可以单独配置LLM并在需要调用时单独配置），主要配置项如下：

```yaml
llm:
  api_type: "openai"  # or azure / ollama / groq etc.
  base_url: "YOUR_BASE_URL"
  api_key: "YOUR_API_KEY"
  model: "gpt-4-turbo"  # or gpt-3.5-turbo
  proxy: "YOUR_PROXY"  # for LLM API requests
```

### step3 使用

#### 智能体初始化方式

dsagent的核心类为`/metagpt/roles/ds_agent/ds_agent.py`，其初始化方式如下：

```python
from metagpt.roles.ds_agent.ds_agent import DSAgent

ds_agent = DSAgent(use_reflection=True, use_rag=True, use_kaggle_exp=True, use_exp_extractor=False)
```

其中参数`use_reflection`控制反思机制，参数`use_rag`控制检索增强，参数`use_kaggle_exp`控制使用现有经验池，参数`use_exp_extractor`控制使用经验提取机制。

#### 执行实际任务

```python
requirement = "..."               # 任务需求
rsp = ds_agent.run(requirement)   # 执行任务
```

可以通过如下方式统计过程中的token开销：

```python
token_cost = ds_agent.llm.get_costs().total_prompt_tokens + ds_agent.llm.get_costs().total_completion_tokens
```



## 开发实验配置

本系统实验主要在`infiagent-dabench`和`ml-benchmark`数据集进行测试，使用者需要收集数据集分别放置在`/data/di_dataset/dabench`和`/data/di_dataset/ml_benchmark`中，以便进行后续实验。数据集可以在链接中直接获取：https://pan.baidu.com/s/18NpXiIaXun6C2IThrUl3zQ?pwd=ipud 。
