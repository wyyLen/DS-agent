# 代码说明

本代码库实现了一个基于MetaGPT框架的数据科学智能体(DS-Agent)系统，包含智能体核心逻辑、检索增强规划、蒙特卡洛树搜索等功能模块。



## 📁 目录结构

### 1. config/
- **功能**：存放所有自定义模型的配置项

### 2. data/
- **功能**：保存实验数据集
- **子目录**：
  - `di_dataset/`：实验数据集目录，可从以下原始数据集中获取，也可以从百度网盘提取（[链接](https://pan.baidu.com/s/18NpXiIaXun6C2IThrUl3zQ?pwd=ipud )）。
    - `da_bench/`：来自[Infiagent-DAbench数据集](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent)的数据
    - `ml_benchmark/`：来自[MLbenchmark数据集](https://drive.google.com/drive/folders/17SpI9WL9kzd260q2DArbXKNcqhidjA7s)的机器学习基准数据

### 3. examples/
- **功能**：包含实验和测试代码
- **子目录**：
  - `ds_agent/`：智能体测试及后端实现代码
    - `agent_service`：智能体后端服务，通过`fastapi`实现
    - `batchInitExpPool.py`：从`kaggle`中提取经验知识核心代码
  - `experiment/`：实验代码及结果分析工具
    - `da_bench/`：对`Infiagent-DAbench`数据集的测试代码（注意`taskweaver`和`autogen`需要自行参照`util`中的工具设计，在相应github仓库代码中完成实验）
    - `ml_benchmark/`：对`ML-benchmark`数据集的测试代码

### 4. metagpt/
- **功能**：智能体核心实现
- **子目录**：
  - `actions/`：定义智能体的行动模块
    - `ds_agent`：本智能体的自定义行为类

  - `prompts/`：智能体提示词目录
    - `ds_agent/`：本智能体的自定义提示词设计
    - `lats/`：树搜索算法相关提示词
    - `ds_task_type.py`：数据科学任务类型设计

  - `provider/`：大语言模型(LLM)接入实现
    - 根据需要自行调整模型接口。当前仅对`zhipuAI`接入有所调整。

  - `rag/`：智能体检索策略目录
    - `engines/`
      - `GraphMatching/`：图匹配辅助类目录
      - `customEmbeddingComparisonEngine.py`：自定义语义检索器
      - `customMixture.py`：自定义混合检索器
      - `customSolutionSamplesGenerate.py`：树搜索算法中的局部工作流检索器
      - `customWorkflowGM.py`：工作流检索器
      - `graphUtils.py`：工具类

    - `retrievers/`          # `engines/`的相应底层实现（主要是混合检索器）

  - `roles/`：智能体角色定义
    - `ds_agent/`：数据科学智能体核心类
      - `ds_agent.py`：常规实现类
      - `ds_agent_stream.py`：流式传输类
      - `lats_react_stream.py`：直接使用树搜索算法的流式传输类

  - `strategy/`：智能体策略实现
    - `ds_planner.py`：规划管理模块
    - `lats_react.py`：树搜索核心类实现


## 数据获取

实验数据集可通过以下方式获取：
1. 从原始项目下载：
   - DA-Agent数据：[GitHub链接](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent)
   - ML Benchmark数据：[Google Drive链接](https://drive.google.com/drive/folders/17SpI9WL9kzd260q2DArbXKNcqhidjA7s)
2. 从百度网盘下载：[百度网盘链接](https://pan.baidu.com/s/18NpXiIaXun6C2IThrUl3zQ?pwd=ipud)

