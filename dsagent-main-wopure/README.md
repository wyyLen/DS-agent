<div align="center">
<h1 align="center">dsagent</h1>

English / [简体中文](./README_CN.md)

</div>

<div align="left">

Welcome to the official repository of the **dsagent** project. **dsagent** is a large language model agent system designed for data science tasks, aiming to enhance the reasoning and decision-making capabilities of language models in complex data science scenarios. While current large language models (LLMs) can execute various tasks through natural language interaction, existing approaches—such as direct code generation, ReAct, and CoT—often encounter issues like incomplete steps, inaccurate instructions, overgeneralization, or missing key process elements when dealing with data science problems that feature complex dependencies, highly nonlinear workflows, and strict task constraints.

To address these challenges, **dsagent** introduces two key technical innovations:

1. **Workflow-centric Retrieval-Augmented Planning**: By building and leveraging a data science experience knowledge base, dsagent combines both textual and workflow knowledge to dynamically enhance the agent’s abilities in task decomposition and execution. This enables efficient, knowledge-driven, and closed-loop task planning, significantly improving reliability and generalizability.
2. **Tree Search-based Autonomous Exploration Mechanism**: For complex tasks, dsagent models both data science workflows and the exploration space, utilizing an efficient process tree search strategy, reflection mechanisms, and node optimization to greatly enhance the agent’s capacity for autonomous exploration and problem-solving in vast solution spaces.

The system comes with a rich built-in knowledge base of data science experiences and supports three agent invocation modes: planning execution, autonomous exploration, and dynamic invocation. dsagent also provides a web-based interactive interface, empowering data scientists and analysts with end-to-end, natural language-enabled data science analysis, and facilitating efficient and reliable automation of data science workflows.



## Requirements

- `Python 3.9+`
- see development requirements in `/requirements-dev.txt`

</div>



## Quick-Start

[check our quick-start demo HERE](./quick_start.md)

To understand the codebase structure, please refer to the [Code Documentation](./docs/folder_structure.md).
