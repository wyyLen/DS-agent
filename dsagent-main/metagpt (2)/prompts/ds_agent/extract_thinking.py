EXTRACT_SEQ_FROM_PLAN = """
# Plan
{plan}
Based on the above plan, summarize the sequential characteristics of the subtasks in the plan. 
If there is a sequential relationship between the tasks, they should be in one list and have a sequential order. Otherwise, they should be in different lists.
Output a list of jsons following the format:
```json
[
    [
        "task_type1": str,
        "task_type2": str,
        ...
    ],
    ...
]
```
# suffix
Keep in mind that Your response MUST follow the valid format above.
"""

EXTRACT_DETAIL_FROM_PLAN = """
# Plan
{plan}
# Available Task Types:
{task_type_desc}
Based on the above plan, summarize the hierarchical thinking in the plan.
Output a list of jsons following the format:
```json
[
    [
        {{
            "task_type": "type of this task, should be one of Available Task Types",
            "instruction": "what you should do in this task, one short phrase or sentence",
            "refinement": bool,   # If this task needs to be further split into subtask sequences, refinement is true, otherwise it is false
            "sub_task_list": [
                {{ "sub_task": "what you should do in this subtask, one short phrase or sentence"}},
                {{ "sub_task": "what you should do in this subtask, one short phrase or sentence"}},
                ...
            ]
        }},
        ...
    ],
    ...
]
```
# Note
1. Only if refinement is true, sub_task_list should be in one list and have a sequential order.
2. Try to merge tasks of the same task_type as much as possible to form a subtask list.
"""

COMBINED_STRUCTURED_PLAN_PROMPT_TEMPLATE = """
# Sequential characteristics
{seq}
# Detailed characteristics
{detail}
Based on the above two kind of characteristics, summarize the hierarchical thinking in the plan.
If there is a sequential relationship between the tasks, they should be in one internal list and have a sequential order. Otherwise, they should be in different internal lists.
Output a list of jsons following the format:
```json
[
    [
        {{
            "task_type": str,
            "instruction": "what you should do in this task, one short phrase or sentence",
            "refinement": bool,   # If this task needs to be further split into subtask sequences, refinement is true, otherwise it is false
            "sub_task_list": [
                {{ "sub_task": "what you should do in this subtask, one short phrase or sentence"}},
                {{ "sub_task": "what you should do in this subtask, one short phrase or sentence"}},
                ...
            ]
        }},
        ...
    ],
    ...
]
```
"""

# desc: The above three prompts are used by old versions and are now @deprecated.

EXTRACT_FROM_PLAN_TEMPLATE = """
Assume that you were a proficient data scientist. Below is a data analysis problem and a plan to successfully solve it. 
Please answer the following questions one by one and **as detailed as possible**. Make sure that another data scientist can successfully solve a similar data science analysis problem based on your answers.
(1) Please summarize the overall design of the plan.
(2) Please explain the dependencies between the tasks in the plan.
(3) Do you think there is a pattern of questions in the current problem that needs to be completed based on multiple steps in the plan?
Note that make sure the answers are directly included from the plan, rather than based on your assumption.

```json
{plan}
```
# data analysis problem
{question}
"""

# desc: kaggle using
EXTRACT_FROM_CODE_TEMPLATE = """
Assume that you were a proficient data scientist. Below is a data analysis problem and a plan to successfully solve it.
# Question
{question}
# Code
{code}
# Available Task Types:
# {task_type_desc}
Please answer the following questions one by one and **as detailed as possible**. Make sure that another data scientist can successfully solve a similar data science analysis problem based on your answers.
(1) Based on the Available Task Types above, please summarize the overall design of the code.
(2) Based on your answer to the first question, please explain the dependencies between the tasks in the code.
(3) Do you think there is a pattern of questions in the current problem that needs to be completed based on multiple steps in the plan?
Note that make sure the answers are directly included from the python code, rather than based on your assumption.
Focus on the key steps in the code and ignore non-critical steps such as environment configuration and result submission
"""

EXTRACT_FROM_CODE_IMPROVEMENT = """
Assume you are a proficient data scientist. Below is a data analysis problem and the code that successfully solves it. Your task is to analyze the code, identify its structure and dependencies, and provide actionable insights for improvement. This will help other data scientists understand and solve similar problems effectively.

# Problem Description
{question}
# Solution Code
{code}
# Available Task Types
{task_type_desc}
# Questions to Answer
1. Overall Design Summary
   - Based on the available task types, summarize the high-level design of the code. 
   - Identify the main tasks implemented in the code and describe the purpose and functionality of each task.
   - Highlight any design patterns or best practices used in the code (e.g., modularization, reusable functions, etc.).
2. Dependencies Between Tasks
   - Explain the dependencies between tasks in the code. 
   - Specify which tasks rely on others and how these dependencies shape the workflow.
   - Discuss whether the dependencies are linear (sequential) or parallel (independent), and suggest ways to optimize them if necessary.
3. Multi-step Task Planning
   - Analyze whether the problem requires a multi-step approach to solve. 
   - If so, break down the steps and explain their logical order, necessity, and relationship to the overall solution.
   - Provide recommendations for improving the clarity or efficiency of these steps.
4. Code Optimization Opportunities
   - Identify areas in the code that could be optimized for better performance, readability, or maintainability.
   - Suggest specific improvements, such as refactoring redundant code, using more efficient algorithms, or leveraging libraries/functions that simplify the implementation.
5. Scalability and Generalization
   - Evaluate whether the current solution is scalable to larger datasets or adaptable to similar problems.
   - Propose modifications to make the code more generalizable or robust to different use cases.

# Guidelines for Responses
- **Directly reference the code**: Use specific lines, functions, or modules from the provided code to support your analysis. Avoid making assumptions not grounded in the code.
- **Focus on key steps**: Prioritize the critical tasks and logic in the code. Ignore non-essential aspects like environment setup or result submission.
- **Clarity and accessibility**: Use clear, concise language and avoid unnecessary jargon. If technical terms are used, provide brief explanations.
- **Actionable insights**: Ensure your recommendations are practical and can be directly implemented by another data scientist.
- **Reproducibility**: Structure your response so that another data scientist can replicate the analysis or apply your suggestions to a similar problem.
"""


# note: kaggle used before, now is @deprecated
EXTRACT_WORKFLOW_FROM_CODE_TEMPLATE = """
Assume that you are a skilled data scientist. The following Python code is a high-performance solution to a Kaggle competition. Please summarize the complete plan for solving the problem from the code **as detailedly as possible**, making sure that another data scientist can accurately reproduce this copy of the code based on your answer.
# goal
{question}
# code
{code}
# Available Task Types:
# {task_type_desc}

Output a list of jsons following the format:
```json
[
    {{
        "task_id": '1',
        "dependent_task_ids": [],
        "instruction": "Load and inspect the abalone dataset",
        "task_type": "pda"
    }},
    {{
        "task_id": str = "unique identifier for a task in plan, can be an ordinal",
        "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
        "instruction": "what you should do in this task, one short phrase or sentence",
        "task_type": "type of this task, should be one of Available Task Types",
    }},
    ...
]
```
# Suffix:
- Keep in mind that Your response MUST follow the valid format above.
- If the specific column information of the data is not clear, it is necessary to execute the pda task first.
Focus on the key steps in the code and ignore non-critical steps such as environment configuration and result submission
"""

# desc: kaggle using now
EXTRACT_WORKFLOW_FROM_CODE_IMPROVEMENT = """
Assume that you are a skilled data scientist with extensive experience in solving complex data analysis problems. Your task is to analyze the provided Python code and extract a complete workflow to solve the problem. The goal is to ensure that another data scientist, even with limited reasoning capabilities, can successfully reproduce the solution using your output.

### Requirements:
1. **Workflow Extraction**:
   - Analyze the provided code and break it down into a series of logical tasks. Each task should represent a clear and distinct step in solving the problem.
   - Ensure that each task includes the following:
     - A unique `task_id`.
     - A list of `dependent_task_ids` (if the task depends on the output of previous tasks).
     - A clear `instruction` describing the task's purpose and expected outcome.
     - A `task_type` selected from the `Available Task Types`.

2. **Define Task Dependencies**:
   - For each task, identify its dependencies. If a task relies on the output of a previous task, include the corresponding `task_id` in the `dependent_task_ids`.

3. **Map Tasks to Types**:
   - Use the following `Available Task Types` to categorize each task:
      - `Available Task Types`: {task_type_desc}
   - If a task's type is unclear, prioritize clarity in the instruction and mark it as `other`.

4. **Focus on Key Steps**:
   - Ignore non-critical steps, such as environment configuration or result submission, unless they directly impact the solution logic.

### Input:
- `question`: {question}.
- `code`: {code}.

### Output:
1. **JSON Workflow**:
   - The output should be a JSON object with the following structure:
     ```json
     [
         {{
             "task_id": "1",
             "dependent_task_ids": [],
             "instruction": "Load and inspect the abalone dataset",
             "task_type": "pda"
         }},
         {{
            "task_id": str = "unique identifier for a task in plan, can be an ordinal",
            "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
            "instruction": "A detailed description of what needs to be accomplished in this task. This should include the objective, any specific requirements or constraints, and the expected output or outcome. Avoid mentioning specific code implementations or technical details.",
            "task_type": "type of this task, should be one of Available Task Types",
        }},
        ...
     ]
     ```

2. **Key Notes**:
   - Each task should be concise, clear, and focused on a single logical step.
   - Dependencies between tasks must be explicitly defined.

3. **Example Output**:
   - Refer to the JSON example above for the expected structure and content.

### Additional Notes:
- Focus on the core logic of the code and ignore non-critical steps such as environment setup or result submission.
- If the specific column information of the dataset is unclear, the first task should involve data inspection (`pda`).
- Use clear and accessible language to ensure the output is understandable to less sophisticated agents.
"""

EXTRACT_WORKFLOW_FROM_EXP_TEMPLATE = """
Assume that you were a proficient data scientist. You need to extract the appropriate workflow paradigm from existing empirical knowledge.
# experience
{exp}
# Available Task Types:
{task_type_desc}
Based on the above experience, please extract the appropriate workflow paradigm from existing empirical knowledge.
Output a json following the format:
```json
[
    {{
        "workflow": "[['task_type',...],['task_type',...],...]"        # task_type should be one of Available Task Types
        "exp": "Summarize the general task paradigm scenario and solution"
    }},
    ...
]
```
below is an example
{workflow_example}

# suffix
Keep in mind that Your response MUST follow the valid format above.
"""

EXTRACT_WORKFLOW_FROM_EXP_TEMPLATE_COT = """
Assume that you were a proficient data scientist. You need to extract the appropriate workflow paradigm to solve a general data science problem from existing empirical knowledge.
# experience
{exp}
# Available Task Types:
{task_type_desc}

Base on the 3rd part of the "solution" in the above experience, please extract the appropriate workflow paradigm to solve the general data science problem from existing empirical knowledge.
If you think there is no any workflow paradigm according to the above experience, please just output an empty list.
Output a json following the format:
```json
[
    {{
        "workflow": "[['task_type',...],['task_type',...],...]"
        "exp": "Summarize the general task paradigm scenario and solution"
    }},
    ...
]
```
Below is an example for a workflow paradigm with its specific question and solution:
{workflow_example}
# suffix
- Keep in mind that Your response MUST follow the valid format above.
"""

EXTRACT_WORKFLOW_FROM_PLAN_TEMPLATE = """
Assume that you were a proficient data scientist. Below is a data analysis problem and a plan to successfully solve it. 
# Question
{question}
# Plan
{plan}

Do you think there is a special pattern of questions in the current problem that needs to be completed based on multiple steps in the plan?
If so, please summarize the general workflow paradigm and its solution.
Below is an example:
{workflow_example}

Output a json following the format:
```json
[
    {{
        "workflow": "[['task_type',...],['task_type',...],...]"
        "exp": "Summarize the general task paradigm scenario and solution"
    }},
    ...
]
```

# suffix
- Keep in mind that Your response MUST follow the valid format above.
- If you think there is no pattern of questions in the current problem that needs to be completed based on multiple steps in the plan, please just output an empty list.
- Make sure the workflow preparation describes the workflow of the current task paradigm without any redundant tasks
"""

EXTRACT_WORKFLOW_FROM_PLAN_TEMPLATE_COT = """
Assume that you were a proficient data scientist. 
# Question
{question}

First, Do you think there is a special general pattern of questions in the above data analysis problem that that requires some specific steps to complete?
If so, please summarize the solution of the above general working paradigm from the plan below. Otherwise, please just output an empty list.
# Plan
{plan}

Below is a workflow paradigm example:
{workflow_example}

Output a json following the format:
```json
[
    {{
        "workflow": "[['task_type',...],['task_type',...],...]"
        "exp": "Summarize the general task paradigm scenario and solution"
    }},
    ...
]
```

# suffix
- Keep in mind that Your response MUST follow the valid format above.
- If you think there is no pattern of questions in the current problem that needs to be completed based on multiple steps in the plan, please just output an empty list.
- Make sure the workflow preparation describes the workflow of the current task paradigm without any redundant tasks
- Avoid complete planning as a workflow paradigm
"""

workflow_example = """
[
    {
         "workflow": "[['machine learning','machine learning'],['machine learning']]"
         "exp": "For the general task paradigm of analyzing the impact of a specific feature on the accuracy of a model, two machine learning tasks should be performed first. These two machine learning tasks generate models using this feature and not using this feature respectively. Then a machine learning task should be performed to calculate the accuracy of the two models respectively."
    }
]
"""

EXTRACT_GOAL_FROM_COMP_DESC = """
You are a proficient data scientist. You need to summarize the data science problem given by the user as concisely and accurately as possible. Be careful not to output irrelevant content.
Below is an example description of a data science analysis problem:
{problem_example}

# Problem
{desc}

# Output Format (MUST FOLLOW):
```
File: "related file names."
Question: "detailed data science analysis task objectives"
Constraints: "all more specific requirements and constraints"
```
# Rules:
1. Language: Always respond in ENGLISH.
2. If any information is missing or unclear, use "Unknown" or "Not specified" in the response.
3. Ensure the response is concise, accurate, and strictly follows the format above.
"""

problem_example = """
File: abalone.csv
Question: Explore the correlation between the length and the weight of the whole abalone. Additionally, perform feature engineering by creating a new feature called "volume" by multiplying the length, diameter, and height of the abalone. Determine if the volume feature improves the accuracy of predicting the number of rings using a linear regression model.
Constraints: Calculate the Pearson correlation coefficient to assess the strength and direction of the linear relationship between length and the weight. The volume feature should be created by multiplying the length, diameter, and height of the abalone. Use the sklearn's linear regression model to predict the number of rings. Split the data into a 70% train set and a 30% test set. Evaluate the models by calculating the root mean squared error (RMSE) with the test set.
"""

# desc: agent using
EXTRACT_EXP_FROM_WORKING_MEMORY_TEMPLATE = """
Assume that you were a proficient data scientist with a rich history of solving complex data analysis problems. Below is a data analysis problem, a plan to solve it, and relevant memory of the attempts, adjustments, and reflections made during past tasks.

Please answer the following questions as thoroughly as possible to provide insights that another data scientist, even with limited reasoning capabilities, could use to solve a similar problem successfully, taking advantage of both successes and errors encountered:

(1) Summarize the overall design of the plan, highlighting any adjustments made to handle challenges or unexpected issues encountered in similar past cases.
(2) Explain the dependencies between tasks in the plan. Describe how each task outcome was anticipated to affect the next, and include any lessons from similar experiences where unexpected dependencies or bottlenecks required adjustment.
(3) Reflect on key trial-and-error experiences relevant to this plan. Describe specific modifications or alternate approaches used in similar situations, and the insights gained about efficient task ordering, potential pitfalls, and valuable checkpoints.
(4) Identify any patterns or steps in the current problem that would benefit from a sequence of tasks. Provide examples of when similar patterns emerged and how they influenced the final outcome.
(5) Provide additional insights from memory on refining the task instructions or managing dependencies that could help prevent similar errors in future workflows.
(6) Based on your answers, detail the core steps in the simplest language possible so that a less sophisticated agent can use your insights to accurately generate a plan. Clarify each task’s essential goal and any specific checks or considerations that are crucial to its successful completion.

Make sure that your answers draw directly from both the current plan and relevant prior experiences included in the agent’s memory, emphasizing learned strategies, error corrections, and adjustments, using clear and accessible language for easier reference.

# Question
{question}

# Working Memory
{working_memory}
"""

EXTRACT_EXP_FROM_WORKING_MEMORY_IMPROVEMENT = """
Assume that you are a proficient data scientist with extensive experience solving complex data analysis problems. Below is a data analysis problem, a proposed plan to solve it, and relevant memory of previous attempts, including challenges faced, solutions applied, and reflections. Use this information to provide a detailed analysis and actionable insights.
Your goal is to help another data scientist, even with limited reasoning capabilities, successfully solve a similar problem by leveraging both successes and errors from past experiences.

### Questions:
(1) **Plan Summary**: What is the overall design of the plan? Highlight key design elements and any adjustments made to address challenges or unexpected issues in similar past cases.
(2) **Task Dependencies**: How do tasks in the plan depend on each other? Describe how each task outcome affects the next and include lessons from past experiences where unexpected dependencies or bottlenecks required adjustment.
(3) **Trial-and-error Insights**: What modifications or alternative approaches were used in similar situations? What insights were gained about efficient task ordering, potential pitfalls, and valuable checkpoints?
(4) **Patterns and Sequences**: Are there recurring patterns or steps in the current problem? Provide examples of when similar patterns emerged and how they influenced the final outcome.
(5) **Refinements**: How can task instructions or dependencies be improved to avoid common errors in future workflows?
(6) **Core Steps**: Based on your answers, detail the simplest, most essential steps to solve this problem. Clarify each task’s goal, key checks, and considerations.
(7) **Problem Type**: What type of data analysis problem is this (e.g., classification, regression, clustering, time series)? How does this influence the choice of methods and task sequence?

### Output Format:
1. Plan Summary:
   - [Key design elements]
   - [Adjustments made]
2. Task Dependencies:
   - [Dependency 1: Task A → Task B]
   - [Lessons learned]
3. Trial-and-error Insights:
   - [Example 1: Problem → Adjustment → Outcome]
   - [Key takeaways]
4. Patterns and Sequences:
   - [Pattern 1: Description → Impact]
   - [Example]
5. Refinements:
   - [Improvement 1: Description]
6. Core Steps:
   - Step 1: [Task description, goal, and key checks]
   - Step 2: ...
7. Problem Type:
   - [Type and its influence on the plan]

Make sure that your answers draw directly from both the current plan and relevant prior experiences included in the agent’s memory, emphasizing learned strategies, error corrections, and adjustments, using clear and accessible language for easier reference.
# Question
{question}
# Working Memory
{working_memory}
"""