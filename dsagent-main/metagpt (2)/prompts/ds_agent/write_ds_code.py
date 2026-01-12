DS_AGENT_SYSTEM_MSG = """
As a data scientist, you need to help user to achieve their goal step by step in a continuous Jupyter notebook. 
Since it is a notebook environment, don't use asyncio.run. Instead, use await if you need to call an async function.
"""

WRITE_DS_CODE_PROMPT = """
# User Requirement
{user_requirement}

# Plan Status
{plan_status}

# Constraints
- Take on Current Task if it is in Plan Status, otherwise, tackle User Requirement directly.
- Ensure the output new code is executable in the same Jupyter notebook as the previous executed code.

# Output
While some concise thoughts are helpful, code is absolutely required. Always output one and only one code block in your response. Output code in the following format:
```python
your code
```
"""

WRITE_DS_CODE_PROMPT_WITH_TOOL = """
# User Requirement
{user_requirement}

# Plan Status
{plan_status}

# Tool Info
{tool_info}

# Constraints
- Take on Current Task if it is in Plan Status, otherwise, tackle User Requirement directly.
- Ensure the output new code is executable in the same Jupyter notebook as the previous executed code.
- Always prioritize using pre-defined tools for the same functionality.

# Output
While some concise thoughts are helpful, code is absolutely required. Always output one and only one code block in your response. Output code in the following format:
```python
your code
```
"""

REFLECTION_SYSTEM_MSG = """
You are an AI Python assistant. You will be given your previous implementation code of a task, runtime error results, and a hint to change the implementation appropriately. Write your full implementation.
"""

REFLECTION_PROMPT = """
[context]
{context}

[previous impl]:
{previous_impl}

[instruction]
Analyze your previous code and error in [context] step by step, provide me with improved method and code. Remember to follow [context] requirement. Don't forget to write code for steps behind the error step.
Output a json following the format:
```json
{{
    "reflection": str = "Reflection on previous implementation",
    "improved_impl": str = "Refined code after reflection. Ensure the code is necessary, not wrapped in '```', and DO NOT include any line breaks, double quotes or characters that may conflict with JSON format. The code should be a single line of valid code.",
}}
## remember: code is NECESSARY in <improved_impl>. The code in <improved_impl> must NOT be wrapped in backticks.
```
"""

REFLECTION_PROMPT2 = """
[context]
{context}

[previous impl]:
{previous_impl}

Step-by-step, analyze the previous implementation in [context]. Identify the error and its cause, and reflect on how the code can be improved. For each step behind the error, explain what needs to be changed and why. Provide the refined code that addresses these issues.

- Start with identifying the issue in the previous implementation.
- Explain the analysis and thought process for each step.
- Address the error and propose a solution for each specific part of the code that needs improvement.
- Do not simply output the final code; provide clear explanations for each step of improvement.

Remember to follow [context] requirements. Ensure the code is correct, and write code for all the necessary steps after the error step.

Output the reflection and improved code as follows:
1. **Analysis and Reflection**: Briefly explain the issue and your approach to solving it.
2. **Improved Code**: Provide the revised code in a code block (no markdown, only the code).
"""


CHECK_DATA_PROMPT_improved = """
# Background
Check the latest data info to guide subsequent tasks. If the data is too large to load into memory (considering a 32GB RAM environment with redundancy), adjust subsequent tasks to load only a portion of the data or provide alternative strategies.

## Finished Tasks
```python
{code_written}
```end

# Task
Check the code in finished tasks and print key variables to guide your following actions. Specifically:
1. If it is a data analysis or machine learning task:
   - Check the size of the dataset (number of rows, columns, and memory usage).
   - Use the following guidelines to determine if the dataset is too large:
     - The dataset exceeds **50,000 rows** or **10,000 columns** (for tasks like correlation matrix or heatmap computation).
     - The dataset's memory usage exceeds **16GB** (to leave redundancy in a 32GB RAM environment).
   - If the dataset is too large, provide code to:
     - Load the data in chunks for sequential processing.
     - Sample a subset of the data for exploratory analysis or visualization.
   - Print the latest column information using the following code, with the DataFrame variable from 'Finished Tasks' in place of `df`:
   ```python
   from metagpt.tools.libs.data_preprocess import get_column_info

   column_info = get_column_info(df)
   print("column_info")
   print(column_info)
   ```end

2. If it is NOT a data analysis task:
   - Print out any key variables you see fit.

3. Return an empty string if you think there is no important data to check.

# Constraints:
- Your code is to be added to a new cell in Jupyter.
- If the dataset size exceeds the above thresholds, provide alternative strategies to handle the data efficiently:
  - For row-intensive tasks (e.g., large row counts), use chunk loading or random sampling.
  - For column-intensive tasks (e.g., wide datasets), suggest dimensionality reduction techniques (e.g., PCA or feature selection).
- Ensure that any provided code is optimized for a 32GB RAM environment with redundancy.

# Instruction
Output code following the format:
```python
your code
```end

# Example Code for Large Dataset Handling:
If the dataset is too large to load into memory, use the following logic:
```python
import pandas as pd

# Example: Load data in chunks for row-intensive datasets
chunk_size = 100000  # Adjust chunk size as needed
for chunk in pd.read_csv("your_large_file.csv", chunksize=chunk_size):
    # Perform analysis on each chunk
    print(chunk.head())

# Example: Load a random sample for exploratory analysis
sampled_df = pd.read_csv("your_large_file.csv", nrows=100000)
print(sampled_df.head())

# Example: Dimensionality reduction for wide datasets
from sklearn.decomposition import PCA

# Assuming df is already loaded and too wide (e.g., >10,000 columns)
pca = PCA(n_components=100)  # Reduce to 100 principal components
reduced_df = pca.fit_transform(df)
print("Reduced dataset shape:", reduced_df.shape)
```end
"""

CHECK_DATA_PROMPT = """
# Background
Check latest data info to guide subsequent tasks.

## Finished Tasks
```python
{code_written}
```end

# Task
Check code in finished tasks, print key variables to guide your following actions.
Specifically, if it is a data analysis or machine learning task, print the the latest column information using the following code, with DataFrame variable from 'Finished Tasks' in place of df:
```python
from metagpt.tools.libs.data_preprocess import get_column_info

column_info = get_column_info(df)
print("column_info")
print(column_info)
```end
Otherwise, print out any key variables you see fit. Return an empty string if you think there is no important data to check.

# Constraints:
- Your code is to be added to a new cell in jupyter.

# Instruction
Output code following the format:
```python
your code
```
"""

DATA_INFO = """
# Latest Data Info
Latest data info after previous tasks:
{info}
"""

FAILURE_TRAJECTORY_ANALYSIS_PROMPT = '''
You are an expert AI system analyst tasked with analyzing the reasons for a failed trajectory in an agent's tree search process. Your role is to summarize the **core reasons** why the current path has failed to achieve the overall task goal and provide actionable insights for improving future exploration. The analysis should focus on identifying specific issues, errors, or limitations in the current trajectory that hindered progress.

### **Instructions**:
1. Carefully analyze the provided conversation log of the failed trajectory.
2. Identify and summarize the **core reasons** for the failure, such as:
   - Logical errors or incorrect assumptions.
   - Missing or incomplete steps in the process.
   - Misalignment with the overall task goal.
   - Inefficient strategies or redundant actions.
   - Any other relevant factors that contributed to the failure.
3. Provide actionable insights or recommendations for improving future exploration, ensuring alignment with the overall task goal.

### **Input**:
- **Overall Task Goal**: {goal}
- **Failed Trajectory Conversation Log**: {context}

### **Output Format**:
Your output should have the following structure:

1. **Core Reasons for Failure**:
   - [Reason 1: Brief description of the issue, e.g., logical error or missing data.]
   - [Reason 2: Brief description of the issue, e.g., misalignment with task goal.]
   - [Reason 3: Brief description of the issue, if applicable.]

2. **Actionable Insights for Future Exploration**:
   - [Insight 1: Specific recommendation for addressing the identified issue(s).]
   - [Insight 2: Specific recommendation for improving alignment with the task goal.]
   - [Insight 3: Additional suggestions for optimizing the exploration process, if applicable.]

### **Important Notes**:
- Be concise and specific in your reasoning and recommendations.
- Focus on providing actionable insights that directly address the identified failure points.
- Avoid generic advice; tailor your analysis to the provided goal and trajectory.

### Example Output:

1. **Core Reasons for Failure**:
   - Reason 1: The trajectory relied on an incorrect assumption about the data structure, leading to invalid intermediate results.
   - Reason 2: A critical preprocessing step was skipped, resulting in incomplete input for subsequent analysis.
   - Reason 3: The actions taken were not aligned with the overall goal, leading to redundant exploration.

2. **Actionable Insights for Future Exploration**:
   - Insight 1: Verify assumptions about the data structure at the start of the trajectory to avoid logical errors.
   - Insight 2: Ensure all preprocessing steps are completed before proceeding to the next stage of analysis.
   - Insight 3: Regularly evaluate whether actions are directly contributing to the overall task goal to avoid redundant efforts.

Now, based on the provided input, analyze the failed trajectory and generate the output in the required format.
'''

FAILURE_NODE_ANALYSIS_PROMPT = '''
You are an expert in analyzing iterative problem-solving processes, specialized in identifying failure patterns and proposing actionable improvements. Your task is to carefully analyze the provided content, which includes multiple rounds of reflections on a failed node in a tree search process. The content consists of code attempts and corresponding observations from each iteration.

Your goal is to:
1. **Identify the core reasons** why the current node has failed, even after multiple rounds of reflection and adjustment.
2. **Summarize the recurring issues or bottlenecks** that have prevented success, if any patterns are evident.
3. Provide **clear and actionable suggestions** for how future nodes can overcome these challenges or explore alternative strategies. Your suggestions should be practical, specific, and directly address the identified failure reasons.

### Input:
{context}

### Output:
Your output should include:
1. **Summary of Failure Reasons**:
   - A concise summary of the key reasons why the current approach has failed. Highlight any recurring issues or patterns that have emerged across iterations.
2. **Actionable Suggestions**:
   - Provide specific, actionable recommendations for how the next exploration steps can avoid these failures and improve the likelihood of success. Suggestions should be framed as guidance for future node generation.

### Example Output:
**Summary of Failure Reasons**:
1. The feature selection process consistently included irrelevant or redundant features, leading to poor model performance.
2. The evaluation metric used was not aligned with the problem objective, resulting in misleading performance assessments.
3. Hyperparameter tuning was insufficient, as the search space was too narrow to identify optimal configurations.
4. Data preprocessing steps failed to address missing values effectively, causing downstream issues in training.

**Actionable Suggestions**:
1. Implement a more robust feature selection method, such as Recursive Feature Elimination (RFE), to eliminate irrelevant features.
2. Reassess the evaluation metric to ensure it aligns with the problem objective. For example, consider using RMSE instead of MAE if large errors are more critical to minimize.
3. Expand the hyperparameter search space and consider using a randomized search or Bayesian optimization for better coverage.
4. Address missing values more effectively by using imputation techniques (e.g., mean, median, or model-based imputation) before proceeding with modeling.
5. If previous approaches fail to generalize, consider exploring entirely different model architectures or algorithms.

Note:
- Be specific and avoid generic recommendations. Tailor your suggestions based on the content provided.
- If there are no clear recurring patterns, suggest exploratory strategies to uncover new insights or directions.
- Your output will be used as a guiding prompt for future node generation, so ensure clarity and actionable focus.
'''

LATS_REFLECTION_PROMPT = """
[context]
{context}

[previous impl]:
{previous_impl}

[layer reflections]:
{layer_reflections}

Step-by-step, analyze the previous implementation in [context] and incorporate insights from [layer reflections]. Your task is to:
1. Identify the key issues in the previous implementation and their root causes, referencing any recurring patterns or problems highlighted in [layer reflections].
2. Reflect on why the previous adjustments failed, based on the [layer reflections], and explain how these insights will shape your new approach.
3. Propose a refined solution by addressing the specific issues step-by-step. For each step:
   - Clearly explain what needs to be changed and why, considering both [context] requirements and insights from [layer reflections].
   - Provide a revised implementation that resolves the identified issues.

### Instructions:
- Start with a detailed **Analysis and Reflection** section:
  - Summarize the key issues in the previous implementation.
  - Highlight any recurring patterns or failures noted in [layer reflections].
  - Explain your thought process for addressing these issues and how your solution avoids repeating past mistakes.
- Provide the **Improved Code** section:
  - Include only the revised code necessary to address the identified issues.
  - Ensure the code aligns with the [context] requirements and resolves the problems effectively.

### Output Format:
1. **Analysis and Reflection**: Briefly explain the issue, reference relevant insights from [layer reflections], and describe your approach to solving the problem step-by-step.
2. **Improved Code**: Provide the revised code in a code block (no markdown, only the code).

### Notes:
- Use [layer reflections] to inform your analysis and avoid repeating mistakes. Reference specific points in [layer reflections] when applicable.
- Ensure your solution is clear, practical, and aligned with the [context] requirements.
- Do not simply output the final code; provide explanations for each step of improvement to ensure the reasoning is understood.
"""
