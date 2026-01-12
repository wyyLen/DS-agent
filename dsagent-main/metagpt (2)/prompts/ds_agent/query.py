GET_QUESTION_TYPE_PROMPT = """
# Question
{question}
# Available Task Types:
{task_type_desc}
Analyze the above user question and the given task_type_desc to determine which task types are included in the question.
Output a json following the format:
```json
["task_type1", "task_type2", ...]    
```
# Suffix:
- Keep in mind that Your response MUST follow the valid format above.
- every task_type in the list should be one of Available Task Types
"""

GET_QA_TYPE_PROMPT_TEMPLATE = """
# Question
{question}
# Solution
{solution}
# Available Task Types:
{task_type_desc}
Analyze the above user question and the given task_type_desc to determine which task types are included in the question.
```json
["task_type1", "task_type2", ...]  
```
# Suffix:
- Keep in mind that Your response MUST follow the valid format above.
- every task_type in the list should be one of Available Task Types
"""