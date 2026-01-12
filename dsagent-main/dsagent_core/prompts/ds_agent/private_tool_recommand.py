PRIVATE_TOOL_RECOMMENDATION_PROMPT = """
## User Requirement:
{current_task}

## Task
Recommend up to {topk} tools from 'Available Tools' that can help solve the 'User Requirement'. 

## Available Tools:
{available_tools}

## Tool Selection and Instructions:
- Select tools most relevant to completing the 'User Requirement'.
- If you believe that no tools are suitable, indicate with an empty list.
- Only list the names of the tools, not the full schema of each tool.
- Ensure selected tools are listed in 'Available Tools'.
- Output a json list of tool names:
```json
["tool_name1", "tool_name2", ...]
```
"""