CONCLUSION_GOAL_RESULTS = """
# Final goal
{final_goal}  # Achieve this goal using the latest results.

# Tasks Result
{tasks_res}  # # Use only the most recent results from executed tasks.

Provide a direct, concise answer addressing the final goal, adhering to these guidelines:
- Keep it brief and focused.
- Retain all relevant values from the results.
- Do NOT add extra explanations, ideas, or interpretations.
- Do NOT include code or pseudo-code.

# Formatting:
- Avoid any additional characters, Markdown, or extra punctuation.
- Respond in plain text, without wrapping in code blocks.
"""