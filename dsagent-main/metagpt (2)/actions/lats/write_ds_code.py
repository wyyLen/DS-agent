from __future__ import annotations

import json

from scipy.special import kwargs

from metagpt.actions import Action
from metagpt.prompts.ds_agent.write_ds_code import WRITE_DS_CODE_PROMPT, DS_AGENT_SYSTEM_MSG, REFLECTION_PROMPT, \
    REFLECTION_SYSTEM_MSG, WRITE_DS_CODE_PROMPT_WITH_TOOL
from metagpt.schema import Message
from metagpt.utils.common import CodeParser


class WriteDsCodeLATS(Action):
    async def _debug_with_reflection(self, context: list[Message], working_memory: list[Message]):
        reflection_prompt = REFLECTION_PROMPT.format(context=context, previous_impl=working_memory)
        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
        reflection = json.loads(CodeParser.parse_code(block=None, text=rsp))
        return reflection["improved_impl"]

    async def run(self, user_requirement: str, working_memory: list[Message] = None, use_reflection: bool = False, **kwargs, ) -> str:
        write_ds_code_prompt = WRITE_DS_CODE_PROMPT.format(user_requirement=user_requirement)
        working_memory = working_memory or []
        context = self.llm.format_msg([Message(content=write_ds_code_prompt, role="user")] + working_memory)
        if use_reflection:
            code = await self._debug_with_reflection(context=context, working_memory=working_memory)
        else:
            rsp = await self.llm.aask(context, system_msgs=[DS_AGENT_SYSTEM_MSG], **kwargs)
            code = CodeParser.parse_code(block=None, text=rsp)
        return code

    async def self_reflection(self, cur_func_impl, feedback) -> str:
        reflection_prompt = REFLECTION_PROMPT.format(context=feedback, previous_impl=cur_func_impl)
        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
        reflection = json.loads(CodeParser.parse_code(block=None, text=rsp))
        return reflection["reflection"]

    async def func_impl(self, goal, prev_func_impl, feedback, self_reflection, accumulated_feedback, accumulated_reflection) -> str:
        PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
        USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:
        [previous impl]:
        ```python
        def add(a: int, b: int) -> int:
            """
            Given integers a and b, return the total value of a and b.
            """
            return a - b
        ```

        [unit test results from previous impl]:
        Tested passed:

        Tests failed:
        assert add(1, 2) == 3 # output: -1
        assert add(1, 2) == 4 # output: -1

        [reflection on previous impl]:
        The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

        [improved impl]:
        ```python
        def add(a: int, b: int) -> int:
            """
            Given integers a and b, return the total value of a and b.
            """
            return a + b
        ```
        '''

        messages = [
            Message(role="system", content=f"{PY_REFLEXION_CHAT_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}"),
            Message(role="user", content=PY_REFLEXION_FEW_SHOT_ADD)
        ]

        def add_code_block(string: str, lang: str) -> str:
            return f"```{lang}\n{string}\n```"
        add_code_block = lambda x: add_code_block(x, "python")

        for impl, feedback, reflection in zip(prev_func_impl, accumulated_feedback, accumulated_reflection):
            messages.append(Message(role="assistant", content=add_code_block(impl)))
            messages.append(Message(role="user",
                                    content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{reflection}"))

        messages.append(Message(role="user", content=f"[improved impl]:\n{goal}"))
        # Build the accumulated context from the provided feedback and reflections
        accumulated_context = "\n\n".join(
            [f"[previous impl {i + 1}]:\n{add_code_block(impl)}\n[unit test results from previous impl {i + 1}]:\n{feedback}\n[reflection on previous impl {i + 1}]:\n{reflection}"
                for i, (impl, feedback, reflection) in enumerate(zip(prev_func_impl, accumulated_feedback, accumulated_reflection))]
        )
        def message2dict(message: Message) -> dict:
            return {"role": message.role, "content": message.content}
        messages_dict = [message2dict(m) for m in messages]
        code = await self.llm.aask(messages_dict, system_msgs=[DS_AGENT_SYSTEM_MSG], **kwargs)
        return code
