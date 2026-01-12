from __future__ import annotations

import json
import re

from metagpt.actions import Action
from metagpt.logs import logger
from dsagent_core.prompts.ds_agent.write_ds_code import WRITE_DS_CODE_PROMPT, DS_AGENT_SYSTEM_MSG, REFLECTION_PROMPT, \
    REFLECTION_SYSTEM_MSG, WRITE_DS_CODE_PROMPT_WITH_TOOL, REFLECTION_PROMPT2, CHECK_DATA_PROMPT
from metagpt.schema import Message, Plan
from metagpt.utils.common import CodeParser, remove_comments


def clean_improved_impl(rsp: str) -> str:
    # note: not used yet
    if rsp.count('`') <= 2:
        return rsp
    pattern = r"```python(.*?)```"
    cleaned_text = re.sub(pattern, lambda m: m.group(1), rsp, flags=re.DOTALL)
    pattern = rf"```json\s+(.*?)```"
    match = re.search(pattern, cleaned_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        res = f"```json\n{repr(code)[1:-1]}\n```"
        return res
    else:
        logger.error(f"{pattern} not match following text:")
        logger.error(cleaned_text)
        res = f"```json\n{repr(cleaned_text)[1:-1]}\n```"
        return res


class WriteDsCode(Action):
    async def _debug_with_reflection(self, context: list[Message], working_memory: list[Message]):
        # note: outdated
        reflection_prompt = REFLECTION_PROMPT.format(context=context, previous_impl=working_memory)
        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
        try:
            reflection = json.loads(CodeParser.parse_code(block=None, text=rsp))
        except json.JSONDecodeError:
            rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
            reflection = json.loads(CodeParser.parse_code(block=None, text=rsp))
        return reflection["improved_impl"]

    async def _improved_debug_with_reflection(self, context: list[Message], previous_impl: list[Message] = None):
        print("--- _improved_debug_with_reflection ---")
        reflection_prompt = REFLECTION_PROMPT2.format(context=context, previous_impl=previous_impl)
        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
        improved_impl = CodeParser.parse_code(block=None, text=rsp)
        return improved_impl

    async def run(self, user_requirement: str, plan_status: str = "", tool_info: str = "",
                  working_memory: list[Message] = None, use_reflection: bool = False, **kwargs, ) -> str:
        if tool_info == "":
            write_ds_code_prompt = WRITE_DS_CODE_PROMPT.format(user_requirement=user_requirement,
                                                               plan_status=plan_status)
        else:
            print("using tools to generate code. tool infos:", tool_info)
            print("---- tool info over ----")
            write_ds_code_prompt = WRITE_DS_CODE_PROMPT_WITH_TOOL.format(user_requirement=user_requirement,
                                                                         plan_status=plan_status, tool_info=tool_info)
        working_memory = working_memory or []
        context = self.llm.format_msg([Message(content=write_ds_code_prompt, role="user")] + working_memory[:-2])
        if use_reflection:
            code = await self._improved_debug_with_reflection(context=context, previous_impl=working_memory[-2:])
        else:
            rsp = await self.llm.aask(context, system_msgs=[DS_AGENT_SYSTEM_MSG], **kwargs)
            code = CodeParser.parse_code(block=None, text=rsp)
        return code


class CheckData(Action):
    async def run(self, plan: Plan):
        finished_tasks = plan.get_finished_tasks()
        code_written = [remove_comments(task.code) for task in finished_tasks]
        code_written = "\n\n".join(code_written)
        prompt = CHECK_DATA_PROMPT.format(code_written=code_written)
        rsp = await self._aask(prompt)
        code = CodeParser.parse_code(block=None, text=rsp)
        return code
