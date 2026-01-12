import asyncio
import json
import re
import subprocess

import fire
from pydantic import Field

from metagpt.actions import Action, ExecuteNbCode
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.utils.common import CodeParser


async def execute_code(code: str):
    execute_code = ExecuteNbCode()
    result, success = await execute_code.run(code)
    await execute_code.terminate()
    return result, success


CODER_PROMPT = """
    Coder. You are a helpful assistant highly skilled in writing code for data analysis. 
    Make sure the generated code is executable.
    While some concise thoughts are helpful, code is absolutely required.
"""


class WriteCode(Action):
    name: str = "CodeWriter"
    profile: str = "Generate code based on user questions other external prompts"
    # execute_code: ExecuteNbCode = Field(default_factory=ExecuteNbCode, exclude=True)

    async def run(self, content: str, working_memory: list[Message] = None, **kwargs):
        rsp = await self._aask(content, system_msgs=[CODER_PROMPT])
        code = self.parse_code(rsp)
        return code

    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text


class Reflect(Action):
    name: str = "Reflector"
    profile: str = "Regenerate code based on error messages and user questions"

    async def run(self, content: str, working_memory: list[Message] = None, **kwargs):
        REFLECTION_PROMPT = """
            [context]
            {context}
            [previous impl]
            {previous_impl}
            Analyze your previous code and error in [context]. Output the corrected python code:
            ```python
            [your code]
            ```
            """
        reflection_prompt = REFLECTION_PROMPT.format(context=content, previous_impl=working_memory)
        rsp = await self._aask(reflection_prompt, system_msgs=[CODER_PROMPT])
        improved_impl = CodeParser.parse_code(block=None, text=rsp)
        return improved_impl


COMMENT_MSG = """
Critic. You are a helpful assistant who analyzes the next steps based on the results of the code execution.
"""


class Critic(Action):
    name: str = "Critic"
    profile: str = "Critic"

    async def run(self, content: str, working_memory: list[Message] = None, **kwargs):
        prompt_template = """
        [goal]
        {goal}
        [working_memory]
        {working_memory}
        Based on overall goals and working memory, decide whether to terminal coding and return, or continue generating code. Note that you only need to complete the overall goal and should not pay extra attention to code robustness.
        Output a json following the format:
        ```json
        {{
            "action": str = "terminal or writecode"   # If the overall task has been completed, it should return to terminal, otherwise it returns to writecode
            "content": str = "If the action is terminal, answer the user's question comprehensively. Otherwise, reflect on the cause of the error based on the current context."
        }}
        ```
        """
        prompt = prompt_template.format(goal=content, working_memory=working_memory)
        rsp = await self._aask(prompt, system_msgs=[COMMENT_MSG])
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp


class ReactAgent(Role):
    name: str = "wbq"
    profile: str = "ReactAgent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteCode, Reflect])
        self._set_react_mode(react_mode=RoleReactMode.REACT.value)

    async def _act(self) -> Message:
        goal, overall_success = self.rc.memory.get()[-1].content, False
        code = await WriteCode(config=self.config).run(goal, self.rc.working_memory.get())
        self.rc.working_memory.add(Message(content=code, role=self.profile, cause_by=WriteCode))
        while not overall_success:
            result, success = await execute_code(code)
            print(f"success: {success}, result: {result}")
            self.rc.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))
            comment = await Critic(config=self.config).run(goal, working_memory=self.rc.working_memory.get())
            comment_msg = Message(content=comment, role="critic", cause_by=Critic)
            comment_json = json.loads(comment)
            self.rc.working_memory.add(comment_msg)
            if comment_json["action"] == "terminal":
                logger.success(f"final answer: {comment_msg}")
                return comment_msg
            else:
                critic_comment = comment_json["content"]
                code = await Reflect(config=self.config).run(critic_comment, self.rc.working_memory.get())
                self.rc.working_memory.add(Message(content=code, role="coder", cause_by=Reflect))
        raise Exception("unreachable")

    async def _react(self) -> Message:
        rsp = await self._act()
        print(f"_react_rsp: {rsp}")
        final_message = Message(content=json.loads(rsp.content)["content"], role="user", cause_by=Critic)
        return final_message
