import json
import re

from metagpt.actions import ExecuteNbCode, Action
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.roles.role import RoleReactMode
from metagpt.schema import Message
from metagpt.utils.common import CodeParser


CODER_PROMPT = """
    Coder. You are a helpful assistant highly skilled in writing code for data analysis. 
    Make sure the generated code is executable.
    While some concise thoughts are helpful, code is absolutely required.
"""


async def execute_code(code: str):
    execute_code = ExecuteNbCode()
    result, success = await execute_code.run(code)
    await execute_code.terminate()
    return result, success


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

    async def run(self, goal: str, working_memory: list[Message] = None, **kwargs):
        REFLECTION_PROMPT = """
            [goal]
            {goal}
            [working_memory]
            {working_memory}
            Output a json following the format:
            ```json
            {{
                "action": str = "terminal or writecode"   # If the overall task has been completed, it should return to terminal, otherwise it returns to writecode
                "content": str = "If the action is terminal, answer the user's question comprehensively. Otherwise, reflect on the cause of the error based on the current context.
                "improved_impl": str = "if the action is terminal, return nothing. Otherwise, return refined code after reflection. And keep in mind that code is NECESSARY"
            }}
            ## remember: code is NECESSARY in <improved_impl>. The code in <improved_impl> must NOT be wrapped in backticks
            """
        reflection_prompt = REFLECTION_PROMPT.format(goal=goal, working_memory=working_memory)
        rsp = await self._aask(reflection_prompt, system_msgs=[CODER_PROMPT])
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp


class ReactRunnerAgent(Role):
    name: str = "wbq"
    profile: str = "ReactRunnerAgent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteCode, Reflect])
        self._set_react_mode(react_mode=RoleReactMode.REACT.value)

    async def _act(self) -> Message:
        goal, overall_success = self.rc.memory.get()[-1].content, False
        code = await WriteCode().run(goal, self.rc.working_memory.get())
        self.rc.working_memory.add(Message(content=code, role=self.profile, cause_by=WriteCode))
        while not overall_success:
            result, success = await execute_code(code)
            print(f"success: {success}, result: {result}")
            self.rc.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))
            reflection = await Reflect().run(goal, self.rc.working_memory.get())
            reflection_msg = Message(content=reflection, role="coder", cause_by=Reflect)
            reflection_json = json.loads(reflection)
            self.rc.working_memory.add(reflection_msg)
            if reflection_json["action"] == "terminal":
                logger.success(f"final answer: {reflection_msg}")
                return reflection_msg
            else:
                code = reflection_json["improved_impl"]
                logger.info(f"new code: {code}")
        raise Exception("unreachable")

    async def _react(self) -> Message:
        rsp = await self._act()
        print(f"_react_rsp: {rsp}")
        final_message = Message(content=json.loads(rsp.content)["content"], role="user", cause_by=Critic)
        return final_message
