from metagpt.actions import Action
from metagpt.schema import Message


class Reflect(Action):
    async def run(self, input, candidate):
        # todo: Reflect and grade the assistant response to the user question below.
        raise NotImplementedError


class InitialAnswer(Action):
    async def run(self, input):
        # todo: GenerateInitialCandidate
        raise NotImplementedError
