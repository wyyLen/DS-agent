import asyncio

from examples.experiment.da_bench.util.DABENCH import DABench
from metagpt.actions.ds_agent.fixed_plan_for_test import get_fixed_plan
from metagpt.strategy.lats_react import LanguageAgentTreeSearch

if __name__ == "__main__":
    bench = DABench()
    task = bench.generate_formatted_prompt(0)
    print(f"{task}\n")
    # print(f"the correct answer is {bench.answers.get(549)}\n")
    lats = LanguageAgentTreeSearch(goal=task)
    best_child, all_nodes = asyncio.run(lats.run(iterations=10))
    print("-----------------------")
    print(f"lats result: {best_child.state['thought']['thought']}")
