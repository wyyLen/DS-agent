from examples.ds_agent.ds_dataset_info import get_format_ds_question
from examples.experiment.da_bench.util.DABENCH import DABench


def get_single_question(query_id):
    bench = DABench()
    for id, question in  bench.answers.items():
        if id == query_id:
            return bench.generate_formatted_prompt(id)
    return "Question not found."


print(get_single_question(0))
print(get_single_question(8))
print(get_single_question(549))

print("\n\n")
# classic cases
print(get_single_question(271))
print("\n")
print(get_single_question(275))
print("\n")
print(get_single_question(604))

# print(get_format_ds_question(181))
