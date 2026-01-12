import re

data = """
[
    {
        "task_id": '1',
        "dependent_task_ids": [],
        "instruction": "Load and inspect the happiness dataset from the specified CSV file",
        "task_type": "pda"
    },
    {
        "task_id": '2',
        "dependent_task_ids": ['1'],
        "instruction": "Identify the 'country' or countries with the highest happiness score in the dataset",
        "task_type": "statistical analysis"
    },
    {
        "task_id": '3',
        "dependent_task_ids": ['2'],
        "instruction": "Format the output to match the required format @country_with_highest_score[country_name]",
        "task_type": "other"
    }
]
"""

data = re.sub(r'"task_id": \'(\d+)\'', r'"task_id": "\1"', data)
def replace_dependent_task_ids(match):
    content = match.group(0).replace("'", '"')
    return content

data = re.sub(r'"dependent_task_ids": \[(\'\d+\'(,\s*\'\d+\')*)?\]', replace_dependent_task_ids, data)

print(data)
