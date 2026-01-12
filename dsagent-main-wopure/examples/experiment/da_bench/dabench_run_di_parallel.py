import sys

from metagpt.const import DA_EVAL_DI_FILE_PATH

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

import asyncio
import json

from util.DABENCH import DABench

from metagpt.logs import logger
from metagpt.roles.di.data_interpreter import DataInterpreter


async def get_prediction(agent, requirement):
    try:
        result = await agent.run(requirement)
        # Parse the result to extract the prediction from the JSON response
        prediction_json = json.loads(str(result).split("Current Plan")[1].split("## Current Task")[0])
        prediction = prediction_json[-1]["result"]  # Extract the last result from the parsed JSON
        return prediction
    except Exception as e:
        # Log an error message if an exception occurs during processing
        logger.info(f"Error processing requirement: {requirement}. Error: {e}")
        return None


async def evaluate_all(agent, k) -> tuple[dict, list]:
    bench = DABench()  # Create an instance of DABench to access its methods and data
    id_list, predictions = [], []  # Initialize lists to store IDs and predictions
    tasks = []  # Initialize a list to hold the tasks

    # Iterate over the answers in DABench to generate tasks
    for key, value in bench.answers.items():
        requirement = bench.generate_formatted_prompt(key)  # Generate a formatted prompt for the current key
        tasks.append(get_prediction(agent, requirement))  # Append the prediction task to the tasks list
        id_list.append(key)  # Append the current key to the ID list

    # Process tasks in groups of size k and execute them concurrently
    for i in range(0, len(tasks), k):
        current_group = tasks[i: i + k]
        group_predictions = await asyncio.gather(*current_group)  # Execute the current group of tasks in parallel
        # Filter out any None values from the predictions and extend the predictions list
        predictions.extend(pred for pred in group_predictions if pred is not None)
        logger.info(f"predictions: {predictions}")
        if i > 3:
            break

    # Evaluate the results using all valid predictions and logger.info the evaluation
    res = bench.eval_all(id_list, predictions)
    logger.info(res)
    return res, list(zip(id_list, predictions))


def main(k=1):
    agent = DataInterpreter()
    res, ids_predictions = asyncio.run(evaluate_all(agent, k))
    res["tokens_cost"] = agent.llm.get_costs()

    output_data = {
        "res": res,
        "ids_predictions": ids_predictions  # 将 ids_predictions 列表直接添加
    }

    logger.info(output_data)

    # with open(DA_EVAL_DI_FILE_PATH, 'w', encoding='utf-8') as file:
    #     json.dump(output_data, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
