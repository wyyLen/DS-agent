from typing import Any

from metagpt.schema import Plan, Task

goal_181 = """
file_name: abalone.csv
question: Explore the correlation between the length and the weight of the whole abalone. Additionally, perform feature engineering by creating a new feature called \"volume\" by multiplying the length, diameter, and height of the abalone. Determine if the volume feature improves the accuracy of predicting the number of rings using a linear regression model.
constraints: Calculate the Pearson correlation coefficient to assess the strength and direction of the linear relationship between length and the weight. The volume feature should be created by multiplying the length, diameter, and height of the abalone. Use the sklearn's linear regression model to predict the number of rings. Split the data into a 70% train set and a 30% test set. Evaluate the models by calculating the root mean squared error (RMSE) with the test set.
"""

goal_57 = """
File: election2016.csv
Question: Question 2: Is there a relationship between the difference in votes received by the Democratic and Republican parties and their percentage point difference?
Constraints: Calculate the Pearson correlation coefficient (r) to assess the strength and direction of the linear relationship between the difference in votes and the percentage point difference. Assess the significance of the correlation using a two-tailed test with a significance level (alpha) of 0.05. Report the p-value associated with the correlation test. Consider the relationship to be linear if the p-value is less than 0.05 and the absolute value of r is greater than or equal to 0.5. Consider the relationship to be nonlinear if the p-value is less than 0.05 and the absolute value of r is less than 0.5. If the p-value is greater than or equal to 0.05, report that there is no significant correlation.
"""

goal_map = {
    "181": goal_181,
    "1812": goal_181,
    "57": goal_57
}

task_list_181 = [
    {"task_id": "1", "dependent_task_ids": [],
     "instruction": "Load and inspect the abalone dataset",
     "task_type": "pda"},
    {"task_id": "2", "dependent_task_ids": ["1"],
     "instruction": "Calculate the Pearson correlation coefficient between the length and the weight of the whole abalone.",
     "task_type": "correlation analysis"
     },
    {"task_id": "3", "dependent_task_ids": ["1"],
     "instruction": "Create a new feature 'volume' by multiplying the length, diameter, and height of the abalone.",
     "task_type": "feature engineering"
     },
    {"task_id": "4", "dependent_task_ids": ["3"],
     "instruction": "Split the dataset into a 70% train set and a 30% test set.",
     "task_type": "data preprocessing"
     },
    {"task_id": "5", "dependent_task_ids": ["4"],
     "instruction": "Train a linear regression model to predict the number of rings using the original features.",
     "task_type": "machine learning"
     },
    {"task_id": "6", "dependent_task_ids": ["4"],
     "instruction": "Train a linear regression model to predict the number of rings using the original features plus the new 'volume' feature.",
     "task_type": "machine learning"
     },
    {"task_id": "7", "dependent_task_ids": ["5", "6"],
     "instruction": "The RMSE of the two trained models is calculated to evaluate their performance.",
     "task_type": "machine learning"
     }
]

task_list_181_2 = [
    {
        "task_id": "1",
        "dependent_task_ids": [],
        "instruction": "Load and inspect the abalone dataset to understand its structure and the available columns.",
        "task_type": "pda"
    },
    {
        "task_id": "2",
        "dependent_task_ids": ["1"],
        "instruction": "Calculate the Pearson correlation coefficient between the length and the weight of the whole abalone.",
        "task_type": "correlation analysis"
    },
    {
        "task_id": "3",
        "dependent_task_ids": ["1"],
        "instruction": "Create a new feature called 'volume' by multiplying the length, diameter, and height of the abalone.",
        "task_type": "feature engineering"
    },
    {
        "task_id": "4",
        "dependent_task_ids": ["1", "3"],
        "instruction": "Split the dataset into a 70% train set and a 30% test set, including the new 'volume' feature.",
        "task_type": "data preprocessing"
    },
    {
        "task_id": "5",
        "dependent_task_ids": ["4"],
        "instruction": "Train a linear regression model to predict the number of rings using the original features without the new 'volume' feature and evaluate its performance using RMSE.",
        "task_type": "machine learning"
    },
    {
        "task_id": "6",
        "dependent_task_ids": ["4"],
        "instruction": "Train a linear regression model to predict the number of rings using the original features plus the new 'volume' feature and evaluate its performance using RMSE.",
        "task_type": "machine learning"
    }
]

task_list_57 = [
    {
        "task_id": "1",
        "dependent_task_ids": [],
        "instruction": "Load and preview the dataset to understand its structure and contents.",
        "task_type": "pda"
    },
    {
        "task_id": "2",
        "dependent_task_ids": ["1"],
        "instruction": "Calculate the difference in votes between the Democratic and Republican parties for each entry.",
        "task_type": "feature engineering"
    },
    {
        "task_id": "3",
        "dependent_task_ids": ["1"],
        "instruction": "Calculate the percentage point difference between the Democratic and Republican parties for each entry.",
        "task_type": "feature engineering"
    },
    {
        "task_id": "4",
        "dependent_task_ids": ["2", "3"],
        "instruction": "Calculate the Pearson correlation coefficient and the associated p-value between the difference in votes and the percentage point difference.",
        "task_type": "correlation analysis"
    },
    {
        "task_id": "5",
        "dependent_task_ids": ["4"],
        "instruction": "Assess the significance of the correlation using the calculated p-value and Pearson correlation coefficient, and report the findings.",
        "task_type": "statistical analysis"
    }
]

"""
Calculate the Pearson correlation coefficient and the p-value between the difference in votes and the percentage point difference.
"""

plan_map = {
    "181": task_list_181,
    "1812": task_list_181_2,
    "57": task_list_57
}


class FixedPlan(Plan):

    def __init__(self, task_id, **data: Any):
        super().__init__(**data)
        task_list: list[Task] = []
        task_map = {}
        for task in plan_map.get(task_id):
            task_list.append(Task(task_id=task["task_id"], instruction=task["instruction"],
                                  dependent_task_ids=task["dependent_task_ids"], task_type=task["task_type"]))
            task_map[task["task_id"]] = task_list[-1]
        self.tasks = task_list
        self.task_map = task_map
        self.current_task_id = task_list[0].task_id


# FIXED_PLAN = FixedPlan(goal=goal_map.get("181"), task_id="181")
# print(FIXED_PLAN.tasks)


def get_fixed_plan(seq: int) -> FixedPlan:
    return FixedPlan(goal=goal_map.get(f"{seq}"), task_id=f"{seq}")
