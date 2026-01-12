from enum import Enum

from pydantic import BaseModel

from metagpt.prompts.ds_task_type import (
    PRE_DA_PROMPT,
    DATA_PREPROCESS_PROMPT,
    FEATURE_ENGINEERING_PROMPT,
    MODEL_EVALUATE_PROMPT,
    MODEL_TRAIN_PROMPT,
    CORRELATION_ANALYSIS_PROMPT,
    OUTLIER_DETECTION_PROMPT,
    STATISTICAL_ANALYSIS_PROMPT,
    DISTRIBUTION_ANALYSIS_PROMPT, MACHINE_LEARNING_PROMPT
)


class TaskTypeDef(BaseModel):
    name: str
    desc: str = ""
    guidance: str = ""


class TaskType(Enum):
    """By identifying specific types of tasks, we can inject human priors (guidance) to help task solving"""
    PDA = TaskTypeDef(
        name="pda",
        desc="For performing pre-analysis data",
        guidance=PRE_DA_PROMPT,
    )
    DATA_PREPROCESS = TaskTypeDef(
        name="data preprocessing",
        desc="For preprocessing dataset in a data analysis or machine learning task ONLY,"
        "general data operation doesn't fall into this type",
        guidance=DATA_PREPROCESS_PROMPT,
    )
    FEATURE_ENGINEERING = TaskTypeDef(
        name="feature engineering",
        desc="Only for creating new columns for input data.",
        guidance=FEATURE_ENGINEERING_PROMPT,
    )
    MACHINE_LEARNING = TaskTypeDef(
        name="machine learning",
        desc="For model training and evaluation tasks",
        guidance=MACHINE_LEARNING_PROMPT,
    )
    # MODEL_TRAIN = TaskTypeDef(
    #     name="model train",
    #     desc="Only for training model.",
    #     guidance=MODEL_TRAIN_PROMPT,
    # )
    # MODEL_EVALUATE = TaskTypeDef(
    #     name="model evaluate",
    #     desc="Only for evaluating model.",
    #     guidance=MODEL_EVALUATE_PROMPT,
    # )
    CORRELATION_ANALYSIS = TaskTypeDef(
        name="correlation analysis",
        desc="Only for analyze the correlations between various variables.",
        guidance=CORRELATION_ANALYSIS_PROMPT,
    )
    OUTLIER_DETECTION = TaskTypeDef(
        name="outlier detection",
        desc="Only for identify and deal with outliers.",
        guidance=OUTLIER_DETECTION_PROMPT,
    )
    STATISTICAL_ANALYSIS = TaskTypeDef(
        name="statistical analysis",
        desc="Only for calculate key summary statistics to get an overview of your data.",
        guidance=STATISTICAL_ANALYSIS_PROMPT,
    )
    DISTRIBUTION_ANALYSIS = TaskTypeDef(
        name="distribution analysis",
        desc="Only for explore the distribution characteristics of data",
        guidance=DISTRIBUTION_ANALYSIS_PROMPT,
    )
    OTHER = TaskTypeDef(name="other", desc="Any tasks not in the defined categories")

    @property
    def type_name(self):
        return self.value.name

    @classmethod
    def get_type(cls, type_name):
        for member in cls:
            if member.type_name == type_name:
                return member.value
        return None
