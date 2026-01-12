import argparse
import json
import os
import re

import yaml
from openai import OpenAI

from metagpt.config2 import Config
from metagpt.const import DATA_PATH, METAGPT_ROOT

gpt4turbo_config_path = METAGPT_ROOT / "config" / "gpt-4-turbo.yaml"
gpt4t_config = Config.from_yaml_file(gpt4turbo_config_path)


def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument('--max_resp', type=int, default=2048)
    args = parser.parse_args()

    args.api_key = gpt4t_config.llm.api_key
    args.model = gpt4t_config.llm.model
    args.url = gpt4t_config.llm.base_url
    return args


demons = """\Format{{
@shapiro_wilk_statistic[test_statistic]
@shapiro_wilk_p_value[p_value]
where "test_statistic" is a number between 0 and 1 representing the Shapiro-Wilk test statistic. Rounding off the answer to two decimal places.
where "p_value" is a number between 0 and 1 representing the p-value from the Shapiro-Wilk test. Rounding off the answer to four decimal places.
}}
\Answer{{
@shapiro_wilk_statistic[0.56]
@shapiro_wilk_p_value[0.0002]   
}}

\Format{{
@total_votes_outliers_num[outlier_num]
where "outlier_num" is an integer representing the number of values considered outliers in the 'total_votes' column.
}}
\Answer{{
@total_votes_outliers[10]   
}}
"""

reformat_template = """You should strictly follow the output requirements in the Format part. Here're some examples: {demons}. 
Your answer should contain all the \"@answer_name[answer]\" in the order mentioned, each \"answer\" should be in the range of value as required. 
The format requirements of this question is: {format}. 
Please give your answer:"""

# label_file = os.path.join(DATA_PATH, "da_bench/da-dev-labels.jsonl")
label_file = DATA_PATH / "di_dataset" / "da_bench" / "da-dev-labels.jsonl"


def get_useful_rsp(chat_history):
    useless_word_list = ["Goodbye", "If you have any more questions", "Have a great day", "Have a wonderful day"]
    for i in range(len(chat_history) - 1, -1, -1):
        # fixed: 所有不是能生成和发送消息的都是assistant，这里应该考虑使用name
        print("chat_history[i]", chat_history[i])
        if 'name' not in chat_history[i]:
            continue
        if chat_history[i]['name'] == 'user_proxy':
            if any(useless in chat_history[i]['content'] for useless in useless_word_list) or chat_history[i]['content'] == "":
                continue
            return chat_history[i]['content']
    return "No useful response found"


def reformat(useful_chat_res, cur_question: str):
    with open(label_file, 'r') as f:
        labels = f.readlines()
    for label in labels:
        label_item = json.loads(label)
        question_item = json.loads(cur_question)
        if label_item['id'] != question_item['id']:
            continue
        question_desc = question_item['question']
        ans_format = question_item['format']

        messages = [{"role": "user", "content": question_desc}]
        messages.append({"role": "assistant", "content": useful_chat_res})
        messages.append({"role": "user", "content": reformat_template.format(demons=demons, format=ans_format)})
        reformatted_response = ask_llm(messages)
        res = evaluate_responses(reformatted_response, label_item)
        print(res)
        return res


def evaluate_responses(reformatted_response, label_item):
    print("reformatted_response", reformatted_response)
    label_id = label_item["id"]
    label_answers = {ans[0]: ans[1] for ans in label_item["common_answers"]}
    answer_names, answers = extract_format(reformatted_response)
    extracted_answers = dict(zip(answer_names, answers))
    correct_answers = {ans_name: is_equal(extracted_answers.get(ans_name), label_answers[ans_name]) for ans_name
                       in label_answers.keys()}
    result = {
        "id": label_id,
        "label_answers": label_answers,
        "predicted_answers": extracted_answers,
        "correctness": correct_answers
    }
    return result


def ask_llm(messages):
    args = define_arguments()
    client = OpenAI(api_key=args.api_key)
    rsp = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=messages,
    )
    return rsp.choices[0].message.content


def extract_format(input_string):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers


def is_equal(response, label):
    if response == label:
        return True
    else:
        try:
            return abs(float(response) - float(label)) < 1e-6
        except:
            return False