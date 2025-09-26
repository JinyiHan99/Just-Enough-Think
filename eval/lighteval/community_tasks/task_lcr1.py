from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
import json
import random
import ast
from lighteval.metrics.normalizations import (
    LogProbCharNorm,
    gsm8k_normalizer,
    harness_triviaqa_normalizer,
    helm_normalizer,
    math_normalizer,
)
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# def prompt_normal(line, task_name: str = None):
#     query = line["question"] + " Please reason step by step, and put your final answer within \\boxed{}."
#     return Doc(
#         task_name=task_name,
#         query=query,  
#         choices=[str(line["std"])],
#         gold_index=0,
#         specific={"std": line["std"]} 
#     )

# def prompt_normal(line, task_name: str = None):

#     system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
#     policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}."
#     question = line["question"]
#     user_content = f"{policy_prompt}\n{question}"
#     query = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_content}
#     ]
#     query = json.dumps(query, ensure_ascii=False)
#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=[str(line["std"])],
#         gold_index=0,
#         specific={"std": line["std"]} 
#     )

math_metrics = CorpusLevelMetricGrouping(
    metric_name=["accuracy"],
    higher_is_better={"accuracy": True},
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeLLMYourBench(),
    corpus_level_fn={"accuracy": np.mean},
)
extend_enum(Metrics, "yourbench_metrics", yourbench_metrics)



def prompt_normal(line, task_name: str = None):
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}."
    return Doc(
        task_name=task_name,
        query=line["question"], 
        choices=[str(line["std"])],
        gold_index=0,
        specific={
            "std": line["std"],
            "system_prompt": f'{system_prompt}\n{policy_prompt}' 
        } 
    )

def prompt_normal_v2(line, task_name: str = None):
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}."
    return Doc(
        task_name=task_name,
        query=line["problem"], 
        choices=[str(line["answer"])],
        gold_index=0,
        specific={
            "std": line["answer"],
            "system_prompt": f'{system_prompt}\n{policy_prompt}' 
        } 
    )

def gpqa_instruct(line, task_name: str = None):
    """Prompt template adapted from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14"""
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query_template = "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}."
    query = query_template.format(
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        Question=line["Question"].strip(),
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        specific={
            "system_prompt": f'{system_prompt}\n{policy_prompt}' 
        } 
    )

# def commonsense_qa(line, task_name: str = None):
#     query = f"The following are multiple choice questions (with answers) about common sense.\nQuestion: {line['question']}\n"
#     query += "".join(
#         [f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, [f" {c}" for c in line["choices"]["text"]])]
#     )
#     query += "Answer:"

#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=LETTER_INDICES[: len(line["choices"]["text"])],
#         gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
#         instruction="The following are multiple choice questions (with answers) about common sense.\n",
#     )

def commonsense_qa(line, task_name: str = None):
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}."
    query = line['question']
    query += "".join(
        [f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, [f" {c}" for c in line["choices"]["text"]])]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(line["choices"]["text"])],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        specific={
            "system_prompt": f'{system_prompt}\n{policy_prompt}' 
        } 
    )


gsm8k_task = LightevalTaskConfig(
    name="gsm8k",
    suite=["community"],
    prompt_function=prompt_normal,
    hf_repo="./data/test/gsm8k_test",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        Metrics.expr_gold_metric,
    ],
    version=0,
)

commonsenseqa_task = LightevalTaskConfig(
    name="commonsenseqa",
    suite=["community"],
    prompt_function=commonsense_qa,
    hf_repo="./data/test/commonsense_qa_validation",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
        Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
        Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
    ],
    stop_sequence=["\n"],
    version=0,
)

mmlu_pro_avg_task = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["community"],
    prompt_function=gpqa_instruct,
    hf_repo="./data/test/MMLU_sample",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[Metrics.gpqa_instruct_avg_at_k(sample_params={"k": 1})],
    stop_sequence=[],
    version=1,
)

math_500_task = LightevalTaskConfig(
    name="math_500",
    suite=["community"],
    prompt_function=prompt_normal,
    hf_repo="./data/test/math500_dataset",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
    ],
    version=2,
)

Olympiad_task = LightevalTaskConfig(
    name="Olympiad",
    suite=["community"],
    prompt_function=prompt_normal_v2,
    hf_repo="./data/test/olympiad",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_k_math(sample_params={"k": 1})],
    version=2,
)

AMC_avg_task = LightevalTaskConfig(
    name="AMC_avg",
    suite=["community"],
    prompt_function=prompt_normal_v2,
    hf_repo="./data/test/AMC",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        Metrics.avg_at_k_math(sample_params={"k": 10}),
    ],
    version=2,
)

aime24_avg_task = LightevalTaskConfig(
    name="aime24_avg",
    suite=["community"],
    prompt_function=prompt_normal,
    hf_repo="./data/test/aime24_dataset",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_k_math(sample_params={"k": 10})],
    version=2,
)

gpqa_diamond_instruct_avg_task = LightevalTaskConfig(
    name="gpqa_diamond_instruct",
    suite=["community"],
    prompt_function=gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[Metrics.gpqa_instruct_avg_at_k(sample_params={"k": 1})],
    stop_sequence=[],
    version=1,
)

aime24_task = LightevalTaskConfig(
    name="aime24",
    prompt_function=prompt_normal,
    hf_repo="./data/test/aime24_dataset",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    suite=["community"],
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1})],
    version=2,
)

TASKS_TABLE = [
               aime24_avg_task, 
               gpqa_diamond_instruct_avg_task, 
               math_500_task,
               commonsenseqa_task,
               gsm8k_task,
               Olympiad_task,
               AMC_avg_task,
               mmlu_pro_avg_task
               ]