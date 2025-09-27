from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import json
import random
import ast
import re
import numpy as np
from math_verify import parse, verify
from lighteval.tasks.requests import Doc
from lighteval.models.model_output import ModelResponse
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import AvgAtK

from lighteval.metrics.utils.metric_utils import SamplingMethod
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SampleLevelComputation

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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
class OutputTokenLengthScorer(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        if model_response.output_tokens and len(model_response.output_tokens) > 0:
            return float(len(model_response.output_tokens[0]))
        else:
            return None

avg_output_token_length_metric = SampleLevelMetric(
    metric_name="avg_output_token_length",
    higher_is_better=False,  
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=OutputTokenLengthScorer(),
    corpus_level_fn=np.mean,  
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

class MyMathVerifyScorer(SampleLevelComputation): 
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        answer = model_response.final_text[0]
        ground_truth = doc.get_golds()[0]
        ground_truth_str = str(ground_truth)
        pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
        boxed_content_ans = re.findall(pattern, answer)

        if not boxed_content_ans:
            return 0.0
            
        final_ans_expr = "\\boxed{" + boxed_content_ans[-1] + "}"
        
        if "\\boxed" not in ground_truth_str:
            final_gt_expr = "\\boxed{" + ground_truth_str + "}"
        else:
            final_gt_expr = ground_truth_str
            
        if final_ans_expr == final_gt_expr:
            return 1.0
            
        try:
            parsed_ans = parse(final_ans_expr)
            parsed_gt = parse(final_gt_expr)
            is_correct = verify(parsed_ans, parsed_gt)
            return 1.0 if is_correct else 0.0
        except Exception:
            return 0.0


def extract_predicted_answer_gpqa(model_response_text: str) -> str | None:
    patterns = [
        (0, re.compile(r"\\boxed\{([A-Z])\}")),
        (1, re.compile(r"ANSWER:\s*([A-Z])\b", re.IGNORECASE)),
        (2, re.compile(r"\bOption ([A-Z])\b", re.IGNORECASE)),
        (3, re.compile(r"\b(?:the\s+)?(?:correct|final|best)?\s*(?:answer|choice)\s*(?:is|would be|should be)\s*:?\s*([A-Z])\b", re.IGNORECASE)),
        (4, re.compile(r"</think>\s*([A-Z])\.\s+[a-zA-Z]{2,}"))
    ]

    found_matches = [] 

    for priority, pattern in patterns:
        for match in pattern.finditer(model_response_text):
            try:
                letter = match.group(1).strip().upper()
                if letter and 'A' <= letter <= 'Z':
                    found_matches.append((match.start(), priority, letter[0]))
            except IndexError:
                continue
    
    if not found_matches:
        return None
    found_matches.sort(key=lambda x: (x[0], -x[1]))
    
    return found_matches[-1][2]

class MyGPQAScorer(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        model_response_text = model_response.final_text[0]
        gold_answer_letter = doc.get_golds()[0]
        predicted_answer_letter = extract_predicted_answer_gpqa(model_response_text)
        if predicted_answer_letter and predicted_answer_letter == gold_answer_letter:
            return 1.0
        return 0.0

def custom_avg_at_k_multi_choice(**kwargs) -> SampleLevelMetric:
    params = kwargs.get("sample_params", {})
    scorer = AvgAtK(
        sample_scoring_function=MyGPQAScorer(),
        **params 
    )
    return SampleLevelMetric(
        metric_name="custom_avg_at_k_multi_choice",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=scorer,
        corpus_level_fn=np.mean, 
    )



my_custom_metric_math = SampleLevelMetric(
    metric_name="custom_math_verify",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=MyMathVerifyScorer(),
    corpus_level_fn=np.mean,
)

gpqa_diamond_instruct_avg = LightevalTaskConfig(
    name="gpqa_diamond_instruct",
    suite=["community"],
    prompt_function=gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        custom_avg_at_k_multi_choice(sample_params={"k": 2}),
        avg_output_token_length_metric
        ],
    stop_sequence=[],
    version=1,
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
    metrics=[
        my_custom_metric_math(
            sample_params={
                "k": 10,  
            }
        ),
        avg_output_token_length_metric
    ],
    version=2,
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
        custom_avg_at_k_multi_choice(sample_params={"k": 1}),
        avg_output_token_length_metric
    ],
    stop_sequence=["\n"],
    version=0,
)

aime24_samples_task = LightevalTaskConfig(
    name="aime24_avg",
    prompt_function=prompt_normal,
    hf_repo="./data/test/aime24_dataset",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    suite=["community"],
    metrics=[
        my_custom_metric_math(
            sample_params={
                "k": 10,  
            }
        ),
        avg_output_token_length_metric
    ],
    generation_size=10, 
    version=2,
)

mmlu_pro_avg_task = LightevalTaskConfig(
    name="mmlu",
    suite=["community"],
    prompt_function=gpqa_instruct,
    hf_repo="./data/test/MMLU_sample",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        custom_avg_at_k_multi_choice(sample_params={"k": 1}),
        avg_output_token_length_metric
        ],
    stop_sequence=[],
    version=1,
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
        my_custom_metric_math(
            sample_params={
                "k": 1,  
            }
        ),
        avg_output_token_length_metric
    ],
    version=0,
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
        my_custom_metric_math(
            sample_params={
                "k": 10,  
            }
        ),
        avg_output_token_length_metric
    ],
    version=2,
)

math500_task = LightevalTaskConfig(
    name="math500",
    prompt_function=prompt_normal,
    hf_repo="./data/test/math500_dataset",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    suite=["community"],
    metrics=[my_custom_metric_math,
    avg_output_token_length_metric
    ],
    version=2,
)


TASKS_TABLE = [
               commonsenseqa_task,
               math500_task,
               aime24_samples_task,
               gsm8k_task,
               Olympiad_task,
               AMC_avg_task,
               mmlu_pro_avg_task,
               gpqa_diamond_instruct_avg
               ]