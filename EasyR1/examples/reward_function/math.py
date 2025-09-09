# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(answer: str) -> float:
    box_match = re.search(r'\\boxed\{.*?\}', answer)
    if not box_match:
        return 0.0
    return 1.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    correct_answers_indices = []
    for i, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
        if accuracy_score == 1.0:
            correct_answers_indices.append(i)
    correct_answers_lengths = [reward_inputs[idx]['response_length'] for idx in correct_answers_indices]

    if correct_answers_indices:
        length_min = min(correct_answers_lengths)
        length_max = max(correct_answers_lengths)
        epsilon_l = 1e-8
        denom = (length_max - length_min + epsilon_l)

        # 计算正确答案的length得分
        for index in correct_answers_indices: 
            alpha=1.2
            delta=0.05
            length_reward = (length_max - reward_inputs[index]['response_length']) / (denom)
            length_reward = length_reward * alpha
            length_reward = length_reward * (1 - delta) + delta
            scores[index]['overall'] += length_reward
            # print("!!! add the lenght score")
    return scores
