import pickle

from src.constants import (
    GEMMA_2B,
    GEMMA_7B,
    LLAMA_3_8B,
    GPT_4o_240806,
    GPT_4o_MINI_240718,
    CEFR_LEVELS,
)
from constants import CERD, CAM_MCQ
from src.evaluators.readability import ReadabilityEvaluator


def readability_evaluation(dataset_name: str, model_name: str, prompt_id: str, target_level: str):
    converted_texts_1 = pickle.load(open(f'data/output/{dataset_name}/{model_name}/{prompt_id}/converted_texts_{target_level}.pkl', 'rb'))
    evaluator = ReadabilityEvaluator()
    print(type(converted_texts_1))
    readability_indexes = evaluator.compute_readability_indexes(converted_texts_1)
    readability_indexes.to_csv(f'data/evaluation/{dataset_name}/{model_name}/readability_indexes_{prompt_id}_target_{target_level}.csv', index=False)


if __name__ == '__main__':
    for dataset_name in [CERD, CAM_MCQ]:
        for model_name in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt_id in ['01', '02', '11', '12', 'w01', 'w02']:
                if prompt_id in ['w01', 'w02']:
                    readability_evaluation(dataset_name, model_name, prompt_id, 'A1')
                else:
                    for target_level in CEFR_LEVELS[:-1]:
                        readability_evaluation(dataset_name, model_name, prompt_id, target_level)
