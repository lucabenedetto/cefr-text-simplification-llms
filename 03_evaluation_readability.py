import pickle
import matplotlib.pyplot as plt
import pandas as pd

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
from src.evaluators.constants import READABILITY_INDEXES
from src.utils_plotting import boxplot_readability_indexes


def readability_evaluation(dataset_name: str, model_name: str, prompt_id: str, target_level: str):
    converted_texts_1 = pickle.load(open(f'data/output/{dataset_name}/{model_name}/{prompt_id}/converted_texts_post_processed_{target_level}.pkl', 'rb'))
    evaluator = ReadabilityEvaluator()
    print("Doing:", dataset_name, model_name, prompt_id, target_level)
    readability_indexes = evaluator.compute_readability_indexes(converted_texts_1)
    len_0 = len(readability_indexes)
    # TODO: make a constant with the list of possible errors, and a constant for each error.
    readability_indexes['error'] = readability_indexes.apply(lambda r: converted_texts_1[r['text_id']] in ['-9', '-10'], axis=1)
    readability_indexes = readability_indexes[~readability_indexes['error']]
    if len(readability_indexes) != len_0:
        print(f"Removed {len_0 - len(readability_indexes)} rows.")
    readability_indexes.to_csv(f'data/evaluation/{dataset_name}/{model_name}/readability_indexes_{prompt_id}_target_{target_level}.csv', index=False)


def compute_and_save_readability_indexes():
    for dataset_name_param in [CERD, CAM_MCQ]:
        for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt_id_param in ['01', '02', '11', '12', 'w01', 'w02']:
                if prompt_id_param in ['w01', 'w02']:
                    readability_evaluation(dataset_name_param, model_name_param, prompt_id_param, 'A1')
                else:
                    for target_level in CEFR_LEVELS[:-1]:
                        readability_evaluation(dataset_name_param, model_name_param, prompt_id_param, target_level)


if __name__ == '__main__':
    # this is to store all the computed readability indexes
    compute_and_save_readability_indexes()
