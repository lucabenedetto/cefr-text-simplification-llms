import pickle
import numpy as np
from collections import defaultdict

import pandas as pd

from src.constants import (
    GEMMA_2B,
    GEMMA_7B,
    LLAMA_3_8B,
    GPT_4o_240806,
    GPT_4o_MINI_240718,
    CEFR_LEVELS, CEFR_TO_INT,
)
from constants import (
    CERD,
    CAM_MCQ,
    MODEL_NAME_TO_STR,
    PROMPT_ID_TO_STR,
)
from src.utils_plotting import boxplot_readability_indexes
from utils import get_readability_indexes_per_target_level
from src.evaluators.readability import ReadabilityEvaluator
from src.evaluators.constants import (
    READABILITY_INDEXES,
    FLESCH_READING_EASE,
    FLESCH_KINCAID_GRADE_LEVEL,
    AUTOMATED_READABILITY_INDEX,
    GUNNING_FOG_INDEX,
    COLEMAN_LIAU,
    SMOG_INDEX,
    LINSEAR_WRITE_FORMULA,
    DALE_CHALL,
)


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

    # this is to plot the results
    for dataset_name_param in [CERD, CAM_MCQ, 'aggregate']:
        for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt_id_param in ['01', '02', '11', '12']:
                if dataset_name_param in [CERD, CAM_MCQ]:
                    readability_indexes_per_level = [
                        pd.read_csv(f'data/evaluation/{dataset_name_param}/{model_name_param}/readability_indexes_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                else:
                    readability_indexes_per_level = [
                        pd.read_csv(f'data/evaluation/{CERD}/{model_name_param}/readability_indexes_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                    readability_indexes_per_level_cam_mcq = [
                        pd.read_csv(f'data/evaluation/{CAM_MCQ}/{model_name_param}/readability_indexes_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                    for idx in range(len(readability_indexes_per_level)):
                        readability_indexes_per_level[idx] = pd.concat(
                            [readability_indexes_per_level[idx], readability_indexes_per_level_cam_mcq[idx]], ignore_index=True)
                boxplot_readability_indexes(
                    readability_indexes_per_level, f"{MODEL_NAME_TO_STR[model_name_param]} | {PROMPT_ID_TO_STR[prompt_id_param]}", f"{dataset_name_param}_{model_name_param}_{prompt_id_param}", figsize=(3, 2.1))

    # this is for the results in the table
    df_cerd = pd.read_csv('data/input/cerd.csv')
    df_cam_mcq = pd.read_csv('data/input/mcq_cupa.csv')
    readability_indexes_per_level_aggregate = get_readability_indexes_per_target_level(pd.concat([df_cerd, df_cam_mcq], axis=0))
    mean_readability_indexes = defaultdict(list)
    median_readability_indexes = defaultdict(list)
    std_readability_indexes = defaultdict(list)
    for idx, cefr in enumerate(CEFR_LEVELS):
        for readability_index in READABILITY_INDEXES:
            mean_readability_indexes[readability_index].append(readability_indexes_per_level_aggregate[idx][readability_index].mean())
            median_readability_indexes[readability_index].append(readability_indexes_per_level_aggregate[idx][readability_index].median())
            std_readability_indexes[readability_index].append(readability_indexes_per_level_aggregate[idx][readability_index].std())
    for readability_index in READABILITY_INDEXES:
        print(readability_index)
        for idx in range(len(CEFR_LEVELS)):
            print("%.2f (%.2f),"
                  % (mean_readability_indexes[readability_index][idx], std_readability_indexes[readability_index][idx]), end=" ")
    # Below the readability indexes of the (aggregated) original datasets.
    # flesch_reading_ease
    # nan (nan), 87.16 (7.69), 74.55 (7.07), 73.32 (8.52), 61.09 (10.99), 61.07 (12.05),
    # flesch_kincaid_grade_level
    # nan (nan), 4.29 (1.57), 7.19 (1.37), 7.39 (1.78), 9.72 (2.26), 10.02 (2.62),
    # automated_readability_index
    # nan (nan), 5.08 (2.15), 8.71 (1.77), 8.98 (2.22), 11.68 (2.69), 12.00 (3.00),
    # gunning_fog_index
    # nan (nan), 5.90 (1.25), 8.93 (1.32), 9.32 (1.70), 11.44 (2.11), 12.11 (2.58),
    # coleman_liau
    # nan (nan), 5.70 (1.84), 7.98 (1.58), 8.11 (1.66), 10.17 (1.87), 9.96 (1.84),
    # smog_index
    # nan (nan), 7.17 (1.63), 9.64 (1.24), 9.91 (1.47), 11.92 (1.83), 12.05 (2.09),
    # linsear_write_formula
    # nan (nan), 6.04 (1.83), 9.34 (2.40), 9.87 (3.35), 11.13 (3.67), 12.73 (4.54),
    # dale_chall
    # nan (nan), 6.64 (0.60), 7.69 (0.69), 7.87 (0.62), 8.71 (0.83), 8.65 (0.72),
    errors_median_result_df = pd.DataFrame(columns=['dataset_name', 'model', 'prompt_id', 'target_level'] + list(READABILITY_INDEXES))
    for dataset_name_param in [CERD, CAM_MCQ, 'aggregate']:
        for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt_id_param in ['01', '02', '11', '12']:
                if dataset_name_param in [CERD, CAM_MCQ]:
                    readability_indexes_per_level = [
                        pd.read_csv(f'data/evaluation/{dataset_name_param}/{model_name_param}/readability_indexes_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                else:
                    readability_indexes_per_level = [
                        pd.read_csv(f'data/evaluation/{CERD}/{model_name_param}/readability_indexes_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                    readability_indexes_per_level_cam_mcq = [
                        pd.read_csv(f'data/evaluation/{CAM_MCQ}/{model_name_param}/readability_indexes_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                    for idx in range(len(readability_indexes_per_level)):
                        readability_indexes_per_level[idx] = pd.concat(
                            [readability_indexes_per_level[idx], readability_indexes_per_level_cam_mcq[idx]], ignore_index=True)
                for idx, cefr in enumerate(CEFR_LEVELS[:-1]):
                    new_row_df = pd.DataFrame({
                        'dataset_name': [dataset_name_param],
                        'model': [model_name_param],
                        'prompt_id': [prompt_id_param],
                        'target_level': [cefr],
                        FLESCH_READING_EASE: [readability_indexes_per_level[idx][FLESCH_READING_EASE].median()],
                        FLESCH_KINCAID_GRADE_LEVEL: [readability_indexes_per_level[idx][FLESCH_KINCAID_GRADE_LEVEL].median()],
                        AUTOMATED_READABILITY_INDEX: [readability_indexes_per_level[idx][AUTOMATED_READABILITY_INDEX].median()],
                        GUNNING_FOG_INDEX: [readability_indexes_per_level[idx][GUNNING_FOG_INDEX].median()],
                        COLEMAN_LIAU: [readability_indexes_per_level[idx][COLEMAN_LIAU].median()],
                        SMOG_INDEX: [readability_indexes_per_level[idx][SMOG_INDEX].median()],
                        LINSEAR_WRITE_FORMULA: [readability_indexes_per_level[idx][LINSEAR_WRITE_FORMULA].median()],
                        DALE_CHALL: [readability_indexes_per_level[idx][DALE_CHALL].median()],
                    })
                    errors_median_result_df = pd.concat([errors_median_result_df, new_row_df], ignore_index=True)
    for read_idx in READABILITY_INDEXES:
        errors_median_result_df[f'{read_idx}_ref_median'] = errors_median_result_df.apply(lambda r: median_readability_indexes[read_idx][CEFR_TO_INT[r['target_level']]], axis=1)
    errors_median_result_df = errors_median_result_df[~errors_median_result_df['target_level'].isin(['A1', 'C2'])]
    for read_idx in READABILITY_INDEXES:
        errors_median_result_df[f'{read_idx}_error'] = errors_median_result_df.apply(lambda r: np.abs(r[read_idx] - r[f'{read_idx}_ref_median']), axis=1)
    errors_median_result_df.to_csv(f'data/evaluation/errors_median_readability_indexes.csv')
