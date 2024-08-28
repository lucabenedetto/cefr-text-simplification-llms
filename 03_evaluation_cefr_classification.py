import pandas as pd
import os
from src.constants import (
    CEFR_LEVELS,
    GEMMA_2B,
    GEMMA_7B,
    GPT_4o_240806,
    GPT_4o_MINI_240718,
    LLAMA_3_8B,
    COLUMN_TEXT,
)
from constants import CERD, CAM_MCQ
from transformers import pipeline


def perform_cefr_classification(classifier, text):
    try:
        return classifier(text)
    except:
        return"{'label': '00', 'score': 0.0}"


def cefr_evaluation_original_datasets():
    classifier = pipeline("text-classification", model="AbdulSami/bert-base-cased-cefr")

    df_cam_mcq = pd.read_csv('data/input/mcq_cupa.csv')
    df_cam_mcq['predictions'] = df_cam_mcq.apply(lambda r: perform_cefr_classification(classifier, r[COLUMN_TEXT]), axis=1)
    df_cam_mcq.to_csv('data/evaluation/cam_mcq/cefr_classification_original_dataset.csv', index=False)
    print("Done Cam MCQ")

    df_cerd = pd.read_csv('data/input/cerd.csv')
    df_cerd['predictions'] = df_cerd.apply(lambda r: perform_cefr_classification(classifier, r[COLUMN_TEXT]), axis=1)
    df_cerd.to_csv('data/evaluation/cerd/cefr_classification_original_dataset.csv', index=False)
    print("Done CERD")


def cefr_evaluation_simplified_texts(dataset_name, model_name, prompt_id, target_level):
    classifier = pipeline("text-classification", model="AbdulSami/bert-base-cased-cefr")

    path = f'data/processed_output/{dataset_name}/{model_name}/{prompt_id}'
    df_post_processed = pd.read_csv(os.path.join(path, f'df_converted_texts_post_processed_{target_level}.csv'))
    df_post_processed = df_post_processed[~df_post_processed[COLUMN_TEXT].isin(['-9', '-10'])]
    df_post_processed['predictions'] = df_post_processed.apply(lambda r: perform_cefr_classification(classifier, r[COLUMN_TEXT]), axis=1)
    df_post_processed.to_csv(f'data/evaluation/{dataset_name}/{model_name}/cefr_classification_{prompt_id}_{target_level}.csv', index=False)


if __name__ == '__main__':
    # cefr_evaluation_original_datasets()
    for param_model_name in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
        for param_prompt_id in ['01', '02', '11', '12', 'w01', 'w02']:
            for param_dataset_name in [CERD, CAM_MCQ]:
                cefr_levels = CEFR_LEVELS[:-1] if param_prompt_id in ['01', '02', '11', '12'] else ['A1']
                for param_target_level in cefr_levels:
                    cefr_evaluation_simplified_texts(param_model_name, param_dataset_name, param_prompt_id, param_target_level)
                    print("Done:", param_model_name, param_dataset_name, param_prompt_id, param_target_level)
