import ast
import numpy as np
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
    COLUMN_TEXT_LEVEL,
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


# TODO I should refactor this and use an Evaluator object
def cefr_evaluation_simplified_texts(dataset_name, model_name, prompt_id, target_level):
    classifier = pipeline("text-classification", model="AbdulSami/bert-base-cased-cefr")

    path = f'data/processed_output/{dataset_name}/{model_name}/{prompt_id}'
    df_post_processed = pd.read_csv(os.path.join(path, f'df_converted_texts_post_processed_{target_level}.csv'))
    df_post_processed = df_post_processed[~df_post_processed[COLUMN_TEXT].isin(['-9', '-10'])]
    df_post_processed['predictions'] = df_post_processed.apply(lambda r: perform_cefr_classification(classifier, r[COLUMN_TEXT]), axis=1)
    df_post_processed.to_csv(f'data/evaluation/{dataset_name}/{model_name}/cefr_classification_{prompt_id}_{target_level}.csv', index=False)


# TODO I should refactor this and move it to src
def within1_score(score, level):
    if level == 'A1' and score in {'A1', 'A2'}:
        return True
    if level == 'A2' and score in {'A1', 'A2', 'B1'}:
        return True
    if level == 'B1' and score in {'A2', 'B1', 'B2'}:
        return True
    if level == 'B2' and score in {'B1', 'B2', 'C1'}:
        return True
    if level == 'C1' and score in {'B2', 'C1', 'C2'}:
        return True
    if level == 'C2' and score in {'C1', 'C2'}:
        return True
    return False


if __name__ == '__main__':
    # perform CEFR evaluation of the original datasets
    cefr_evaluation_original_datasets()
    for dataset_name in [CERD, CAM_MCQ]:
        df = pd.read_csv(f'data/evaluation/{dataset_name}/cefr_classification_original_dataset.csv',)
        df['not_evaluated'] = df.apply(lambda r: r['predictions'] == "{'label': '00', 'score': 0.0}", axis=1)
        df['score'] = df.apply(lambda r: ast.literal_eval(r['predictions'][1:-1])['label'] if not r['not_evaluated'] else 'NA', axis=1)
        df['score_correct'] = df.apply(lambda r: r['score'] == r[COLUMN_TEXT_LEVEL], axis=1)
        df['score_within1'] = df.apply(lambda r: within1_score(r['score'], r[COLUMN_TEXT_LEVEL]), axis=1)
        print(dataset_name, np.mean(df['score_correct']), np.mean(df['score_within1']), len(df[~df['not_evaluated']]))
        for cefr_level in CEFR_LEVELS:
            local_df = df[df[COLUMN_TEXT_LEVEL] == cefr_level]
            print(dataset_name, cefr_level, np.mean(local_df['score_correct']), np.mean(local_df['score_within1']), len(local_df[~local_df['not_evaluated']]))
    # cerd 0.22356495468277945 0.3323262839879154 117
    # cerd A1 nan nan 0
    # cerd A2 0.578125 0.953125 64
    # cerd B1 0.6166666666666667 0.7666666666666667 46
    # cerd B2 0.0 0.0 0
    # cerd C1 0.0 0.04477611940298507 4
    # cerd C2 0.0 0.0 3
    # cam_mcq 0.16981132075471697 0.1761006289308176 140
    # cam_mcq A1 nan nan 0
    # cam_mcq A2 nan nan 0
    # cam_mcq B1 0.9642857142857143 1.0 140
    # cam_mcq B2 0.0 0.0 0
    # cam_mcq C1 0.0 0.0 0
    # cam_mcq C2 0.0 0.0 0

    # perform the CEFR evaluation and store the results
    for param_model_name in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
        for param_prompt_id in ['01', '02', '11', '12', 'w01', 'w02']:
            for param_dataset_name in [CERD, CAM_MCQ]:
                cefr_levels = CEFR_LEVELS[:-1] if param_prompt_id in ['01', '02', '11', '12'] else ['A1']
                for param_target_level in cefr_levels:
                    cefr_evaluation_simplified_texts(param_dataset_name, param_model_name, param_prompt_id, param_target_level)
                    print("Done:", param_model_name, param_dataset_name, param_prompt_id, param_target_level)

    # measure the accuracy of the TS task (comparing the target level and the level from the classifier).
    result_df = pd.DataFrame(columns=['dataset_name', 'model', 'prompt_id', 'target_level', 'accuracy', 'within1', 'support', 'not_evaluated'])
    for param_dataset_name in [CERD, CAM_MCQ, 'aggregate']:
        for param_model_name in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for param_prompt_id in ['01', '02', '11', '12', 'w01', 'w02']:
                cefr_levels = CEFR_LEVELS[:-1] if param_prompt_id in ['01', '02', '11', '12'] else ['A1']
                for target_level in cefr_levels:
                    if param_dataset_name != 'aggregate':
                        local_df = pd.read_csv(
                            f'data/evaluation/{param_dataset_name}/{param_model_name}/cefr_classification_{param_prompt_id}_{target_level}.csv'
                        )
                    else:
                        local_df = pd.concat([
                            pd.read_csv(f'data/evaluation/{CERD}/{param_model_name}/cefr_classification_{param_prompt_id}_{target_level}.csv'),
                            pd.read_csv(f'data/evaluation/{CAM_MCQ}/{param_model_name}/cefr_classification_{param_prompt_id}_{target_level}.csv'),
                        ],
                        ignore_index=True,)
                    # [{'label': 'B1', 'score': 0.9546597003936768}] or {'label': '00', 'score': 0.0}
                    len_0 = len(local_df)
                    local_df = local_df[local_df['predictions'] != "{'label': '00', 'score': 0.0}"]
                    len_1 = len(local_df)
                    if len_1 > 0:
                        local_df['score'] = local_df.apply(lambda r: ast.literal_eval(r['predictions'][1:-1])['label'], axis=1)
                        local_df['within1'] = local_df.apply(lambda r: within1_score(r['score'], r['text_level']), axis=1)
                    else:
                        local_df['score'] = False
                        local_df['within1'] = False
                    new_row_df = pd.DataFrame({
                        'dataset_name': [param_dataset_name],
                        'model': [param_model_name],
                        'prompt_id': [param_prompt_id],
                        'target_level': [target_level],
                        'accuracy': [np.mean(local_df['score'] == local_df['text_level'])],
                        'within1': [np.mean(local_df['within1'])],
                        'support': [len_1],
                        'not_evaluated': [len_0 - len_1],
                    })
                    result_df = pd.concat([result_df, new_row_df], ignore_index=True)
    result_df.to_csv(f'data/evaluation/evaluation_cefr_classification.csv')
