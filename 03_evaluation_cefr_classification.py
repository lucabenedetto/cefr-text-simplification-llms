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
)
from constants import CERD, CAM_MCQ
from transformers import pipeline


# def eval_verbosity_responses(model_name, dataset_name, prompt_id, target_level):
#     path = f'data/output/{dataset_name}/{model_name}/{prompt_id}'
#     df_post_processed = pd.read_csv(os.path.join(path, f'df_converted_texts_post_processed_{target_level}.csv'))
#     df_post_processed = df_post_processed[~df_post_processed['text'].isin(['-9', '-10'])]
#     df_raw = pd.read_csv(os.path.join(path, f'df_converted_texts_{target_level}.csv'))
#     df_post_processed = pd.merge(df_post_processed, df_raw, on='text_id', how='left')
#     df_post_processed['processed_len'] = df_post_processed.apply(lambda r: len(r['text_x'].split()), axis=1)
#     df_post_processed['raw_len'] = df_post_processed.apply(lambda r: len(r['text_y'].split()), axis=1)
#     df_post_processed['diff'] = df_post_processed.apply(lambda r: r['raw_len'] - r['processed_len'], axis=1)
#     return df_post_processed['diff'].tolist()

def main():
    model_name = "AbdulSami/bert-base-cased-cefr"
    classifier = pipeline("text-classification", model=model_name)

    df_cam_mcq = pd.read_csv('data/input/mcq_cupa.csv')
    predictions = classifier(df_cam_mcq[COLUMN_TEXT])
    df_cam_mcq['prediction_label'] = [x['label'] for x in predictions]
    df_cam_mcq['prediction_score'] = [x['score'] for x in predictions]
    df_cam_mcq.to_csv('data/evaluation/cam_mcq/cefr_classification_original_dataset.csv', index=False)
    print("Done Cam MCQ")

    df_cerd = pd.read_csv('data/input/cerd.csv')
    predictions = classifier(df_cerd[COLUMN_TEXT])
    df_cerd['prediction_label'] = [x['label'] for x in predictions]
    df_cerd['prediction_score'] = [x['score'] for x in predictions]
    df_cerd.to_csv('data/evaluation/cerd/cefr_classification_original_dataset.csv', index=False)
    print("Done CERD")


if __name__ == '__main__':
    main()
    # test_strings = ['this is a text.', 'My apologies, I reckoned that this contemporary appliance would work.']
    # model_name = "AbdulSami/bert-base-cased-cefr"
    # classifier = pipeline("text-classification", model=model_name)
    # predictions = classifier(test_strings)
    # print(predictions)
    # print([(x['label'], x['score']) for x in predictions])

    # for param_model_name in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
    #     for param_prompt_id in ['01', '02', '11', '12', 'w01', 'w02']:
    #         list_differences = []
    #         for param_dataset_name in [CERD, CAM_MCQ]:
    #             cefr_levels = CEFR_LEVELS[:-1] if param_prompt_id in ['01', '02', '11', '12'] else ['A1']
    #             for param_target_level in cefr_levels:
    #                 differences = eval_verbosity_responses(param_model_name, param_dataset_name, param_prompt_id, param_target_level)
    #                 list_differences.extend(differences)
    #         print(
    #             "Model %s | prompt %s -> diff mu = %.1f | std = %.1f | support = %d"
    #             % (param_model_name, param_prompt_id, np.mean(list_differences), np.std(list_differences), len(list_differences))
    #         )
