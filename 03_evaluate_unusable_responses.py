import pandas as pd
import os
from src.constants import (
    CEFR_LEVELS,
    GEMMA_2B,
    GEMMA_7B,
    GPT_4o_240806,
    GPT_4o_MINI_240718,
    LLAMA_3_8B,
)
from constants import CERD, CAM_MCQ


def count_unusable_responses(model_name, dataset_name, prompt_id, target_level):
    path = f'data/output/{dataset_name}/{model_name}/{prompt_id}'
    df = pd.read_csv(os.path.join(path, f'df_converted_texts_post_processed_{target_level}.csv'))
    filtered_df = df[df['text'].isin(['-9', '-10'])]
    # print(model_name, dataset_name, prompt_id, target_level, "Unusable:", len(filtered_df), "| Total:", len(df))
    return len(filtered_df), len(df)


if __name__ == '__main__':
    for param_model_name in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
        for param_prompt_id in ['01', '02', '11', '12', 'w01', 'w02']:
            n_unusable = 0
            n_total = 0
            for param_dataset_name in [CERD, CAM_MCQ]:
                cefr_levels = CEFR_LEVELS[:-1] if param_prompt_id in ['01', '02', '11', '12'] else ['A1']
                for param_target_level in cefr_levels:
                    n_u, n_t = count_unusable_responses(param_model_name, param_dataset_name, param_prompt_id, param_target_level)
                    n_unusable += n_u
                    n_total += n_t
            print("Model %s | prompt %s -> unusable = %.3f | Total = %d" % (param_model_name, param_prompt_id, n_unusable/n_total, n_total))
