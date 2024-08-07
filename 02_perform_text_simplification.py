"""
This script contains the code for performing the text simplification task (converting the given text into a different
CEFR level). This can be done with different prompts, which might include different information.
TODO: I still have to define the prompts.
TODO: I still have to choose which models to use.
Prompts can have different components:
- could have or not have the CEFR level of the given text
- could have or not have the word list for the original/target CEFR level.
- could have or not have the questions which have to be answerable given the text.
"""
import os
import pandas as pd
import pickle
from src.utils import init_adapter
from src.constants import (
    GPT_3_5_0613,
    GPT_4o_MINI_240718,
    GEMMA_2B,
    GEMMA_7B,
    LLAMA_3_8B,
    MINICPM_v2_6,
    MISTRAL_7B_v02,
)
from constants import CERD, CAM_MCQ
from utils import get_key_from_model_name, get_dataset


def main(model_name: str,
         prompt_id: str,
         dataset_name: str,
         target_cefr_level: str,
         access_key: str,
         save_as_dataframe: bool = True,
         ):
    adapter = init_adapter(model_name, access_key=access_key, use_gpu=True)
    dataset = get_dataset(dataset_name)  # the [:10] is TMP for preliminary experimetns.
    converted_texts = adapter.convert_texts(prompt_id, dataset, target_cefr_level)
    output_path = f'data/output/{dataset_name}/{model_name}/{prompt_id}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pickle.dump(converted_texts, open(os.path.join(output_path, f'converted_texts_{target_cefr_level}.pkl'), 'wb'))
    if save_as_dataframe:
        df_converted_texts = pd.DataFrame.from_dict(converted_texts, orient='index')
        df_converted_texts = df_converted_texts.reset_index()
        df_converted_texts = df_converted_texts.rename(columns={'index': 'text_id', 0: 'text'})
        df_converted_texts.to_csv(os.path.join(output_path, f'df_converted_texts_{target_cefr_level}.csv'), index=False)


if __name__ == '__main__':
    for param_model_name in (GEMMA_2B, GEMMA_7B, LLAMA_3_8B):
        param_prompt_id = '00'
        param_dataset_name = CERD  # CERD, CAM_MCQ
        param_target_level = 'A2'  # A1, A2, ..., C2
        param_access_key = get_key_from_model_name(param_model_name)
        main(param_model_name, param_prompt_id, param_dataset_name, param_target_level, param_access_key)
