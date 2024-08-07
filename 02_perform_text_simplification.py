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
import pickle
from src.utils import init_adapter


def main(model_name, prompt_id, dataset_name, target_cefr_level, access_key):
    adapter = init_adapter(model_name, access_key=access_key)
    converted_texts = adapter.convert_texts(prompt_id, dataset_name, target_cefr_level)
    pickle.dump(
        converted_texts,
        open(os.path.join(f'data/output/{dataset_name}/{model_name}/{prompt_id}/converted_texts_{target_cefr_level}.pkl'), 'wb')
    )


if __name__ == '__main__':
    param_model_name = 'model_name'
    param_prompt_id = 'prompt_id'
    param_dataset_name = 'dataset_name'  # CERD, CAM_MCQ
    param_target_level = 'target_level'  # A1, A2, ..., C2
    param_access_key = 'asdf'
    main(param_model_name, param_prompt_id, param_dataset_name, param_target_level, param_access_key)
