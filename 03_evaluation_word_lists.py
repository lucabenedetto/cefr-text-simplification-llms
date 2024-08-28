import pickle
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
from src.evaluators.word_list import WordListEvaluator
from src.utils_plotting import line_plot_word_lists_count


def word_list_evaluation(dataset_name: str, model_name: str, prompt_id: str, target_level: str):
    converted_texts_1 = pickle.load(open(f'data/output/{dataset_name}/{model_name}/{prompt_id}/converted_texts_post_processed_{target_level}.pkl', 'rb'))
    print("Doing:", dataset_name, model_name, prompt_id, target_level)
    word_lists = WordListEvaluator().measure_word_frequency(converted_texts_1)
    len_0 = len(word_lists)
    # TODO: make a constant with the list of possible errors, and a constant for each error.
    word_lists['error'] = word_lists.apply(lambda r: converted_texts_1[r['text_id']] in ['-9', '-10'], axis=1)
    word_lists = word_lists[~word_lists['error']]
    if len(word_lists) != len_0:
        print(f"Removed {len_0 - len(word_lists)} rows.")
    for level in CEFR_LEVELS:
        word_lists[level + '_frac'] = word_lists.apply(lambda r: r[level]/r['text_length_n_words'], axis=1)
    word_lists.to_csv(f'data/evaluation/{dataset_name}/{model_name}/word_lists_{prompt_id}_target_{target_level}.csv', index=False)


def compute_and_save_word_list_evaluation():
    for dataset_name_param in [CERD, CAM_MCQ]:
        for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt_id_param in ['01', '02', '11', '12', 'w01', 'w02']:
                if prompt_id_param in ['w01', 'w02']:
                    word_list_evaluation(dataset_name_param, model_name_param, prompt_id_param, 'A1')
                else:
                    for target_level in CEFR_LEVELS[:-1]:
                        word_list_evaluation(dataset_name_param, model_name_param, prompt_id_param, target_level)


if __name__ == '__main__':
    compute_and_save_word_list_evaluation()

    # this is to plot the results
    for dataset_name_param in [CERD, CAM_MCQ, 'aggregate']:
        for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt_id_param in ['01', '02', '11', '12']:
                if dataset_name_param in [CERD, CAM_MCQ]:
                    word_lists_count_per_level = [
                        pd.read_csv(f'data/evaluation/{dataset_name_param}/{model_name_param}/word_lists_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                    line_plot_word_lists_count(word_lists_count_per_level, f"{model_name_param} | {prompt_id_param}", f"{dataset_name_param}_{model_name_param}_{prompt_id_param}", figsize=(3, 2.1))
                else:
                    word_lists_count_per_level = [
                        pd.read_csv(f'data/evaluation/{CERD}/{model_name_param}/word_lists_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                    word_lists_count_per_level_cam_mcq = [
                        pd.read_csv(f'data/evaluation/{CAM_MCQ}/{model_name_param}/word_lists_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                    for idx in range(len(word_lists_count_per_level)):
                        word_lists_count_per_level[idx] = pd.concat(
                            [word_lists_count_per_level[idx], word_lists_count_per_level_cam_mcq[idx]], ignore_index=True)
                    line_plot_word_lists_count(word_lists_count_per_level, f"{model_name_param} | {prompt_id_param}", f"{dataset_name_param}_{model_name_param}_{prompt_id_param}", figsize=(3, 2.1))
