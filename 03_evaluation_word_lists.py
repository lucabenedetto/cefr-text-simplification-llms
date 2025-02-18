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
from constants import (
    CERD,
    CAM_MCQ,
    MODEL_NAME_TO_STR,
    PROMPT_ID_TO_STR,
)
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



def get_word_list_count_and_frac(word_lists_per_level, model, prompt):  #, title, filename=None, figsize=(4, 3)):
    # word_lists_per_level is a list of 5 DF (one for each CEFR level from A1 to C1). Each dataframe contains, for the
    # reading passages simplified at that CEFR level, the fraction of words from different CEFR levels (according to
    # CEFRj).
    out_df = pd.DataFrame(columns=['model', 'prompt', 'target_cefr', 'word_level', 'avg_frac'])
    for level in CEFR_LEVELS:  # This is the CEFR level of the words from the wordlist.
        for idx, local_df in enumerate(word_lists_per_level):
            # this is for the target levels of the simplification
            new_row = {
                'model': [model],
                'prompt': [prompt],
                'target_cefr': [CEFR_LEVELS[idx]],
                'word_level': [level],
                'avg_frac': [local_df[level + '_frac'].mean()],
            }
            if len(out_df) == 0:
                out_df = pd.DataFrame(new_row)
            else:
                out_df = pd.concat([out_df, pd.DataFrame(new_row)], ignore_index=True)
    return out_df


if __name__ == '__main__':
    # The line below is meant to be run only once, not to be re-run if you update the evaluation.
    # compute_and_save_word_list_evaluation()

    out_df = pd.DataFrame(columns=['model', 'prompt', 'target_cefr', 'word_level', 'avg_frac'])

    # this is to plot the results
    for dataset_name_param in ['aggregate']:  # [CERD, CAM_MCQ, 'aggregate']:
        for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt_id_param in ['01', '02', '11', '12']:
                if dataset_name_param in [CERD, CAM_MCQ]:
                    word_lists_count_per_level = [
                        pd.read_csv(f'data/evaluation/{dataset_name_param}/{model_name_param}/word_lists_{prompt_id_param}_target_{level}.csv')
                        for level in CEFR_LEVELS[:-1]
                    ]
                else:
                    # This is for the AGGREGATE case.
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

                # TODO method that computes the count/frac of words per level.
                new_df = get_word_list_count_and_frac(word_lists_count_per_level, model_name_param, prompt_id_param)
                if len(out_df) == 0:
                    out_df = new_df.copy()
                else:
                    out_df = pd.concat([out_df, new_df], ignore_index=True)

                # This to save the results as plot
                # line_plot_word_lists_count(word_lists_count_per_level, f"{MODEL_NAME_TO_STR[model_name_param]} | {PROMPT_ID_TO_STR[prompt_id_param]}", f"{dataset_name_param}_{model_name_param}_{prompt_id_param}", figsize=(3, 2.1))


    # This is to print the latex table that has as columns the target levels of the simplification, and as rows the
    # model-prompt pairs (different tables for each CEFR level from the word list).
    for word_level in CEFR_LEVELS:
        print(f"Word level: {word_level}")
        output_str = ""
        for model in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            for prompt in ['01', '02', '11', '12']:
                output_str += f"{MODEL_NAME_TO_STR[model]} | {PROMPT_ID_TO_STR[prompt]} | "
                local_df = out_df[(out_df['word_level'] == word_level) & (out_df['model'] == model) & (out_df['prompt'] == prompt)].sort_values('target_cefr', ascending=True)
                for target_cefr, avg_frac in local_df[['target_cefr', 'avg_frac']].values:
                    output_str += "%s | %.4f | " % (target_cefr, avg_frac)
                # print(local_df)
                output_str += "\n"
        print(output_str)

    # This is the same as above, but averaging the results for the different prompts
    for word_level in CEFR_LEVELS:
        print("\\hline")
        print(f"% Word level: {word_level}")
        output_str = ""
        for model in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
            output_str += f"%13s " % MODEL_NAME_TO_STR[model]
            frac_dict = {cefr: 0 for cefr in CEFR_LEVELS[:-1]}
            for prompt in ['01', '02', '11', '12']:
                local_df = out_df[(out_df['word_level'] == word_level) & (out_df['model'] == model) & (out_df['prompt'] == prompt)].sort_values('target_cefr', ascending=True)
                for target_cefr, avg_frac in local_df[['target_cefr', 'avg_frac']].values:
                    frac_dict[target_cefr] += avg_frac
            for target_cefr, frac in frac_dict.items():
                # output_str += "%s & %.4f & " % (target_cefr, frac/4.0)
                output_str += "& %.4f " % (frac/4.0)
            output_str += "\\\\ \n"
        print(output_str)
