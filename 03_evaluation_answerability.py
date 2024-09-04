import pandas as pd

from src.constants import (
    GEMMA_2B,
    GEMMA_7B,
    LLAMA_3_8B,
    GPT_4o_240806,
    GPT_4o_MINI_240718,
    CEFR_LEVELS,
    COLUMN_LLM_ANSWER_CORRECT,
)
from constants import CAM_MCQ
from src.evaluators.answerability import AnswerabilityEvaluator
from utils import get_key_from_model_name


def answerability_evaluation(dataset_name: str, model_name: str, prompt_id: str, target_level: str):
    converted_texts_1 = pd.read_csv(f'data/output/{dataset_name}/{model_name}/{prompt_id}/df_converted_texts_post_processed_{target_level}.csv')
    len_0 = len(converted_texts_1)
    # TODO: make a constant with the list of possible errors, and a constant for each error.
    converted_texts_1['error'] = converted_texts_1.apply(lambda r: r['text'] in ['-9', '-10'], axis=1)
    out_df = converted_texts_1[~converted_texts_1['error']]
    if len(out_df) != len_0:
        print(f"Removed {len_0 - len(out_df)} rows.")
    questions_df = pd.read_csv(f'data/input/mcq_cupa.csv')
    questions_df = questions_df.drop(columns=['text', 'text_level'])
    questions_df = pd.merge(questions_df, converted_texts_1, on='text_id', how='left')
    print("Doing:", dataset_name, model_name, prompt_id, target_level)
    api_key = get_key_from_model_name(GPT_4o_MINI_240718)
    out_df = AnswerabilityEvaluator().evaluate_answerability(questions_df, api_key=api_key)
    out_df.to_csv(f'data/evaluation/{dataset_name}/{model_name}/answerability_{prompt_id}_target_{target_level}.csv', index=False)


def compute_and_save_answerability_evaluation():
    dataset_name_param = CAM_MCQ
    for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
        for prompt_id_param in ['01', '02', '11', '12']:
            for target_level in CEFR_LEVELS[:-1]:
                answerability_evaluation(dataset_name_param, model_name_param, prompt_id_param, target_level)


if __name__ == '__main__':
    compute_and_save_answerability_evaluation()

    # this is to plot the results
    result_df = pd.DataFrame(columns=['dataset_name', 'model', 'prompt_id', 'target_level', 'accuracy', 'support'])
    dataset_name_param = CAM_MCQ
    for model_name_param in [GEMMA_2B, GEMMA_7B, LLAMA_3_8B, GPT_4o_240806, GPT_4o_MINI_240718]:
        for prompt_id_param in ['01', '02', '11', '12']:
            # evaluation single CEFR levels
            for level in CEFR_LEVELS[:-1]:
                answerability_eval_per_level = pd.read_csv(f'data/evaluation/{dataset_name_param}/{model_name_param}/answerability_{prompt_id_param}_target_{level}.csv')
                new_row_df = pd.DataFrame({
                    'dataset_name': [dataset_name_param],
                    'model': [model_name_param],
                    'prompt_id': [prompt_id_param],
                    'target_level': [level],
                    'accuracy': [answerability_eval_per_level['llm_answer_correct'].mean()],
                    'support': [len(answerability_eval_per_level)],
                })
                result_df = pd.concat([result_df, new_row_df], ignore_index=True)
            # aggregate evaluation
            local_df = pd.concat([
                pd.read_csv(f'data/evaluation/{dataset_name_param}/{model_name_param}/answerability_{prompt_id_param}_target_{level}.csv')
                for level in CEFR_LEVELS[:-1]
            ], ignore_index=True)
            new_row_df = pd.DataFrame({
                'dataset_name': [dataset_name_param],
                'model': [model_name_param],
                'prompt_id': [prompt_id_param],
                'target_level': ['all'],
                'accuracy': [local_df['llm_answer_correct'].mean()],
                'support': [len(local_df)],
            })
            result_df = pd.concat([result_df, new_row_df], ignore_index=True)

    result_df.to_csv(f'data/evaluation/answerability_evaluation.csv', index=False)

    questions_df = pd.read_csv(f'data/input/mcq_cupa.csv')
    api_key = get_key_from_model_name(GPT_4o_MINI_240718)
    for level in CEFR_LEVELS:
        out_df = AnswerabilityEvaluator().evaluate_answerability(questions_df[questions_df['text_level'] == level], api_key=api_key)
        print(f"{level}: ACC: {out_df[COLUMN_LLM_ANSWER_CORRECT].mean()}, Support: {len(out_df)}")
    # A1: ACC: nan, Support: 0
    # A2: ACC: nan, Support: 0
    # B1: ACC: 0.9714285714285714, Support: 140
    # B2: ACC: 0.919431279620853, Support: 422
    # C1: ACC: 0.8947368421052632, Support: 171
    # C2: ACC: 0.9354838709677419, Support: 62
