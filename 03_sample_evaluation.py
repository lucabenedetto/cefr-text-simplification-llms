import pickle

import pandas as pd
import numpy as np
from src.constants import (
    COLUMN_TEXT_ID,
    GPT_4o_MINI_240718,
    OPENAI_MODEL_NAMES,
)
from src.evaluators.readability import ReadabilityEvaluator
from src.evaluators.answerability import AnswerabilityEvaluator
from src.evaluators.word_list import WordListEvaluator
from utils import get_key_from_model_name


def readability_evaluation():
    # first test ReadabilityEvaluator
    converted_texts_1 = pickle.load(open('data/output/cerd/gpt-3.5-turbo-0613/00/converted_texts_A1.pkl', 'rb'))
    evaluator = ReadabilityEvaluator()
    readability_indexes = evaluator.compute_readability_indexes(converted_texts_1)
    print(readability_indexes)


def answerability_evaluation():
    # first test AnswerabilityEvaluator
    converted_texts_1 = pickle.load(open('data/output/cam_mcq/gpt_4o_mini_240718/00/converted_texts_A1.pkl', 'rb'))
    df = pd.read_csv('data/input/mcq_cupa.csv')
    df = df[df[COLUMN_TEXT_ID].isin(converted_texts_1.keys())].copy()
    evaluator = AnswerabilityEvaluator()
    api_key = get_key_from_model_name(GPT_4o_MINI_240718)
    llm_responses_original_df = evaluator.evaluate_answerability(df, api_key, OPENAI_MODEL_NAMES[GPT_4o_MINI_240718], 1)
    print(np.mean(llm_responses_original_df['llm_answer_correct']))
    df['text'] = df.apply(lambda r: converted_texts_1[r['text_id']], axis=1)
    llm_responses_simplified_df = evaluator.evaluate_answerability(df, api_key, OPENAI_MODEL_NAMES[GPT_4o_MINI_240718], 1)
    print(np.mean(llm_responses_simplified_df['llm_answer_correct']))
    print(llm_responses_simplified_df)


def word_lists_evaluation():
    # first test WordListEvaluator
    converted_texts_1 = pickle.load(open('data/output/cam_mcq/gpt_4o_mini_240718/00/converted_texts_A1.pkl', 'rb'))
    evaluator = WordListEvaluator()
    word_frequency_per_level = evaluator.measure_word_frequency(converted_texts_1)
    print(word_frequency_per_level)


if __name__ == '__main__':
    readability_evaluation()
    answerability_evaluation()
    word_lists_evaluation()
