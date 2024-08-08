from typing import Tuple
import json
import openai
import pandas as pd

from src.constants import (
    COLUMN_NAMES,
    COLUMN_TEXT_ID,
    COLUMN_TEXT,
    COLUMN_QUESTION_ID,
    COLUMN_QUESTION,
    COLUMN_OPTION_A,
    COLUMN_OPTION_B,
    COLUMN_OPTION_C,
    COLUMN_OPTION_D,
    COLUMN_CORRECT_ANSWER,
    # GPT_4o_240806,
    COLUMN_LLM_ANSWER_INDEX,
    COLUMN_LLM_ANSWER_TEXT,
    COLUMN_LLM_ANSWER_CORRECT,
)


class AnswerabilityEvaluator:

    def __init__(self):
        pass

    def evaluate_answerability(self,
                               questions_df: pd.DataFrame,
                               api_key: str,
                               model_name: str = 'gpt-4o-2024-08-06',
                               temperature: float = 1.0,
                               ) -> pd.DataFrame:
        """
        Evaluate the effects of simplifying the texts by using an LLM to answer the reading comprehension questions.
        Returns a dataframe with the responses (and whether they are correct).
        At this stage, it only works with OpenAI models (and they need to have the param that you can set a json output)
        :param questions_df:
        :param api_key:
        :param model_name:
        :param temperature:
        :return:
        """
        for column in COLUMN_NAMES:
            if column not in questions_df.columns:
                raise ValueError(f"Column {column} not found in questions dataframe.")

        openai.api_key = api_key

        out_df = pd.DataFrame(columns=[
            COLUMN_QUESTION_ID, COLUMN_TEXT_ID, COLUMN_CORRECT_ANSWER,
            COLUMN_LLM_ANSWER_INDEX, COLUMN_LLM_ANSWER_TEXT, COLUMN_LLM_ANSWER_CORRECT
        ])

        system_prompt = f"""You will be shown a multiple choice question from an English reading comprehension exam.
You have to select the correct answer from the given options.
Provide only a JSON file with the following structure:
{{"index": "index of the correct answer", "text": "text of the correct answer"}}
"""
        # iterate over all the questions in the dataframe
        for index, row in questions_df.iterrows():
            user_prompt = f"""Reading passage: "{row[COLUMN_TEXT]}"
Question: "{row[COLUMN_QUESTION]}"
Options: "['{row[COLUMN_OPTION_A]}', '{row[COLUMN_OPTION_B]}', '{row[COLUMN_OPTION_C]}', '{row[COLUMN_OPTION_D]}']"
"""
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
                    temperature=temperature,
                    response_format={"type": 'json_object'}
                )
                answer = response['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                # this if the GPT model did not produce a response
                answer = "{'index': -9, 'text': 'None'}"
            llm_answer_index, llm_answer_text = self.validate_answer(answer)
            llm_answer_index = {'0': 'a', '1': 'b', '2': 'c', '3': 'd'}.get(llm_answer_index, 'x')
            new_row_df = pd.DataFrame({COLUMN_QUESTION_ID: [row[COLUMN_QUESTION_ID]],
                                       COLUMN_TEXT_ID: [row[COLUMN_TEXT_ID]],
                                       COLUMN_CORRECT_ANSWER: row[COLUMN_CORRECT_ANSWER],
                                       COLUMN_LLM_ANSWER_INDEX: [llm_answer_index],
                                       COLUMN_LLM_ANSWER_TEXT: [llm_answer_text],
                                       COLUMN_LLM_ANSWER_CORRECT: [llm_answer_index == row[COLUMN_CORRECT_ANSWER]],
                                       })
            out_df = pd.concat([out_df, new_row_df], ignore_index=True)
        return out_df

    @staticmethod
    def validate_answer(answer: str) -> Tuple[str, str]:
        try:
            answer_json = json.loads(answer)
            index_str = str(answer_json['index'])
            answer_text = str(answer_json['text'])
            if not index_str.isdigit():
                print("The index is not an integer.")
                return '-8', 'None'
            return str(index_str), answer_text
        except json.JSONDecodeError:
            print("The answer is not a valid JSON string.")
            return '-7', 'None'
        except KeyError:
            print("'index' not in keys.")
            return '-6', 'None'
