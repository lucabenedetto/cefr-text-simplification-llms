"""
Script for data preparation. It takes the two raw datasets and converts them in a common format.

Datasets:
- CERD
- Cambridge reading MCQs -- for this one for the questions as well

Input format depends on the dataset, columns of the output dataframes are defined in the COLUMN_NAMES constant:
- text_id
- text
- text_level
- question_id (this is unique!)
- question
- options
- correct_answer
- question_difficulty

"""
import json
import pandas as pd
import os
import re

from src.constants import (
    COLUMN_NAMES,
    COLUMN_TEXT_ID,
    COLUMN_TEXT,
    COLUMN_TEXT_LEVEL,
    COLUMN_QUESTION_ID,
    COLUMN_QUESTION,
    COLUMN_OPTIONS,
    COLUMN_OPTION_A,
    COLUMN_OPTION_B,
    COLUMN_OPTION_C,
    COLUMN_OPTION_D,
    COLUMN_CORRECT_ANSWER,
    COLUMN_QUESTION_DIFFICULTY,
)


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)


def prepare_dataset_cam_mcq():
    out_df = pd.DataFrame(columns=COLUMN_NAMES)

    # Iterate over all the texts
    for line in read_jsonl('data/raw/Cambridge Multiple-Choice Questions Reading Dataset.jsonl'):
        text_id = line['id']
        text = line['text']
        text = re.sub('\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text_level = line['level']
        # The following fields are not used in this work.
        # text_title = line['title']
        # text_difficulty = line['diff']
        # text_facility = line['fac']
        for local_question_id in line['questions'].keys():
            question_text = line['questions'][local_question_id]['text']
            question_correct_answer = line['questions'][local_question_id]['answer']
            question_difficulty = line['questions'][local_question_id]['diff']
            option_a = line['questions'][local_question_id]['options']['a']['text']
            option_b = line['questions'][local_question_id]['options']['b']['text']
            option_c = line['questions'][local_question_id]['options']['c']['text']
            option_d = line['questions'][local_question_id]['options']['d']['text']
            # Below other fields which are not used for this work. Disc and fac are available for all options.
            # question_discrimination = line['questions'][local_question_id]['disc']
            # question_facility = line['questions'][local_question_id]['fac']
            # option_a_disc = line['questions'][local_question_id]['options']['a']['disc']
            # option_a_fac = line['questions'][local_question_id]['options']['a']['fac']

            new_row_df = pd.DataFrame({
                COLUMN_TEXT_ID: [text_id],
                COLUMN_TEXT: [text],
                COLUMN_TEXT_LEVEL: [text_level],
                COLUMN_QUESTION_ID: [str(text_id) + '_' + str(local_question_id)],
                COLUMN_QUESTION: [question_text],
                COLUMN_OPTIONS: [str([option_a, option_b, option_c, option_d])],
                COLUMN_OPTION_A: [option_a],
                COLUMN_OPTION_B: [option_b],
                COLUMN_OPTION_C: [option_c],
                COLUMN_OPTION_D: [option_d],
                COLUMN_CORRECT_ANSWER: [question_correct_answer],
                COLUMN_QUESTION_DIFFICULTY: [question_difficulty],
            })
            out_df = pd.concat([out_df, new_row_df], ignore_index=True)
    out_df.to_csv('data/input/mcq_cupa.csv', index=False)
    out_df.drop_duplicates(COLUMN_TEXT_ID).to_csv('data/input/mcq_cupa_texts.csv', index=False)


def prepare_dataset_cerd():
    # List of folders to iterate over
    folders_and_target_levels = (
        ('CPE', 'C2'),  # https://www.cambridgeenglish.org/exams-and-tests/proficiency/
        ('CAE', 'C1'),  # https://www.cambridgeenglish.org/exams-and-tests/advanced/
        ('FCE', 'B2'),  # https://www.cambridgeenglish.org/exams-and-tests/first/
        ('PET', 'B1'),  # https://www.cambridgeenglish.org/exams-and-tests/preliminary/
        ('KET', 'A2'),  # https://www.cambridgeenglish.org/exams-and-tests/key/
    )

    out_df = pd.DataFrame(columns=COLUMN_NAMES)
    for folder, target_level in folders_and_target_levels:
        print(folder, target_level)
        # Iterate over all files in the current folder
        for file in os.listdir(os.path.join('data/raw/Readability_dataset', folder)):
            text_id = os.path.join(folder, file).replace('.txt', '').replace('/', '_')
            # print(text_id)
            with open(os.path.join('data/raw/Readability_dataset', folder, file), 'r') as f:
                text = f.read()
                text = re.sub('\n', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                out_df = pd.concat(
                    [
                        out_df,
                        pd.DataFrame({COLUMN_TEXT: [text], COLUMN_TEXT_ID: [text_id], COLUMN_TEXT_LEVEL: [target_level]})
                    ],
                    ignore_index=True
                )
    out_df.to_csv('data/input/cerd.csv', index=False)


if __name__ == '__main__':
    prepare_dataset_cam_mcq()
    prepare_dataset_cerd()
