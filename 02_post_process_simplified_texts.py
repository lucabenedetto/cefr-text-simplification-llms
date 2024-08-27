import os
import pandas as pd
from src.constants import (
    CEFR_LEVELS,
    COLUMN_TEXT_ID,
    COLUMN_TEXT_LEVEL,
    GEMMA_2B,
    GEMMA_7B,
    GPT_4o_240806,
    GPT_4o_MINI_240718,
    LLAMA_3_8B,
)
from utils import get_dataset
from constants import CERD, CAM_MCQ


def try_split_by(text, split_by, keep_idx):
    try:
        return text.split(split_by)[keep_idx]
    except IndexError:
        return text


def _post_process_llama_simplified_text(text, text_id):
    text = try_split_by(text, 'end_header_id|>', 1)
    text = try_split_by(text, '<|eot_id|>', 0)
    text = text.replace("\n\n", "\n")
    text = try_split_by(text, f'Here is the simplified text:', 1)
    # it often refers to the requested level
    for level in CEFR_LEVELS[:-1]:
        if level in ['A1', 'A2']:
            ax = 'an'
        else:
            ax = 'a'
        text = try_split_by(text, f'Here is a simplified version of the text for {ax} {level} level learner:', 1)
        text = try_split_by(text, f'Here is a simplified version of the text for {ax} {level} learner:', 1)
        text = try_split_by(text, f'Here is a simplified version of the text, adapted for {ax} {level} level learner:', 1)
        text = try_split_by(text, f'Here is a simplified version of the text, suitable for {ax} {level} learner:', 1)
        text = try_split_by(text, f'Here is a simplified version of the text, suitable for {ax} {level} level learner:', 1)
        text = try_split_by(text, f'Here is the simplified text for {ax} {level} learner:', 1)
        text = try_split_by(text, f'Here is the simplified text for {ax} {level} level learner:', 1)
        text = try_split_by(text, f'Here is the simplified text, adapted for {ax} {level} level learner:', 1)
        text = try_split_by(text, f'Here is the simplified text, adapted for {ax} {level} learner:', 1)
        text = try_split_by(text, f'Here is the simplified text, suitable for {ax} {level} level learner:', 1)
        text = try_split_by(text, f'Here is the simplified text, suitable for {ax} {level} learner:', 1)
    # It often describes what it did, although this was not requested.
    text = try_split_by(text, 'Note:', 0)
    text = try_split_by(text, 'Changes made:', 0)
    text = try_split_by(text, 'I changed the text by:', 0)
    text = try_split_by(text, 'I simplified the text by:', 0)
    text = try_split_by(text, 'I made the following changes:', 0)
    text = try_split_by(text, 'I changed the following:', 0)
    text = try_split_by(text, 'I changed the text to make it easier to understand by:', 0)
    text = try_split_by(text, 'I made the following changes to simplify the text:', 0)
    text = try_split_by(text, 'I changed the text to make it easier to understand', 0)  # "...for a learer of level XX by:"
    text = try_split_by(text, 'I minimized changes to the factual content while making the text easier to understand', 0)  # "...for a learer of level XX by:"
    text = try_split_by(text, 'I made some changes to the original text to make it more accessible to a B1 learner, including:', 0)  # "...for a learer of level XX by:"
    # Some "motivational" comments
    text = try_split_by(text, "Let me know if you'd like me to simplify anything further!", 0)
    text = try_split_by(text, "I hope this simplified text meets your requirements!", 0)
    for level in CEFR_LEVELS[:-1]:
        text = try_split_by(text, f'I hope this simplified version helps {level} learners understand the text better!', 0)
    # text = try_split_by(text, '**Simplified Text:**', 1)
    if len(text) < 5:
        return '-10'
    return text


def _post_process_gpt_simplified_text(text, text_id):
    if text == "{'index': -9, 'text': 'None'}":
        return '-9'
    text = text.replace("\n\n", "\n")
    text = try_split_by(text, '**Simplified Text:**', 1)
    text = try_split_by(text, 'Text: ', 1)
    if len(text) < 5:
        return '-10'
    return text


def _post_process_gemma_simplified_text(text, text_id):
    # print(f"\n- - - - - - - - - -\nDoing text_id {text_id}.")
    # # These are some common errors which I found by manually looking at the texts
    if '**Questions:**' in text:
        return "-1"
    if 'Please note that this is just a sample text' in text:
        # I found (Gemma 2B):
        # Please note that this is just a sample text and may not be representative of all texts on the same topic.
        # "Please note that this is just a sample text and may not be representative of all texts at this level."
        return "-2"
    if 'What is the main idea of the passage?' in text:
        return "-3"
    text = text[3:]
    text = text.replace("\n\n", "\n")
    text = text.split('<eos>')[0]
    text = try_split_by(text, '**Simplified Text:**', 1)
    text = try_split_by(text, '**Simplified version:**', 1)
    text = try_split_by(text, 'Simplified Text:', 1)
    text = try_split_by(text, 'Simplified text:', 1)
    text = try_split_by(text, '**Answer:**', 1)  # Sometimes answer is used to refer to the simplified text
    text = try_split_by(text, '**Please simplify the passage below:**', 1)
    for level in CEFR_LEVELS[:-1]:
        text = try_split_by(text, f'Please help me simplify the text for a learner of {level} level on the CEFR.', 1)
        text = try_split_by(text, f'Please help me simplify the text for a learner of {level} on the CEFR.', 1)
        text = try_split_by(text, f'Please help me simplify the text for a learner of {level} English.', 1)
    text = try_split_by(text, 'Sure, here is the simplified text:', 1)
    text = try_split_by(text, 'Sure, here is the simplified passage:', 1)
    text = try_split_by(text, 'Sure, here is the simplified version of the text:', 1)
    text = try_split_by(text, 'Sure, here is the simplified text you requested:', 1)
    text = try_split_by(text, 'Here is the simplified text:', 1)
    text = try_split_by(text, 'The following is a simplified version of the text:', 1)
    text = try_split_by(text, '## Reading Comprehension Questions', 0)
    # Provides some descriptors which are not needed nor requested
    text = try_split_by(text, 'It is written in a style that', 0)
    text = try_split_by(text, 'It is written in a clear and concise style', 0)
    text = try_split_by(text, 'The text is written in a style that', 0)
    text = try_split_by(text, 'The text is written in a complex style', 0)

    # An example (from Gemma 7B):
    # ## Reading Comprehension Questions
    # 1. Where is Lake Nipissing located?
    # 2. What do the fishermen build on the ice?
    # 3. What do you need to wear when you go fishing on Lake Nipissing in winter?
    # 4. What is the process of catching fish on Lake Nipissing?
    # 5. What does Bob Marvisch like to do when he has caught fish?"
    text = try_split_by(text, '**Additional Notes:**', 0)
    # An example (from Gemma 2B):
    # **Additional Notes:**
    # * The passage uses simple vocabulary and short sentences.
    # * The passage provides a variety of activities to choose from, catering to different interests.
    # * The passage emphasizes the beauty and wonder of Australia.
    text = try_split_by(text, '**Note:**', 0)

    # Comments for Gemma 2B
    # # # Other texts which I found for Gemma 2B (where it actually performed the simplification)
    # I hope this is the information you were looking for.
    # Please note that this simplified text has been modified to ensure it is appropriate for a learner of A1 on the CEFR.
    # Sure, here's a simplified version of the text for an A1 learner:
    # Please simplify the passage to a level A1 on the CEFR. <-- but actually does the simplification
    # # # Here it actually performed a very strict summarisation.
    # "This passage tells about the favourite memories of five celebrities of different nationalities when they took train journeys."
    if len(text) < 5:
        return '-10'
    if 'Answer the following questions based on the text:' in text:
        return '-1'
    return text


def post_process_responses(model_name, dataset_name, prompt_id, target_level):
    path = f'data/output/{dataset_name}/{model_name}/{prompt_id}'
    df_simplified_texts = pd.read_csv(os.path.join(path, f'df_converted_texts_{target_level}.csv'))
    df_simplified_texts = filter_df_simplified_texts_by_cefr_level(df_simplified_texts, dataset_name, target_level)
    if model_name in [GEMMA_2B, GEMMA_7B]:
        df_simplified_texts['processed_text'] = df_simplified_texts.apply(lambda r: _post_process_gemma_simplified_text(r['text'], r['text_id']), axis=1)
    elif model_name in [GPT_4o_240806, GPT_4o_MINI_240718]:
        df_simplified_texts['processed_text'] = df_simplified_texts.apply(lambda r: _post_process_gpt_simplified_text(r['text'], r['text_id']), axis=1)
    elif model_name in [LLAMA_3_8B]:
        df_simplified_texts['processed_text'] = df_simplified_texts.apply(lambda r: _post_process_llama_simplified_text(r['text'], r['text_id']), axis=1)
    else:
        raise ValueError(f'Model {model_name} not recognized.')
    df_simplified_texts = df_simplified_texts.drop('text', axis=1)
    df_simplified_texts = df_simplified_texts.rename(columns={'processed_text': 'text'})
    df_simplified_texts.to_csv(os.path.join(path, f'df_converted_texts_post_processed_{target_level}.csv'), index=False)


def filter_df_simplified_texts_by_cefr_level(df_simplified_texts, dataset_name, target_level):
    input_df = get_dataset(dataset_name)
    dict_target_level_by_text_id = {text_id: target_level for text_id, target_level in input_df[[COLUMN_TEXT_ID, COLUMN_TEXT_LEVEL]].values}
    df_simplified_texts[COLUMN_TEXT_LEVEL] = df_simplified_texts.apply(lambda r: dict_target_level_by_text_id[r[COLUMN_TEXT_ID]], axis=1)
    len_0 = len(df_simplified_texts)
    df_simplified_texts = df_simplified_texts[df_simplified_texts[COLUMN_TEXT_LEVEL] > target_level]  # This works because 'A1' < 'A2' < 'B1' < 'B2' < 'C1' < 'C2' is True
    print(f"Removed {len_0-len(df_simplified_texts)} because of target level ({target_level}) being higher or equal to original level.")
    return df_simplified_texts


if __name__ == '__main__':
    for param_dataset_name in [CERD, CAM_MCQ]:
        for param_target_level in CEFR_LEVELS[:-1]:  # Because I don't perform text simplification to level C2
            for param_prompt_id in ['01', '02', '11', '12']:
                post_process_responses(GEMMA_2B, param_dataset_name, param_prompt_id, param_target_level)
                post_process_responses(GEMMA_7B, param_dataset_name, param_prompt_id, param_target_level)
                post_process_responses(GPT_4o_MINI_240718, param_dataset_name, param_prompt_id, param_target_level)
                post_process_responses(GPT_4o_240806, param_dataset_name, param_prompt_id, param_target_level)
                post_process_responses(LLAMA_3_8B, param_dataset_name, param_prompt_id, param_target_level)
        for param_prompt_id in ['w01', 'w02']:
            post_process_responses(GEMMA_2B, param_dataset_name, param_prompt_id, 'A1')
            post_process_responses(GEMMA_7B, param_dataset_name, param_prompt_id, 'A1')
            post_process_responses(GPT_4o_MINI_240718, param_dataset_name, param_prompt_id, 'A1')
            post_process_responses(GPT_4o_240806, param_dataset_name, param_prompt_id, 'A1')
            post_process_responses(LLAMA_3_8B, param_dataset_name, param_prompt_id, 'A1')
