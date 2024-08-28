import json
import pandas as pd
from src.constants import OPENAI_MODEL_NAMES, PRETRAINED_MODEL_NAMES, COLUMN_TEXT_ID, COLUMN_TEXT, COLUMN_TEXT_LEVEL, \
    CEFR_LEVELS
from constants import CERD, CAM_MCQ
from src.evaluators.readability import ReadabilityEvaluator


def get_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == CERD:
        return pd.read_csv('data/input/cerd.csv')
    if dataset_name == CAM_MCQ:
        return pd.read_csv('data/input/mcq_cupa_texts.csv')
    raise ValueError(f'Dataset {dataset_name} not supported.')


def get_key_from_model_name(model_name: str) -> str:
    if model_name in OPENAI_MODEL_NAMES:
        with open('/home/luca/.keys/openai_key.json', 'r') as f:
            data = json.load(f)
        api_key = data['key']
        return api_key
    if model_name in PRETRAINED_MODEL_NAMES:
        # with open('/home/luca/.keys/hf_access_token.json', 'r') as f:  # This is for my local machine
        with open('/home/lb990/.keys/hf_access_token.json', 'r') as f:  # This is for the server
            data = json.load(f)
        access_token = data['key']
        return access_token
    else:
        raise ValueError(f'Model name {model_name} not recognized.')


def get_readability_indexes_per_target_level(df):
    texts_dict = {text_id: text for text_id, text in df[[COLUMN_TEXT_ID, COLUMN_TEXT]].values}
    target_level_dict = {text_id: level for text_id, level in df[[COLUMN_TEXT_ID, COLUMN_TEXT_LEVEL]].values}
    read_idxs = ReadabilityEvaluator().compute_readability_indexes(texts_dict)
    read_idxs[COLUMN_TEXT_LEVEL] = read_idxs.apply(lambda r: target_level_dict[r[COLUMN_TEXT_ID]], axis=1)
    read_idxs_level = [read_idxs[read_idxs[COLUMN_TEXT_LEVEL] == level] for level in CEFR_LEVELS]
    return read_idxs_level
