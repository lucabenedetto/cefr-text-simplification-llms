import json
import pandas as pd
from src.constants import OPENAI_MODEL_NAMES, PRETRAINED_MODEL_NAMES
from constants import CERD, CAM_MCQ


def get_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == CERD:
        return pd.read_csv('data/input/cerd.csv')
    if dataset_name == CAM_MCQ:
        return pd.read_csv('data/input/mcq_cupa.csv')
    raise ValueError(f'Dataset {dataset_name} not supported.')


def get_key_from_model_name(model_name: str) -> str:
    if model_name in OPENAI_MODEL_NAMES:
        with open('/home/luca/.keys/openai_key.json', 'r') as f:
            data = json.load(f)
        api_key = data['key']
        return api_key
    if model_name in PRETRAINED_MODEL_NAMES:
        # with open('/home/lb990/.keys/hf_access_token.json', 'r') as f:  # This is for my local machine
        with open('/home/lb990/.keys/hf_access_token.json', 'r') as f:  # This is for the server
            data = json.load(f)
        access_token = data['key']
        return access_token
    else:
        raise ValueError(f'Model name {model_name} not recognized.')
