from typing import Dict
import pandas as pd
from src.constants import COLUMN_TEXT, COLUMN_TEXT_ID
from src.prompts import get_prompt_from_prompt_id


class BaseLLMAdapter(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.prompt = None  # this is defined later in the convert_texts method

    def convert_texts(self, prompt_id: str, dataset: pd.DataFrame, target_cefr_level: str) -> Dict[str, str]:
        self.prompt = get_prompt_from_prompt_id(prompt_id=prompt_id, target_cefr_level=target_cefr_level)
        converted_texts = dict()
        for text_id, text in dataset[[COLUMN_TEXT_ID, COLUMN_TEXT]].values:
            if text_id not in converted_texts.keys():
                print(f"Processing text_id {text_id}")  # TODO make this logging.
                converted_texts[text_id] = self.convert_single_text(text)
        return converted_texts

    def convert_single_text(self, text) -> str:
        raise NotImplementedError("convert_single_text not implemented for the Base class, use the other classes.")
