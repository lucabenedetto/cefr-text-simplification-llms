from typing import Dict

import pandas as pd
import textstat
from src.evaluators.constants import (
    FLESCH_READING_EASE,
    FLESCH_KINCAID_GRADE_LEVEL,
    AUTOMATED_READABILITY_INDEX,
    GUNNING_FOG_INDEX,
    COLEMAN_LIAU,
    SMOG_INDEX,
    LINSEAR_WRITE_FORMULA,
    DALE_CHALL,
    READABILITY_INDEXES,
)


class ReadabilityEvaluator:

    def __init__(self):
        pass

    def compute_readability_indexes(self, dict_texts: Dict[str, 'str']) -> pd.DataFrame:
        df = pd.DataFrame(columns=READABILITY_INDEXES)
        for text_id, text in dict_texts.items():
            readability_scores = self.compute_readability_indexes_single_text(text)
            new_row_df = pd.DataFrame({
                FLESCH_READING_EASE: [readability_scores[FLESCH_READING_EASE]],
                FLESCH_KINCAID_GRADE_LEVEL: [readability_scores[FLESCH_KINCAID_GRADE_LEVEL]],
                AUTOMATED_READABILITY_INDEX: [readability_scores[AUTOMATED_READABILITY_INDEX]],
                GUNNING_FOG_INDEX: [readability_scores[GUNNING_FOG_INDEX]],
                COLEMAN_LIAU: [readability_scores[COLEMAN_LIAU]],
                SMOG_INDEX: [readability_scores[SMOG_INDEX]],
                LINSEAR_WRITE_FORMULA: [readability_scores[LINSEAR_WRITE_FORMULA]],
                DALE_CHALL: [readability_scores[DALE_CHALL]],
            })
            df = pd.concat([df, new_row_df], ignore_index=True)
        return df

    @staticmethod
    def compute_readability_indexes_single_text(text: str) -> Dict[str, float]:
        """
        This function computes the readability indexes of a given text.
        :param text:
        :return:
        """
        readability_indexes = dict()
        readability_indexes[FLESCH_READING_EASE] = textstat.flesch_reading_ease(text)
        readability_indexes[FLESCH_KINCAID_GRADE_LEVEL] = textstat.flesch_kincaid_grade(text)
        readability_indexes[AUTOMATED_READABILITY_INDEX] = textstat.automated_readability_index(text)
        readability_indexes[GUNNING_FOG_INDEX] = textstat.gunning_fog(text)
        readability_indexes[COLEMAN_LIAU] = textstat.coleman_liau_index(text)
        # if self.use_smog:
        readability_indexes[SMOG_INDEX] = textstat.smog_index(text)
        # if self.version > 1:
        readability_indexes[LINSEAR_WRITE_FORMULA] = textstat.linsear_write_formula(text)
        readability_indexes[DALE_CHALL] = textstat.dale_chall_readability_score(text)

        return readability_indexes
