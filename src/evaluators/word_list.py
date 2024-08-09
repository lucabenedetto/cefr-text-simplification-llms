from typing import Dict, Set, Tuple
import pandas as pd
import string

from src.constants import CEFR_LEVELS, A1, A2, B1, B2, C1, C2, COLUMN_TEXT_LENGTH


class WordListEvaluator:
    def __init__(self):
        pass

    def measure_word_frequency(self,
                               dict_texts: Dict[str, 'str'],
                               word_list_source: str = 'olp_en_cefrj',
                               by_pos: bool = False,
                               ) -> pd.DataFrame:
        if by_pos:
            raise NotImplementedError()

        # get word lists from files
        dict_word_lists = dict()
        for cefr in CEFR_LEVELS:
            cefr_df = pd.read_csv(f'data/input/{word_list_source}_vocabulary_profile_{cefr}.csv')
            dict_word_lists[cefr] = set(cefr_df['headword'].values)

        df = pd.DataFrame(columns=CEFR_LEVELS + [COLUMN_TEXT_LENGTH])
        for text_id, text in dict_texts.items():
            word_list_frequencies, text_length = self.measure_word_frequency_single_text(text, dict_word_lists)
            new_row_df = pd.DataFrame({
                A1: [word_list_frequencies[A1]],
                A2: [word_list_frequencies[A2]],
                B1: [word_list_frequencies[B1]],
                B2: [word_list_frequencies[B2]],
                C1: [word_list_frequencies[C1]],
                C2: [word_list_frequencies[C2]],
                COLUMN_TEXT_LENGTH: [text_length],
            })
            df = pd.concat([df, new_row_df], ignore_index=True)
        return df

    @staticmethod
    def measure_word_frequency_single_text(text: str, word_lists: Dict[str, Set[str]]) -> Tuple[Dict[str, int], int]:
        # remove punctuation
        text = text.replace('\n', ' ')
        text = text.replace('   ', ' ')
        text = text.replace('  ', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        # split text into single words
        text = text.split(sep=' ')
        text_length = len(text)
        word_frequencies = dict()
        for cefr in CEFR_LEVELS:
            word_frequencies[cefr] = len([word for word in text if word in word_lists[cefr]])
        return word_frequencies, text_length
