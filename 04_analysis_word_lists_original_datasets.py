import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from src.evaluators.word_list import WordListEvaluator
from src.constants import COLUMN_TEXT_ID, COLUMN_TEXT, COLUMN_TEXT_LEVEL, CEFR_LEVELS
from constants import CERD, CAM_MCQ


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def get_word_lists_per_text(df):
    texts_dict = {text_id: text for text_id, text in df[[COLUMN_TEXT_ID, COLUMN_TEXT]].values}
    target_level_dict = {text_id: level for text_id, level in df[[COLUMN_TEXT_ID, COLUMN_TEXT_LEVEL]].values}
    word_lists = WordListEvaluator().measure_word_frequency(texts_dict)
    word_lists[COLUMN_TEXT_LEVEL] = word_lists.apply(lambda r: target_level_dict[r[COLUMN_TEXT_ID]], axis=1)
    for level in CEFR_LEVELS:
        word_lists[level + '_frac'] = word_lists.apply(lambda r: r[level]/r['text_length_n_words'], axis=1)
    word_lists_per_level = [word_lists[word_lists[COLUMN_TEXT_LEVEL] == level] for level in CEFR_LEVELS]
    return word_lists_per_level


def boxplot_text_length(word_lists_per_level, title, filename=None):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.boxplot([local_df['text_length_n_words'] for local_df in word_lists_per_level])
    ax.set_title(f"Text length (n. words) | {title}")
    ax.set_xticks(range(1, len(CEFR_LEVELS)+1))
    ax.set_xticklabels(CEFR_LEVELS)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'output_figures/boxplot_text_length/{filename}.png')
    plt.close(fig)


def boxplot_count_words_per_level(word_lists_per_level, title, filename=None):
    for level in CEFR_LEVELS:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.boxplot([local_df[level + '_frac'] for local_df in word_lists_per_level])
        ax.set_title(f"Words from level {level} | {title}")
        ax.set_xticks(range(1, len(CEFR_LEVELS)+1))
        ax.set_xticklabels(CEFR_LEVELS)
        ax.set_ylabel(f"Fraction of words in text.")
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(f'output_figures/boxplot_count_words_per_level/{filename}_{level}.png')
        plt.close(fig)


def main():
    df_cerd = pd.read_csv('data/input/cerd.csv')
    df_cam_mcq = pd.read_csv('data/input/mcq_cupa.csv')

    word_lists_count_per_level_cerd = get_word_lists_per_text(df_cerd)
    word_lists_count_per_level_cam_mcq = get_word_lists_per_text(df_cam_mcq)
    word_lists_count_per_level_aggregate = get_word_lists_per_text(pd.concat([df_cerd, df_cam_mcq], axis=0))

    boxplot_text_length(word_lists_count_per_level_cerd, title='CERD', filename=CERD)
    boxplot_text_length(word_lists_count_per_level_cam_mcq, title='Cambridge MCQ', filename=CAM_MCQ)
    boxplot_text_length(word_lists_count_per_level_aggregate, title='Aggregate', filename='aggregate')

    boxplot_count_words_per_level(word_lists_count_per_level_cerd, title='CERD', filename=CERD)
    boxplot_count_words_per_level(word_lists_count_per_level_cam_mcq, title='Cambridge MCQ', filename=CAM_MCQ)
    boxplot_count_words_per_level(word_lists_count_per_level_aggregate, title='Aggregate', filename='aggregate')


if __name__ == '__main__':
    main()
