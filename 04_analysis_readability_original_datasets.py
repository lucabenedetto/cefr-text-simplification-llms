import matplotlib

import pandas as pd
from src.evaluators.readability import ReadabilityEvaluator
from src.constants import COLUMN_TEXT_ID, COLUMN_TEXT, COLUMN_TEXT_LEVEL, CEFR_LEVELS
from constants import CERD, CAM_MCQ
from src.utils_plotting import boxplot_readability_indexes

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def get_readability_indexes_per_target_level(df):
    texts_dict = {text_id: text for text_id, text in df[[COLUMN_TEXT_ID, COLUMN_TEXT]].values}
    target_level_dict = {text_id: level for text_id, level in df[[COLUMN_TEXT_ID, COLUMN_TEXT_LEVEL]].values}
    read_idxs = ReadabilityEvaluator().compute_readability_indexes(texts_dict)
    read_idxs[COLUMN_TEXT_LEVEL] = read_idxs.apply(lambda r: target_level_dict[r[COLUMN_TEXT_ID]], axis=1)
    read_idxs_level = [read_idxs[read_idxs[COLUMN_TEXT_LEVEL] == level] for level in CEFR_LEVELS]
    return read_idxs_level


def main():
    df_cerd = pd.read_csv('data/input/cerd.csv')
    df_cam_mcq = pd.read_csv('data/input/mcq_cupa.csv')
    readability_indexes_per_level_cerd = get_readability_indexes_per_target_level(df_cerd)
    readability_indexes_per_level_cam_mcq = get_readability_indexes_per_target_level(df_cam_mcq)
    readability_indexes_per_level_aggregate = get_readability_indexes_per_target_level(pd.concat([df_cerd, df_cam_mcq], axis=0))
    boxplot_readability_indexes(readability_indexes_per_level_cerd, title='CERD', filename=CERD)
    boxplot_readability_indexes(readability_indexes_per_level_cam_mcq, title='Cambridge MCQ', filename=CAM_MCQ)
    boxplot_readability_indexes(readability_indexes_per_level_aggregate, title='Aggregate', filename='aggregate')


if __name__ == '__main__':
    main()

# Num. texts per level.
# CERD
# 0, 64, 60, 71, 67, 69
# Cam MCQ
# 0, 0, 28, 58, 25, 9
# Aggregate
# 0, 64, 88, 129, 92, 78