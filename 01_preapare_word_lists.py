import pandas as pd


def main():
    df = pd.read_csv('data/raw/cefrj-vocabulary-profile-1.5.csv')
    for cefr in ['A1', 'A2', 'B1', 'B2']:
        tmp_df = df[df['CEFR'] == cefr][['headword', 'pos', 'CEFR']]
        tmp_df.to_csv(f'data/input/olp_en_cefrj_vocabulary_profile_{cefr}.csv', index=False)
    df = pd.read_csv('data/raw/octanove-vocabulary-profile-c1c2-1.0.csv')
    for cefr in ['C1', 'C2']:
        tmp_df = df[df['CEFR'] == cefr][['headword', 'pos', 'CEFR']]
        tmp_df.to_csv(f'data/input/olp_en_cefrj_vocabulary_profile_{cefr}.csv', index=False)


if __name__ == '__main__':
    main()
