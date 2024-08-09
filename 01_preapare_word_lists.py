"""
This script prepares the word lists for different CEFR levels and stores them in the 'data/input/' directory.
At this stage it only works on the olp_en_cefrj vocabulary profile (as taken from
https://github.com/openlanguageprofiles/olp-en-cefrj).
Possibly TODO: get the word lists from Cambridge Assessment.
These are some useful links if found:
- from pre-A1 to A2: https://www.cambridgeenglish.org/images/149681-yle-flyers-word-list.pdf
- A2: https://www.cambridgeenglish.org/Images/506886-a2-key-2020-vocabulary-list.pdf
- B1: https://www.cambridgeenglish.org/Images/506887-b1-preliminary-2020-vocabulary-list.pdf
The second and third are also the ones which we have linked in our github repo.
"""
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
