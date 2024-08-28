from matplotlib import pyplot as plt

from src.constants import CEFR_LEVELS
from src.evaluators.constants import READABILITY_INDEXES


def boxplot_readability_indexes(read_idxs_level, title, filename=None, figsize=(6, 4.2)):
    for readability_index in READABILITY_INDEXES:
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot([local_df[readability_index] for local_df in read_idxs_level])
        # ax.violinplot([local_df['flesch_kincaid_grade_level'] for local_df in read_idxs_level[1:]], showmeans=False, showmedians=True)
        ax.set_title(f"{readability_index} | {title}")
        ax.set_xticks(range(1, len(CEFR_LEVELS)+1))
        ax.set_xticklabels(CEFR_LEVELS)
        ax.grid(axis='y')
        # this is to use if I want the num. of texts per level in the plot.
        # ax.set_xticklabels([cefr + f'\n(n.={len(read_idxs_level[i])})' for i, cefr in enumerate(CEFR_LEVELS)])
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(f'output_figures/boxplot_readability_indexes/{filename}_{readability_index}.png')
        plt.close(fig)


def line_plot_word_lists_count(word_lists_per_level, title, filename=None, figsize=(6, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    for level in CEFR_LEVELS:
        ax.plot([local_df[level + '_frac'].mean() for local_df in word_lists_per_level], label=level)
    # ax.set_title(f"Frequency of words from vocabulary lists | {title}")
    ax.set_title(title)
    ax.set_xticks(range(0, len(CEFR_LEVELS)))
    ax.set_xticklabels(CEFR_LEVELS)
    if figsize==(6, 4.2):
        # ax.set_ylabel(f"Fraction of text made of words from vocabulary list of a specific CEFR level.")
        ax.set_ylabel(f"Frequency of words from vocabulary list")
        ax.set_xlabel("CEFR level of the reading passage")
        ax.legend(ncols=2)
    ax.set_yscale('log')
    ax.set_ylim([10**-4, 10**0])
    ax.grid(axis='y')
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'output_figures/line_plot_evaluation_vocabulary_lists/line_plot_evaluation_vocabulary_lists_{filename}_frac.png')
    plt.close(fig)
