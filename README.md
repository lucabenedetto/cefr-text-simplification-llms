# CEFR-targeted Text Simplification for Question Adaptation with LLMs

This repo contains the code to reproduce the results of the paper "_Towards CEFR-targeted Text Simplification for 
Question Adaptation_", presented at [RANLP 25](https://ranlp.org/ranlp2025/) (September 2025, Varna, Bulgaria).

The paper is available on the ACL Anthology: \[link will be added when the proceedings are online\]

❗️ If you have any questions, or would be interested in collaborating on a follow-up work, please reach out. ❗️

---

## Citation
If you use this code, please cite the RANLP paper:
```
[to add when the proceedings are online]
```

---

## How to use
This repo contains the codebase that is needed to reproduce the results shown in the paper, as well as the outputs of 
the models.
However, the data is not shared here and has to be downloaded separately.

### Data
The datasets are available at the following links:
- [Cambridge Multiple-Choice Questions Reading Dataset (Cam. MCQ)](https://englishlanguageitutoring.com/datasets/cambridge-multiple-choice-questions-reading-dataset)
- [Cambridge English Readsability Dataset (CERD)](https://ilexir.co.uk/datasets/index.html)

And the vocabulary list is available at:
- [CEFR-J Vocabulary List](https://github.com/openlanguageprofiles/olp-en-cefrj)

### Data Preparation
Data preparation is performed with the two scripts `01_data_preparation.py` and `01_prepare_word_lists.py`. 
They need the dataset and the vocabulary list to be located in the `./data/raw` folder.

## Text Simplification and Post-processing
Text simplification is done with the script `02_perform_text_simplification.py`.
It currently works with OpenAI models (via the OpenAI API) and models available on HuggingFace.
For both, the code uses the API keys stored in `~/.keys/openai_key.json` and `~/.keys/hf_access_token.json`; please 
store your keys there, or update the code as necessary.

Post-processing is done with `02_post_process_simplified_texts.py`. 
Since we are not currently using a structured outputter, if you want to experiment with other models, you will most 
likely have to update this script to correctly parse the output of the new models you experiment with.

The output files for both the simplification and post-processing scripts are stored in the folder `data/output`, in 
different folders for different datasets, models, and promtps.

## Analysis
All the results/plots shown on the paper are obtained with the analysis scripts: `03_*` for the analysis on the 
simplified texts, and `04_*` for the analysis on the original datasets.

Evaluation files are saved into the `data/evaluation` folder, and the output figures in `output_figures`. 

## Misc.
- The promtps are available in `src/prompts.py`
- The model names (as used for the APIs) are in `src/constants.py`
- The models' output is available in `data/output`.
