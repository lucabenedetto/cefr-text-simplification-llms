# Names of the models used which are available on Hugging Face
GEMMA_2B = 'gemma_2b'
GEMMA_7B = 'gemma_7b'
LLAMA_3_8B = 'llama3_8b'
LLAMA_3_1_8B = 'llama3_1_8b'
MISTRAL_7B_v02 = 'mistral_v02_7b'
MISTRAL_7B_v03 = 'mistral_v03_7b'

PRETRAINED_MODEL_NAMES = {
    GEMMA_2B: "google/gemma-2b-it",
    GEMMA_7B: "google/gemma-7b-it",
    LLAMA_3_8B: "meta-llama/Meta-Llama-3-8B-Instruct",
    LLAMA_3_1_8B: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    MISTRAL_7B_v02: "mistralai/Mistral-7B-Instruct-v0.2",
    MISTRAL_7B_v03: "mistralai/Mistral-7B-Instruct-v0.3",
}

# Names of OpenAI models
GPT_3_5_0613 = 'gpt_3_5_0613'  # This is an older deprecated model and should not be used.
GPT_3_5_1106 = 'gpt_3_5_1106'
GPT_4o_240806 = 'gpt_4o_240806'
GPT_4o_MINI_240718 = 'gpt_4o_mini_240718'

OPENAI_MODEL_NAMES = {
    GPT_3_5_0613: "gpt-3.5-turbo-0613",
    GPT_3_5_1106: 'gpt-3.5-turbo-1106',
    GPT_4o_240806: 'gpt-4o-2024-08-06',
    GPT_4o_MINI_240718: 'gpt-4o-mini-2024-07-18',
}

# COLUMN NAMES
COLUMN_TEXT_ID = 'text_id'
COLUMN_TEXT = 'text'
COLUMN_TEXT_LEVEL = 'text_level'
COLUMN_QUESTION_ID = 'question_id'
COLUMN_QUESTION = 'question'
COLUMN_OPTIONS = 'options'
COLUMN_OPTION_A = 'option_a'
COLUMN_OPTION_B = 'option_b'
COLUMN_OPTION_C = 'option_c'
COLUMN_OPTION_D = 'option_d'
COLUMN_CORRECT_ANSWER = 'correct_answer'
COLUMN_QUESTION_DIFFICULTY = 'question_difficulty'

COLUMN_NAMES = [
    COLUMN_TEXT_ID,
    COLUMN_TEXT,
    COLUMN_TEXT_LEVEL,
    COLUMN_QUESTION_ID,
    COLUMN_QUESTION,
    COLUMN_OPTIONS,
    COLUMN_OPTION_A,
    COLUMN_OPTION_B,
    COLUMN_OPTION_C,
    COLUMN_OPTION_D,
    COLUMN_CORRECT_ANSWER,
    COLUMN_QUESTION_DIFFICULTY,
]

COLUMN_LLM_ANSWER_INDEX = 'llm_answer_index'
COLUMN_LLM_ANSWER_TEXT = 'llm_answer_text'
COLUMN_LLM_ANSWER_CORRECT = 'llm_answer_correct'

COLUMN_TEXT_LENGTH = 'text_length_n_words'

A1 = 'A1'
A2 = 'A2'
B1 = 'B1'
B2 = 'B2'
C1 = 'C1'
C2 = 'C2'
CEFR_LEVELS = [A1, A2, B1, B2, C1, C2]
