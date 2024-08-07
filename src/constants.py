# Names of pretrained models available on Hugging Face
GEMMA_2B = 'gemma_2b'
GEMMA_7B = 'gemma_7b'
LLAMA_3_8B = 'llama3_8b'
MISTRAL_7B_v02 = 'mistral_v02_7b'

PRETRAINED_MODEL_NAMES = {
    GEMMA_2B: "google/gemma-2b-it",
    GEMMA_7B: "google/gemma-7b-it",
    LLAMA_3_8B: "meta-llama/Meta-Llama-3-8B-Instruct",
    MISTRAL_7B_v02: "mistralai/Mistral-7B-Instruct-v0.2"
}

# Names of OpenAI models
GPT_3_5_0613 = 'gpt-3.5-turbo-0613'

OPENAI_MODEL_NAMES = {
    GPT_3_5_0613: "gpt-3.5-turbo-0613",
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
