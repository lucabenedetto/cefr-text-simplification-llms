def get_prompt_from_prompt_id(prompt_id: str, target_cefr_level: str) -> str:
    # Prompts 0x_xx
    if prompt_id == f'00_{target_cefr_level}':
        # simplest prompt, only the reference to the CEFR and ask to simplify the text.
        return f"""You will be shown a reading passage used to evaluate the reading proficiency of learners of English as a foreign language. 
Your task is to simplify the text to make it appropriate for a learner of level {target_cefr_level} on the Common European Framework of Reference for Languages (CEFR)."""
    if prompt_id == f'01_{target_cefr_level}':
        # With respect to the previous one, adds the request for minimising the changes to the factual content.
        return f"""You will be shown a reading passage used to evaluate the reading proficiency of learners of English as a foreign language. 
Your task is to simplify the text to make it appropriate for a learner of level {target_cefr_level} on the Common European Framework of Reference for Languages (CEFR).
Please minimize changes to the factual content of the reading passage while ensuring the simplified text is clear and easy to understand for an {target_cefr_level} learner."""
    if prompt_id == f'02_{target_cefr_level}':
        # With respect to the previous one, adds the information about the knowledge of the learners of a specific CEFR level.
        return f"""You will be shown a reading passage used to evaluate the reading proficiency of learners of English as a foreign language. 
Your task is to simplify the text to make it appropriate for a learner of level {target_cefr_level} on the Common European Framework of Reference for Languages (CEFR).
A student of level {target_cefr_level} {get_cefr_levels_description(target_cefr_level)}
Please minimize changes to the factual content of the reading passage while ensuring the simplified text is clear and easy to understand for an {target_cefr_level} learner."""
    # Prompts 1x_xx
    if prompt_id == f'10_{target_cefr_level}':
        return f"""You are a skilled English teacher preparing an exam to evaluate the reading proficiency of learners of English as a foreign language.
Starting from the given text, you have to simplify to make it appropriate for a learner of level {target_cefr_level} on the Common European Framework of Reference for Languages (CEFR)."""
    if prompt_id == f'11_{target_cefr_level}':
        return f"""You are a skilled English teacher preparing an exam to evaluate the reading proficiency of learners of English as a foreign language.
Starting from the given text, you have to simplify to make it appropriate for a learner of level {target_cefr_level} on the Common European Framework of Reference for Languages (CEFR).
You have to minimize the changes to the factual content of the reading passage, in order not to affect the answerability of the questions associated with it."""
    if prompt_id == f'12_{target_cefr_level}':
        return f"""You are a skilled English teacher preparing an exam to evaluate the reading proficiency of learners of English as a foreign language.
Starting from the given text, you have to simplify to make it appropriate for a learner of level {target_cefr_level} on the Common European Framework of Reference for Languages (CEFR).
A student of level {target_cefr_level} {get_cefr_levels_description(target_cefr_level)}
You have to minimize the changes to the factual content of the reading passage, in order not to affect the answerability of the questions associated with it."""
    raise ValueError(f'Invalid prompt id: {prompt_id}')


def get_cefr_levels_description(student_level):
    dict_cefr_level_descriptions = {
        'A1': 'can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type; can introduce him/herself and others and can ask and answer questions about personal details such as where he/she lives, people he/she knows and things he/she has; can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.',
        'A2': 'can understand sentences and frequently used expressions related to areas of most immediate relevance (e.g. very basic personal and family information, shopping, local geography, employment); can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters; can describe in simple terms aspects of his/her background, immediate environment and matters in areas of immediate need.',
        'B1': 'can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc; can deal with most situations likely to arise whilst travelling in an area where the language is spoken; can produce simple connected text on topics which are familiar or of personal interest; can describe experiences and events, dreams, hopes & ambitions and briefly give reasons and explanations for opinions and plans.',
        'B2': 'can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialisation; can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party; can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.',
        'C1': 'can understand a wide range of demanding, longer texts, and recognise implicit meaning; can express him/herself fluently and spontaneously without much obvious searching for expressions; can use language flexibly and effectively for social, academic and professional purposes; can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.',
        'C2': 'can understand with ease virtually everything heard or read; can summarise information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation; can express him/herself spontaneously, very fluently and precisely, differentiating finer shades of meaning even in more complex situations.',
    }
    return dict_cefr_level_descriptions[student_level]
