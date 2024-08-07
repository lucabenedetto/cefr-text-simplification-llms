import openai

from src.llm_adapter.base import BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):

    def __init__(self, model_name: str, api_key: str = None, temperature: float = 0):
        super().__init__(model_name)
        openai.api_key = api_key
        self.temperature = temperature

    def convert_single_text(self, text: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{'role': 'system', 'content': self.prompt}, {'role': 'user', 'content': f"Text: '{text}'"}],
                temperature=self.temperature,
                response_format={"type": 'text'}
            )
            answer = response['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            # this if the GPT model did not produce a response
            answer = "{'index': -9, 'text': 'None'}"
        return answer
