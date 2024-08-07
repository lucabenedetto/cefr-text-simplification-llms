from src.llm_adapter import BaseLLMAdapter, OpenAIAdapter, HuggingFaceAdapter
from src.constants import PRETRAINED_MODEL_NAMES, OPENAI_MODEL_NAMES


def init_adapter(model_name: str,
                 access_key: str = None,
                 use_gpu: bool = None,
                 temperature: float = 0
                 ) -> BaseLLMAdapter:
    if model_name in OPENAI_MODEL_NAMES.keys():
        return OpenAIAdapter(model_name=model_name, api_key=access_key, temperature=temperature)
    if model_name in PRETRAINED_MODEL_NAMES.keys():
        return HuggingFaceAdapter(model_name=model_name, access_token=access_key, use_gpu=use_gpu)
    raise ValueError(f'Model name {model_name} not recognized.')
