import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class LlmConfig:
    base_url: str
    api_key : str
    lv_model: str
    llm_model: str
    llm_temperature: float

llm_config = LlmConfig(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    lv_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
)