from typing import TypedDict
from typing import TypedDict, Literal, Optional, List, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
Mode = Literal["rag", "benchmark", "analysis", "unknown"]

class AgentState(TypedDict):
    prompt: str
    mode: Optional[Mode]
    answer: str
    cuda_files: List[Dict]


classifier_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

rag_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

benchmark_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3
)

analysis_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0 
)

sanitize_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3
)