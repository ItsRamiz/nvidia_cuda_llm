from typing import TypedDict, Optional, List, Dict

class AgentState(TypedDict):
    prompt: str
    mode: Optional[str]
    is_build_benchmark: bool
    answer: str
    cuda_files: List[Dict]
    run_id: str
