from graph.state import AgentState
from src.tools import (
    decide_workflow,
    run_rag,
    run_analysis,
    run_benchmark,
    is_build_benchmark,
    run_sanitization,
)
from src.cuda_tools import run_build_script
from observability.observability import Trace


def run_sanitization(state: AgentState) -> AgentState:
    with Trace("sanitization_node", run_id=state["run_id"]):
        return run_sanitization(state)

def workflow_node(state: AgentState) -> AgentState:
    with Trace("workflow_node", run_id=state["run_id"]):
        return decide_workflow(state)


def rag_node(state: AgentState) -> AgentState:
    with Trace("rag_node", run_id=state["run_id"]):
        return run_rag(state)


def analysis_node(state: AgentState) -> AgentState:
    with Trace("analysis_node", run_id=state["run_id"]):
        return run_analysis(state)


def benchmark_probe_node(state: AgentState) -> AgentState:
    """
    Updates state["is_build_benchmark"] by inspecting compiled executables.
    """
    with Trace("benchmark_probe_node", run_id=state["run_id"]):
        return is_build_benchmark(state)


def build_benchmark_node(state: AgentState) -> AgentState:
    with Trace("build_benchmark_node", run_id=state["run_id"]):
        return run_build_script(state)


def run_benchmark_node(state: AgentState) -> AgentState:
    with Trace("run_benchmark_node", run_id=state["run_id"]):
        return run_benchmark(state)


def noop_node(state: AgentState) -> AgentState:
    with Trace("noop_node", run_id=state["run_id"]):
        return state
