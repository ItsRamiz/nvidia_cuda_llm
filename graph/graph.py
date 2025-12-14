from langgraph.graph import StateGraph, START, END
from graph.state import AgentState
from graph.nodes import (
    workflow_node,
    rag_node,
    analysis_node,
    benchmark_probe_node,
    build_benchmark_node,
    run_benchmark_node,
    noop_node,
)
from graph.routing import route_main, route_benchmark
from src.tools import run_sanitization


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("workflow", workflow_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("analysis_node", analysis_node)

    graph.add_node("benchmark_probe", benchmark_probe_node)
    graph.add_node("build_benchmark", build_benchmark_node)
    graph.add_node("run_benchmark", run_benchmark_node)

    graph.add_node("sanitize_node", run_sanitization)

    graph.add_edge(START, "workflow")

    # Main intent routing
    graph.add_conditional_edges(
        "workflow",
        route_main,
        {
            "rag": "rag_node",
            "analysis": "analysis_node",
            "benchmark": "benchmark_probe",
        }
    )

    # Benchmark decision routing
    graph.add_conditional_edges(
        "benchmark_probe",
        route_benchmark,
        {
            0: "build_benchmark",
            1: "run_benchmark",
        }
    )

    graph.add_edge("build_benchmark", "run_benchmark")

    graph.add_edge("run_benchmark", END)
    graph.add_edge("rag_node", "sanitize_node")
    graph.add_edge("sanitize_node", END)
    graph.add_edge("analysis_node", END)

    return graph.compile()
