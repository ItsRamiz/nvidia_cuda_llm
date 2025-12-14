from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from models import AgentState
from tools import decide_workflow, run_rag, route_by_mode, run_benchmark, run_analysis, load_cuda_sources, expand_topics_with_rag,extract_optimization_topics

load_dotenv()

def test_app(app):
    query = input("Ask? ")

    initial_state: AgentState = {
        "prompt": query,
        "mode": "unknown",
        "answer": ""
    }

    final_state = app.invoke(initial_state)

    print(final_state["answer"])

def main():

    graph = StateGraph(AgentState)
    graph.add_node("workflow", decide_workflow)
    graph.add_node("rag_node", run_rag)
    graph.add_node("benchmark_node", run_benchmark)
    graph.add_node("analysis_node", run_analysis)
    graph.add_node("analysis_rag", extract_optimization_topics)
    graph.add_node("analysis_reason", expand_topics_with_rag)

    graph.add_edge(START, "workflow")
    graph.add_conditional_edges(
        "workflow",
        route_by_mode,
        {
            "rag": "rag_node",
            "benchmark": "benchmark_node",
            "analysis": "analysis_node",
            #"unknown": "fallback_node",
        }
    )

    graph.add_edge("rag_node", END)
    graph.add_edge("benchmark_node", END)
    graph.add_edge("analysis_node", END)

    app = graph.compile()
    while True:
        test_app(app)



if __name__ == "__main__":
    main()
