from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from models import AgentState
from src.tools import decide_workflow,route_for_benchmark, run_rag, route_by_mode, run_benchmark, run_analysis, load_cuda_sources, expand_topics_with_rag, extract_optimization_topics, is_build_benchmark
from src.cuda_tools import run_build_script
import json
from datetime import datetime

load_dotenv()

app = Flask(__name__)

# Initialize the graph
graph = StateGraph(AgentState)
graph.add_node("workflow", decide_workflow)
graph.add_node("rag_node", run_rag)
graph.add_node("benchmark_node", lambda state: state)
graph.add_node("run_benchmark", run_benchmark)
graph.add_node("analysis_node", run_analysis)
# -----
graph.add_node("build_benchmark_node", run_build_script)
# -----

graph.add_edge(START, "workflow")
graph.add_conditional_edges(
    "workflow",
    route_by_mode,
    {
        "rag": "rag_node",
        "benchmark": "build_benchmark_node",
        "analysis": "analysis_node",
    }
)

graph.add_conditional_edges(
    "benchmark_node",
    route_for_benchmark,
    {
        0: "build_benchmark_node",
        1: "run_benchmark"
    }
)

graph.add_edge("rag_node", END)
graph.add_edge("build_benchmark_node", "run_benchmark")
graph.add_edge("run_benchmark", END)
graph.add_edge("analysis_node", END)

app_graph = graph.compile()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Create initial state
        initial_state: AgentState = {
            "prompt": user_message,
            "mode": "unknown",
            "is_build_benchmark": False,
            "answer": "",
            "cuda_files": []
        }
        
        # Invoke the graph
        final_state = app_graph.invoke(initial_state)
        
        # Return the response
        return jsonify({
            'answer': final_state.get("answer", "No response generated."),
            'mode': final_state.get("mode", "unknown")
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

