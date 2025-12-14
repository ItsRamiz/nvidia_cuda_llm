from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from graph.graph import build_graph
from graph.state import AgentState
import uuid
from observability.observability import Trace

load_dotenv()

app = Flask(__name__)

app_graph = build_graph()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    run_id = str(uuid.uuid4())

    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        initial_state: AgentState = {
            "prompt": user_message,
            "mode": "unknown",
            "is_build_benchmark": False,
            "answer": "",
            "cuda_files": [],
            "run_id": run_id, 
        }

        # Optional: trace the entire request
        with Trace("api_chat", run_id=run_id):
            final_state = app_graph.invoke(initial_state)

        return jsonify({
            "answer": final_state.get("answer", ""),
            "mode": final_state.get("mode", "unknown"),
            "run_id": run_id, 
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "run_id": run_id,
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
