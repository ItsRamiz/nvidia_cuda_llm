from prompts import classifier_system_prompt, rag_system_prompt, cuda_optimize,CUDA_EXTRA_OPT, CUDA_TOPIC_SYSTEM_PROMPT, CUDA_OPTIMIZATION_SYSTEM_PROMPT
from models import classifier_llm, rag_llm, AgentState, Mode, analysis_llm
import subprocess, json
import time
from observability.observability import Trace
from pathlib import Path
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from src.cuda_tools import has_compiled_executables, run_build_script

load_dotenv()

SEVERITY_KEYWORDS = {
    "launch": 5,
    "grid": 5,
    "block": 5,
    "scalable": 4,
    "synchronization": 4,
    "cudaDeviceSynchronize": 4,
    "correctness": 4,
    "index": 3,
    "memory access": 2,
    "coales": 2,
    "branch": 1,
    "cudaMemcpy": 1,
    "pinned": 1,
}
MODE_BENCHMARK: Mode = "benchmark"
MODE_ANALYSIS: Mode = "analysis"
MODE_RAG: Mode = "rag"
MODE_UNKNOWN: Mode = "unknown"

ROOT = Path(__file__).resolve().parents[1]  # Go up from src/tools.py to project root

def decide_workflow(state: AgentState) -> AgentState:
    """
    Uses an LLM to choose between:
    - rag       → answer question
    - benchmark → run performance tests
    - analysis  → analyze bottlenecks
    """
    with Trace("decide_workflow"):
        prompt = classifier_system_prompt + state["prompt"]

        result = classifier_llm.invoke(prompt)
        intent = result.content.strip().lower()
        match intent:
            case "benchmark":
                state["mode"] = MODE_BENCHMARK
            case "analysis":
                state["mode"] = MODE_ANALYSIS
            case "rag":
                state["mode"] = MODE_RAG

        return state


def run_rag(state: AgentState) -> AgentState:
    """
    Perform a similarity search over PGVector using cosine distance.
    """
    with Trace("run_rag"):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        query_embedding = embeddings.embed_query(state["prompt"])

        with psycopg.connect(
            "dbname=vectordb user=postgres password=postgres"
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        document,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM langchain_pg_embedding
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5;
                """, (query_embedding, query_embedding))

                rows = cur.fetchall()

        search_results = "\n".join(
            f"{score:.3f} → {doc[:200]}"
            for doc, score in rows
        )

        rag_prompt = rag_system_prompt.format(
            search_results=search_results,
            user_question=state["prompt"]
        )

        response = rag_llm.invoke(rag_prompt)
        state["answer"] = response.content

        return state


def is_build_benchmark(state: AgentState) -> AgentState:
    """
    Checks whether the project root /code directory contains any .exe files or has any compiled executables.
    If the function returns 1, then the benchmark exe has been built and the benchmark can be run, Dont do Anything!.
    If the function returns 0, then the benchmark has not been built and the build script must be run.
    """
    if has_compiled_executables():
        state["is_build_benchmark"] = True
    else:
        state["is_build_benchmark"] = False
    return state

def run_benchmark(state: AgentState) -> AgentState:
    """
    Run all .exe files under /code and store benchmark results in state.
    """
    with Trace("run_benchmark"):
        code_dir = ROOT / "code"
        exe_files = sorted(code_dir.glob("*.exe"))

        if not exe_files:
            state["benchmarks"] = []
            state["answer"] = f"No .exe files found in {code_dir}"
            return state

        results = []

        for exe in exe_files:
            start = time.perf_counter()

            completed = subprocess.run(
                [str(exe)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            end = time.perf_counter()

            results.append({
                "name": exe.name,
                "runtime_ms": (end - start) * 1000,
                "returncode": completed.returncode,
            })

        state["benchmarks"] = results

        state["answer"] = "\n".join(
            f"{r['name']}: {r['runtime_ms']:.2f} ms"
            for r in results
        )

    return state


def run_analysis(state: AgentState) -> AgentState:
    cuda_files = load_cuda_sources("code")
    topics_text = extract_optimization_topics(cuda_files)
    state["answer"] = topics_text
    return state


def extract_optimization_topics(cuda_files: list[dict]) -> str:
    if not cuda_files:
        return "No CUDA files provided."

    # Use only the first file for now
    cuda_file = cuda_files[0]

    payload = f"""
    You are analyzing CUDA code.
    
    Rules:
    - List optimization or correctness issues as bullet points.
    - Focus on runtime performance, scalability, or benchmarking correctness.
    - Do NOT include more than 5 issues.
    - For each issue, include a short title and a 1-line explanation.
    
    CUDA Code:
    {cuda_file['code']}
    """
    response = analysis_llm.invoke(
        [
            {"role": "system", "content": CUDA_TOPIC_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
    )

    raw_text = response.content
    lines = [line.strip("-• ").strip() for line in raw_text.splitlines() if line.strip()]

    scored = []
    for line in lines:
        score = 0
        lower = line.lower()
        for key, weight in SEVERITY_KEYWORDS.items():
            if key in lower:
                score = max(score, weight)

        scored.append((score, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_two = [line for score, line in scored if score > 0][:2]

    if not top_two:
        return "No meaningful optimization issues found."

    return "\n".join(f"- {issue}" for issue in top_two)


def expand_topics_with_rag(topics: list[dict]) -> list[str]:
    results = []
    for item in topics:
        query = f"{item['topic']} CUDA optimization"
        docs = retrieve_docs(query)
        context = "\n\n".join(docs)
        prompt = f"""
            Topic: {item['topic']}
            Code evidence:
            {item['evidence']}
            
            Retrieved CUDA documentation:
            {context}
            """

        response = analysis_llm.invoke(
            [
                {"role": "system", "content": CUDA_OPTIMIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        results.append(response.content)
    return results


def route_by_mode(state: AgentState) -> Mode:
    return state.get("mode", "unknown")


def route_for_benchmark():
    return has_compiled_executables()


def load_cuda_sources(path: str) -> List[Dict[str, str]]:
    """
    Load CUDA (.cu) source files.

    Args:
        path: Path to a .cu file OR a directory containing .cu files

    Returns:
        List of dicts:
        [
            {
                "filename": "kernel.cu",
                "code": "<full source code>"
            },
            ...
        ]
    """
    p = Path(path)
    results = []

    if p.is_file() and p.suffix == ".cu":
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.cu"))
    else:
        raise ValueError("Path must be a .cu file or a directory containing .cu files")

    for file in files:
        results.append({
            "filename": file.name,
            "code": file.read_text(encoding="utf-8", errors="ignore")
        })

    return results


def retrieve_docs(query: str, k: int = 5) -> list[str]:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings.embed_query(query)

    with psycopg.connect(
        "dbname=vectordb user=postgres password=postgres"
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document
                FROM langchain_pg_embedding
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, k))

            rows = cur.fetchall()

    return [row[0] for row in rows]

