from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from models import cuda_build_llm

CUDA_BUILD_SYSTEM_PROMPT = """
You are a build-and-benchmark orchestration agent.

Rules (MANDATORY):
1. First, call has_compiled_executables.
2. If the result is 1:
   - Immediately call run_benchmark.
3. If the result is 0:
   - First call run_build_script.
   - Then call run_benchmark.
4. Do NOT explain your reasoning.
5. Do NOT skip steps.
6. Use tools exactly as required.

Return only the final benchmark output.
"""



if __name__ == "__main__":
    response = cuda_build_llm.invoke([
        {"role": "system", "content": CUDA_BUILD_SYSTEM_PROMPT},
        {"role": "user", "content": "Run the CUDA benchmark workflow."}
    ])

    print(response.content)

