"""
SIMPLE LANGGRAPH DEMO (ONE FILE)

What this teaches:
- StateGraph
- Nodes
- Conditional edges
- Routing function
- LLM prompts
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()



if __name__ == "__main__":

