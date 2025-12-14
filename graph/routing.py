from graph.state import AgentState
from src.tools import route_for_benchmark


def route_main(state: AgentState):
    return state.get("mode", "unknown")


def route_benchmark(state: AgentState):
    """
    route_for_benchmark() returns:
    - 1 → already built
    - 0 → needs build
    """
    return route_for_benchmark()
