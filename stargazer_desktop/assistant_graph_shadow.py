"""Optional LangGraph prototype, never imported by desktop server or quote monitor.

Run with a dedicated checkpoint database. Business records remain the fact source.
No API calls, trading, notification or model training in this prototype.
"""
from typing import TypedDict


class State(TypedDict, total=False):
    snapshot_id: str
    expires_at: str
    now: str
    towers: dict
    draft: dict
    decision: str


def deterministic_decision(state, approved):
    if approved is not True:
        return 'REJECTED'
    if not state.get('snapshot_id') or state.get('now', '') >= state.get('expires_at', ''):
        return 'STALE_REQUIRES_NEW_DRAFT'
    return 'HUMAN_CONFIRMED_OBSERVATION_ONLY'


def build_graph(checkpointer):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt

    def approve(state):
        answer = interrupt({'draft': state.get('draft'), 'towers': state.get('towers'),
                            'scope': '观察计划，非买卖决策'})
        current = dict(state, now=answer['now'])
        return {'decision': deterministic_decision(current, answer.get('approved'))}

    builder = StateGraph(State)
    builder.add_node('human_confirmation', approve)
    builder.add_edge(START, 'human_confirmation')
    builder.add_edge('human_confirmation', END)
    return builder.compile(checkpointer=checkpointer)
