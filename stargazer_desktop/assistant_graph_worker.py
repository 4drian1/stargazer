"""Isolated local LangGraph runtime. Reads only immutable plan/review drafts."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    draft_id: str
    kind: str
    snapshot_hash: str
    expires_at: str
    models: dict
    decision: str


def build(checkpointer):
    def reference(state):
        return {'decision': 'WAITING_FOR_HUMAN'}
    def status(state):
        return {'models': {'selection': 'RULE_OBSERVATION', 'information': 'NOT_CONNECTED', 'risk': 'NOT_READY'}}
    def approve(state):
        answer = interrupt({'draft_id': state['draft_id'], 'kind': state['kind'], 'models': state['models']})
        now, expiry = datetime.fromisoformat(answer['now']), datetime.fromisoformat(state['expires_at'])
        decision = ('REJECTED' if answer.get('approved') is not True else
                    'STALE_REQUIRES_NEW_DRAFT' if now >= expiry else 'HUMAN_CONFIRMED_OBSERVATION_ONLY')
        return {'decision': decision}
    graph = StateGraph(State)
    graph.add_node('immutable_reference', reference)
    graph.add_node('model_readiness', status)
    graph.add_node('human_confirmation', approve)
    graph.add_edge(START, 'immutable_reference')
    graph.add_edge('immutable_reference', 'model_readiness')
    graph.add_edge('model_readiness', 'human_confirmation')
    graph.add_edge('human_confirmation', END)
    return graph.compile(checkpointer=checkpointer)


def run(body):
    root = Path(body['root']).resolve()
    identifier, kind = body['draft_id'], body['kind']
    if kind not in ('plan', 'review') or len(identifier) != 64 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('INVALID_REFERENCE')
    draft = json.loads((root / f'{kind}_{identifier}.json').read_text(encoding='utf-8'))
    raw = json.dumps(draft, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != identifier:
        raise ValueError('DRAFT_HASH_MISMATCH')
    db = root / 'graph_checkpoint.sqlite'
    with closing(sqlite3.connect(db, check_same_thread=False)) as conn:
        conn.execute('PRAGMA max_page_count=256')  # <=1MiB, inside assistant 10MiB total.
        saver = SqliteSaver(conn)
        graph = build(saver)
        config = {'configurable': {'thread_id': kind + ':' + identifier}}
        existing = graph.get_state(config)
        if not existing.values:
            graph.invoke({'draft_id': identifier, 'kind': kind,
                          'snapshot_hash': draft.get('snapshot_hash', draft.get('facts_hash')),
                          'expires_at': body['expires_at']}, config)
        if body['action'] == 'confirm' and graph.get_state(config).interrupts:
            graph.invoke(Command(resume={'approved': body['approved'], 'now': body['now']}), config)
        return {'ok': True, 'decision': graph.get_state(config).values['decision'],
                'runtime': 'LANGGRAPH_ISOLATED', 'draft_id': identifier}


if __name__ == '__main__':
    try:
        body = json.loads(sys.stdin.read(8192))
        print(json.dumps(run(body), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': type(exc).__name__}))
        sys.exit(2)
