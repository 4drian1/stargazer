"""Causal node-event context; never carries a repair-strength label forward."""


def trace_events(rows):
    events=[];trace=[];active=None;last_time=0
    for row in rows:
        if rows:
            assert (row['index'],row['day'])==(rows[0]['index'],rows[0]['day'])
        assert row['asof']>last_time
        last_time=row['asof'];broken_id=None
        if active is not None and row['close']<active['node']:
            active.update(end_at=row['asof'],end_reason='CLOSE_BELOW_NODE')
            broken_id=active['event_id'];active=None
        if active is not None and row['path']!='NONE' and row['node']!=active['node']:
            active.update(end_at=row['asof'],end_reason='CONFIRMED_NODE_CHANGED')
            active=None
        if active is None and row['path']!='NONE':
            assert row['node'] is not None and row['close']>=row['node']
            active=dict(event_id=f"{row['index']}:{row['day']}:{len(events)+1}",
                index=row['index'],day=row['day'],node=row['node'],start_at=row['asof'],
                start_path=row['path'],entry_known_day_low=row['known_day_low'],steps=0)
            events.append(active)
        if active is not None:
            active['steps']+=1
            active['observed_low']=min(active.get('observed_low',row['low']),row['low'])
        trace.append(dict(asof=row['asof'],event_id=None if active is None else active['event_id'],
            event_age=None if active is None else active['steps'],
            event_entry_known_day_low=None if active is None else active['entry_known_day_low'],
            event_post_entry_new_low=None if active is None else active['observed_low']<active['entry_known_day_low'],
            wick_below_active_node=None if active is None else row['low']<active['node'],
            broken_event_id=broken_id))
    if active is not None:
        active.update(end_at=last_time,end_reason='INPUT_END_NOT_OUTCOME')
    return trace,events
