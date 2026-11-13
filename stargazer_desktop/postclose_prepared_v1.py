"""Post-close preparation manifest: roster, version binding, next-day backfill.

One manifest per observation day lives in the runtime directory
(dist/data/postclose_risk/prepared_manifest.json). It records who must be
prepared (plan stocks, manual observations, holdings, watch), per-item status
with evidence, the structured version binding (load_version/source_hash/
prepared_at), and the explicitly open full-board Pyxis supply item. Loading
side verifies it on start of day and re-schedules only the missing pieces.

Merge discipline: refreshing never degrades an unchanged item — status and
evidence are kept unless the requirement (kind/history days/turnover spec)
actually changed. Readiness is proven strictly: a target must match this
contract's P/E, contain every required code with per-row READY, cover the
required history days, and carry source receipts.
"""
import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

CONTRACT = 'POSTCLOSE_PREPARED_MANIFEST_V1'
MANIFEST_NAME = 'prepared_manifest.json'
OPEN_PYXIS_ITEM = dict(
    id='PYXIS_FULL_BOARD_SUPPLY',
    status='VERIFY_AT_EACH_LIVE_CUTOFF',
    note='已接入免费真实5分钟板块供给；按每个盘中截止核验覆盖及必需输入，不以昨日准备冒充今日Pyxis就绪')

_lock = threading.Lock()
_wl_state = {'at': 0.0, 'pending': False, 'cfg': None}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def runtime_dir():
    import postclose_risk_runtime
    return postclose_risk_runtime.directory()


def manifest_path():
    return runtime_dir() / MANIFEST_NAME


def _sha_text(value) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _req_hash(item) -> str:
    """Requirement identity: kind + history days + turnover spec."""
    return _sha_text(json.dumps(
        [item.get('kind'), item.get('history_days') or [], item.get('turnover_reference')],
        sort_keys=True, ensure_ascii=False, default=str))


def _group_key(history_days, turnover_reference):
    """Hashable grouping key matching minute_history's cohort semantics."""
    days = tuple(int(d) for d in (history_days or []))
    if turnover_reference is None:
        spec = None
    elif isinstance(turnover_reference, dict):
        spec = json.dumps(turnover_reference, sort_keys=True, ensure_ascii=False)
    else:
        spec = str(turnover_reference)
    return days, spec


def source_version():
    """Acceptance-run binding when the daily source pipeline passed."""
    state_file = runtime_dir() / 'state.json'
    try:
        state = json.loads(state_file.read_bytes())
    except (OSError, ValueError):
        return None
    if state.get('status') == 'COMPLETE' and state.get('acceptance_status') == 'PASS':
        return [state.get('run_dir'), state.get('packs_sha256')]
    return None


def roster_from_plan(plan):
    """Normalized preparation roster from a confirmed assistant plan shape."""
    items = {}
    for row in plan.get('rows') or []:
        code = str(row.get('code') or '').zfill(6)
        if len(code) != 6 or not code.isdigit() or code in items:
            continue
        entry = row.get('entry_rule') or {}
        items[code] = dict(kind='PLAN',
                           history_days=list(entry.get('history_days') or []),
                           turnover_reference=entry.get('turnover_reference'))
    for obs in plan.get('manual_observations') or []:
        code = str(obs.get('code') or '').zfill(6)
        if len(code) == 6 and code.isdigit() and code not in items:
            items[code] = dict(kind='MANUAL', history_days=[],
                               turnover_reference=None)
    for code in plan.get('selected_manual_codes') or []:
        code = str(code).zfill(6)
        if len(code) == 6 and code.isdigit() and code not in items:
            items[code] = dict(kind='MANUAL', history_days=[],
                               turnover_reference=None)
    return items


def roster_from_plan_frozen(exported_plan, default_days):
    """Roster from a stargazer_plan CLI export (sessions[].candidates[].code)."""
    items = {}
    for session in exported_plan.get('sessions') or []:
        for cand in session.get('candidates') or []:
            code = str(cand.get('code') or '').zfill(6)
            if len(code) == 6 and code.isdigit() and code not in items:
                items[code] = dict(kind='PLAN', history_days=list(default_days),
                                   turnover_reference=None)
    return items


def roster_from_holdings(holdings, calendar_days, P):
    """Positions roster: last 4 trading days up to P, mirroring prepare_holdings."""
    days = sorted({int(d) for d in (calendar_days or [])})
    history = [d for d in days if d <= int(P)][-4:]
    items = {}
    for pos in holdings or []:
        code = str(pos.get('code') or '').zfill(6)
        if len(code) == 6 and code.isdigit():
            items[code] = dict(kind='HOLDINGS', history_days=history,
                               turnover_reference=None)
    return items


def roster_from_watch(watch):
    items = {}
    for w in watch or []:
        code = str(w.get('code') or '').zfill(6)
        if len(code) == 6 and code.isdigit():
            items[code] = dict(kind='WATCH', history_days=[],
                               turnover_reference=None)
    return items


def full_roster(cfg, plan, P, calendar_days):
    """Authoritative union: plan codes + holdings + watch. Overlapping codes
    merge by requirement union (kind precedence, history-day union), never
    overwritten by the later roster."""
    import services
    wl = services.wl_load(cfg)
    roster = {}
    _union_into(roster, roster_from_plan(plan or {}))
    _union_into(roster, roster_from_holdings(services.active_positions(cfg, wl.get('positions') or []), calendar_days, P))
    _union_into(roster, roster_from_watch(wl.get('watch') or []))
    return roster


def schedule_roster(items, P, E):
    """Group the roster by preparation shape and schedule each group once.
    minute_history_v1.schedule is idempotent per cohort key, so re-scheduling
    an unchanged group is a no-op; only genuinely changed cohorts do work."""
    from minute_history_v1 import schedule
    groups = {}
    for code, item in sorted(items.items()):
        key = _group_key(item.get('history_days'), item.get('turnover_reference'))
        groups.setdefault(key, []).append(code)
    targets = []
    for (history_days, _spec), codes in groups.items():
        rows = []
        for c in codes:
            item = items[c]
            rows.append(dict(code=c,
                             entry_rule=dict(history_days=list(history_days)),
                             turnover_reference=item.get('turnover_reference')))
        target = schedule(dict(data_date=int(P), monitor_date=int(E), rows=rows,
                               prepare_role='INCREMENTAL'))
        targets.append((codes, list(history_days), target))
    return targets


def _target_ready(target, codes, P, E, days_map):
    """Strict readiness proof for one preparation target file."""
    try:
        value = json.loads(Path(target).read_bytes())
    except (OSError, ValueError) as exc:
        return 'SCHEDULED', 'TARGET_UNREADABLE:' + type(exc).__name__
    if value.get('data_date') != int(P) or value.get('monitor_date') != int(E):
        return 'MISMATCH_DATE', 'target P/E 与清单不一致'
    if value.get('status') != 'READY':
        return value.get('status', 'SCHEDULED'), value.get('reason') or 'PREPARING'
    rows = value.get('rows') or {}
    for code in codes:
        row = rows.get(code)
        if row is None:
            return 'MISSING_CODE', code + ' 不在 target 行内'
        if row.get('status') != 'READY':
            return row.get('status', 'SCHEDULED'), (row.get('reason') or 'PREPARING') + ':' + code
        have = {int(d) for d in (row.get('days') or [])}
        need = {int(d) for d in (days_map.get(code) or [int(P)])}
        if not need <= have:
            return 'DAYS_INCOMPLETE', code + ' 历史日不足'
        if not row.get('receipts'):
            return 'NO_SOURCE_RECEIPTS', code + ' 无来源证据'
    return 'READY', None


def item_ready(item, P, E):
    """A manifest item is ready only with a target that strictly proves it."""
    target = item.get('target')
    if not target or item.get('status') != 'READY':
        return False
    status, _ = _target_ready(target, [item.get('_code', '')], P, E,
                              {item.get('_code', ''): item.get('history_days') or [int(P)]})
    return status == 'READY'


def refresh_manifest(P, E, roster, targets, extra_items=None):
    """Write the prepared manifest. Unchanged items keep status/evidence; new
    or changed requirements are (re)scheduled and proven against their target."""
    from cross_process_lock_v1 import FileLock
    with FileLock(manifest_path()):
        return _refresh_manifest(P, E, roster, targets, extra_items)


def _refresh_manifest(P, E, roster, targets, extra_items):
    prev = current_manifest()
    prev_items = ((prev or {}).get('items') or {}) if (prev or {}).get('observation_date') == int(E) else {}
    items = {}
    for code, spec in sorted(roster.items()):
        req = _req_hash(spec)
        entry = dict(kind=spec.get('kind', 'PLAN'),
                     history_days=list(spec.get('history_days') or []),
                     req_hash=req, status='SCHEDULED')
        old = prev_items.get(code)
        if old and old.get('req_hash') == req:
            entry['status'] = old.get('status', 'SCHEDULED')
            for k in ('reason', 'target'):
                if old.get(k):
                    entry[k] = old[k]
        items[code] = entry
    for codes, history_days, target in targets or []:
        if not target:
            for code in codes:
                if code in items:
                    items[code]['status'] = 'SCHEDULED'
            continue
        days_map = {c: list(history_days) for c in codes}
        status, reason = _target_ready(target, codes, P, E, days_map)
        for code in codes:
            if code not in items:
                continue
            items[code]['status'] = status
            items[code].pop('reason', None)
            if reason:
                items[code]['reason'] = reason
            items[code]['target'] = str(target)
    roster_hash = _sha_text(json.dumps(
        {c: i['req_hash'] for c, i in sorted(items.items())}, sort_keys=True))
    extras = [i for i in (extra_items or [])
              if i.get('id') != OPEN_PYXIS_ITEM['id']]
    manifest = dict(
        contract=CONTRACT,
        load_version='P1',
        data_date=int(P), observation_date=int(E),
        source_hash=roster_hash, source_version=source_version(),
        prepared_at=datetime.now().isoformat(timespec='seconds'),
        status='READY' if items and all(i['status'] == 'READY' for i in items.values())
               else 'PARTIAL',
        items=items,
        open_items=[dict(OPEN_PYXIS_ITEM)] + extras)
    _write(manifest_path(), manifest)
    return manifest


def current_manifest():
    try:
        value = json.loads(manifest_path().read_bytes())
    except (OSError, ValueError):
        return None
    return value if value.get('contract') == CONTRACT else None


def reconcile_targets(target=None):
    """Advance registered tasks after publication or after an interrupted startup."""
    from cross_process_lock_v1 import FileLock
    if not manifest_path().is_file():
        return
    with FileLock(manifest_path()):
        value = current_manifest()
        if not value:
            return
        items = value.get('items') or {}
        changed = False
        for code, item in items.items():
            path = item.get('target')
            if not path or (target is not None and Path(path) != Path(target)):
                continue
            status, reason = _target_ready(path, [code], value['data_date'], value['observation_date'],
                                           {code: item.get('history_days')})
            if status == item.get('status') and reason == item.get('reason'):
                continue
            item['status'] = status
            item.pop('reason', None)
            if reason:
                item['reason'] = reason
            changed = True
        if changed:
            value['status'] = 'READY' if items and all(i['status'] == 'READY' for i in items.values()) else 'PARTIAL'
            _write(manifest_path(), value)


def binding(day=None):
    """Structured version binding for loaders; None when absent or stale."""
    manifest = current_manifest()
    if not manifest:
        return None
    day = int(day or datetime.now().strftime('%Y%m%d'))
    fresh = int(manifest.get('observation_date') or 0) == day
    return dict(load_version=manifest.get('load_version'),
                source_hash=manifest.get('source_hash'),
                source_version=manifest.get('source_version'),
                prepared_at=manifest.get('prepared_at'),
                data_date=manifest.get('data_date'),
                observation_date=manifest.get('observation_date'),
                status=manifest.get('status'),
                fresh=fresh,
                open_items=manifest.get('open_items') or [])


def _calendar_days(cfg, P):
    try:
        import paths
        rel = (cfg.get('plan') or {}).get('calendar') or 'data/calendar.json'
        raw = paths.data_file(rel, seed_from_bundle=False)
        return json.loads(Path(raw).read_bytes())
    except (OSError, ValueError, AttributeError):
        return []


def next_trading_day(calendar_days, P):
    """Validated E: P must be a calendar day with a successor; no silent P fallback."""
    days = sorted({int(d) for d in (calendar_days or [])})
    P = int(P)
    if P not in days or days.index(P) + 1 >= len(days):
        raise ValueError('CALENDAR_UNAVAILABLE:P_NOT_IN_CALENDAR')
    return days[days.index(P) + 1]


def on_plan_frozen(cfg, exported_path, trading_date, plan=None):
    """W2a: after a plan is frozen/exported, sync it to the runtime data dir,
    stamp the version binding, and prepare the full roster. E is validated
    against the trading calendar; failure raises (caller surfaces a warning)."""
    exported = Path(exported_path)
    body = exported.read_bytes()
    stamp = dict(load_version='P1', source_path=str(exported),
                 source_sha256=_sha_bytes(body), trading_date=int(trading_date),
                 exported_at=datetime.now().isoformat(timespec='seconds'))
    runtime_data = runtime_dir().parent
    synced = None
    try:
        if Path(exported).resolve() != (runtime_data / exported.name).resolve():
            target = runtime_data / exported.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + '.tmp')
            temp.write_bytes(body)
            temp.replace(target)
            synced = str(target)
    except OSError:
        synced = None
    stamp['synced_to'] = synced
    _write(runtime_data / (exported.stem + '.meta.json'), stamp)

    P = int(trading_date)
    calendar = _calendar_days(cfg, P)
    E = next_trading_day(calendar, P)
    days = sorted({int(d) for d in calendar})
    default_days = [d for d in days if d <= P][-4:]
    roster = {}
    if plan is not None:
        roster = roster_from_plan(plan)
    else:
        try:
            exported_plan = json.loads(body)
        except ValueError:
            exported_plan = {}
        roster = roster_from_plan_frozen(exported_plan, default_days)
    _union_into(roster, roster_from_holdings(
        _wl(cfg).get('positions') or [], days, P))
    _union_into(roster, roster_from_watch(_wl(cfg).get('watch') or []))
    targets = schedule_roster(roster, P, E) if roster else []
    manifest = refresh_manifest(P, E, roster, targets) if roster else None
    return dict(stamp=stamp, roster_size=len(roster), manifest=manifest)


def _wl(cfg):
    import services
    value = services.wl_load(cfg)
    return dict(value, positions=services.active_positions(cfg, value.get('positions') or []))


def on_watchlist_saved(cfg):
    """W2b: holdings/watch changed — coalesce bursts into one converging worker:
    the thread processes the latest state until no new save arrives. Requirement
    changes for an already-listed code are (re)prepared, not just new codes."""
    now = time.monotonic()
    with _lock:
        _wl_state['pending'] = True
        _wl_state['cfg'] = cfg
        _wl_state['at'] = now
    if _wl_state.get('worker') and _wl_state['worker'].is_alive():
        return
    worker = threading.Thread(target=_watchlist_worker, daemon=True,
                              name='prepared-watchlist-backfill')
    _wl_state['worker'] = worker
    worker.start()


def _watchlist_worker():
    while True:
        with _lock:
            if not _wl_state['pending']:
                return
            _wl_state['pending'] = False
            cfg = _wl_state['cfg']
        try:
            _watchlist_backfill(cfg)
        except Exception:
            return  # startup verify_and_backfill is the safety net


_KIND_ORDER = ('PLAN', 'MANUAL', 'HOLDINGS', 'WATCH')


def _merge_requirement(old, wanted):
    """Union of two roster specs for the same code: kind by precedence,
    history days union, turnover from the more specific spec."""
    if old is None:
        return dict(wanted)
    kind = old['kind'] if _KIND_ORDER.index(old['kind']) <= _KIND_ORDER.index(
        wanted.get('kind', 'WATCH')) else wanted['kind']
    days = sorted({int(d) for d in (old.get('history_days') or [])} |
                  {int(d) for d in (wanted.get('history_days') or [])})
    return dict(kind=kind, history_days=days,
                turnover_reference=(old.get('turnover_reference')
                                    if old.get('turnover_reference') is not None
                                    else wanted.get('turnover_reference')))


def _union_into(base, items):
    """Merge a roster into base by requirement union — later rosters must
    never overwrite an existing code's requirement."""
    for code, item in (items or {}).items():
        base[code] = _merge_requirement(base.get(code), item)
    return base


def _watchlist_backfill(cfg):
    manifest = current_manifest()
    today = int(datetime.now().strftime('%Y%m%d'))
    # 观察日早于今天才算过期；当晚为次日观察日准备的窗口必须放行。
    if not manifest or int(manifest.get('observation_date') or 0) < today:
        return  # stale/absent manifest: selection or startup backfill owns it
    P = manifest.get('data_date')
    E = manifest.get('observation_date')
    roster = {c: i for c, i in (manifest.get('items') or {}).items()}
    wanted = {}
    wanted.update(roster_from_holdings(_wl(cfg).get('positions') or [],
                                       _calendar_days(cfg, P), P))
    wanted.update(roster_from_watch(_wl(cfg).get('watch') or []))
    template_days = next((i.get('history_days') for i in roster.values()
                          if i.get('history_days')), [])
    changed = {}
    for code, item in wanted.items():
        old = roster.get(code)
        if old is None:
            item = dict(item, history_days=(item.get('history_days') or template_days))
            changed[code] = item
            continue
        merged = _merge_requirement(old, item)
        if not merged['history_days']:
            merged['history_days'] = list(template_days)
        if _req_hash(merged) != old.get('req_hash'):
            changed[code] = merged
    if not changed:
        return
    targets = schedule_roster(changed, P, E)
    merged_roster = dict(roster, **changed)
    refresh_manifest(P, E, merged_roster, targets)


def rebuild_from_authority(cfg):
    """No manifest (first boot / lost file): rebuild from the authoritative
    rosters — the confirmed assistant plan plus holdings/watch. Returns the
    manifest, or None when no authority source exists."""
    import paths
    plan = None
    try:
        raw = paths.data_file('data/pentacles_demo/assistant_plan.json',
                              seed_from_bundle=False)
        candidate = json.loads(Path(raw).read_bytes())
        if candidate.get('rows') or candidate.get('manual_observations') or \
                candidate.get('selected_manual_codes'):
            plan = candidate
    except (OSError, ValueError, AttributeError):
        plan = None
    P = int(plan['data_date']) if plan else None
    if P is None:
        holdings = _wl(cfg).get('positions') or []
        if not holdings:
            return None
        today = datetime.now().strftime('%Y%m%d')
        calendar = _calendar_days(cfg, int(today))
        if calendar:
            days = sorted({int(d) for d in calendar})
            P = max((d for d in days if d <= int(today)), default=None)
    if P is None:
        return None
    calendar = _calendar_days(cfg, P)
    E = int(plan['monitor_date']) if plan else next_trading_day(calendar, P)
    roster = full_roster(cfg, plan, P, calendar)
    if not roster:
        return None
    targets = schedule_roster(roster, P, E)
    return refresh_manifest(P, E, roster, targets)


def verify_and_backfill(cfg, day=None):
    """W2d: next-day continuation. A fresh manifest is validated item by item;
    anything not strictly READY is re-scheduled (incremental). Never fails
    the caller; NO_MANIFEST means the caller may try rebuild_from_authority."""
    day = int(day or datetime.now().strftime('%Y%m%d'))
    manifest = current_manifest()
    if manifest is None:
        return dict(status='NO_MANIFEST', day=day)
    if int(manifest.get('observation_date') or 0) != day:
        return dict(status='STALE_MANIFEST', day=day,
                    observation_date=manifest.get('observation_date'))
    P, E = manifest['data_date'], manifest['observation_date']
    roster = {c: i for c, i in (manifest.get('items') or {}).items()}
    backlog = {}
    for code, item in roster.items():
        probe = dict(item, _code=code)
        if not item_ready(probe, P, E):
            backlog[code] = item
    if backlog:
        targets = schedule_roster(backlog, P, E)
        manifest = refresh_manifest(P, E, roster, targets,
                                    extra_items=manifest.get('open_items'))
        return dict(status='BACKFILLED', day=day, backlog=sorted(backlog),
                    manifest_status=manifest.get('status'))
    return dict(status='READY', day=day, items=len(roster),
                open_items=manifest.get('open_items') or [])
