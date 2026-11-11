"""Sparse quote-only references from a P-frozen map; never buy qualifications."""
import math


def downside_limit(close):
    """Largest cent strictly below yesterday close minus 3 percent."""
    return math.floor(math.nextafter(close*.97, 0.)*100)/100


def watch_groups(rule):
    mapping=rule.get('observation_map') or {}
    bands=mapping.get('bands') or rule.get('observation_bands') or []
    atr=rule.get('atr_ref');close=rule.get('close_p')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (atr,close)):
        return []
    limit=downside_limit(close)
    candidates=[b for b in bands if all(isinstance(b.get(k),(int,float)) and math.isfinite(b[k])
        for k in ('lower','upper','source_waves')) and 0<b['lower']<=b['upper']<=close and b['lower']<=limit and b['source_waves']>=1]
    groups=[]
    for band in sorted(candidates,key=lambda b:-b['lower']):
        if not groups or groups[-1][0]['lower']-band['lower']>.5*atr:
            groups.append([band])
        else:groups[-1].append(band)
    selected=[]
    for members in groups:
        midpoint=(members[0]['lower']+members[-1]['lower'])/2
        band=min(members,key=lambda b:(-b['source_waves'],round(abs(b['lower']-midpoint),10),-b['lower']))
        item={k:band[k] for k in ('lower','upper','source_waves','shared_peak_groups') if k in band}
        item.update(zone_upper=min(limit,band['upper']+.2*atr),quote_only=True,
            execution_qualified=bool(band.get('observable',False)),
            member_nodes=[b['lower'] for b in members],spacing_atr=.5)
        # Adjacent group representatives may approach their boundary. Keep them sparse.
        if selected and selected[-1]['lower']-item['lower']<.5*atr:
            if item['source_waves']>selected[-1]['source_waves']:selected[-1]=item
        else:selected.append(item)
    return selected[:8]
