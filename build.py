#!/usr/bin/env python3
"""Render paper.md from paper.md.tmpl with every number read from the live corpus.

    python3 build.py            # write paper.md
    python3 build.py --check    # fail if paper.md is stale (for a build gate)

WHY THE NUMBERS ARE NOT TYPED

The global instructions for this machine open with a note about documents that stopped
agreeing with the systems they described: a handoff pointing at a run that had already
failed four times, a pricing ladder sixteen days out of date, a spectral analyzer marked
"awaiting deployment" while it served live traffic. The rule that came out of it is that a
file which can drift will drift, and the fix is to stop hand-maintaining it.

A paper is the worst case of this. It gets written once, quoted for months, and every
figure in it is a hostage to a corpus that grows daily. So no figure here is typed. Every
one is computed from the drift log and the outcome labels at render time, and `--check`
turns staleness into a failing build rather than a thing someone notices later.

The prose still has to be true, and nothing here can check that. What this removes is the
narrower failure where the prose was true when written and the arithmetic beside it has
since moved.
"""
import argparse
import bisect
import collections
import datetime
import glob
import json
import math
import pathlib
import sys

DRIFT = pathlib.Path.home() / '.config/laserbrain/drift-log.jsonl'
TRANSCRIPTS = pathlib.Path.home() / '.claude/projects'
OUT = pathlib.Path.home() / '.config/laserbrain/verdict-outcomes.jsonl'
HERE = pathlib.Path(__file__).resolve().parent
TMPL = HERE / 'paper.md.tmpl'
PAPER = HERE / 'paper.md'


def load(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pct(n, d, places=1):
    return f'{n / d * 100:.{places}f}' if d else 'n/a'



def speech_bands():
    """Drift rate against how long the agent has been running unattended.

    THE JOIN, AND WHY IT IS THE ONLY ONE THAT SEPARATES A REDIRECT FROM A DRIFT

    The corpus cannot label a fire as "the user redirected me" or "I wandered", because both
    produce the same reading: overlap with the ground collapses. §5.1 is that admission. But
    the transcript records every user message as a `queue-operation`/`enqueue` row with a
    timestamp — top-level turns and ones typed mid-work alike — and the drift log is
    timestamped too. Joining them does not label anything. It CONDITIONS on the one variable
    that distinguishes the two causes: whether a person had just spoken.

    Two corrections are applied, and both cost the result rather than flattering it:

      FRESH GROUND    a reading taken straight after `reground` or `grounded` is scored
                      against a ground that was just reset, so it cannot drift by
                      construction. Every such reading is dropped, along with the resets
                      themselves. Keeping them would manufacture the low end of the curve.
      ATTRIBUTION     a reading is only assigned to a message if that message is the most
                      recent one before it. No window is imposed beyond the band edges.

    Returns None when no transcript is readable — on another machine, or with the projects
    directory absent, the section says so rather than rendering an empty table.
    """
    files = glob.glob(str(TRANSCRIPTS / '**' / '*.jsonl'), recursive=True)
    if not files:
        return None

    def when(s):
        return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))

    spoke = set()
    for f in files:
        try:
            for line in open(f, errors='replace'):
                if '"queue-operation"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get('operation') == 'enqueue' and d.get('timestamp'):
                    spoke.add(d['timestamp'])
        except OSError:
            continue
    if not spoke:
        return None
    speaks = sorted(when(s) for s in spoke)

    rows = [r for r in load(DRIFT) if r.get('ts')]
    rows.sort(key=lambda r: r['ts'])
    by_run = collections.defaultdict(list)
    for r in rows:
        by_run[r.get('run')].append(r)

    kept = []
    for seq in by_run.values():
        for i, r in enumerate(seq):
            if r.get('reason') in ('reground', 'grounded'):
                continue
            if i and seq[i - 1].get('reason') in ('reground', 'grounded'):
                continue
            if r.get('drifting') is None:
                continue
            kept.append(r)

    BANDS = [(0, 60, 'under 1 minute'), (60, 300, '1–5 minutes'),
             (300, 1800, '5–30 minutes'), (1800, 10 ** 9, 'over 30 minutes')]
    out = []
    for lo, hi, label in BANDS:
        g = n = 0
        for r in kept:
            t = when(r['ts'])
            i = bisect.bisect_right(speaks, t)
            if not i:
                continue
            gap = (t - speaks[i - 1]).total_seconds()
            if lo <= gap < hi:
                n += 1
                g += r.get('reason') == 'goal-drift'
        out.append({'label': label, 'drift': g, 'n': n})

    # Two-proportion z between the two best-powered adjacent bands. Reported because the
    # extreme band is small and must not be what the claim rests on.
    a, b = out[1], out[2]
    z = None
    if a['n'] and b['n']:
        pool = (a['drift'] + b['drift']) / (a['n'] + b['n'])
        se = math.sqrt(pool * (1 - pool) * (1 / a['n'] + 1 / b['n']))
        if se:
            z = (b['drift'] / b['n'] - a['drift'] / a['n']) / se
    return {'bands': out, 'z': z, 'kept': len(kept), 'messages': len(speaks)}


def numbers():
    rows, labels = load(DRIFT), load(OUT)
    if not rows:
        print(f'  no corpus at {DRIFT} — refusing to render a paper with no evidence '
              'behind it.')
        sys.exit(1)

    # THE ERA SPLIT COMES FIRST, because pooling the two is how three statistics in this
    # project were reported wrong at once. Before 2026-07-28 only drift moments were
    # logged, so `drifting` did not exist as a field and every row was a fire; afterwards
    # every step is logged. A rate across both has a numerator from one policy and a
    # denominator from the other.
    era = [r for r in rows if r.get('drifting') is not None]
    fires = [r for r in era if r['drifting']]
    n = {}
    n['readings_all'] = len(rows)
    n['readings'] = len(era)
    n['pre_era'] = len(rows) - len(era)
    n['fires'] = len(fires)
    n['fire_rate'] = pct(len(fires), len(era))
    n['runs'] = len({r.get('run') for r in rows})
    ts = sorted(r['ts'] for r in rows if r.get('ts'))
    n['span_from'], n['span_to'] = (ts[0][:10], ts[-1][:10]) if ts else ('?', '?')

    mix = collections.Counter(r.get('reason') for r in era)
    n['verdict_table'] = '\n'.join(
        f'| `{k}` | {v} | {pct(v, len(era))}% |'
        for k, v in mix.most_common())
    fm = collections.Counter(r.get('reason') for r in fires)
    n['fire_table'] = '\n'.join(
        f'| `{k}` | {v} | {pct(v, len(fires))}% |'
        for k, v in fm.most_common())
    n['goaldrift_share'] = pct(fm.get('goal-drift', 0), len(fires), 0)
    n['reground_share'] = pct(mix.get('reground', 0), len(era), 0)

    # LABELS. Only fires are labelled — a quiet reading interrupted nothing, so there is
    # nothing to judge — which is why this yields precision and not sensitivity.
    key = lambda r: (r.get('run'), r.get('step'))          # noqa: E731
    firekeys = {key(r) for r in fires}
    lab = {key(l): l for l in labels if key(l) in firekeys}
    oc = collections.Counter(v.get('outcome') for v in lab.values())
    n['labelled'] = len(lab)
    n['labelled_pct'] = pct(len(lab), len(fires))
    n['useful'], n['false'], n['unclear'] = oc['useful'], oc['false'], oc['unclear']
    clear = oc['useful'] + oc['false']
    n['clear'] = clear
    n['precision'] = pct(oc['useful'], clear)
    n['unclear_pct'] = pct(oc['unclear'], len(lab))

    # The excursion slot: spelled almost never, fired never.
    par = [r for r in rows if isinstance(r.get('laserscore'), str) and '⊂' in r['laserscore']]
    n['parent_spelled'] = len(par)
    n['parent_pct'] = pct(len(par), len(rows), 2)
    n['excursions'] = sum(1 for r in rows if r.get('reason') == 'excursion')

    # Chains: does a goal-drift fire resolve, or does the next reading drift again?
    runs = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: (r.get('run') or '', r.get('step') or 0)):
        runs[r.get('run')].append(r)
    gd = nxt = 0
    for seq in runs.values():
        for i, r in enumerate(seq[:-1]):
            if r.get('reason') == 'goal-drift':
                gd += 1
                nxt += seq[i + 1].get('reason') == 'goal-drift'
    n['chain_n'], n['chain_d'], n['chain_pct'] = nxt, gd, pct(nxt, gd, 0)

    # Agent comparison — reported so its own weakness is visible rather than assumed away.
    by = collections.Counter(r.get('agent') for r in era)
    n['agent_table'] = '\n'.join(
        f'| `{a}` | {c} | {sum(1 for r in era if r.get("agent") == a and r["drifting"])} |'
        for a, c in by.most_common())
    top = by.most_common(2)
    n['agent_second'] = f'{top[1][0]} ({top[1][1]} readings)' if len(top) > 1 else 'none'

    # ── drift against unattended runtime ─────────────────────────────────────
    sb = speech_bands()
    if sb is None:
        n['speech_available'] = ''
        n['speech_table'] = '| — | — | — | — |'
        n['speech_z'] = 'n/a'
        n['speech_kept'] = n['speech_messages'] = '0'
        n['speech_note'] = ('No transcript was readable on this machine, so this section '
                            'could not be computed. The table above is empty by design '
                            'rather than by accident.')
    else:
        n['speech_available'] = '1'
        n['speech_table'] = '\n'.join(
            f"| {b['label']} | {b['drift']} | {b['n']} | **{pct(b['drift'], b['n'])}%** |"
            for b in sb['bands'])
        n['speech_z'] = f"{sb['z']:.2f}" if sb['z'] is not None else 'n/a'
        n['speech_kept'] = f"{sb['kept']:,}"
        n['speech_messages'] = f"{sb['messages']:,}"
        a, b = sb['bands'][1], sb['bands'][2]
        n['speech_note'] = (
            f"The claim rests on the middle two bands, which carry {a['n'] + b['n']} of the "
            f"{sb['kept']} readings: {a['drift']}/{a['n']} = {pct(a['drift'], a['n'])}% "
            f"against {b['drift']}/{b['n']} = {pct(b['drift'], b['n'])}%, "
            f"z = {sb['z']:.2f}.")
        n['speech_lo'] = pct(sb['bands'][0]['drift'], sb['bands'][0]['n'])
        n['speech_hi_n'] = str(sb['bands'][3]['n'])
        n['speech_hi'] = pct(sb['bands'][3]['drift'], sb['bands'][3]['n'])

    # Sensitivity: the join landed 2026-08-01, so this is expected to be 0 for a while.
    n['joinable'] = sum(1 for r in rows if r.get('run') and r.get('drifting') is not None)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='exit 1 if paper.md differs from a fresh render')
    a = ap.parse_args()
    if not TMPL.exists():
        print(f'  no template at {TMPL}')
        return 1
    try:
        text = TMPL.read_text().format(**numbers())
    except KeyError as e:
        # A placeholder with no number behind it must break the build. Rendering it as the
        # literal "{whatever}" would ship a paper with a hole in it that reads like a typo.
        print(f'  template references {e} and numbers() does not compute it.')
        return 1
    if a.check:
        cur = PAPER.read_text() if PAPER.exists() else ''
        if cur != text:
            print('  STALE — paper.md does not match the corpus. Run: python3 build.py')
            return 1
        print('  paper.md is current with the corpus.')
        return 0
    PAPER.write_text(text)
    print(f'  wrote {PAPER} ({len(text.splitlines())} lines)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
