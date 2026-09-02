"""Export final trajectories and diagnostic state violations from saved run JSONs."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

STATES = {
    'CO2': ('co2_realized', 300.0, 1800.0, 'ppm'),
    'RH': ('rh_realized', 30.0, 95.0, '%'),
    'SWC': ('soil_water_content_hourly', 0.12, 0.35, 'm3/m3'),
}


def state_metrics(values, low, high):
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not math.isfinite(v) for v in values):
        raise ValueError('state samples must be nonempty, finite numbers')
    below = sum(v < low for v in values)
    above = sum(v > high for v in values)
    return {'hours': len(values), 'min': min(values), 'max': max(values),
            'below_hours': below, 'above_hours': above,
            'violation_hours': below + above,
            'violation_rate_pct': 100 * (below + above) / len(values),
            'mean_outside_distance': sum(max(low-v, v-high, 0) for v in values) / len(values),
            'all_hours_in_range': below + above == 0}


def collect(runs):
    files = sorted(Path(runs).glob('*.json'))
    if not files:
        raise ValueError('no run JSON files found')
    hourly, audits, joint = [], [], []
    identities, provenance = set(), set()
    for path in files:
        r = json.loads(path.read_text(encoding='utf-8'))
        ident = {k: r[k] for k in ('method', 'scenario', 'seed')}
        identity = tuple(ident.values())
        if identity in identities:
            raise ValueError(f'duplicate method/scenario/seed: {identity}')
        identities.add(identity)
        provenance.add(tuple(r['provenance'][k] for k in ('protocol_hash', 'source_hash', 'landscape_hash')))
        plan = r['plan']
        if len(plan) != 24 or any(len(row) != 4 for row in plan):
            raise ValueError(f'{path.name}: expected a 24x4 plan')
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for row in plan for v in row):
            raise ValueError(f'{path.name}: invalid control value')
        control_ok = r['penalty'] <= 1e-6
        if control_ok != r['threshold_qualified']:
            raise ValueError(f'{path.name}: inconsistent control qualification')
        all_ok = True
        for state, (key, low, high, unit) in STATES.items():
            values = r['details'][key]
            if len(values) != len(plan):
                raise ValueError(f'{path.name}: trajectory length mismatch')
            metrics = state_metrics(values, low, high)
            all_ok = all_ok and metrics['all_hours_in_range']
            audits.append({**ident, 'state': state, 'unit': unit,
                           'lower': low, 'upper': high, **metrics})
        joint.append({**ident, 'profit': r['profit'], 'control_qualified': control_ok,
                      'state_qualified': all_ok, 'joint_qualified': control_ok and all_ok})
        for h, control in enumerate(plan):
            hourly.append({**ident, 'hour_index_zero_based': h,
                **dict(zip(('temperature_setpoint', 'supplemental_light', 'co2_request', 'rh_setpoint'), control)),
                **{state: r['details'][spec[0]][h] for state, spec in STATES.items()}})
    if len(provenance) != 1:
        raise ValueError('mixed protocol, source or landscape fingerprints')
    return hourly, audits, joint


def summarize(audits):
    groups = defaultdict(list)
    for r in audits:
        groups[(r['method'], r['scenario'], r['state'])].append(r)
        groups[(r['method'], 'ALL', r['state'])].append(r)
    result = []
    for (method, scenario, state), rows in sorted(groups.items()):
        hours = sum(r['hours'] for r in rows)
        violations = sum(r['violation_hours'] for r in rows)
        result.append({'method': method, 'scenario': scenario, 'state': state,
            'runs': len(rows), 'hours': hours, 'lower': rows[0]['lower'], 'upper': rows[0]['upper'],
            'unit': rows[0]['unit'], 'min': min(r['min'] for r in rows), 'max': max(r['max'] for r in rows),
            'below_hours': sum(r['below_hours'] for r in rows),
            'above_hours': sum(r['above_hours'] for r in rows),
            'violation_hours': violations, 'violation_rate_pct': 100*violations/hours,
            'runs_all_hours_in_range': sum(r['all_hours_in_range'] for r in rows),
            'mean_outside_distance': sum(r['mean_outside_distance']*r['hours'] for r in rows)/hours})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.resolve() == args.runs.resolve() or args.out.resolve().is_relative_to(args.runs.resolve()):
        parser.error('audit output must be separate from the run directory')
    try:
        hourly, audits, joint = collect(args.runs)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('hourly_trajectories.csv', hourly), ('per_run_states.csv', audits),
                       ('joint_qualification.csv', joint), ('state_summary.csv', summarize(audits))]:
        with (args.out/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'runs': len(joint), 'joint_qualified': sum(r['joint_qualified'] for r in joint),
                      'out': str(args.out.resolve()), 'ranges': STATES,
                      'interpretation': 'Inclusive diagnostic ranges; post-update hourly samples; no safety certification.'}, indent=2))


if __name__ == '__main__':
    main()
