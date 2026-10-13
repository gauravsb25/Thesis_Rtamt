"""Compare the three isolated RTAMT experiments without changing any monitor."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import project_code as project

def read(directory):
    path = ROOT / 'experiments' / directory / 'results.json'
    if not path.exists():
        raise SystemExit(f'Experiment is incomplete: {path}')
    return json.loads(path.read_text())

def table(title, rows, headers=('Evaluation', 'Existing', 'Candidate', 'Change')):
    print('\n  ' + title)
    print('  ' + '-' * 77)
    print(f'  {headers[0]:<32} {headers[1]:>12} {headers[2]:>14} {headers[3]:>11}')
    for name, a, b in rows:
        print(f"  {name:<32} {a['ba']:>11.2f}% {b['ba']:>13.2f}% {b['ba']-a['ba']:>+10.2f}")

def main():
    live = read('live_filter_context_2026_09_30')
    offline = read('offline_context_2026_09_30')
    csv = read('live_csv_context_2026_09_30')
    shapelets = read('joint_shapelets_2026_09_30')
    scattering = read('scattering_rules_2026_09_30')
    full_scattering = read('scattering_full_decision_2026_09_30')
    expected = project.fingerprint()
    offline_expected = project.offline_fingerprint()
    if live['fingerprint'] != expected or csv['fingerprint'] != expected:
        raise SystemExit('Monitor changed: experiment results no longer match current code.')
    if offline['fingerprint'] != offline_expected or csv['csv_fingerprint'] != offline_expected:
        raise SystemExit('Offline code or source recordings changed: results need recomputation.')
    for result in (shapelets, scattering, full_scattering):
        if result['fingerprint'] != expected or result['offline_fingerprint'] != offline_expected:
            raise SystemExit('A learned-rule result no longer matches the current monitor or dataset.')
    print('\n' + '=' * 83)
    print('  RTAMT MONITOR ACCURACY - 30 SEPTEMBER 2026')
    print('=' * 83)
    print('  BALANCED ACCURACY: average recall of LEFT, RIGHT, DOWN and STRAIGHT.')
    print('  All windows count. No abstention. External checks are retrospective.')
    print('\n  NEW SIGNAL METHODS: OUTER SESSION-VALIDATION BALANCED ACCURACY')
    print('  Controls show whether the new features actually help learned rules.')
    print(f"  {'Fixed family':<49} {'Online':>11} {'Offline':>11}")
    for f in shapelets['results']['live']['ablations']:
        a = shapelets['results']['live']['ablations'][f]
        b = shapelets['results']['offline']['ablations'][f]
        print(f"  {'Shapelet study / '+f:<49} {a['ba']:>10.2f}% {b['ba']:>10.2f}%")
    for f in scattering['modes']['online']['ablations']:
        a = scattering['modes']['online']['ablations'][f]
        b = scattering['modes']['offline']['ablations'][f]
        print(f"  {'Scattering study / '+f:<49} {a['ba']:>10.2f}% {b['ba']:>10.2f}%")
    print(f"  {'Existing fixed monitor':<49} {live['baseline']['ba']:>10.2f}% {offline['baseline']['ba']:>10.2f}%")
    print('  Both studies selected the original monitor in every outer block.')
    for mode, result in full_scattering['modes'].items():
        table('SCATTERING FOUR-CLASS CORRECTIONS: ' + mode.upper(),
              [('Whole-session validation', result['baseline'], result['nested'])] +
              [(name, v['baseline'], v['candidate']) for name, v in result['external'].items()])
        print('  Frozen choice: ' + result['frozen']['family'])
        print('  Candidate acceptance: ' + ('PASSED' if result['adopt'] else 'REJECTED'))
    for title, result in [('ONLINE: reconstructed dataset history', live),
                          ('OFFLINE: completed-recording context', offline),
                          ('ONLINE: genuine preceding CSV samples', csv)]:
        rows = [('Whole-session validation', result['baseline'], result['nested'])]
        rows += [(name, v['baseline'], v['candidate']) for name, v in result['external'].items()]
        table(title, rows)
        print('  Candidate acceptance: ' + ('PASSED' if result['adopt'] else 'REJECTED'))
    table('INPUT CONTEXT CHECK: existing formulas, existing thresholds',
          [('Development (fixed rule)', csv['published'], csv['baseline'])] +
          [(name, v['published'], v['baseline']) for name, v in csv['external'].items()],
          ('Evaluation', 'NPZ history', 'CSV history', 'Change'))
    print('\n  CSV history changes the information supplied to the same formulas.')
    print('  It is a separate input protocol, not proof that a new rule is better.')
    print('  Dataset windows use their completed-window mean; the stream wrapper')
    print('  applies that same preprocessing. Existing raw StreamMonitor differs.')
    print('  Causality concerns decisions on already ICA-cleaned EEG; live ICA')
    print('  preprocessing and a prospective continuous-stream test are unvalidated.')
    print('\n  RAW COUNTS FOR THE FROZEN CSV CANDIDATE ON EXTERNAL SETS')
    for name, result in csv['external'].items():
        candidate = result['candidate']
        print(f"\n  {name}: {candidate['correct']}/{candidate['n']} correct; "
              f"ordinary accuracy {100*candidate['correct']/candidate['n']:.2f}%")
        for i, cls in enumerate(project.CLASSES):
            row = candidate['confusion'][i]
            print(f"  {cls:<10} {row[i]:>5}/{sum(row):<5} recall {candidate['recall'][i]:>6.2f}%")
    print('\n  Current monitor fingerprint: ' + expected)
    print('  Current offline fingerprint: ' + offline_expected)
    print('  Production monitor files were preserved throughout these experiments.')
    print('=' * 83 + '\n')

if __name__ == '__main__':
    main()
