"""Print saved measurements; does not refit or select on test data."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    r=json.loads((HERE/'results.json').read_text())
    lines=['# Spectral-pattern experiment: measured results','',
        '**No candidate met the adoption rule. Production monitors were not changed.**','',
        'All percentages below are measured. Development means predictions for sessions excluded from fitting, not training accuracy.',
        'Movement macro is the average recall for LEFT, RIGHT and DOWN, including movements incorrectly called STRAIGHT.','',
        '| Development method | Correct / total | Ordinary accuracy | Balanced accuracy | Movement macro |',
        '|---|---:|---:|---:|---:|']
    for name,v in r['development'].items():
        lines.append(f"| {name} | {v['correct']}/{v['n']} | {v['accuracy']:.2f}% | {v['balanced_accuracy']:.2f}% | {v['movement_macro']:.2f}% |")
    lines+=['','The winner under the predeclared movement-first ranking was `'+r['selected']+'`, the control using only existing monitor measurements. Its 0.33-point movement gain was below the 0.5-point requirement and its balanced accuracy fell. It was not eligible for adoption.',
        'Only that frozen winner was scored externally, as a diagnostic, despite failing the adoption gate. No spectral candidate was selected using external results. These external sets have been exposed in previous project experiments and are not blind tests.','',
        '| Split / model | Correct / total | Ordinary accuracy | Balanced accuracy | Movement macro |',
        '|---|---:|---:|---:|---:|']
    train=r['training_fit']
    lines.append(f"| Training fit / selected control | {train['correct']}/{train['n']} | {train['accuracy']:.2f}% | {train['balanced_accuracy']:.2f}% | {train['movement_macro']:.2f}% |")
    for split,models in r['external'].items():
        for name,v in models.items():
            lines.append(f"| {split} / {name} | {v['correct']}/{v['n']} | {v['accuracy']:.2f}% | {v['balanced_accuracy']:.2f}% | {v['movement_macro']:.2f}% |")
    lines+=['','## Class counts','', '| Split / model | LEFT | RIGHT | DOWN | STRAIGHT |','|---|---:|---:|---:|---:|']
    groups=[('Development',r['development'])]+list(r['external'].items())
    for split,models in groups:
        for name,v in models.items():
            cells=[f'{row[k]}/{sum(row)} ({v["recall"][k]:.2f}%)' for k,row in enumerate(v['confusion'])]
            lines.append('| '+split+' / '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## Interpretation','',
        '- Temporal ordering helps relative to static bandpower in this fixed comparison, but neither beats the existing monitor.',
        '- Monitor plus spectral features has the best ordinary development accuracy (96.25%), but worse movement recognition. The STRAIGHT majority makes ordinary accuracy a poor selection target.',
        '- The filter-bank CSP implementation learns spatial filters from training sessions only; it is not another sweep over hand-picked electrode differences.',
        '- These are seven specific model/feature configurations, not proof that all frequency-domain methods fail. Only two subjects are available.',
        '- Development comparisons reuse historically inspected sessions. Method selection adds optimism; fresh recordings are needed for a genuinely prospective evaluation.',
        '- These research classifiers are not newly validated RTAMT monitors. No classifier is disguised as a temporal-logic rule.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'+'='*91)
    print(' EEG SPECTRAL PATTERNS | SESSION-HELD-OUT VALIDATION')
    print('='*91)
    print(f"{'Method':28} {'Correct / N':17} {'Accuracy':>10} {'Balanced':>10} {'L/R/D':>10}")
    print('-'*91)
    for name,v in r['development'].items():
        print(f"{name:28} {str(v['correct'])+'/'+str(v['n']):17} {v['accuracy']:9.2f}% {v['balanced_accuracy']:9.2f}% {v['movement_macro']:9.2f}%")
    print('='*91)
    print('ADOPTION: FAILED. Current production monitors retained.')
    print('Full counts and retrospective external results: '+str(HERE/'RESULTS.md'))

if __name__=='__main__': main()
