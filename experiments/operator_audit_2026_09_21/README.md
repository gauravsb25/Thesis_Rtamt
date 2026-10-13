# Independent operator audit — 21 September 2026

This audit complements the concurrent operator sweep. It writes only in this directory;
it does not modify the production monitors, other experiments, datasets, or shared caches.
`signals_snapshot.py` is a byte-for-byte snapshot for reproducibility. Its original script
entry point can write the shared bank: do not execute that snapshot directly.

Run from the project root:

```powershell
python -B experiments\operator_audit_2026_09_21\audit.py
```

## What was independently executed

40 representative formulas, each in three modes: offline, direct online, and online after
`pastify()`. These are compatibility smoke tests on short synthetic traces, not accuracy
experiments, not every alias, and not a proof of all boundary semantics.

* Offline: 39/40 execute. Unbounded `unless` fails with an AttributeError in this bundled runtime.
* Direct online: 30/40 execute. Future-time formulas fail; bounded future operators are not
  directly usable just because the parser accepts them.
* Online after pastification: 36/40 execute. Bounded future formulas and next execute;
  unbounded always/eventually/until cannot be pastified, and unbounded unless still fails.
* Arithmetic, mathematical functions, comparisons, Boolean connectives, past-time operators,
  rise and fall execute on the tested inputs. This does not establish numerical safety for
  arbitrary inputs (division by zero, overflow, negative square roots).

Full outcomes and exceptions are in `audit_results.json`. Delayed alignment after pastification
still needs verification before deployment. The existing monitor uses sample-index timestamps:
`[0:767]` spans its 768 samples. Do not reinterpret those numbers as physical seconds or change
the sampling period without changing the bounds consistently.

## Reproducible discrepancies in the concurrent sweep

1. **Response feature is not equivalent to its documented formula.** On 12 zeros,
   `always[0:11]((x >= 2) implies eventually[0:3](x <= -2))` is satisfied: there is no
   positive excursion requiring a response. `_reduce` returns `resp=1`, making its proposed
   test `resp*12 <= 3` false. The same mismatch occurs with negative-only excursions.
   Either implement vacuous satisfaction for this feature or explicitly describe a different
   rule requiring a triggering event. This issue does not prove its classifier predictions
   were miscomputed; it invalidates the asserted equivalence to this STL formula.
2. **Rise verification checks a different signal.** `_reduce` counts crossings of `abs(s)>=2`.
   The verifier compares `rise(s>=2)` and a separate signed NumPy count, rather than comparing
   RTAMT against the stored rise feature. On `[0,-3,-3,0]` repeated three times, the feature
   counts 3, signed RTAMT rise counts 0, and absolute-value RTAMT rise counts 3. Correct the
   description and compare the actual bank feature to `rise(abs(s)>=2)`. Also test exact
   threshold equality and the initial sample separately.
3. **The monotonicity statement is too broad.** `abs(x)` and `x²` are not monotone over signed
   EEG. At x=-3 and threshold 4, x²>=4 is true while x>=sqrt(4) is false. Positive-domain
   monotone transforms with transformed thresholds can be redundant. The bank already
   includes absmax, so this wording error does not mean all absolute-value features were omitted.
4. **Some features are preprocessing summaries, not native STL operators.** Occupancy fraction,
   event counts, argmax ordering, and separation need explicit derived signals or additional
   constructions. `historically` alone means every sample, not a percentage of samples.
5. **The interval descriptions reverse some thirds.** At the final sample, once[0:255] looks
   at the last third, not the first. `_reduce` itself slices e1/e2/e3 chronologically; its OPS
   documentation should distinguish forward-time from backward-time bounds.
6. **The noise conclusions are stronger than the evidence.** Eight null replicates cannot
   establish a significant effect merely because the observed value beats all eight: the
   basic plus-one Monte Carlo p-value would be 1/9, before accounting for multiple searches.
   Shuffling individual overlapping windows also destroys temporal dependence; recording-level
   or appropriate block permutations are preferable. This is not proof of a real accuracy gain.

## Accuracy evidence and what remains workable

The concurrent README reports development balanced-accuracy gains of +0.15 points for response,
+0.08 for sustained excursions, +0.18 for xor/iff, and -0.91 for the full search. Those numbers
were read from its report, **not independently rerun here**. They do not demonstrate >90% test
accuracy, and the response-formula discrepancy should be fixed before interpreting that family.

Earlier saved nested-STL results tested 145 candidates: best development balanced accuracy
85.50% versus 85.01%, below their adoption rule. Earlier 703-feature synthesis reached 87.26%
development balanced accuracy, but its external results were mixed. Those are historical results,
not new measurements from this audit. Repeating their names as new methods would be misleading.

Workable engineering directions include correctly specified sustained excursions, onset/offset
events, ordered channel responses, and fixed-delay future formulas. Their accuracy benefit remains
unproven. Additional arithmetic involving two signals can add features; logical aliases alone do
not add expressive power. Equality is generally unhelpful for raw EEG, but is meaningful for
discrete derived state signals, so 'never usable' is too strong.

Before another accuracy run: correct the semantic discrepancies, test quiet/positive-only/
negative-only/boundary traces, freeze feature definitions, and select candidates using development
sessions only. Compare final frozen candidates against the same original monitor on the same
recordings. Previously examined external datasets are retrospective tests, not fresh blind tests.
Keep ordinary and balanced accuracy separate. There is no justified production replacement or
new accuracy improvement from this audit. These failures do not establish that no method can work.
