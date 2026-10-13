# Development-only stage selection

Fixed before running this continuation: use the same 15 fitting configurations and cached
inner-training fits from the balanced search. For each, consider seven nonempty subsets of changed
stages (gate, nod, direction); omitted stages retain the original monitor. Thus there are 105
configuration/subset options plus the unchanged baseline.

Choose the configuration and subset by pooled inner-validation four-class balanced accuracy;
prefer fewer changed stages on ties. Refit that configuration on the corresponding outer training
set, discard nonselected stage changes, and evaluate on the outer session block. Never choose an
outer fold's subset from its labels. No further held-out evaluation. This is a further exploratory
development search informed by the earlier stage ablation, not independent confirmation.
