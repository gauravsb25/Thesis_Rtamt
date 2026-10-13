# Frozen learned challenger formulas

These formulas were learned automatically and failed adoption. Original production monitors are unchanged. Time bounds are sample indices on the preprocessed 32 Hz trace; divide by 32 for seconds. Evaluate at index 0 of the completed window. The last trace sample is at 95/32 seconds. All formula horizons fit within it.

Channel signals are centered and MAD-normalized filtered EEG; pair signals normalize the channel difference; energy signals are log(1 + smoothed energy / within-window median energy). Exact preprocessing is in data.py. These are not raw-voltage thresholds.

Each seed supplies four class formulas. Their normalized robustness values are averaged, then the highest movement-class score is used only when the original monitor detects movement. Class formulas may overlap; they are not a mutually exclusive partition.

## positive, seed 11, LEFT

Robustness divisor: 0.27307582. Horizon: 46 samples. AST nodes: 4.

```text
eventually[0:31](not(eventually[0:15]((energy_F4 <= 0.23926147622288288))))
```

## positive, seed 11, RIGHT

Robustness divisor: 0.1. Horizon: 30 samples. AST nodes: 7.

```text
(eventually[0:15]((energy_T3 <= 0.23925882430298892)) until[0:15] eventually[0:15]((energy_T4 <= 0.23926225540247509)))
```

## positive, seed 11, DOWN

Robustness divisor: 0.1. Horizon: 30 samples. AST nodes: 7.

```text
(eventually[0:15]((energy_T3 >= -0.23925333492498352)) until[0:15] eventually[0:15]((energy_Fp2 <= 0.23926439540913927)))
```

## positive, seed 11, STRAIGHT

Robustness divisor: 0.1. Horizon: 30 samples. AST nodes: 7.

```text
(eventually[0:15]((energy_F4 <= 0.23926319524754966)) until[0:15] eventually[0:15]((energy_T4 <= 0.23927535520040982)))
```

## positive, seed 29, LEFT

Robustness divisor: 0.30507962. Horizon: 46 samples. AST nodes: 4.

```text
eventually[0:31](not(eventually[0:15]((energy_Fp1 <= 0.23926705399170184))))
```

## positive, seed 29, RIGHT

Robustness divisor: 0.2693248. Horizon: 46 samples. AST nodes: 4.

```text
eventually[0:31](not(eventually[0:15]((energy_T4 <= 0.23926758691131325))))
```

## positive, seed 29, DOWN

Robustness divisor: 0.1. Horizon: 18 samples. AST nodes: 4.

```text
always[0:3](eventually[0:15]((energy_F4 <= 0.23926204846792593)))
```

## positive, seed 29, STRAIGHT

Robustness divisor: 0.1. Horizon: 18 samples. AST nodes: 4.

```text
always[0:3](eventually[0:15]((energy_F4 <= 0.23926590939079592)))
```

## supervised, seed 11, LEFT

Robustness divisor: 0.76267578. Horizon: 95 samples. AST nodes: 5.

```text
(always[64:95]((pair_F4_Cz <= 1.8784976662610209)) and eventually[0:95]((energy_T4 >= 1.8941314458847045)))
```

## supervised, seed 11, RIGHT

Robustness divisor: 0.50508464. Horizon: 95 samples. AST nodes: 5.

```text
(always[0:15]((pair_Fp1_Fp2 >= -0.98347271651041768)) and eventually[0:95]((ch_T3 <= -2.8198639392852782)))
```

## supervised, seed 11, DOWN

Robustness divisor: 1.2291176. Horizon: 63 samples. AST nodes: 5.

```text
(always[0:63]((pair_F3_F4 <= 2.8681246815700963)) and always[16:47]((pair_Fp2_T3 <= 2.3869328434698533)))
```

## supervised, seed 11, STRAIGHT

Robustness divisor: 0.8401373. Horizon: 95 samples. AST nodes: 5.

```text
(always[32:63]((ch_Fp2 >= -2.2550655679907554)) and always[0:95]((energy_Fp2 <= 1.7891433870298354)))
```

## supervised, seed 29, LEFT

Robustness divisor: 1.4676035. Horizon: 95 samples. AST nodes: 5.

```text
(always[0:31]((pair_Fz_T3 <= 1.4789411777402122)) and always[64:95]((ch_F4 <= 2.0989903917979746)))
```

## supervised, seed 29, RIGHT

Robustness divisor: 0.56958841. Horizon: 47 samples. AST nodes: 5.

```text
(always[16:47]((pair_Fz_Cz <= 2.0748841973468664)) and always[0:15]((pair_Fz_T4 <= 1.3350641489028932)))
```

## supervised, seed 29, DOWN

Robustness divisor: 1.1621619. Horizon: 95 samples. AST nodes: 5.

```text
(always[0:15]((pair_Fp1_Cz <= 1.1145232495312332)) and always[0:95]((pair_T3_T4 <= 3.5893315562418735)))
```

## supervised, seed 29, STRAIGHT

Robustness divisor: 0.66138512. Horizon: 95 samples. AST nodes: 2.

```text
always[0:95]((energy_T4 <= 1.9340058546208949))
```
