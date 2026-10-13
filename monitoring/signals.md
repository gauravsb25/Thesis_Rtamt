# The monitor's signals, in depth

The seven signals the four STL rules read, explained in plain words and written as formulas. Everything here is
taken from `compute_signals` and its helpers in [`rtamt_signals.py`](rtamt_signals.py) (line numbers as of
2026-10-14). Thresholds are the current default, selected on sub_01 sessions 1–20.

Two common misreadings, corrected up front:

- **Contrast does not "cut" the window into separate half-second pieces.** It slides a half-second block along the
  window **one sample at a time**, so there are 641 overlapping blocks, not 6.
- **Delta's "1–4 Hz" is coarser than it sounds.** A half-second slice can only resolve frequencies in steps of 2 Hz,
  so in practice the "slow" band is the single 2 Hz bin, spread a little by the window taper.

---

## Notation

| Symbol | Meaning |
|---|---|
| $x_c[t]$ | raw EEG of channel $c$ at sample $t$, in µV; $c \in$ {Fp1, Fp2, F3, Fz, F4, T3, Cz, T4} |
| $N = 768$ | samples in one window (3 s at $f_s = 256$ Hz); $t = 0, \dots, N-1$ |
| $C = 8$ | number of channels |
| $\mathrm{BP}_{[a,b]}(\cdot)$ | 2nd-order Butterworth band-pass, $a$ to $b$ Hz, applied causally (left to right) |
| $\operatorname{med}(\cdot)$ | median |
| $\hat\sigma(y)$ | robust spread of a signal $y$ (defined below) |
| $W = 128$ | half a second, in samples |

---

## Part 0: four building blocks every signal uses

**1. Band-pass filters.** Each signal first filters the EEG to keep only a range of speeds. All four are 2nd-order
Butterworth filters, applied causally (left to right, like a live system):

| Filter | Range | Used by |
|---|---|---|
| detect | 0.5–3 Hz | contrast, background (head-movement speed) |
| class | 0.5–6 Hz | deflect, nod |
| prominence | 0.3–4 Hz | sideways |
| direction | 0.3–6 Hz | turn |

$$
y_c^{\mathrm{det}} = \mathrm{BP}_{[0.5,\,3]}(x_c),\quad
y_c^{\mathrm{cls}} = \mathrm{BP}_{[0.5,\,6]}(x_c),\quad
y_c^{\mathrm{prom}} = \mathrm{BP}_{[0.3,\,4]}(x_c),\quad
y_c^{\mathrm{dir}} = \mathrm{BP}_{[0.3,\,6]}(x_c)
$$

**2. The robust "normal size" (spread).** For a channel: take its median, measure how far each sample is from it,
and take the **median of those distances × 1.4826**. This is the MAD (median absolute deviation). For ordinary noise
it equals the standard deviation, but a single big swing barely changes it. So it measures **"how big this channel's
wiggle normally is"**, unaffected by the movement itself.

$$
\hat\sigma(y) = 1.4826 \cdot \operatorname{med}_t \big|\, y[t] - \operatorname{med}(y) \,\big|
$$

**3. Energy.** Energy = the signal **squared**. Squaring makes positive and negative swings both count, and makes big
swings count much more than small ones.

The **pooled energy** used by contrast and to find "the loudest part" of a window: each channel filtered to 0.5–3 Hz,
divided by its own spread, squared, and averaged over the 8 channels:

$$
p[t] = \frac{1}{C} \sum_{c} \left( \frac{y_c^{\mathrm{det}}[t]}{\hat\sigma\!\left(y_c^{\mathrm{det}}\right)} \right)^{2}
$$

**4. Ratios make the monitor unit-free.** Almost every signal is something divided by something from **the same
window or the same person**. If a person's EEG is ten times bigger, both parts grow ten times and the ratio doesn't
change. That's why one set of thresholds can work for both subjects. The microvolt experiment showed what happens
without ratios: sub_11 fell 11 points.

**"So far" values.** Most signals are computed **up to each moment $t$** (a running value), so RTAMT's `once` and
`historically` can check them, and the row-by-row monitor can use them live. At the window's last sample, "so far" =
the whole window.

---

## 1. Contrast: is part of this window much louder than its calmest part?

**Steps** (lines 349–352, helper `_pooled_energy`):
1. Filter each of the 8 channels to **0.5–3 Hz**.
2. Divide each channel by **its own spread**, so a naturally big channel (like sub_11's T4) doesn't dominate.
3. Square and **average over the 8 channels**. That gives one "pooled energy" number per sample.
4. Sum it over every **half-second block (128 samples)**, sliding one sample at a time.
5. **contrast at moment t = (energy of the half-second ending at t) ÷ (energy of the quietest half-second in the whole
   window)**. Before the first half-second is complete, contrast is 1.

**Formula:**

$$
E[k] = \sum_{i=k}^{k+W-1} p[i], \qquad k = 0, \dots, N-W
$$

$$
\mathrm{contrast}[t] =
\begin{cases}
\dfrac{E[t-W+1]}{\min_{k} E[k]} & t \ge W-1 \\[2ex]
1 & t < W-1
\end{cases}
$$

**Rule:** `once(contrast >= 5.0)`, meaning at some moment a half-second was at least 5× the calmest one. **Only used
when there's no history** (`background_valid = 0`).

**Example:** calmest half-second energy 2, a turn makes 50 → contrast 25 → passes easily. A calm window with one
small blink: calmest 0.5, blink 3 → contrast 6 → **also passes**. That's the weakness.

**Why it's built this way:** a movement is a burst on top of a calm stretch, and dividing by the calmest part needs
no knowledge of the person.
**Weakness:** it only knows this window. A very calm window makes every bump look big (false alarms), and a busy
window hides a real nod (missed nods).

---

## 2. Delta: did the share of slow waves rise?

**Steps** (lines 365–382):
1. Cut the window into **half-second slices that overlap**: a new slice every quarter-second, so about 11 slices.
2. For each slice and each channel: apply a **Hann taper** (fades the slice edges to avoid artificial frequencies),
   then take the **FFT** (the frequency content).
3. **slow share = power in 1–4 Hz ÷ power in 0.5–45 Hz.** Average the share over the 8 channels.
4. **delta = this slice's share − the lowest share of any slice in the window.**
5. Each value holds from the end of its slice until the next slice ends.

**Formula:** slices start at $s_j = 64\,j$ (every quarter-second) and are $L = 128$ samples long. For slice $j$ and
channel $c$, with the slice mean removed and a Hann window $w$:

$$
P_{c,j}(f) = \Big|\, \mathrm{FFT}\big( (x_c[s_j \dots s_j+L-1] - \overline{x}_{c,j}) \cdot w \big)(f) \,\Big|^{2}
$$

$$
\mathrm{share}_j = \frac{1}{C} \sum_{c}
\frac{\sum_{1 \le f < 4\ \mathrm{Hz}} P_{c,j}(f)}{\sum_{0.5 \le f \le 45\ \mathrm{Hz}} P_{c,j}(f)}
$$

$$
\mathrm{delta}[t] = \mathrm{share}_{j(t)} - \min_{j} \mathrm{share}_j
$$

where $j(t)$ is the latest slice that has ended by sample $t$ ($s_j + L - 1 \le t$); before the first slice ends,
delta is 0.

**Rule:** `once(delta >= 0.16)`, meaning at some point the slow share rose by at least 0.16 above the calmest slice.

**Example:** calm slices have a slow share of 0.20; during a turn a slice reaches 0.62 → delta 0.42 → passes.

**Why:** head movement shifts electrode contact and drags the signal slowly, which adds **slow** power. A blink or a
muscle twitch adds energy, but not mostly slow energy.
**Detail at depth:** a half-second slice has a frequency resolution of 2 Hz (its FFT bins sit at 0, 2, 4, 6 … Hz).
So "1–4 Hz" is effectively the **2 Hz bin**, about 1–3 Hz after the taper's spreading. It's a coarse "how much slow
power" measure, not a precise band.
**Note:** delta is a **share** (a fraction), not energy, so it's unit-free on its own.

---

## 3. Deflect: did the electrodes swing well beyond their normal size?

**Steps** (lines 384–387):
1. Filter each channel to **0.5–6 Hz** and subtract its median.
2. **spread** = 1.4826 × the median of |signal|, the channel's normal wiggle size.
3. For each channel: the **biggest |swing| so far ÷ spread** (a running maximum, so it only grows).
4. **Average over the 8 channels.**

**Formula:**

$$
z_c[t] = y_c^{\mathrm{cls}}[t] - \operatorname{med}\!\left(y_c^{\mathrm{cls}}\right),
\qquad
\sigma_c = 1.4826 \cdot \operatorname{med}_t |z_c[t]|
$$

$$
\mathrm{deflect}[t] = \frac{1}{C} \sum_{c} \frac{\max_{i \le t} |z_c[i]|}{\sigma_c}
$$

**Rule:** `once(deflect >= 2.8)`. Because it's a running maximum, this is the same as "the final value ≥ 2.8".

**Example:** T3 normally wiggles ±3 µV and swings to 15 µV → 5 for T3. If the 8 channels give 5, 4, 3, 2, 2, 3, 2,
2, the average is 2.9 → passes.

**Why:** a real head movement makes **many** electrodes swing far beyond normal at once. A local artifact on one
electrode raises only one of the eight numbers, so the average stays low.
**Likely (not measured directly):** even plain noise has a biggest swing of about 3 times its normal size over 768
samples. So 2.8 isn't strict, and deflect mostly removes windows where nothing at all swung. In the error analysis,
missed movements were mostly due to contrast being too low, not deflect.

---

## 4. Nod: midline-versus-temples balance plus F3/F4 moving together

**Steps** (lines 389–406). It's the **sum of two parts**, both on the 0.5–6 Hz signals:

**Part A, balance (running, from the window start):**
- midline energy = Fz² + Cz², summed so far; temple energy = T3² + T4², summed so far.
- **balance = log(midline ÷ temples)**.
- Nod: the whole head moves, so the midline is about as big as the temples → near **0**.
- Turn: the temples (over the neck and jaw muscles, and the side of the electrode cap) swing much more → **strongly
  negative**.

**Part B, symmetry (in the loudest 1 second so far):**
- Find the loudest 1-s stretch (by pooled energy).
- common = average of (F3 + F4)²; differential = average of (F3 − F4)².
- **symmetry = log(common ÷ differential)**.
- Nod: F3 and F4 move **together**, so the sum is big and the difference small → **positive**.
- Turn: they move **oppositely** → **negative**.

**nod = balance + symmetry.** The rule reads its **final** value: `nod >= −0.6` → nod-like.

**Formula** (using $z_c$ from deflect):

$$
\mathrm{balance}[t] = \ln \frac{\sum_{i \le t} \left( z_{Fz}[i]^2 + z_{Cz}[i]^2 \right)}
                              {\sum_{i \le t} \left( z_{T3}[i]^2 + z_{T4}[i]^2 \right)}
$$

Let $B_1(t)$ be the loudest 1-second block (256 samples) of pooled energy $p$ among the blocks finished by $t$:

$$
\mathrm{symmetry}[t] = \ln \frac{\operatorname{mean}_{i \in B_1(t)} \left( z_{F3}[i] + z_{F4}[i] \right)^2}
                               {\operatorname{mean}_{i \in B_1(t)} \left( z_{F3}[i] - z_{F4}[i] \right)^2}
$$

$$
\mathrm{nod}[t] = \mathrm{balance}[t] + \mathrm{symmetry}[t], \qquad
\text{the rule reads } \mathrm{nod}[N-1]
$$

**Example:** turn: balance −2.0, symmetry −1.0 → nod −3.0 → not a nod. Nod: balance −0.4, symmetry +1.2 → nod
+0.8 → nod.

**Why two parts:** each alone is fooled sometimes. A strong turn can be oddly symmetric, and a nod can be uneven
across the midline. Adding them makes a decision only when the evidence agrees.
**Note:** it's a ratio of energies (unit-free overall), but it isn't divided by each channel's spread. So it depends
on how strongly each electrode picks up movement for that person.

---

## 5. Sideways: was there ever a big left–right swing?

**Steps** (lines 408–416):
1. Filter to **0.3–4 Hz** and build the pair **F4 − T3** (a front-right electrode minus a left-temple one).
2. Subtract its median; **spread** = 1.4826 × the median |pair|.
3. **sideways at moment t = |F4 − T3 at t| ÷ spread**, at **every sample**, not a running maximum.

**Formula:**

$$
v[t] = y_{F4}^{\mathrm{prom}}[t] - y_{T3}^{\mathrm{prom}}[t],
\qquad
\tilde v[t] = v[t] - \operatorname{med}(v)
$$

$$
\mathrm{sideways}[t] = \frac{|\tilde v[t]|}{1.4826 \cdot \operatorname{med}_t |\tilde v[t]|}
$$

**Rules:**
- DOWN needs `historically(sideways <= 7.5)`: **never** a big sideways swing.
- "Not a nod" includes `once(sideways > 7.5)`: **any** big sideways swing.

**Example:** at rest the pair moves about ±4 µV; a turn pushes it to 40 µV → sideways 10 → this can't be a nod,
even if nod looked nod-like.

**Why:** it's a **veto**. Nod evidence can be fooled; an extreme left–right swing almost never happens in a pure nod.
So it overrules nod and sends the window to LEFT/RIGHT.
**Why F4 − T3:** it spans the head diagonally, so a sideways turn moves its two ends in opposite directions and the
difference becomes big. T3−T4 was tested and rejected because of sub_11's huge T4 swings.

---

## 6. Turn: which way was the biggest F4 − T3 swing?

**Steps** (lines 420–434):
1. Filter to **0.3–6 Hz** and build the pair **F4 − T3**.
2. Find the **loudest 1.5 seconds so far** (by pooled energy). That's where the movement is.
3. Inside it: subtract the median, then find the **single sample with the largest |value|**, keeping its sign.
4. **turn = −(that peak) ÷ (1.4826 × the median |segment|)**.
5. An exact tie at zero is read as LEFT (a technical fallback).

**Formula:** let $B_{1.5}(t)$ be the loudest 1.5-second block (384 samples) of pooled energy $p$ among the blocks
finished by $t$:

$$
d[t] = y_{F4}^{\mathrm{dir}}[t] - y_{T3}^{\mathrm{dir}}[t],
\qquad
s[i] = d[i] - \operatorname{med}_{i \in B_{1.5}(t)} d[i] \quad \text{for } i \in B_{1.5}(t)
$$

$$
i^{*} = \arg\max_{i \in B_{1.5}(t)} |s[i]|,
\qquad
\mathrm{turn}[t] = -\,\frac{s[i^{*}]}{1.4826 \cdot \operatorname{med}_{i \in B_{1.5}(t)} |s[i]|}
$$

**Rule:** read at the **last sample**: $\mathrm{turn}[N-1] > 0$ → **LEFT**, $\mathrm{turn}[N-1] < 0$ → **RIGHT**.

**Why the minus sign:** in a LEFT turn, T3 (left temple) rises, so F4 − T3 goes **down**: a negative peak. Flipping it
makes **positive = LEFT**.
**Why the loudest 1.5 s:** outside the movement the pair is just noise; looking only where the energy is avoids
reading a random wiggle.
**Weakness:** a turn is often **turn-and-return**, two swings in opposite directions. If both are about equal,
picking "the biggest" becomes close to a coin toss. The SINCE rule tried to fix this and was rejected; this is also
where most LEFT/RIGHT swaps come from.

---

## 7. Background: is this window much louder than the seconds before it?

**Steps** (lines 291–332):
1. Take the **history** (up to 5 s before the window) **and** the window **together**, and filter them to
   **0.5–3 Hz** in one go, so the filter runs continuously into the window.
2. Per channel: the **energy of the half-second ending at each sample**.
3. Per channel: the **reference** = the **median** of that energy over the history, **skipping its first second**
   (the filter is still settling) and using **at most the last 4 s**, sampled every 8th sample.
4. **background at moment t = average over the 8 channels of (energy at t ÷ that channel's reference)**.
5. **background_valid** = 1 if at least 1 s of usable history existed, else 0 (then background = 0 everywhere).

**Formula:** let $h_c$ be the $H$ history samples before the window ($H \le 1280$, i.e. at most 5 s). Filter history
and window together, then take the half-second energy ending at each sample $\tau$:

$$
u_c = \mathrm{BP}_{[0.5,\,3]}\big( [\,h_c,\ x_c\,] \big),
\qquad
e_c[\tau] = \frac{1}{W} \sum_{i=\tau-W+1}^{\tau} u_c[i]^2
$$

The reference uses the history after a 1-second settle (256 samples), at most the last 4 s (1024 samples), every
8th sample:

$$
\ell = \max(256,\ H - 1024),
\qquad
\mathcal{P} = \lbrace\, \ell,\ \ell+8,\ \ell+16,\ \dots \rbrace \cap [\,\ell,\ H-1\,],
\qquad
r_c = \operatorname{med}_{\tau \in \mathcal{P}} e_c[\tau]
$$

$$
\mathrm{background\_valid} =
\begin{cases} 1 & H - \ell \ge 256 \ \ (\text{at least 1 s of usable history}) \\ 0 & \text{otherwise} \end{cases}
$$

$$
\mathrm{background}[t] = \frac{1}{C} \sum_{c} \frac{e_c[H + t]}{r_c}
\qquad (\text{and } 0 \text{ for all } t \text{ when not valid})
$$

**Rule:** `once(background >= 8.0)`, meaning some half-second was at least 8× this person's level just before. When
valid, it **replaces** contrast.

**Example:** chewing before the window gives a reference of 5. A blink reaches 6 → background 1.2 → not loud ✅. A
turn reaches 60 → background 12 → loud ✅. Contrast would have called the blink loud.

**Why the median:** a single loud moment in the history (another blink) barely shifts a median, so the reference
stays "typical".
**Why per channel, then averaged:** each electrode is compared with itself (unit-free), and averaging needs several
electrodes to rise together.
**Weakness:** if a movement happened in the history (two movements close together), the reference is too high and
the second movement can be missed. The "censored background" option was built to fix this, but it raised false
alarms, so it's off.

---

## How they fit together

```
Did the head move?   LOUD  =  background ≥ 8.0   (or, with no history, contrast ≥ 5.0)
                     AND   delta ≥ 0.16          (slow waves rose)
                     AND   deflect ≥ 2.8         (electrodes swung)          → if not: STRAIGHT
Was it a nod?        nod ≥ −0.6  AND  sideways never > 7.5                    → DOWN
Which side?          turn > 0 → LEFT,   turn < 0 → RIGHT
```

The same, as the four STL rules RTAMT checks ($\mathbf{O}$ = `once[0:767]`, $\mathbf{H}$ = `historically[0:767]`):

$$
\mathrm{LOUD} = \mathbf{O}(\mathrm{background} \ge 8.0) \ \lor\
\big( \mathbf{H}(\mathrm{background\_valid} < 0.5) \land \mathbf{O}(\mathrm{contrast} \ge 5.0) \big)
$$

$$
\mathrm{MOVE} = \mathrm{LOUD} \ \land\ \mathbf{O}(\mathrm{delta} \ge 0.16) \ \land\ \mathbf{O}(\mathrm{deflect} \ge 2.8)
$$

$$
\begin{aligned}
\mathrm{STRAIGHT} &= \lnot\, \mathrm{MOVE} \\
\mathrm{DOWN} &= \mathrm{MOVE} \land (\mathrm{nod} \ge -0.6) \land \mathbf{H}(\mathrm{sideways} \le 7.5) \\
\mathrm{LEFT} &= \mathrm{MOVE} \land \big( (\mathrm{nod} < -0.6) \lor \mathbf{O}(\mathrm{sideways} > 7.5) \big) \land (\mathrm{turn} > 0) \\
\mathrm{RIGHT} &= \mathrm{MOVE} \land \big( (\mathrm{nod} < -0.6) \lor \mathbf{O}(\mathrm{sideways} > 7.5) \big) \land (\mathrm{turn} < 0)
\end{aligned}
$$

The three "moved?" signals each look at a **different property**: **how much** energy (contrast/background), **what
kind** of energy (delta: slow), and **how widespread** the swing is (deflect: many electrodes). A blink can fake one
of them more easily than all three at once.
