# 5-Minute Demo Script (Conversational English)

## Runtime target
- Target length: 4:35 to 4:55
- Style: natural, spoken, not formal reading
- Demo URL: `video/final/demo/index.html`

---

## 0:00-0:20 | Quick intro
**On screen**
- Demo home header.

**Say this (conversational)**
Hi everyone, this is my project on risk-sensitive ETF trading with temporal-difference learning.
The key point is simple: I’m not only chasing return, I’m selecting policies that still look acceptable when we enforce downside risk constraints.

---

## 0:20-0:55 | Why XLF and why these 3 algorithms (Panel 1)
**On screen**
- Open tab: `1) Why XLF + Project Purpose`
- Point at KPI cards and headline table.

**Say this**
I use XLF because it is liquid and diversified, but still has real drawdown and recovery periods.
So it’s a good stress environment for risk-return tradeoffs.
I compare Q-learning, SG-Sarsa, and n-step Sarsa because they represent three different learning behaviors:
off-policy tabular, on-policy linear approximation, and multi-step credit assignment.
That gives me a meaningful comparison under the same gate.

---

## 0:55-1:45 | Make the algorithm mechanics tangible (Panel 2)
**On screen**
- Switch to `2) TD Mechanics Lab`
- Click each method once and move 1-2 sliders.

**Say this**
This panel is here to make the methods intuitive.
For Q-learning, the target uses the greedy max term.
You can see when I move alpha or gamma, the update reacts quickly.
For SG-Sarsa, updates are on-policy and often smoother, but more sensitive to schedule settings.
For n-step Sarsa, the target includes longer-horizon information, which helps align updates with trading horizon.
So this is exactly why the three-way comparison is useful.

---

## 1:45-2:35 | Core evidence: gate replay (Panel 3)
**On screen**
- Switch to `3) Gate Replay Demo`
- Let autoplay run, or click `Next` manually.

**Say this**
Here is the main result.
Q-learning gives the biggest raw uplift, reward delta about +0.456 and return delta about +0.620.
But risk checks fail, so it gets rejected.
SG-Sarsa improves less and still fails risk checks.
n-step Sarsa has smaller utility gain, around +0.022 reward delta and +0.024 return delta,
but it passes all gate checks and gets promoted.
So the selected algorithm is the one that is feasible under constraints, not the one with the biggest unconstrained jump.

---

## 2:35-3:25 | Practical ETF use case (Panel 4)
**On screen**
- Switch to `4) ETF Application Case`
- Keep initial capital at 10,000.
- Toggle risk profile: `Balanced` -> `Aggressive` -> `Conservative`.

**Say this**
This is the practical part.
I map the same model outputs to different ETF operation profiles.
Balanced mode follows promoted-first logic, so it picks n-step Sarsa.
Aggressive mode is return-first, which may pick Q-learning even if risk checks fail.
Conservative mode emphasizes downside control and again tends to select n-step Sarsa.
So this is how the comparison becomes actionable: policy selection depends on risk mandate, not just one metric.

---

## 3:25-4:05 | Trials and setbacks + tuning logic (Panel 5)
**On screen**
- Switch to `5) Parameter / Setback`
- Change metric dropdown: `return_delta` -> `cvar_delta` -> `train_seconds`.

**Say this**
This panel shows why tuning was not trivial.
SG can be feasible in one regime, but if I extend episode budget without coordinated retuning, utility can degrade.
So more training is not automatically better.
The key parameters are alpha, epsilon decay, epsilon minimum, and budget together.

---

## 4:05-4:35 | Wrap-up with one clear takeaway
**On screen**
- Stay on Panel 5 table or return to Panel 3 for final visual.

**Say this**
Final takeaway: under the synchronized risk gate on XLF, n-step Sarsa is the most robust promoted method.
This project is less about one flashy best number, and more about a transparent selection process under explicit risk constraints.

---

## 4:35-4:50 | Closing
**On screen**
- Repo link card.

**Say this**
Thanks for watching.
All code, figures, and data are in this repository:
https://github.com/tianhangzhu03/RL
