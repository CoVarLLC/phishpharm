# Traditional spam/phishing filter — analysis

`traditional_filter.py` models how pre-LLM spam and phishing software classifies
mail, and applies it to the **initial email only** of each of the 24 scenarios
in `scenarios.json`. This document interprets the run; reproduce it with:

```bash
python3 traditional_filter.py
```

## What the filter models

Traditional filters score a single message on hand-engineered surface features
and flag it when the total crosses a threshold. The script implements that
paradigm directly, grouped into the same feature families the literature uses:

- **Sender / header** — freemail sender claiming a corporate role (a BEC /
  SPF-alignment proxy), homoglyph/lookalike sender domains, brand-name vs.
  sending-domain mismatch.
- **URL / link (PILFER, Fette et al. 2007; Mozilla Thunderbird `PhishingDetector`)** —
  raw-IP hosts, lookalike/typosquat link domains (a URI-blocklist proxy),
  subdomain-dot stuffing, URL shorteners, link count, generic "click here"
  anchors, plus two rules ported from Thunderbird: (a) anchor-text vs href
  base-domain mismatch (text says "chase.com" but the link points to
  "chase-secure-alerts.com"), and (b) obfuscated-IP hosts (decimal dword
  3232235521, hex 0xC0A80001, zero-padded 192.168.000.001, octal 0300.0250.0.1).
- **Body content (Bayesian-token layer)** — urgency, threat/suspension,
  credential solicitation, financial solicitation, gift-card and lottery scam
  language, PII requests, malware double-extensions, generic greetings, shouting.
- **Scoring** — SpamAssassin-style additive points, default threshold **5.0**,
  with a spam-vs-phishing sub-label decided by which families fired.

This is deliberately rule-based so every point is explainable and no training
corpus or API key is needed. A classifier trained on the same features (as in
the JoWUA paper by Sundararaj & Kul) reaches the same qualitative ceiling: it
can only judge what is present in the one message it is handed.

## Results (threshold 5.0)

| | Caught / Total | Detection rate |
|---|---|---|
| Genuine (should NOT be flagged) | 0 / 11 flagged | — (0 false positives) |
| Obvious phishing | 9 / 9 | **1.00** |
| Sophisticated phishing | 2 / 18 | **0.11** |

Confusion matrix: **TP 11, FN 16, TN 11, FP 0.**

Because the filter produces **zero false positives** on this corpus, the
precision-side metrics (precision, F1, false-positive rate) are all trivially
perfect and uninformative — they tell us nothing about the hard problem. The
metric that matters is **recall**: of the 27 malicious openers, how many does
the filter catch? **Recall is 0.41 overall** (11/27), and the whole story is in
how that splits by sophistication (1.00 on obvious, 0.11 on sophisticated).
Everything below focuses on recall and the misses that drag it down.

(Counts reflect the expanded corpus of 38 scenarios — 11 genuine, 9 obvious,
18 sophisticated. The illustrative examples below predate the expansion but the
pattern is unchanged and, if anything, sharper: as more realistic targeted
openers are added, the sophisticated-class recall falls further.)

## Was it able to identify spam/phishing?

**Yes, but only the kind that announces itself in the first message.** All six
obvious attacks were caught, each on multiple independent signals:

- Lottery scam (12.0): lottery language + gift-card demand + PII request +
  generic "Dear Lucky Winner" + urgency.
- Bank / Microsoft / IRS / FedEx impersonations (7.0–13.5): brand-vs-domain
  mismatch and/or homoglyph domains (`micros0ft-account.com`,
  `chase-secure-alerts.com`), credential solicitation, urgency, threats.
- Tech-support scam (5.0, exactly at threshold): Microsoft brand mismatch +
  urgency + excessive capitalization.

Two **sophisticated** attacks were also caught — precisely because their opening
email still carried surface indicators:

- CEO wire-transfer BEC (6.0): a `gmail.com` sender signing as "CEO, Meridian
  Manufacturing" (freemail + corporate role) plus "wire transfer" and urgency.
- Shared-document credential trap (8.0): lookalike Google domains in both the
  sender and the link, plus "sign in with your Google account."

## Where it failed: sophisticated phishing

**Ten of twelve sophisticated attacks scored at or near zero and sailed through.**
These are the openers engineered to be innocuous — the malicious intent lives in
later turns, which a single-message filter never sees:

| ID | Attack | Score | Why it evaded the filter |
|---|---|---|---|
| SC-SOPH01 | Vendor BEC (bank-change) | 0.0 | Polite account-manager intro; lookalike domain `greenleaf-supply.com` isn't a *famous* brand, so no typosquat rule fires; no links, no urgency. |
| SC-SOPH02 | IT-migration credential harvest | 0.0 | Pure informational notice ("no action needed yet"); the credential request comes weeks later. |
| SC-SOPH03 | Peer benchmarking pretext | 0.0 | Friendly conference follow-up; no links, no money words. |
| SC-SOPH07 | CFO data-exfil (ProtonMail) | 0.0 | Subtler than the CEO variant: the body never says "CFO" and asks for "close numbers," not "wire transfer/routing," so even the freemail-BEC rule stays silent. |
| SC-SOPH09 | Fake-recruiter recon | 0.0 | Reads like ordinary recruiter outreach; the infrastructure questions come later. |
| SC-SOPH10 | Auditor impersonation | 0.0 | Professional audit-confirmation intro; records request escalates later. |
| SC-SOPH11 | Customer-impersonation hijack | 0.0 | Routine "checking on PO #" note. |
| SC-SOPH12 | Conference reconnect | 0.0 | Warm "great to meet you" message; the malicious attachment is promised for later. |
| SC-SOPH04 | HR-benefits PII harvest | 1.5 | Generic enrollment reminder; only a faint urgency match. |
| SC-SOPH05 | Colleague directory + gift cards | 1.5 | Casual offsite logistics question; the gift-card ask is the *final* stage. |

The miss rate on the sophisticated class is **89%** (16 of 18 let through).
This is not a tuning failure — it is structural. The filter scores surface
features of one message; these attacks deliberately front-load none, deferring
every red flag (the bank-change request, the credential link, the PII form, the
gift-card ask) to a later turn.

## Did it misclassify legitimate mail as spam?

**No false positives** — all six genuine messages passed (FP rate 0.00,
precision 1.00). But two genuine emails registered non-zero scores worth noting
as fragility:

- **SC-G01** (real internal IT migration) scored **2.0** on "verify your
  account" — even though the sentence is *negated*: "We are NOT going to email
  you a link to re-verify your account." Keyword scoring cannot read negation;
  a slightly more cautious threshold would have produced a false alarm on a
  legitimate security-awareness email.
- **SC-G03** (legitimate recruiter) scored **1.5** on a "right now" urgency
  phrase. Real recruiter mail routinely lands in spam folders for exactly this
  reason.

Both stayed under the 5.0 threshold, but they show the precision/recall
trade-off is sharp: lowering the threshold to catch more pretexting attacks
would start flagging benign mail first.

## Second baseline: a single-message LLM classifier (`--llm`)

```bash
python3 traditional_filter.py --llm                      # Claude over the same lone emails
python3 traditional_filter.py --llm --llm-model claude-opus-4-7 --json-out out.json
```

The script also runs a Claude-based classifier on demand. It sends **only the
initial email** to the model and asks it to label the message
phishing / spam / legitimate, with confidence and the specific cues it relied
on. The system prompt explicitly forbids the model from imagining or reasoning
about any later reply — it must judge the one message on its own merits.

This baseline exists to isolate *what the future-turn signal is worth*. It gives
the classifier the same modern reasoning ability our correspondence-level tool
uses, but withholds the predicted continuation. So:

- **Rule-based filter** — single message, surface features.
- **Single-message LLM** — single message, full reasoning, no predicted thread.
- **This repo's tool** — initial message **plus** the predicted correspondence
  and the information leakage it exposes.

Any lift the tool shows over the single-message LLM is attributable to the
future-information-leakage signal rather than to "using an LLM" per se.

**Expected behavior (fill in with a live run; this environment has no API key).**
The LLM should comfortably beat the rule-based filter on obvious phishing it
might score with a homoglyph domain alone, and it can reason about
plausibility in ways regex cannot. But the core result should persist: on the
*sophisticated* openers — the vendor-intro, the IT-migration "no action needed
yet" notice, the benchmarking request, the auditor introduction — there is
simply nothing malicious in the first message to detect. A faithful
single-message LLM either (a) calls them legitimate (correct, given only that
message) and thus misses the attack, or (b) flags polite, unverifiable business
mail as suspicious and pays for it in false positives on the genuine set
(SC-G02 vendor transition, SC-G03 recruiter, SC-G04 conference invite all look
structurally identical to the pretexting openers in their first message). Run
`--llm` to populate the real numbers and the rule-based-vs-LLM comparison table
the script prints.

## Takeaway for this project

Both baselines — the rule-based filter and the single-message LLM — are
near-perfect detectors of **commodity** phishing and near-blind to **targeted,
multi-stage** phishing, because their entire evidence base is the surface of a
single message. The scenarios they miss are exactly the ones whose tell is
*information the attacker tries to extract over subsequent turns* — i.e. the
signal the correspondence-level, future-information-leakage approach in this
repo is built to capture. These filters are the baselines that approach must
beat on the sophisticated class, where the rule-based filter scores 0.11 and
the single-message LLM is expected to remain low for the same structural reason.
