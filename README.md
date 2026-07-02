# phishpharm — phishing-detection training data

This repository builds a training set of two-sided email correspondences for a
phishing-detection approach based on **information leakage in possible future
responses**. The intuition: phishing senders are often constrained — they
can't truthfully answer plausible recipient follow-ups, they can't be reached
through legitimate side channels, and they can't sustain consistent
context across turns. Generating realistic full correspondences exposes those
constraints in a way that single-shot phishing emails do not.

## Files

| File | Purpose |
|---|---|
| `scenarios.json` | 38 scenarios spanning genuine business / personal communications, obvious phishing, and sophisticated multi-stage phishing. Each scenario defines a sender persona, an initial email, target recipient personas, and (for non-genuine scenarios) the attacker's goal and the information they seek. |
| `phishing_response_personas.json` | 37 recipient personas grouped into 6 organizations (manufacturing, logistics, SaaS, financial services, healthcare, higher-ed) plus unaffiliated consumer personas. Each persona has a `system_prompt` that puts an LLM into character, plus an `information_store` (facts the persona knows) and an `information_vault` (role-specific items they should not reveal). Personas in the same organization can be targeted by the same workplace scenario. **Used for training-data generation.** |
| `generic_personas.json` | 12 trait-based personas — no name, role, or organization, only psychological/behavioral qualities. **Used at runtime by the predictive classifier** to simulate responses for any scenario. Generic personas inherit their disclosure rules from the recipient's organization vault (see below) rather than carrying their own; unaffiliated recipients pair with no vault. |
| `predictive_classifier.py` | **The actual detection approach this repo is built around.** Classifies an initial email by *predicting how the conversation would unfold* and watching for information leakage in the predicted future turns — it never sees the scenario type, sender, or attacker goal. Sweeps the 12 generic personas (×N branches) replying to the email while an LLM continues the apparent sender's side; analyzes each predicted correspondence for disclosure of the recipient org's abstracted vault categories; then a risk-assessor LLM produces the verdict from the aggregated leakage. All calls use the Claude API. |
| `organization_information_vaults.json` | Per-organization vaults — categories of information that any persona acting on behalf of that organization should not reveal. Generic personas in `generic_personas.json` inherit these at runtime when paired with an organization context. Per-persona vaults in `phishing_response_personas.json` layer role-specific additions on top. |
| `generate_correspondences.py` | Generation script. For every (scenario × target persona) pair, simulates a multi-turn email exchange between two Claude calls (one playing the sender with full knowledge of the scenario, one playing the recipient with only their persona). Writes one JSON record per correspondence to a JSONL file. |
| `annotate_leakage.py` | Post-generation script that asks Claude to identify exact text spans in recipient replies that constitute information leakage (cross-referenced against each scenario's `sensitive_information_sought` list). Writes a flat annotations JSON file. |
| `viewer.html` | Self-contained browser UI for reading the correspondences. Gmail-style thread view, scenario context in the side rail, leakage spans highlighted in-line, manual highlighting with note tooltips, filter by type / scenario / persona / free-text. No build step — open in any browser. |
| `traditional_filter.py` | Baseline detectors applied to the **initial email only**: (1) a rule-based filter modeling pre-LLM software — PILFER structural features + SpamAssassin-style additive scoring, with anchor/href base-domain mismatch and encoded-IP detection ported from Mozilla Thunderbird's `PhishingDetector.sys.mjs`; no API key — and (2) an optional single-message LLM classifier (`--llm`, Claude API) that judges the same lone email. Reports confusion matrices, per-type detection rates, and a side-by-side comparison. Neither baseline uses the predicted correspondence — that is the signal this repo's tool adds on top. |
| `traditional_filter_analysis.md` | Written analysis of the baseline runs: what they catch, what they miss, and why single-message classifiers (rule-based *or* LLM) are structurally blind to multi-stage phishing. |
| `plot_confusion_matrices.py` | Renders a confusion matrix per classifier (rule-based + LLM) from the `filter_results.json` that `traditional_filter.py --json-out` writes. Outputs a PNG and also prints the matrices as text. Requires matplotlib. |
| `phishing_training_data.json` | Earlier single-email dataset, retained for reference. Not consumed by the new pipeline. |
| `requirements.txt` | Python dependencies. |

## Schema highlights

### Scenarios
Each scenario object carries:
- `scenario_type`: `genuine` | `obvious_phishing` | `sophisticated_phishing`
- `attacker_goal` and `sensitive_information_sought` (null/empty for genuine)
- `is_spear_phishing`: targets specific individuals using personal/organizational detail
- `target_personas`: list of persona IDs that the scenario plausibly targets — the generator pairs the scenario with each of these
- `sender_persona`: sender's claimed identity, background, communication style, objective, escalation strategy, and explicit constraints on what they cannot do (used to keep them in character across turns)
- `initial_email`: the first message. Supports `{recipient_first_name}`, `{recipient_email}`, `{recipient_org_domain}`, `{colleague_name}` placeholders, substituted at generation time.

### Personas
Each persona carries Big Five profile, vulnerability/resilience factors, a
detailed `system_prompt` written in second person, and an `organization_id`
linking to one of the workplaces. Workplace scenarios can be sent to any
subset of an organization's personas; consumer scenarios target unaffiliated
personas.

## Running the generator

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

# Dry run — list what would be generated
python generate_correspondences.py --dry-run

# Generate everything (one correspondence per scenario × target persona pair)
python generate_correspondences.py

# Smoke test: a single sophisticated scenario, one persona
python generate_correspondences.py \
    --scenarios-filter SC-SOPH01 \
    --personas-filter MER-AP \
    --limit 1

# Multiple variants per pair (useful for diversity)
python generate_correspondences.py --runs-per-pair 3

# Resume after interruption
python generate_correspondences.py --resume
```

Each correspondence is appended to `correspondences.jsonl` as one JSON line
with the full thread, both agents' in-character reasoning per turn, red flags
the recipient noticed, and the termination reason. The script uses Claude's
structured-output feature to validate every turn against a schema, so partial
results are usable even if the run is killed.

## How the simulation works

Each turn alternates:

1. The **recipient agent** receives the email thread so far. Its system
   prompt is the persona's `system_prompt` — nothing about the scenario or
   the sender's hidden objective leaks across. It produces a structured
   response: an action (`reply`, `no_reply`, `report_as_phishing`,
   `verify_out_of_band`, `disengage`), the email body if replying, and an
   in-character reasoning string plus any red flags it noticed.
2. The **sender agent** receives the same thread plus the full scenario
   context, including the attacker's hidden goal and escalation strategy. It
   decides whether to send another email (and at what escalation stage), end
   successfully, or give up.

The thread terminates when either side disengages or `--max-turns` is reached.

## Inspecting correspondences for information leakage

The premise of this dataset is that phishing reveals itself through information
leakage in possible future responses. To look at correspondences and see where
leakage actually occurred:

1. **Auto-annotate (optional but recommended).** Run the annotator over the
   generated JSONL to identify leakage spans in recipient replies:

   ```bash
   python annotate_leakage.py \
       --input correspondences.jsonl \
       --output annotations.json
   ```

   The annotator reads each phishing correspondence, sees the attacker's hidden
   goal and the categories of information they were trying to extract, and
   asks Claude to point at exact substrings in the recipient's outgoing
   emails where leakage occurred. The output is a flat JSON list, one record
   per highlight, with category and explanatory note.

2. **Open the viewer.** Open `viewer.html` in any browser (no server, no build
   step). Click **Load JSONL** and pick `correspondences.jsonl`, then click
   **Load annotations** and pick `annotations.json`.

3. Each correspondence renders as a Gmail-style thread. Highlighted spans show
   on hover the category and explanation of what was leaked. The right rail
   shows the attacker's goal, the information they sought, and both personas'
   metadata. The filter bar narrows the list by scenario type, scenario ID,
   recipient, or full-text search. Each turn can optionally show the in-
   character reasoning the agent provided when generating the email.

   To add highlights manually: select text in a recipient email, click
   "Highlight as leakage" in the bottom bar, and add a note. Manual annotations
   persist to `localStorage` and can be exported as JSON via the same bar.

## Baseline classifiers (what our tool must beat)

Two single-message baselines classify the **initial email only** — neither uses
the predicted correspondence. They establish the floor the information-leakage
tool has to improve on.

```bash
# Rule-based filter alone (offline, no API key):
python3 traditional_filter.py

# Rule-based + single-message LLM, with the side-by-side comparison:
export ANTHROPIC_API_KEY=...
python3 traditional_filter.py --llm --json-out filter_results.json
```

`traditional_filter.py` writes a confusion-matrix PNG automatically
(`filter_confusion_matrices.png`, one panel per classifier that ran) — install
matplotlib (`pip install matplotlib`) to enable it. Control it with:

```bash
python3 traditional_filter.py --llm --plot-out cm.png   # custom path
python3 traditional_filter.py --llm --normalize         # add row-normalized rates
python3 traditional_filter.py --no-plot                 # skip plotting

# Or regenerate the plot later from a saved results file:
python3 plot_confusion_matrices.py --input filter_results.json
```

`traditional_filter.py` prints per-email verdicts, confusion-matrix counts,
metrics, and per-type detection rates for each classifier (see
`traditional_filter_analysis.md` for interpretation). `plot_confusion_matrices.py`
turns the `--json-out` file into a two-panel PNG and also prints the matrices
as text.

## Predictive classifier (the real approach)

`predictive_classifier.py` is the detector the baselines exist to be measured
against. It does not score the initial email's surface features. It predicts
the conversation and looks for information leakage in the predicted future:

1. **Predict correspondences.** For each of the 12 generic personas it generates
   `--branches` predicted correspondences. The persona replies to the email; an
   LLM playing "the original sender" — given only the thread, never the scenario
   or any goal — continues naturally, pursuing whatever the sender appears to
   want. They go back and forth up to `--max-iterations` rounds.
2. **Analyze leakage.** Each predicted correspondence is reviewed by an analyst
   LLM together with the recipient organization's **abstracted** vault
   (`organization_information_vaults.json` — category + type-level description,
   never any real secret), which reports whether the recipient disclosed any
   protected category.
3. **Assess risk.** A final assessor LLM reads the aggregated leakage findings
   across all personas and branches and produces the verdict. Legitimate mail,
   continued naturally, does not steer recipients into disclosing protected
   info; phishing does — across personalities, escalating, reaching sensitive
   categories.

The classifier is told only the initial email and (optionally) the recipient's
organization — the party being defended, which selects the vault. It never sees
the scenario type, sender persona, or attacker goal.

```bash
export ANTHROPIC_API_KEY=...

# Cheap smoke test (one scenario, two personas, single branch):
python3 predictive_classifier.py --limit 1 --personas GP-01,GP-07 --branches 1

# See the work plan + Claude-call estimate without spending anything:
python3 predictive_classifier.py --dry-run

# Full sweep (expensive — read the printed estimate first):
python3 predictive_classifier.py --json-out predictive_results.json
```

It prints a verdict per scenario, a confusion matrix, and recall (the metric
that matters — see `traditional_filter_analysis.md`), and writes a
`predictive_confusion_matrix.png`. The `--json-out` file is loadable by
`plot_confusion_matrices.py`. Cost scales with scenarios × personas × branches
× rounds; bound it with `--limit`, `--personas`, `--branches`,
`--max-iterations`, and parallelize with `--concurrency`.

## Adding scenarios or personas

- New scenarios: append to `scenarios.scenarios` in `scenarios.json`. Pick an
  unused `id`. The `target_personas` list determines who the generator will
  pair the scenario with — for spear-phishing scenarios this is meaningful
  (the sender only knows enough to plausibly target certain people); for
  generic blast phishing, list anyone who might receive it.
- New personas: append to `personas` in `phishing_response_personas.json`.
  Set `organization_id` to one of the existing org IDs to make the persona
  available for workplace scenarios, or `null` for a consumer persona.
- New organizations: append to `organizations` in
  `phishing_response_personas.json`, then create at least 2-3 personas with
  that `organization_id` so workplace scenarios have multiple targets to
  choose from.
