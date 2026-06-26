# Auditing Counterfactual Recourse for EU AI Act Compliance in Financial Credit Scoring

A reproducible pipeline that audits algorithmic recourse (counterfactual explanations) from credit-scoring models against five recourse-quality criteria, framed around EU AI Act compliance. The study spans seven research questions (RQ1–RQ7), two public credit datasets, two model families, and multiple counterfactual-generation methods, and exports publication-ready figures and statistical tables.

## Overview

For each applicant profile, the pipeline generates counterfactual recourse, grades it against five criteria, and runs statistical analyses to compare methods, models, and datasets:

- **C1 — Immutability**: recourse does not require changing protected/immutable features
- **C2 — Actionability**: changes are realistically actionable by the applicant
- **C3 — Sparsity**: few features changed
- **C4 — Causal**: changes respect plausible causal structure
- **C5 — Diversity**: diverse recourse options offered

Analyses include bootstrap confidence intervals, Cohen's *d* effect sizes, Wilcoxon and Friedman/Nemenyi tests, co-failure (phi) analysis, Pareto cost–compliance trade-offs, Cronbach's alpha, and sensitivity checks.

## Repository structure

```
.
├── main_pdf.py        # Full pipeline (RQ1–RQ7); exports PNG + PDF figures and CSV tables
├── check_results.py   # Quick sanity check that the pipeline completed correctly
├── requirements.txt   # Python dependencies
├── data/              # Input datasets (see below)
└── results/           # Generated outputs
    ├── figures/       # Figures (PNG)
    ├── master_scorecard.csv
    ├── rq*_*.csv      # Per-research-question result tables
    └── nemenyi_*.csv  # Post-hoc test outputs
```

## Datasets

Place the following files in `data/`:

- `german.data` — German Credit dataset (UCI Machine Learning Repository)
- `default_of_credit_card_clients.xls` — Taiwan / Default of Credit Card Clients dataset (UCI)

Both are publicly available from the UCI Machine Learning Repository and are not redistributed here.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Run the full pipeline
python main_pdf.py

# Verify outputs
python check_results.py
```

Results are written to `results/` (CSV tables + `results/figures/` PNGs). Publication-quality PDF copies of every figure are also produced.

## Key configuration

Set near the top of `main_pdf.py`:

- `N_PROFILES = 50` — applicant profiles sampled per dataset
- `N_COUNTERFACTUALS = 5` — counterfactuals generated per profile
- `RANDOM_STATE = 42` — seed for reproducibility

## Requirements

Python 3.9+ with: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, openpyxl, dice-ml, tqdm, xlrd (see `requirements.txt`).

## Citation

If you use this code, please cite the associated thesis:

> *Auditing Counterfactual Recourse for EU AI Act Compliance in Financial Credit Scoring.*

## License

Add a license of your choice (e.g. MIT) before publishing.
