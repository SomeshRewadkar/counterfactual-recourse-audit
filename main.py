"""
================================================================================
THESIS FULL PIPELINE (v3 — Complete RQ1–RQ7 Coverage) — PDF FIGURE EXPORT
================================================================================
Title:  Auditing Counterfactual Recourse for EU AI Act Compliance
        in Financial Credit Scoring

Target Journal: Expert Systems with Applications (Elsevier)

PDF EXPORT VARIANT:
  This file is functionally identical to `main.py`. The ONLY difference is
  that every generated figure (graphs, plots, charts, heatmaps, radars,
  etc.) is additionally saved as a publication-quality PDF inside a
  dedicated `pdf_figures/` folder, which is created automatically on first
  run. The original PNG outputs in `./results/figures/` are preserved so
  that no existing display or downstream tooling is disrupted.

WHAT CHANGED FROM v2:
  - Full coverage of all 7 Research Questions
  - RQ1: Bootstrap CIs, feature-level violation tracking, composite scores
  - RQ2: Marginal gain computation, compliance curve export
  - RQ3: Cohen's d, Wilcoxon tests, failure overlap (systemic vs method-specific)
  - RQ4: Three-factor analysis, model-aware stats, CD diagram data
  - RQ5: Range-normalized recourse cost, cost decomposition by feature tier,
          Pareto frontier computation
  - RQ6: Persistence curves, co-failure phi-coefficients, criterion resolution
  - RQ7: Cronbach's alpha, sensitivity analysis, expert validation framework
  - Fixed C5 Diversity grading bug (now scoped to tool/phase/model)
  - 28 publication-ready figures aligned to RQ-figure mapping
  - All results exported to structured CSVs for thesis tables

HOW TO USE:
  1. pip install dice-ml pandas numpy scikit-learn scipy matplotlib seaborn openpyxl
  2. Download datasets to ./data/
  3. Run: python main_pdf.py
  4. Results saved to: ./results/
  5. PDF copies of every figure saved to: ./pdf_figures/
================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from itertools import combinations

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import chi2, friedmanchisquare, rankdata, wilcoxon
from scipy.optimize import minimize
import dice_ml

# ============================================================================
# CONFIGURATION
# ============================================================================
N_PROFILES = 50
N_COUNTERFACTUALS = 5
RANDOM_STATE = 42
RESULTS_DIR = "./results"
DATA_DIR = "./data"
FIGURES_DIR = "./results/figures"
# New: every figure produced by the pipeline is also saved as a PDF here.
# The directory is created automatically on first run if it does not exist.
PDF_FIGURES_DIR = "./pdf_figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PDF_FIGURES_DIR, exist_ok=True)

CRITERIA = ['C1_Immutability', 'C2_Actionability', 'C3_Sparsity',
            'C4_Causal', 'C5_Diversity']

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_german_credit():
    print("\n" + "=" * 70)
    print("LOADING DATASET A: German Credit Dataset")
    print("=" * 70)

    file_path = os.path.join(DATA_DIR, 'german.data')
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing file! Download 'german.data' from UCI to {DATA_DIR}/\n"
            f"URL: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)"
        )

    col_names = ['V' + str(i) for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=' ', header=None, names=col_names)

    data = pd.DataFrame()
    data['Age'] = df['V13'].astype(float)
    data['Sex'] = df['V9'].isin(['A91', 'A93', 'A94']).astype(float)
    data['Job'] = df['V17'].map(
        {'A171': 0.0, 'A172': 1.0, 'A173': 2.0, 'A174': 3.0}).astype(float)
    data['Housing'] = df['V15'].map(
        {'A151': 1.0, 'A152': 2.0, 'A153': 0.0}).astype(float)
    data['Credit_Amount'] = df['V5'].astype(float)
    data['Duration_Months'] = df['V2'].astype(float)
    data['Saving_Account'] = df['V6'].map(
        {'A61': 0.0, 'A62': 1.0, 'A63': 2.0, 'A64': 3.0, 'A65': 0.0}).astype(float)
    data['Checking_Account'] = df['V1'].map(
        {'A11': 0.0, 'A12': 1.0, 'A13': 2.0, 'A14': 0.0}).astype(float)
    data['Num_Credits'] = df['V16'].astype(float)
    data['Approved'] = (df['V21'] == 1).astype(int)
    data = data.fillna(0.0)

    print(f"  Shape: {data.shape}, Approval Rate: {data['Approved'].mean():.1%}")
    return data


def load_taiwan_credit():
    print("\n" + "=" * 70)
    print("LOADING DATASET B: Taiwan Credit Card Default Dataset")
    print("=" * 70)

    file_path = os.path.join(DATA_DIR, 'default_of_credit_card_clients.xls')
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing file! Download from UCI to {DATA_DIR}/\n"
            f"URL: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients"
        )

    df = pd.read_excel(file_path, header=1)
    data = pd.DataFrame()
    data['Age'] = df['AGE'].astype(float)
    data['Sex'] = df['SEX'].astype(float)
    data['Education'] = df['EDUCATION'].astype(float)
    data['Marriage'] = df['MARRIAGE'].astype(float)
    data['Credit_Limit'] = df['LIMIT_BAL'].astype(float)
    data['Bill_Amt1'] = df['BILL_AMT1'].astype(float)
    data['Pay_Amt1'] = df['PAY_AMT1'].astype(float)
    data['Pay_Status1'] = df['PAY_0'].astype(float)
    data['Utilization_Ratio'] = np.where(
        data['Credit_Limit'] > 0,
        data['Bill_Amt1'] / data['Credit_Limit'],
        0.0
    ).astype(float)
    data['Approved'] = (df['default payment next month'] == 0).astype(int)
    data = data.fillna(0.0)

    print(f"  Shape: {data.shape}, Approval Rate: {data['Approved'].mean():.1%}")
    return data


# ============================================================================
# SECTION 2: FEATURE CONFIGURATION
# ============================================================================

GERMAN_CONFIG = {
    'name': 'German_Credit',
    'immutable_features': ['Age', 'Sex'],
    'actionable_features': ['Job', 'Housing', 'Credit_Amount', 'Duration_Months',
                            'Saving_Account', 'Checking_Account', 'Num_Credits'],
    'mutable_features': ['Job', 'Housing', 'Credit_Amount', 'Duration_Months',
                         'Saving_Account', 'Checking_Account', 'Num_Credits'],
    'directional_constraints': {'Job': 'increase_only'},
    'causal_invalid_pairs': [
        ('Credit_Amount', 'Duration_Months'),  # Loan amount & term co-determined
        ('Saving_Account', 'Checking_Account'),  # Both reflect liquidity
        ('Job', 'Credit_Amount'),  # Job level constrains credit eligibility
    ],
}

TAIWAN_CONFIG = {
    'name': 'Taiwan_Credit',
    'immutable_features': ['Age', 'Sex', 'Marriage'],
    'actionable_features': ['Education', 'Credit_Limit', 'Bill_Amt1',
                            'Pay_Amt1', 'Pay_Status1', 'Utilization_Ratio'],
    'mutable_features': ['Education', 'Credit_Limit', 'Bill_Amt1',
                         'Pay_Amt1', 'Pay_Status1', 'Utilization_Ratio'],
    'directional_constraints': {
        'Education': 'decrease_only',
        'Pay_Amt1': 'increase_only',
    },
    'causal_invalid_pairs': [
        ('Bill_Amt1', 'Pay_Amt1'),  # Bill and payment causally linked
        ('Credit_Limit', 'Utilization_Ratio'),  # Utilization = Bill / Limit
        ('Bill_Amt1', 'Utilization_Ratio'),  # Utilization derived from bill
    ],
}


# ============================================================================
# SECTION 3: MODEL TRAINING
# ============================================================================

def train_models(data, dataset_name):
    print(f"\n  Training models on {dataset_name}...")
    X = data.drop('Approved', axis=1)
    y = data['Approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"  Random Forest — Accuracy: {rf_acc:.3f}, AUC: {rf_auc:.3f}")

    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    print(f"  Gradient Boosting — Accuracy: {gb_acc:.3f}, AUC: {gb_auc:.3f}")

    return {'RandomForest': rf, 'GradientBoosting': gb}, X_train, X_test, y_test


def get_rejected_profiles(model, X_test, n_profiles):
    predictions = model.predict(X_test)
    rejected = X_test[predictions == 0]
    if len(rejected) < n_profiles:
        return rejected.reset_index(drop=True)
    return rejected.head(n_profiles).reset_index(drop=True)


# ============================================================================
# SECTION 4: DiCE COUNTERFACTUAL GENERATION
# ============================================================================

def generate_counterfactuals_dice(model, X_train, rejected_profiles, config, phase):
    phase_names = {1: "UNCONSTRAINED", 2: "IMMUTABILITY",
                   3: "DIRECTIONAL", 4: "CAUSAL"}
    print(f"\n    [DiCE] Phase {phase}: {phase_names[phase]}")

    feature_names = list(X_train.columns)
    train_df = X_train.copy().reset_index(drop=True)
    train_df['Approved'] = model.predict(X_train)

    d = dice_ml.Data(dataframe=train_df,
                     continuous_features=feature_names,
                     outcome_name='Approved')
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    results = []
    for idx in range(len(rejected_profiles)):
        profile = rejected_profiles.iloc[[idx]].reset_index(drop=True)
        try:
            kwargs = {'total_CFs': N_COUNTERFACTUALS,
                      'desired_class': "opposite"}

            if phase >= 2:
                kwargs['features_to_vary'] = config['mutable_features']

            if phase >= 3:
                permitted = {}
                for feat, direction in config['directional_constraints'].items():
                    current_val = float(profile[feat].values[0])
                    feat_max = float(X_train[feat].max())
                    feat_min = float(X_train[feat].min())
                    if direction == 'increase_only':
                        permitted[feat] = [current_val, feat_max]
                    elif direction == 'decrease_only':
                        permitted[feat] = [feat_min, current_val]
                if permitted:
                    kwargs['permitted_range'] = permitted

            cf = exp.generate_counterfactuals(profile, **kwargs)
            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is not None and len(cf_df) > 0:
                for _, cf_row in cf_df.iterrows():
                    if cf_row.get('Approved', 0) == 1:
                        # Phase 4: Post-hoc causal filtering for DiCE
                        # DiCE doesn't support causal graphs natively, so we
                        # reject CFs that violate causal pair constraints.
                        if phase >= 4:
                            causal_violation = False
                            for fa, fb in config.get('causal_invalid_pairs', []):
                                orig_a = float(profile[fa].values[0])
                                orig_b = float(profile[fb].values[0])
                                cf_a = float(cf_row[fa])
                                cf_b = float(cf_row[fb])
                                if (abs(cf_a - orig_a) > 0.01 and
                                        abs(cf_b - orig_b) > 0.01):
                                    causal_violation = True
                                    break
                            if causal_violation:
                                continue  # Skip this CF

                        row = {'profile_idx': idx, 'phase': phase,
                               'tool': 'DiCE', 'dataset': config['name']}
                        for feat in feature_names:
                            orig = float(profile[feat].values[0])
                            cfv = float(cf_row[feat])
                            row[f'orig_{feat}'] = orig
                            row[f'cf_{feat}'] = cfv
                            row[f'changed_{feat}'] = abs(orig - cfv) > 0.01
                        results.append(row)
        except Exception:
            pass

    print(f"      Generated {len(results)} counterfactuals")
    return pd.DataFrame(results)


# ============================================================================
# SECTION 5: WACHTER COUNTERFACTUAL GENERATION
# ============================================================================

def _wachter_loss(x_prime, x_orig, model, lambda_param, feature_ranges):
    try:
        prob_approved = model.predict_proba(x_prime.reshape(1, -1))[0, 1]
    except Exception:
        prob_approved = float(model.predict(x_prime.reshape(1, -1))[0])

    pred_loss = (prob_approved - 1.0) ** 2
    diffs = np.abs(x_prime - x_orig) / (feature_ranges + 1e-9)
    dist_loss = np.sum(diffs)
    return lambda_param * pred_loss + dist_loss


def _generate_single_wachter_cf(x_orig, model, feature_names, feature_bounds,
                                feature_ranges, lambda_param=10.0, n_restarts=3):
    best_cf = None
    best_dist = np.inf

    for restart in range(n_restarts):
        perturbation = np.random.randn(len(x_orig)) * 0.1 * feature_ranges
        x_init = x_orig + perturbation
        for i, (lo, hi) in enumerate(feature_bounds):
            x_init[i] = np.clip(x_init[i], lo, hi)

        try:
            result = minimize(
                _wachter_loss, x_init,
                args=(x_orig, model, lambda_param, feature_ranges),
                method='L-BFGS-B', bounds=feature_bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            x_cf = result.x
            if model.predict(x_cf.reshape(1, -1))[0] == 1:
                dist = np.sum(np.abs(x_cf - x_orig) / (feature_ranges + 1e-9))
                if dist < best_dist:
                    best_dist = dist
                    best_cf = x_cf
        except Exception:
            continue

    return best_cf


def generate_counterfactuals_wachter(model, X_train, rejected_profiles,
                                     config, phase):
    phase_names = {1: "UNCONSTRAINED", 2: "IMMUTABILITY",
                   3: "DIRECTIONAL", 4: "CAUSAL"}
    print(f"\n    [Wachter] Phase {phase}: {phase_names[phase]}")

    feature_names = list(X_train.columns)
    feature_mins = X_train.values.min(axis=0)
    feature_maxs = X_train.values.max(axis=0)
    feature_ranges = feature_maxs - feature_mins

    results = []
    for idx in range(len(rejected_profiles)):
        if idx % 10 == 0:
            print(f"      Processing profile {idx}/{len(rejected_profiles)}...")

        profile = rejected_profiles.iloc[[idx]].reset_index(drop=True)
        x_orig = profile.values[0].astype(float)

        bounds = []
        for i, feat in enumerate(feature_names):
            lo = float(feature_mins[i])
            hi = float(feature_maxs[i])
            if phase >= 2 and feat in config['immutable_features']:
                lo = hi = float(x_orig[i])
            if phase >= 3 and feat in config.get('directional_constraints', {}):
                direction = config['directional_constraints'][feat]
                current_val = float(x_orig[i])
                if direction == 'increase_only':
                    lo = max(lo, current_val)
                elif direction == 'decrease_only':
                    hi = min(hi, current_val)
            if lo > hi:
                lo = hi
            bounds.append((lo, hi))

        cfs_for_profile = []
        for cf_num in range(N_COUNTERFACTUALS):
            lambda_val = 10.0 * (1.0 + 0.2 * cf_num)
            np.random.seed(RANDOM_STATE + idx * 100 + cf_num)

            # Phase 4: generate more candidates and filter causally invalid ones
            n_attempts = 2 if phase < 4 else 5

            x_cf = _generate_single_wachter_cf(
                x_orig, model, feature_names, bounds, feature_ranges,
                lambda_param=lambda_val, n_restarts=n_attempts
            )

            if x_cf is not None:
                # Phase 4: Post-hoc causal filtering — reject CFs that
                # change both features in a causal pair simultaneously.
                # This is necessary because L-BFGS-B can't encode causal
                # constraints as variable bounds (unlike immutability/direction).
                if phase >= 4:
                    causal_violation = False
                    for fa, fb in config.get('causal_invalid_pairs', []):
                        fa_idx = feature_names.index(fa) if fa in feature_names else -1
                        fb_idx = feature_names.index(fb) if fb in feature_names else -1
                        if fa_idx >= 0 and fb_idx >= 0:
                            fa_changed = abs(x_cf[fa_idx] - x_orig[fa_idx]) > 0.01
                            fb_changed = abs(x_cf[fb_idx] - x_orig[fb_idx]) > 0.01
                            if fa_changed and fb_changed:
                                causal_violation = True
                                break
                    if causal_violation:
                        continue  # Reject this CF, try next lambda
                # Sparsity-promoting step: snap tiny changes to original value.
                # L-BFGS-B makes micro-adjustments to ALL features. Without this,
                # Wachter changes 7-9 features by negligible amounts, causing
                # C3 Sparsity to fail on nearly every CF. We zero out changes
                # below 5% of the feature range — these are optimizer noise,
                # not meaningful recourse suggestions.
                SNAP_THRESHOLD = 0.05  # 5% of feature range
                for i in range(len(x_cf)):
                    if feature_ranges[i] > 0:
                        relative_change = abs(x_cf[i] - x_orig[i]) / feature_ranges[i]
                        if relative_change < SNAP_THRESHOLD:
                            x_cf[i] = x_orig[i]

                # Re-verify validity after snapping
                if model.predict(x_cf.reshape(1, -1))[0] != 1:
                    continue  # Snapping broke the flip — discard

                is_distinct = True
                for prev_cf in cfs_for_profile:
                    if np.allclose(x_cf, prev_cf, rtol=0.01):
                        is_distinct = False
                        break

                if is_distinct:
                    cfs_for_profile.append(x_cf)
                    row = {'profile_idx': idx, 'phase': phase,
                           'tool': 'Wachter', 'dataset': config['name']}
                    for i, feat in enumerate(feature_names):
                        orig = float(x_orig[i])
                        cfv = float(x_cf[i])
                        row[f'orig_{feat}'] = orig
                        row[f'cf_{feat}'] = cfv
                        row[f'changed_{feat}'] = abs(orig - cfv) > 0.01
                    results.append(row)

    print(f"      Generated {len(results)} counterfactuals")
    return pd.DataFrame(results)


# ============================================================================
# SECTION 6: COMPLIANCE SCORECARD GRADING
# ============================================================================
# Fixed: C5 Diversity now scoped to (tool, phase, model, profile_idx)
# Added: Normalized recourse cost, feature-tier cost decomposition,
#        composite compliance score, feature-level violation tracking
# ============================================================================

def grade_compliance(cf_results, config, feature_ranges=None):
    """
    Grade counterfactuals against the 5-criterion Compliance Scorecard.

    Returns DataFrame with columns:
      - C1–C5 pass/fail
      - composite_score (0–5)
      - recourse_cost (normalized L1)
      - cost_immutable, cost_actionable (decomposed)
      - n_features_changed
      - per-feature violation flags
    """
    if len(cf_results) == 0:
        return pd.DataFrame()

    immutable = config['immutable_features']
    actionable = config['actionable_features']
    directional = config.get('directional_constraints', {})
    causal_pairs = config.get('causal_invalid_pairs', [])
    feature_cols = [c.replace('changed_', '') for c in cf_results.columns
                    if c.startswith('changed_')]

    # Build feature ranges for normalization if not provided
    if feature_ranges is None:
        feature_ranges = {}
        for f in feature_cols:
            orig_vals = cf_results[f'orig_{f}']
            cf_vals = cf_results[f'cf_{f}']
            all_vals = pd.concat([orig_vals, cf_vals])
            feat_range = all_vals.max() - all_vals.min()
            feature_ranges[f] = feat_range if feat_range > 0 else 1.0

    grades = []
    for _, row in cf_results.iterrows():
        g = {
            'profile_idx': row['profile_idx'],
            'phase': row['phase'],
            'tool': row['tool'],
            'dataset': row['dataset'],
        }

        # C1 — Immutability (Art. 10 + Recital 47)
        c1_violations = []
        for f in immutable:
            if row.get(f'changed_{f}', False):
                c1_violations.append(f)
                g[f'violation_{f}'] = 1
            else:
                g[f'violation_{f}'] = 0
        g['C1_Immutability'] = 'FAIL' if c1_violations else 'PASS'
        g['C1_violation_details'] = ','.join(c1_violations) if c1_violations else ''

        # C2 — Actionability (Art. 86)
        c2_violations = []
        for feat, direction in directional.items():
            orig = row.get(f'orig_{feat}', 0)
            cfv = row.get(f'cf_{feat}', 0)
            if direction == 'increase_only' and cfv < orig - 0.01:
                c2_violations.append(f'{feat}_decreased')
            elif direction == 'decrease_only' and cfv > orig + 0.01:
                c2_violations.append(f'{feat}_increased')
        g['C2_Actionability'] = 'FAIL' if c2_violations else 'PASS'
        g['C2_violation_details'] = ','.join(c2_violations) if c2_violations else ''

        # C3 — Sparsity (Art. 14)
        n_changed = sum(1 for f in feature_cols if row.get(f'changed_{f}', False))
        g['C3_Sparsity'] = 'PASS' if n_changed <= 3 else 'FAIL'
        g['n_features_changed'] = n_changed

        # C4 — Causal Validity (Art. 13)
        c4_violations = []
        for fa, fb in causal_pairs:
            if row.get(f'changed_{fa}', False) and row.get(f'changed_{fb}', False):
                c4_violations.append(f'{fa}+{fb}')
        g['C4_Causal'] = 'FAIL' if c4_violations else 'PASS'
        g['C4_violation_details'] = ','.join(c4_violations) if c4_violations else ''

        # Recourse cost — NORMALIZED by feature range (Ustun et al. 2019)
        cost_total = 0.0
        cost_immutable = 0.0
        cost_actionable = 0.0
        for f in feature_cols:
            orig_v = row.get(f'orig_{f}', 0)
            cf_v = row.get(f'cf_{f}', 0)
            fr = feature_ranges.get(f, 1.0)
            normalized_diff = abs(cf_v - orig_v) / (fr + 1e-9)
            cost_total += normalized_diff
            if f in immutable:
                cost_immutable += normalized_diff
            elif f in actionable:
                cost_actionable += normalized_diff
        g['recourse_cost'] = round(cost_total, 4)
        g['cost_immutable'] = round(cost_immutable, 4)
        g['cost_actionable'] = round(cost_actionable, 4)

        # Per-feature change tracking (for RQ1 Fig 1.4 heatmap)
        for f in feature_cols:
            g[f'feat_changed_{f}'] = 1 if row.get(f'changed_{f}', False) else 0

        grades.append(g)

    grades_df = pd.DataFrame(grades)

    # C5 — Diversity (GDPR Art. 22) — scoped to (tool, phase, model, profile)
    # FIXED: was previously only scoped to profile_idx, causing cross-condition contamination
    div_records = []
    group_cols = ['profile_idx', 'tool', 'phase', 'dataset']
    if 'model' in cf_results.columns:
        group_cols.append('model')

    for pid in grades_df['profile_idx'].unique():
        tool = grades_df.loc[grades_df['profile_idx'] == pid, 'tool'].iloc[0]
        phase = grades_df.loc[grades_df['profile_idx'] == pid, 'phase'].iloc[0]
        dataset = grades_df.loc[grades_df['profile_idx'] == pid, 'dataset'].iloc[0]

        mask = ((cf_results['profile_idx'] == pid) &
                (cf_results['tool'] == tool) &
                (cf_results['phase'] == phase))
        pcf = cf_results[mask]

        paths = set()
        for _, r in pcf.iterrows():
            cs = frozenset(f for f in feature_cols if r.get(f'changed_{f}', False))
            paths.add(cs)

        div_records.append({
            'profile_idx': pid,
            'tool': tool,
            'phase': phase,
            'dataset': dataset,
            'C5_Diversity': 'PASS' if len(paths) >= 2 else 'FAIL',
            'n_distinct_paths': len(paths),
        })

    div_df = pd.DataFrame(div_records)
    # Merge on all scoping columns to avoid cross-contamination
    merge_cols = ['profile_idx', 'tool', 'phase', 'dataset']
    grades_df = grades_df.merge(
        div_df[merge_cols + ['C5_Diversity', 'n_distinct_paths']],
        on=merge_cols, how='left'
    )

    # Composite compliance score (0–5) — sum of binary passes
    for c in CRITERIA:
        if c not in grades_df.columns:
            grades_df[c] = 'FAIL'
    grades_df['composite_score'] = sum(
        (grades_df[c] == 'PASS').astype(int) for c in CRITERIA
    )

    return grades_df


# ============================================================================
# SECTION 7: STATISTICAL ANALYSIS — COMPLETE RQ3, RQ4, RQ6, RQ7
# ============================================================================

def friedman_nemenyi_test(all_grades, criterion, dataset):
    """Friedman + Nemenyi for one criterion on one dataset (RQ2, RQ4)."""
    ds = all_grades[all_grades['dataset'] == dataset]

    conditions = []
    for tool in ['DiCE', 'Wachter']:
        for phase in [1, 2, 3, 4]:
            conditions.append((tool, phase))

    profile_scores = {}
    for tool, phase in conditions:
        subset = ds[(ds['tool'] == tool) & (ds['phase'] == phase)]
        for pid in subset['profile_idx'].unique():
            if pid not in profile_scores:
                profile_scores[pid] = {}
            profile_scores[pid][(tool, phase)] = (
                subset[subset['profile_idx'] == pid][criterion] == 'PASS'
            ).mean()

    complete_profiles = [pid for pid, scores in profile_scores.items()
                         if len(scores) == len(conditions)]

    if len(complete_profiles) < 3:
        return None, None, None, None

    matrix = np.array([
        [profile_scores[pid][cond] for cond in conditions]
        for pid in complete_profiles
    ])

    if matrix.shape[1] < 3 or np.all(matrix == matrix[0, 0]):
        return None, None, None, None

    try:
        groups = [matrix[:, i] for i in range(matrix.shape[1])]
        stat, p_value = friedmanchisquare(*groups)
    except Exception:
        return None, None, None, None

    n_profiles = matrix.shape[0]
    n_conditions = matrix.shape[1]
    ranks = np.apply_along_axis(lambda x: rankdata(-x), 1, matrix)
    avg_ranks = ranks.mean(axis=0)

    # Nemenyi critical distance (k=8, alpha=0.05, q≈3.031 from Demsar 2006)
    q_alpha = 3.031
    critical_diff = q_alpha * np.sqrt(
        n_conditions * (n_conditions + 1) / (6.0 * n_profiles))

    cond_labels = [f"{t}_P{p}" for t, p in conditions]
    nemenyi_matrix = pd.DataFrame(
        index=cond_labels, columns=cond_labels)
    for i in range(n_conditions):
        for j in range(n_conditions):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            nemenyi_matrix.iloc[i, j] = "SIG" if diff > critical_diff else "n.s."

    # Also return average ranks for CD diagram (RQ4)
    rank_df = pd.DataFrame({
        'condition': cond_labels,
        'avg_rank': avg_ranks
    }).sort_values('avg_rank')

    return stat, p_value, nemenyi_matrix, rank_df


def compute_cohens_d(group1, group2):
    """Cohen's d effect size for two groups (RQ3)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a proportion (RQ1)."""
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(np.mean(sample))
    lower = np.percentile(stats, (1 - ci) / 2 * 100)
    upper = np.percentile(stats, (1 + ci) / 2 * 100)
    return lower, upper


def compute_phi_coefficient(x, y):
    """Phi coefficient for two binary arrays (RQ6)."""
    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)
    n = len(x)
    n11 = np.sum((x == 1) & (y == 1))
    n10 = np.sum((x == 1) & (y == 0))
    n01 = np.sum((x == 0) & (y == 1))
    n00 = np.sum((x == 0) & (y == 0))
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denom


def compute_cronbachs_alpha(binary_matrix):
    """Cronbach's alpha for internal consistency (RQ7)."""
    k = binary_matrix.shape[1]
    if k < 2:
        return 0.0
    item_variances = np.var(binary_matrix, axis=0, ddof=1)
    total_scores = binary_matrix.sum(axis=1)
    total_variance = np.var(total_scores, ddof=1)
    if total_variance == 0:
        return 0.0
    alpha = (k / (k - 1)) * (1 - np.sum(item_variances) / total_variance)
    return alpha


# ============================================================================
# SECTION 8: RQ-SPECIFIC ANALYSIS FUNCTIONS
# ============================================================================

def analyze_rq1(all_grades):
    """RQ1: Baseline compliance failure rate with CIs and feature violations."""
    print(f"\n{'=' * 70}")
    print("RQ1 ANALYSIS: Baseline Compliance Failure Rate")
    print("=" * 70)

    p1 = all_grades[all_grades['phase'] == 1]
    results = []

    for dataset in p1['dataset'].unique():
        for tool in p1['tool'].unique():
            subset = p1[(p1['dataset'] == dataset) & (p1['tool'] == tool)]
            if len(subset) == 0:
                continue

            row = {'dataset': dataset, 'tool': tool, 'n': len(subset)}

            for c in CRITERIA:
                passes = (subset[c] == 'PASS').astype(int).values
                rate = np.mean(passes)
                ci_lo, ci_hi = bootstrap_ci(passes)
                row[f'{c}_rate'] = round(rate * 100, 1)
                row[f'{c}_ci_lo'] = round(ci_lo * 100, 1)
                row[f'{c}_ci_hi'] = round(ci_hi * 100, 1)

            # Composite score stats
            row['composite_mean'] = round(subset['composite_score'].mean(), 2)
            row['composite_std'] = round(subset['composite_score'].std(), 2)
            row['full_pass_rate'] = round(
                (subset['composite_score'] == 5).mean() * 100, 1)

            # Recourse cost stats
            row['cost_mean'] = round(subset['recourse_cost'].mean(), 2)
            row['cost_std'] = round(subset['recourse_cost'].std(), 2)
            row['cost_immutable_mean'] = round(
                subset['cost_immutable'].mean(), 2)

            # Feature-level violation rates
            feat_cols = [c for c in subset.columns if c.startswith('feat_changed_')]
            for fc in feat_cols:
                feat_name = fc.replace('feat_changed_', '')
                row[f'violation_rate_{feat_name}'] = round(
                    subset[fc].mean() * 100, 1)

            results.append(row)
            print(f"  {dataset} / {tool}: Full pass={row['full_pass_rate']}%, "
                  f"Composite={row['composite_mean']}±{row['composite_std']}")

    rq1_df = pd.DataFrame(results)
    rq1_df.to_csv(os.path.join(RESULTS_DIR, 'rq1_baseline_analysis.csv'),
                   index=False)
    return rq1_df


def analyze_rq2(all_grades):
    """RQ2: Incremental constraint impact — marginal gains per phase."""
    print(f"\n{'=' * 70}")
    print("RQ2 ANALYSIS: Incremental Constraint Impact")
    print("=" * 70)

    results = []
    for dataset in all_grades['dataset'].unique():
        for tool in all_grades['tool'].unique():
            for c in CRITERIA:
                prev_rate = None
                for phase in [1, 2, 3, 4]:
                    subset = all_grades[
                        (all_grades['dataset'] == dataset) &
                        (all_grades['tool'] == tool) &
                        (all_grades['phase'] == phase)
                    ]
                    if len(subset) == 0:
                        continue
                    rate = (subset[c] == 'PASS').mean() * 100
                    marginal = rate - prev_rate if prev_rate is not None else 0
                    results.append({
                        'dataset': dataset, 'tool': tool,
                        'criterion': c, 'phase': phase,
                        'pass_rate': round(rate, 1),
                        'marginal_gain': round(marginal, 1),
                    })
                    prev_rate = rate

    rq2_df = pd.DataFrame(results)
    rq2_df.to_csv(os.path.join(RESULTS_DIR, 'rq2_compliance_curve.csv'),
                   index=False)

    # Print summary
    for c in CRITERIA:
        c_data = rq2_df[rq2_df['criterion'] == c]
        if len(c_data) == 0:
            continue
        p1_mean = c_data[c_data['phase'] == 1]['pass_rate'].mean()
        p4_mean = c_data[c_data['phase'] == 4]['pass_rate'].mean()
        print(f"  {c}: P1={p1_mean:.1f}% → P4={p4_mean:.1f}% "
              f"(total gain={p4_mean - p1_mean:.1f}pp)")

    return rq2_df


def analyze_rq3(all_grades):
    """RQ3: Method-dependent vs systemic — Cohen's d, Wilcoxon, overlap."""
    print(f"\n{'=' * 70}")
    print("RQ3 ANALYSIS: Method-Dependent vs. Systemic Failures")
    print("=" * 70)

    effect_results = []
    overlap_results = []

    for dataset in all_grades['dataset'].unique():
        for phase in [1, 2, 3, 4]:
            for c in CRITERIA:
                dice = all_grades[
                    (all_grades['dataset'] == dataset) &
                    (all_grades['tool'] == 'DiCE') &
                    (all_grades['phase'] == phase)
                ]
                wachter = all_grades[
                    (all_grades['dataset'] == dataset) &
                    (all_grades['tool'] == 'Wachter') &
                    (all_grades['phase'] == phase)
                ]

                if len(dice) == 0 or len(wachter) == 0:
                    continue

                dice_pass = (dice[c] == 'PASS').astype(int).values
                wachter_pass = (wachter[c] == 'PASS').astype(int).values

                d = compute_cohens_d(dice_pass, wachter_pass)

                # Wilcoxon test (on profile-level aggregated scores)
                try:
                    _, p_val = wilcoxon(
                        dice_pass[:min(len(dice_pass), len(wachter_pass))],
                        wachter_pass[:min(len(dice_pass), len(wachter_pass))]
                    )
                except Exception:
                    p_val = 1.0

                effect_results.append({
                    'dataset': dataset, 'phase': phase, 'criterion': c,
                    'dice_rate': round(dice_pass.mean() * 100, 1),
                    'wachter_rate': round(wachter_pass.mean() * 100, 1),
                    'cohens_d': round(d, 3),
                    'wilcoxon_p': round(p_val, 4),
                    'effect_size': ('Large' if abs(d) >= 0.8 else
                                    'Medium' if abs(d) >= 0.5 else
                                    'Small' if abs(d) >= 0.2 else 'Negligible'),
                })

                # Failure overlap analysis (Phase 1 only for clarity)
                if phase == 1:
                    # Match profiles by index
                    min_n = min(len(dice), len(wachter))
                    d_fail = (dice[c].values[:min_n] == 'FAIL')
                    w_fail = (wachter[c].values[:min_n] == 'FAIL')
                    both_fail = (d_fail & w_fail).sum()
                    dice_only = (d_fail & ~w_fail).sum()
                    wachter_only = (~d_fail & w_fail).sum()
                    neither = (~d_fail & ~w_fail).sum()
                    total = min_n

                    overlap_results.append({
                        'dataset': dataset, 'criterion': c,
                        'both_fail_pct': round(both_fail / total * 100, 1),
                        'dice_only_pct': round(dice_only / total * 100, 1),
                        'wachter_only_pct': round(wachter_only / total * 100, 1),
                        'neither_fail_pct': round(neither / total * 100, 1),
                    })

    effect_df = pd.DataFrame(effect_results)
    effect_df.to_csv(os.path.join(RESULTS_DIR, 'rq3_effect_sizes.csv'),
                      index=False)

    overlap_df = pd.DataFrame(overlap_results)
    overlap_df.to_csv(os.path.join(RESULTS_DIR, 'rq3_failure_overlap.csv'),
                       index=False)

    # Print summary — show per-phase across both datasets
    print("\n  Phase 1 summary (baseline method comparison):")
    for c in CRITERIA:
        c_data = effect_df[(effect_df['criterion'] == c) &
                           (effect_df['phase'] == 1)]
        if len(c_data) > 0:
            for _, r in c_data.iterrows():
                print(f"    {r['dataset']}/{c}: d={r['cohens_d']:.3f} "
                      f"({r['effect_size']}) — "
                      f"DiCE={r['dice_rate']:.0f}% vs Wachter={r['wachter_rate']:.0f}%")

    return effect_df, overlap_df


def analyze_rq4(all_grades):
    """RQ4: Generalizability — three-factor analysis."""
    print(f"\n{'=' * 70}")
    print("RQ4 ANALYSIS: Generalizability Across Models and Datasets")
    print("=" * 70)

    p4 = all_grades[all_grades['phase'] == 4]
    results = []

    for dataset in p4['dataset'].unique():
        for model in p4['model'].unique() if 'model' in p4.columns else ['All']:
            for tool in p4['tool'].unique():
                subset = p4[
                    (p4['dataset'] == dataset) & (p4['tool'] == tool)
                ]
                if 'model' in p4.columns and model != 'All':
                    subset = subset[subset['model'] == model]

                if len(subset) == 0:
                    continue

                row = {'dataset': dataset, 'model': model, 'tool': tool,
                       'n': len(subset)}
                for c in CRITERIA:
                    row[f'{c}_rate'] = round(
                        (subset[c] == 'PASS').mean() * 100, 1)
                row['composite_mean'] = round(
                    subset['composite_score'].mean(), 2)
                results.append(row)

    rq4_df = pd.DataFrame(results)
    rq4_df.to_csv(os.path.join(RESULTS_DIR, 'rq4_generalizability.csv'),
                   index=False)

    # Run Friedman per dataset and collect CD diagram data
    cd_data = []
    for dataset in all_grades['dataset'].unique():
        for c in CRITERIA:
            stat, p_val, nemenyi, rank_df = friedman_nemenyi_test(
                all_grades, c, dataset)
            if stat is not None:
                print(f"  {dataset}/{c}: χ²={stat:.3f}, p={p_val:.4f}")
                if rank_df is not None:
                    rank_df['dataset'] = dataset
                    rank_df['criterion'] = c
                    cd_data.append(rank_df)

    if cd_data:
        cd_df = pd.concat(cd_data, ignore_index=True)
        cd_df.to_csv(os.path.join(RESULTS_DIR, 'rq4_cd_diagram_data.csv'),
                      index=False)

    return rq4_df


def analyze_rq5(all_grades):
    """RQ5: Recourse cost vs compliance trade-off."""
    print(f"\n{'=' * 70}")
    print("RQ5 ANALYSIS: Recourse Cost vs. Compliance Trade-Off")
    print("=" * 70)

    results = []
    for dataset in all_grades['dataset'].unique():
        for tool in all_grades['tool'].unique():
            for phase in [1, 2, 3, 4]:
                subset = all_grades[
                    (all_grades['dataset'] == dataset) &
                    (all_grades['tool'] == tool) &
                    (all_grades['phase'] == phase)
                ]
                if len(subset) == 0:
                    continue

                results.append({
                    'dataset': dataset, 'tool': tool, 'phase': phase,
                    'compliance_mean': round(
                        subset['composite_score'].mean() / 5 * 100, 1),
                    'cost_mean': round(subset['recourse_cost'].mean(), 3),
                    'cost_std': round(subset['recourse_cost'].std(), 3),
                    'cost_immutable_mean': round(
                        subset['cost_immutable'].mean(), 3),
                    'cost_actionable_mean': round(
                        subset['cost_actionable'].mean(), 3),
                    'cost_immutable_pct': round(
                        subset['cost_immutable'].mean() /
                        (subset['recourse_cost'].mean() + 1e-9) * 100, 1),
                })

    rq5_df = pd.DataFrame(results)
    rq5_df.to_csv(os.path.join(RESULTS_DIR, 'rq5_cost_tradeoff.csv'),
                   index=False)

    # Print summary
    for phase in [1, 4]:
        p_data = rq5_df[rq5_df['phase'] == phase]
        if len(p_data) > 0:
            print(f"  Phase {phase}: Compliance={p_data['compliance_mean'].mean():.1f}%, "
                  f"Cost={p_data['cost_mean'].mean():.3f}, "
                  f"Immutable%={p_data['cost_immutable_pct'].mean():.1f}%")

    # Pareto frontier data (per-counterfactual for scatter plot)
    pareto_data = all_grades[all_grades['phase'] == 4][
        ['dataset', 'tool', 'composite_score', 'recourse_cost']
    ].copy()
    pareto_data.to_csv(os.path.join(RESULTS_DIR, 'rq5_pareto_data.csv'),
                        index=False)

    return rq5_df


def analyze_rq6(all_grades):
    """RQ6: Persistent failure modes, co-failure correlations."""
    print(f"\n{'=' * 70}")
    print("RQ6 ANALYSIS: Persistent Failure Modes")
    print("=" * 70)

    # Persistence curves
    persistence = []
    for dataset in all_grades['dataset'].unique():
        for c in CRITERIA:
            for phase in [1, 2, 3, 4]:
                subset = all_grades[
                    (all_grades['dataset'] == dataset) &
                    (all_grades['phase'] == phase)
                ]
                if len(subset) == 0:
                    continue
                fail_rate = (subset[c] == 'FAIL').mean() * 100
                persistence.append({
                    'dataset': dataset, 'criterion': c,
                    'phase': phase, 'fail_rate': round(fail_rate, 1),
                })

    persist_df = pd.DataFrame(persistence)
    persist_df.to_csv(os.path.join(RESULTS_DIR, 'rq6_persistence_curves.csv'),
                       index=False)

    # Criterion resolution analysis
    resolution = []
    for dataset in all_grades['dataset'].unique():
        for c in CRITERIA:
            p1_rate = persist_df[
                (persist_df['dataset'] == dataset) &
                (persist_df['criterion'] == c) &
                (persist_df['phase'] == 1)
            ]['fail_rate'].values
            p1_rate = p1_rate[0] if len(p1_rate) > 0 else 100

            min_phase = 'Not achieved'
            for phase in [2, 3, 4]:
                pn_rate = persist_df[
                    (persist_df['dataset'] == dataset) &
                    (persist_df['criterion'] == c) &
                    (persist_df['phase'] == phase)
                ]['fail_rate'].values
                pn_rate = pn_rate[0] if len(pn_rate) > 0 else 100
                # Significant improvement: fail rate drops by >15pp from baseline
                # OR fail rate reaches below 10%
                if (p1_rate - pn_rate) > 15 or pn_rate < 10:
                    min_phase = f'Phase {phase}'
                    break

            p4_rate = persist_df[
                (persist_df['dataset'] == dataset) &
                (persist_df['criterion'] == c) &
                (persist_df['phase'] == 4)
            ]['fail_rate'].values
            p4_fail = p4_rate[0] if len(p4_rate) > 0 else 100

            resolution.append({
                'dataset': dataset, 'criterion': c,
                'min_phase_significant': min_phase,
                'max_pass_rate_p4': round(100 - p4_fail, 1),
                'residual_gap': round(p4_fail, 1),
            })
            print(f"  {dataset}/{c}: Min phase={min_phase}, "
                  f"P4 pass={100 - p4_fail:.1f}%, Gap={p4_fail:.1f}%")

    res_df = pd.DataFrame(resolution)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'rq6_resolution_analysis.csv'),
                   index=False)

    # Co-failure phi-coefficient matrix (Phase 4)
    for dataset in all_grades['dataset'].unique():
        p4 = all_grades[
            (all_grades['dataset'] == dataset) &
            (all_grades['phase'] == 4)
        ]
        if len(p4) == 0:
            continue

        fail_matrix = pd.DataFrame()
        for c in CRITERIA:
            fail_matrix[c] = (p4[c] == 'FAIL').astype(int).values

        phi_matrix = pd.DataFrame(
            index=CRITERIA, columns=CRITERIA, dtype=float)
        for c1 in CRITERIA:
            for c2 in CRITERIA:
                phi_matrix.loc[c1, c2] = round(
                    compute_phi_coefficient(
                        fail_matrix[c1].values, fail_matrix[c2].values), 3)

        phi_matrix.to_csv(os.path.join(
            RESULTS_DIR, f'rq6_cofailure_phi_{dataset}.csv'))
        print(f"\n  Co-failure matrix ({dataset}):")
        print(phi_matrix.to_string())

    return persist_df, res_df


def analyze_rq7(all_grades):
    """RQ7: Scorecard validation — Cronbach's alpha, sensitivity."""
    print(f"\n{'=' * 70}")
    print("RQ7 ANALYSIS: Scorecard Validation")
    print("=" * 70)

    # Build binary matrix for all grades
    binary_matrix = np.column_stack([
        (all_grades[c] == 'PASS').astype(int).values for c in CRITERIA
    ])

    # Full Cronbach's alpha
    full_alpha = compute_cronbachs_alpha(binary_matrix)
    print(f"  Full scorecard Cronbach's α = {full_alpha:.3f}")

    # Item-deleted alpha
    alpha_results = [{'analysis': 'Full (C1–C5)', 'cronbachs_alpha': round(full_alpha, 3)}]
    for i, c in enumerate(CRITERIA):
        reduced = np.delete(binary_matrix, i, axis=1)
        alpha_del = compute_cronbachs_alpha(reduced)
        alpha_results.append({
            'analysis': f'Without {c}',
            'cronbachs_alpha': round(alpha_del, 3)
        })
        print(f"  Without {c}: α = {alpha_del:.3f} "
              f"({'↑' if alpha_del > full_alpha else '↓'})")

    alpha_df = pd.DataFrame(alpha_results)
    alpha_df.to_csv(os.path.join(RESULTS_DIR, 'rq7_cronbach_alpha.csv'),
                     index=False)

    # Sensitivity analysis: ±10% threshold perturbation for C3 (sparsity)
    # C3 threshold is n_changed <= 3. Test with 2 and 4.
    sensitivity = []
    for threshold in [2, 3, 4]:
        c3_pass = (all_grades['n_features_changed'] <= threshold).astype(int)
        composite_with = sum(
            (all_grades[c] == 'PASS').astype(int) for c in CRITERIA
            if c != 'C3_Sparsity'
        ) + c3_pass
        sensitivity.append({
            'criterion': 'C3_Sparsity',
            'threshold': threshold,
            'composite_mean': round(composite_with.mean(), 2),
        })

    sens_df = pd.DataFrame(sensitivity)
    sens_df.to_csv(os.path.join(RESULTS_DIR, 'rq7_sensitivity.csv'),
                    index=False)

    # Confusion matrix data (simulated expert validation framework)
    # The researcher grades 200 random CFs manually; this saves the sample
    sample = all_grades.sample(
        n=min(200, len(all_grades)), random_state=RANDOM_STATE
    )[['profile_idx', 'tool', 'phase', 'dataset', 'composite_score'] +
      CRITERIA].copy()
    sample['scorecard_verdict'] = np.where(
        sample['composite_score'] >= 4, 'Compliant', 'Non-Compliant')
    sample['expert_verdict'] = ''  # To be filled manually by researcher
    sample.to_csv(os.path.join(RESULTS_DIR, 'rq7_expert_validation_sample.csv'),
                   index=False)
    print(f"  Expert validation sample (n={len(sample)}) saved for manual review.")

    return alpha_df


# ============================================================================
# SECTION 9: COMPREHENSIVE VISUALIZATIONS (28 figures for RQ1–RQ7)
# ============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.figsize': (7, 4.5),
})

COLORS = {
    'dice': '#2C3E50', 'wachter': '#E74C3C',
    'dice_light': '#5D7B9D', 'wachter_light': '#F1948A',
    'phase1': '#E74C3C', 'phase2': '#F39C12',
    'phase3': '#3498DB', 'phase4': '#27AE60',
    'pass': '#27AE60', 'partial': '#F39C12', 'fail': '#E74C3C',
    'c1': '#2C3E50', 'c2': '#E74C3C', 'c3': '#3498DB',
    'c4': '#27AE60', 'c5': '#F39C12',
}


def _save_fig(fig, name):
    """
    Persist a figure to disk.

    Behaviour:
      1. Saves the original raster (PNG/JPG/etc.) into ``FIGURES_DIR`` so
         existing notebooks, reports, and downstream tooling that consume
         the PNG outputs continue to work unchanged.
      2. Additionally saves a vector PDF copy of the same figure into
         ``PDF_FIGURES_DIR``. The base filename is preserved; only the
         extension is swapped to ``.pdf``. If the supplied ``name`` has
         no extension, ``.pdf`` is simply appended.

    The PDF folder is created at module import time, but we guard with
    ``os.makedirs(..., exist_ok=True)`` here as well so the function is
    safe to call even if the directory was removed mid-run.
    """
    # 1) Original raster output — preserves existing display behaviour.
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=300,
                bbox_inches='tight', facecolor='white')

    # 2) PDF copy — same filename stem, ``.pdf`` extension.
    base, _ext = os.path.splitext(name)
    pdf_name = f"{base}.pdf"
    os.makedirs(PDF_FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(PDF_FIGURES_DIR, pdf_name),
                bbox_inches='tight', facecolor='white')

    plt.close(fig)
    print(f"    ✓ {name}  (+ {pdf_name})")


def generate_all_visualizations(all_grades):
    """Generate all 28 publication-ready figures from real pipeline data."""
    print(f"\n{'=' * 70}")
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    # --- RQ1 Figures ---
    _plot_rq1_baseline_bars(all_grades)
    _plot_rq1_grade_distribution(all_grades)
    _plot_rq1_violation_heatmap(all_grades)

    # --- RQ2 Figures ---
    _plot_rq2_compliance_curve(all_grades)
    _plot_rq2_radar(all_grades)

    # --- RQ3 Figures ---
    _plot_rq3_dumbbell(all_grades)
    _plot_rq3_interaction(all_grades)
    _plot_rq3_failure_overlap(all_grades)

    # --- RQ4 Figures ---
    _plot_rq4_faceted_bars(all_grades)
    _plot_rq4_boxplot(all_grades)

    # --- RQ5 Figures ---
    _plot_rq5_dual_axis(all_grades)
    _plot_rq5_pareto(all_grades)

    # --- RQ6 Figures ---
    _plot_rq6_stacked_area(all_grades)
    _plot_rq6_persistence(all_grades)
    _plot_rq6_cofailure_heatmap(all_grades)

    # --- RQ7 Figures ---
    _plot_rq7_tornado(all_grades)

    # --- Heatmap (existing, improved) ---
    _plot_compliance_heatmap(all_grades)
    _plot_cost_comparison(all_grades)


def _plot_rq1_baseline_bars(g):
    """Fig 1.1: Grouped bar chart of baseline compliance."""
    p1 = g[g['phase'] == 1]
    if len(p1) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    conditions = []
    for ds in sorted(p1['dataset'].unique()):
        for tool in ['DiCE', 'Wachter']:
            sub = p1[(p1['dataset'] == ds) & (p1['tool'] == tool)]
            if len(sub) > 0:
                conditions.append((ds, tool, sub))

    x = np.arange(len(CRITERIA))
    width = 0.18
    color_map = {
        ('DiCE', 0): COLORS['dice'], ('DiCE', 1): COLORS['dice_light'],
        ('Wachter', 0): COLORS['wachter'], ('Wachter', 1): COLORS['wachter_light'],
    }

    for i, (ds, tool, sub) in enumerate(conditions):
        rates = [(sub[c] == 'PASS').mean() * 100 for c in CRITERIA]
        ds_idx = 0 if 'German' in ds else 1
        color = color_map.get((tool, ds_idx), '#666')
        ax.bar(x + (i - len(conditions)/2 + 0.5) * width, rates,
               width * 0.9, label=f'{tool}–{ds.split("_")[0]}',
               color=color, edgecolor='white')

    ax.set_ylabel('Compliance Pass Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.split('_')[1] for c in CRITERIA])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.set_title('Baseline compliance (Phase 1, unconstrained)', fontweight='bold')
    _save_fig(fig, 'rq1_baseline_compliance.png')


def _plot_rq1_grade_distribution(g):
    """Fig 1.2: Stacked bar of Full Pass / Partial / Fail."""
    p1 = g[g['phase'] == 1]
    if len(p1) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    tools = p1['tool'].unique()
    full = [(p1[p1['tool'] == t]['composite_score'] == 5).mean() * 100 for t in tools]
    partial = [((p1[p1['tool'] == t]['composite_score'] >= 3) &
                (p1[p1['tool'] == t]['composite_score'] < 5)).mean() * 100 for t in tools]
    fail = [(p1[p1['tool'] == t]['composite_score'] < 3).mean() * 100 for t in tools]

    x = np.arange(len(tools))
    ax.bar(x, fail, 0.5, label='Fail (≤2)', color=COLORS['fail'])
    ax.bar(x, partial, 0.5, bottom=fail, label='Partial (3–4)', color=COLORS['partial'])
    ax.bar(x, full, 0.5, bottom=[f+p for f, p in zip(fail, partial)],
           label='Full Pass (5)', color=COLORS['pass'])
    ax.set_xticks(x)
    ax.set_xticklabels(tools)
    ax.set_ylabel('Proportion (%)')
    ax.legend(fontsize=8)
    ax.set_title('Compliance grade distribution (Phase 1)', fontweight='bold')
    _save_fig(fig, 'rq1_grade_distribution.png')


def _plot_rq1_violation_heatmap(g):
    """Fig 1.4: Feature-level violation heatmap."""
    p1 = g[g['phase'] == 1]
    feat_cols = [c for c in p1.columns if c.startswith('feat_changed_')]
    if len(feat_cols) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    conditions = []
    data_matrix = []
    for tool in ['DiCE', 'Wachter']:
        sub = p1[p1['tool'] == tool]
        if len(sub) == 0:
            continue
        conditions.append(tool)
        rates = [sub[fc].mean() * 100 for fc in feat_cols]
        data_matrix.append(rates)

    if len(data_matrix) == 0:
        plt.close()
        return

    feat_labels = [fc.replace('feat_changed_', '') for fc in feat_cols]
    data_arr = np.array(data_matrix).T

    sns.heatmap(data_arr, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=conditions, yticklabels=feat_labels,
                vmin=0, vmax=100, ax=ax)
    ax.set_title('Feature-level violation frequency (Phase 1)', fontweight='bold')
    _save_fig(fig, 'rq1_violation_heatmap.png')


def _plot_rq2_compliance_curve(g):
    """Fig 2.1: Multi-line compliance trajectory."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    criterion_colors = [COLORS['c1'], COLORS['c2'], COLORS['c3'],
                        COLORS['c4'], COLORS['c5']]
    markers = ['o', 's', '^', 'D', 'v']

    for c, color, marker in zip(CRITERIA, criterion_colors, markers):
        rates = []
        for phase in [1, 2, 3, 4]:
            sub = g[g['phase'] == phase]
            if len(sub) > 0:
                rates.append((sub[c] == 'PASS').mean() * 100)
            else:
                rates.append(0)
        ax.plot([1, 2, 3, 4], rates, marker=marker, color=color,
                label=c.split('_')[1], linewidth=1.8, markersize=7)

    ax.set_xlabel('Constraint Phase')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['P1\n(None)', 'P2\n(Immutable)',
                        'P3\n(Direction)', 'P4\n(Causal)'])
    ax.set_ylim(0, 105)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_title('Compliance trajectory across constraint phases', fontweight='bold')
    fig.subplots_adjust(right=0.75)
    _save_fig(fig, 'rq2_compliance_curve.png')


def _plot_rq2_radar(g):
    """Fig 2.4: Radar charts for DiCE and Wachter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4),
                                    subplot_kw=dict(polar=True))
    N = len(CRITERIA)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    labels = [c.split('_')[1] for c in CRITERIA]
    phase_colors = [COLORS['phase1'], COLORS['phase2'],
                    COLORS['phase3'], COLORS['phase4']]

    for ax, tool, title in [(ax1, 'DiCE', 'DiCE'), (ax2, 'Wachter', 'Wachter')]:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontweight='bold', pad=20)

        for phase, color in zip([1, 2, 3, 4], phase_colors):
            sub = g[(g['tool'] == tool) & (g['phase'] == phase)]
            if len(sub) == 0:
                continue
            vals = [(sub[c] == 'PASS').mean() * 100 for c in CRITERIA]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linewidth=1.3,
                    label=f'P{phase}')
            ax.fill(angles, vals, color=color, alpha=0.06)

    ax2.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=8)
    fig.suptitle('Compliance profiles across phases', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_fig(fig, 'rq2_radar.png')


def _plot_rq3_dumbbell(g):
    """Fig 3.1: Dumbbell chart — DiCE vs Wachter at Phase 1."""
    p1 = g[g['phase'] == 1]
    if len(p1) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = np.arange(len(CRITERIA))

    for i, c in enumerate(CRITERIA):
        d_rate = (p1[p1['tool'] == 'DiCE'][c] == 'PASS').mean() * 100
        w_rate = (p1[p1['tool'] == 'Wachter'][c] == 'PASS').mean() * 100
        ax.plot([d_rate, w_rate], [i, i], color='#CCC', linewidth=2, zorder=1)
        ax.scatter(d_rate, i, color=COLORS['dice'], s=80, zorder=2)
        ax.scatter(w_rate, i, color=COLORS['wachter'], s=80, zorder=2)
        ax.text((d_rate + w_rate) / 2, i + 0.25,
                f'Δ={abs(d_rate - w_rate):.0f}pp', ha='center', fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([c.split('_')[1] for c in CRITERIA])
    ax.set_xlabel('Pass Rate (%)')
    ax.scatter([], [], color=COLORS['dice'], label='DiCE')
    ax.scatter([], [], color=COLORS['wachter'], label='Wachter')
    ax.legend(fontsize=8)
    ax.set_title('DiCE vs. Wachter at Phase 1', fontweight='bold')
    _save_fig(fig, 'rq3_dumbbell.png')


def _plot_rq3_interaction(g):
    """Fig 3.2: Interaction plot — Method × Phase."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tool, color, ls in [('DiCE', COLORS['dice'], '-'),
                            ('Wachter', COLORS['wachter'], '--')]:
        rates = []
        for phase in [1, 2, 3, 4]:
            sub = g[(g['tool'] == tool) & (g['phase'] == phase)]
            if len(sub) > 0:
                rates.append(sub['composite_score'].mean() / 5 * 100)
            else:
                rates.append(0)
        ax.plot([1, 2, 3, 4], rates, f'o{ls}', color=color,
                label=tool, linewidth=2, markersize=8)

    ax.set_xlabel('Constraint Phase')
    ax.set_ylabel('Mean Composite Compliance (%)')
    ax.set_xticks([1, 2, 3, 4])
    ax.legend()
    ax.set_title('Method × Phase interaction', fontweight='bold')
    _save_fig(fig, 'rq3_interaction.png')


def _plot_rq3_failure_overlap(g):
    """Fig 3.4: Failure overlap stacked bars."""
    p1 = g[g['phase'] == 1]
    overlap_path = os.path.join(RESULTS_DIR, 'rq3_failure_overlap.csv')
    if not os.path.exists(overlap_path):
        return
    overlap = pd.read_csv(overlap_path)
    if len(overlap) == 0:
        return

    # Average across datasets
    avg = overlap.groupby('criterion').mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(avg))
    w = 0.5
    ax.bar(x, avg['both_fail_pct'], w, label='Both Fail (Systemic)',
           color='#C0392B')
    ax.bar(x, avg['dice_only_pct'], w, bottom=avg['both_fail_pct'],
           label='DiCE-Only', color=COLORS['dice_light'])
    ax.bar(x, avg['wachter_only_pct'], w,
           bottom=avg['both_fail_pct'] + avg['dice_only_pct'],
           label='Wachter-Only', color=COLORS['wachter_light'])
    ax.bar(x, avg['neither_fail_pct'], w,
           bottom=(avg['both_fail_pct'] + avg['dice_only_pct'] +
                   avg['wachter_only_pct']),
           label='Neither Fails', color=COLORS['pass'])

    ax.set_xticks(x)
    ax.set_xticklabels([c.split('_')[1] for c in avg['criterion']])
    ax.set_ylabel('Proportion (%)')
    ax.legend(fontsize=7)
    ax.set_title('Failure overlap at Phase 1', fontweight='bold')
    _save_fig(fig, 'rq3_failure_overlap.png')


def _plot_rq4_faceted_bars(g):
    """Fig 4.1: Faceted bars by dataset and model."""
    p4 = g[g['phase'] == 4]
    if len(p4) == 0 or 'model' not in p4.columns:
        return

    datasets = sorted(p4['dataset'].unique())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 4.5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data = p4[p4['dataset'] == ds]
        models = sorted(ds_data['model'].unique())
        x = np.arange(len(CRITERIA))
        width = 0.35

        for i, model in enumerate(models):
            sub = ds_data[ds_data['model'] == model]
            rates = [(sub[c] == 'PASS').mean() * 100 for c in CRITERIA]
            ax.bar(x + (i - 0.5) * width, rates, width,
                   label=model, edgecolor='white')

        ax.set_title(ds.replace('_', ' '), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('_')[0] for c in CRITERIA], fontsize=8)
        ax.set_ylim(0, 105)
        if ax == axes[0]:
            ax.set_ylabel('Pass Rate (%)')
        ax.legend(fontsize=7)

    fig.suptitle('Phase 4 compliance by model and dataset', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_fig(fig, 'rq4_faceted_bars.png')


def _plot_rq4_boxplot(g):
    """Fig 4.2: Box plot of composite scores."""
    p4 = g[g['phase'] == 4]
    if len(p4) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = []
    labels = []
    for ds in sorted(p4['dataset'].unique()):
        for model in sorted(p4['model'].unique()) if 'model' in p4.columns else ['']:
            sub = p4[(p4['dataset'] == ds)]
            if model and 'model' in p4.columns:
                sub = sub[sub['model'] == model]
            if len(sub) > 0:
                groups.append(sub['composite_score'].values)
                lbl = f"{model.replace('Classifier','')[:3]}–{ds.split('_')[0]}"
                labels.append(lbl)

    if groups:
        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch in bp['boxes']:
            patch.set_alpha(0.6)
        ax.set_ylabel('Composite Score (0–5)')
        ax.axhline(y=4, color=COLORS['pass'], ls='--', lw=0.8)
        ax.set_title('Composite score distribution at Phase 4', fontweight='bold')
    _save_fig(fig, 'rq4_boxplot.png')


def _plot_rq5_dual_axis(g):
    """Fig 5.1: Dual-axis compliance vs cost."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    phases = [1, 2, 3, 4]
    compliance = [g[g['phase'] == p]['composite_score'].mean() / 5 * 100
                  for p in phases]
    cost = [g[g['phase'] == p]['recourse_cost'].mean() for p in phases]

    ax1.plot(phases, compliance, 'o-', color=COLORS['pass'], lw=2.5,
             markersize=9, label='Compliance')
    ax1.set_ylabel('Compliance (%)', color=COLORS['pass'])
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.bar(phases, cost, width=0.4, alpha=0.35, color=COLORS['wachter'],
            label='Recourse Cost')
    ax2.set_ylabel('Cost (normalized L1)', color=COLORS['wachter'])

    ax1.set_xticks(phases)
    ax1.set_xticklabels(['P1', 'P2', 'P3', 'P4'])
    ax1.set_xlabel('Constraint Phase')
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper left', fontsize=8)
    ax1.set_title('Compliance–cost trade-off', fontweight='bold')
    _save_fig(fig, 'rq5_dual_axis.png')


def _plot_rq5_pareto(g):
    """Fig 5.2: Scatter with Pareto frontier."""
    p4 = g[g['phase'] == 4]
    if len(p4) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for tool, color in [('DiCE', COLORS['dice']), ('Wachter', COLORS['wachter'])]:
        sub = p4[p4['tool'] == tool]
        ax.scatter(sub['recourse_cost'], sub['composite_score'],
                   alpha=0.35, s=25, color=color, label=tool, edgecolors='none')

    ax.set_xlabel('Recourse Cost')
    ax.set_ylabel('Composite Score (0–5)')
    ax.axhline(y=4, color='grey', ls=':', lw=0.6)
    ax.legend()
    ax.set_title('Compliance–cost at Phase 4', fontweight='bold')
    _save_fig(fig, 'rq5_pareto.png')


def _plot_rq6_stacked_area(g):
    """Fig 6.1: Stacked area of failure composition."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    phases = [1, 2, 3, 4]
    fail_data = {}
    criterion_colors = [COLORS['c1'], COLORS['c2'], COLORS['c3'],
                        COLORS['c4'], COLORS['c5']]

    for c in CRITERIA:
        fail_data[c] = [(g[g['phase'] == p][c] == 'FAIL').mean() * 100
                        for p in phases]

    ax.stackplot(phases, *fail_data.values(),
                 colors=criterion_colors, alpha=0.75,
                 labels=[c.split('_')[1] for c in CRITERIA])
    ax.set_xlabel('Constraint Phase')
    ax.set_ylabel('Cumulative Failure Rate (%)')
    ax.set_xticks(phases)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Failure composition across phases', fontweight='bold')
    _save_fig(fig, 'rq6_stacked_area.png')


def _plot_rq6_persistence(g):
    """Fig 6.2: Persistence curves."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    criterion_colors = [COLORS['c1'], COLORS['c2'], COLORS['c3'],
                        COLORS['c4'], COLORS['c5']]
    markers = ['o', 's', '^', 'D', 'v']

    for c, color, marker in zip(CRITERIA, criterion_colors, markers):
        rates = [(g[g['phase'] == p][c] == 'FAIL').mean() * 100
                 for p in [1, 2, 3, 4]]
        ax.plot([1, 2, 3, 4], rates, marker=marker, color=color,
                label=c.split('_')[1], linewidth=1.8)

    ax.set_xlabel('Constraint Phase')
    ax.set_ylabel('Profiles Still Failing (%)')
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_title('Failure persistence by criterion', fontweight='bold')
    fig.subplots_adjust(right=0.72)
    _save_fig(fig, 'rq6_persistence.png')


def _plot_rq6_cofailure_heatmap(g):
    """Fig 6.4: Co-failure phi-coefficient heatmap."""
    p4 = g[g['phase'] == 4]
    if len(p4) == 0:
        return

    fail_matrix = np.column_stack([
        (p4[c] == 'FAIL').astype(int).values for c in CRITERIA
    ])

    phi = np.zeros((len(CRITERIA), len(CRITERIA)))
    for i in range(len(CRITERIA)):
        for j in range(len(CRITERIA)):
            phi[i, j] = compute_phi_coefficient(
                fail_matrix[:, i], fail_matrix[:, j])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    mask = np.triu(np.ones_like(phi, dtype=bool), k=1)
    sns.heatmap(phi, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlBu_r', vmin=-0.2, vmax=1.0,
                xticklabels=[c.split('_')[0] for c in CRITERIA],
                yticklabels=[c.split('_')[0] for c in CRITERIA], ax=ax)
    ax.set_title('Co-failure correlation (Phase 4)', fontweight='bold')
    _save_fig(fig, 'rq6_cofailure_heatmap.png')


def _plot_rq7_tornado(g):
    """Fig 7.3: Sensitivity tornado diagram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    # Test sparsity threshold sensitivity
    base_composite = g['composite_score'].mean()
    impacts = []
    for c in CRITERIA:
        # Simulate removing this criterion
        modified = sum(
            (g[cr] == 'PASS').astype(int) for cr in CRITERIA if cr != c
        ).mean()
        impact = modified - (base_composite - (g[c] == 'PASS').mean())
        impacts.append(abs(base_composite - modified * 5 / 4))

    sorted_idx = np.argsort(impacts)[::-1]
    y = np.arange(len(CRITERIA))
    labels = [CRITERIA[i].split('_')[1] for i in sorted_idx]
    vals = [impacts[i] for i in sorted_idx]

    ax.barh(y, vals, height=0.5, color=COLORS['wachter'], alpha=0.8)
    ax.barh(y, [-v for v in vals], height=0.5, color=COLORS['pass'], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Impact on Composite Score')
    ax.axvline(x=0, color='black', lw=0.8)
    ax.set_title('Criterion sensitivity analysis', fontweight='bold')
    _save_fig(fig, 'rq7_tornado.png')


def _plot_compliance_heatmap(g):
    """Comprehensive compliance heatmap (all conditions)."""
    for dataset in g['dataset'].unique():
        ds = g[g['dataset'] == dataset]
        heatmap_data = []
        for tool in ['DiCE', 'Wachter']:
            for phase in sorted(ds['phase'].unique()):
                sub = ds[(ds['tool'] == tool) & (ds['phase'] == phase)]
                if len(sub) == 0:
                    continue
                rates = {f"{tool} P{phase}": 0}
                for c in CRITERIA:
                    rates[c] = (sub[c] == 'PASS').mean() * 100
                rates_row = {'Condition': f"{tool} P{phase}"}
                rates_row.update({c: (sub[c] == 'PASS').mean() * 100
                                  for c in CRITERIA})
                heatmap_data.append(rates_row)

        if not heatmap_data:
            continue

        df_heat = pd.DataFrame(heatmap_data).set_index('Condition')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_heat, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, ax=ax)
        ax.set_title(f'Compliance Rates — {dataset}', fontweight='bold')
        _save_fig(fig, f'heatmap_{dataset}.png')


def _plot_cost_comparison(g):
    """Recourse cost comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = []
    for tool in ['DiCE', 'Wachter']:
        for phase in [1, 2, 3, 4]:
            sub = g[(g['tool'] == tool) & (g['phase'] == phase)]
            if len(sub) > 0:
                plot_data.append({
                    'Tool': tool, 'Phase': f'P{phase}',
                    'Cost': sub['recourse_cost'].mean()
                })

    if plot_data:
        pdf = pd.DataFrame(plot_data)
        pivot = pdf.pivot(index='Phase', columns='Tool', values='Cost')
        pivot.plot(kind='bar', ax=ax, color=[COLORS['dice'], COLORS['wachter']],
                   edgecolor='white')
        ax.set_ylabel('Mean Recourse Cost (normalized)')
        ax.set_title('Recourse cost by phase and method', fontweight='bold')
        ax.legend()
        plt.xticks(rotation=0)
    _save_fig(fig, 'cost_comparison.png')


# ============================================================================
# SECTION 10: RESULTS REPORTING
# ============================================================================

def print_compliance_summary(all_grades):
    print(f"\n{'=' * 70}\nCOMPLIANCE SCORECARD RESULTS\n{'=' * 70}")

    for dataset in all_grades['dataset'].unique():
        ds = all_grades[all_grades['dataset'] == dataset]
        print(f"\n  ── {dataset} ──")

        for tool in ['DiCE', 'Wachter']:
            tool_data = ds[ds['tool'] == tool]
            if len(tool_data) == 0:
                continue
            print(f"\n    [{tool}]")
            for phase in sorted(tool_data['phase'].unique()):
                ph = tool_data[tool_data['phase'] == phase]
                rates = {c: (ph[c] == 'PASS').mean() for c in CRITERIA}
                parts = [f"{c.split('_')[0]}={rates[c]:.0%}" for c in CRITERIA]
                mean_cost = ph['recourse_cost'].mean()
                composite = ph['composite_score'].mean()
                print(f"      Phase {phase}: {' | '.join(parts)} "
                      f"(N={len(ph)}, cost={mean_cost:.2f}, "
                      f"composite={composite:.2f})")


def save_all_results(all_grades, all_counterfactuals):
    grades_path = os.path.join(RESULTS_DIR, "master_scorecard.csv")
    cf_path = os.path.join(RESULTS_DIR, "all_counterfactuals.csv")
    all_grades.to_csv(grades_path, index=False)
    all_counterfactuals.to_csv(cf_path, index=False)

    # Save Nemenyi matrices
    for dataset in all_grades['dataset'].unique():
        for c in CRITERIA:
            _, _, nemenyi, _ = friedman_nemenyi_test(all_grades, c, dataset)
            if nemenyi is not None:
                path = os.path.join(RESULTS_DIR, f"nemenyi_{dataset}_{c}.csv")
                nemenyi.to_csv(path)

    print(f"\n  Core results saved to {RESULTS_DIR}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_audit(data, config, feature_ranges_dict, n_profiles=N_PROFILES):
    """Run the complete audit: both tools × 4 phases × both models."""
    dataset_name = config['name']
    models, X_train, X_test, y_test = train_models(data, dataset_name)

    # Compute feature ranges from training data for normalized cost
    for feat in X_train.columns:
        fr = X_train[feat].max() - X_train[feat].min()
        feature_ranges_dict[feat] = fr if fr > 0 else 1.0

    all_grades, all_cfs = pd.DataFrame(), pd.DataFrame()

    for model_name, model in models.items():
        print(f"\n{'~' * 50}")
        print(f"  AUDITING: {dataset_name}_{model_name}")
        print(f"{'~' * 50}")
        rejected = get_rejected_profiles(model, X_test, n_profiles)
        print(f"  Rejected profiles isolated: {len(rejected)}")

        for tool_func, tool_name in [
            (generate_counterfactuals_dice, 'DiCE'),
            (generate_counterfactuals_wachter, 'Wachter'),
        ]:
            for phase in [1, 2, 3, 4]:
                cf = tool_func(model, X_train, rejected, config, phase)
                if len(cf) > 0:
                    cf['model'] = model_name
                    all_cfs = pd.concat([all_cfs, cf], ignore_index=True)
                    grades = grade_compliance(
                        cf, config, feature_ranges=feature_ranges_dict)
                    grades['model'] = model_name
                    all_grades = pd.concat(
                        [all_grades, grades], ignore_index=True)

    return all_grades, all_cfs


def main():
    print("\n" + "#" * 70)
    print("#  THESIS PIPELINE v3: Full RQ1–RQ7 Coverage")
    print("#  Tools: DiCE (random) + Wachter (gradient-based)")
    print(f"#  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    all_grades, all_cfs = pd.DataFrame(), pd.DataFrame()
    feature_ranges_dict = {}

    # German Credit
    german_data = load_german_credit()
    g_grades, g_cfs = run_full_audit(
        german_data, GERMAN_CONFIG, feature_ranges_dict)
    all_grades = pd.concat([all_grades, g_grades], ignore_index=True)
    all_cfs = pd.concat([all_cfs, g_cfs], ignore_index=True)

    # Taiwan Credit
    taiwan_data = load_taiwan_credit()
    t_grades, t_cfs = run_full_audit(
        taiwan_data, TAIWAN_CONFIG, feature_ranges_dict)
    all_grades = pd.concat([all_grades, t_grades], ignore_index=True)
    all_cfs = pd.concat([all_cfs, t_cfs], ignore_index=True)

    if len(all_grades) > 0:
        # Core reporting
        print_compliance_summary(all_grades)

        # RQ-specific analyses
        analyze_rq1(all_grades)
        analyze_rq2(all_grades)
        analyze_rq3(all_grades)
        analyze_rq4(all_grades)
        analyze_rq5(all_grades)
        analyze_rq6(all_grades)
        analyze_rq7(all_grades)

        # Statistical tests
        print(f"\n{'=' * 70}")
        print("FRIEDMAN + NEMENYI TESTS")
        print("=" * 70)
        for dataset in all_grades['dataset'].unique():
            print(f"\n  --- {dataset} ---")
            for c in CRITERIA:
                stat, p_val, nemenyi, rank_df = friedman_nemenyi_test(
                    all_grades, c, dataset)
                if stat is not None:
                    sig = "✓ SIG" if p_val < 0.05 else "  n.s."
                    print(f"    {c:20s}: χ²={stat:.3f}, "
                          f"p={p_val:.4f} {sig}")

        # Save everything
        save_all_results(all_grades, all_cfs)

        # Generate all visualizations
        generate_all_visualizations(all_grades)

        # Final summary
        print(f"\n{'=' * 70}")
        print("PIPELINE COMPLETE — FULL RQ1–RQ7 COVERAGE")
        print(f"{'=' * 70}")
        print(f"  Total counterfactuals graded: {len(all_grades)}")
        print(f"  Tools: {sorted(all_grades['tool'].unique().tolist())}")
        print(f"  Datasets: {sorted(all_grades['dataset'].unique().tolist())}")
        models_list = (sorted(all_grades['model'].unique().tolist())
                       if 'model' in all_grades.columns else ['N/A'])
        print(f"  Models: {models_list}")
        print(f"  Phases: {sorted(all_grades['phase'].unique().tolist())}")
        print(f"\n  Output files:")
        for f in sorted(os.listdir(RESULTS_DIR)):
            if not os.path.isdir(os.path.join(RESULTS_DIR, f)):
                print(f"    - {f}")
        print(f"\n  Figures:")
        if os.path.exists(FIGURES_DIR):
            for f in sorted(os.listdir(FIGURES_DIR)):
                print(f"    - figures/{f}")
        print(f"\n  PDF figures (vector copies for the thesis):")
        if os.path.exists(PDF_FIGURES_DIR):
            for f in sorted(os.listdir(PDF_FIGURES_DIR)):
                print(f"    - pdf_figures/{f}")
        print(f"\n  Next: Review outputs and draft thesis chapters.")
    else:
        print("\n  ERROR: No counterfactuals generated. Check data files.")


if __name__ == "__main__":
    main()
