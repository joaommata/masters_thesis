#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ======================
# ----- CONFIG ----------
# ======================
RESULTS_DIR = '/zhome/d0/a/221493/thesis/results'
N_FOLDS = 5
CONFIGS = ['B1','B2','B3','B4','B5','M1','M2','M3','M4','M5','M6']

KEY_ORGAN_STRUCTURES = ['Left Lung', 'Right Lung', 'Heart', 'Mediastinum',
                       'Facies Diaphragmatica', 'Aorta',
                       'Left Hilus Pulmonis', 'Right Hilus Pulmonis']
OTHER_ORGAN_STRUCTURES = ['Left Scapula', 'Right Scapula',
                         'Left Clavicle', 'Right Clavicle', 'Weasand', 'Spine']
KEY_ORGAN_RATIOS = ['cardiothoracic_ratio', 'lung_area_ratio', 'lung_height_ratio',
                   'lung_width_ratio', 'left_lung_fraction', 'right_lung_fraction']
DEMO_COLS = ['age_pred', 'sex_male', 'sex_female', 'race_white', 'race_black', 'race_asian']
ORGANS = KEY_ORGAN_STRUCTURES + OTHER_ORGAN_STRUCTURES
NORMALIZED_ORGANS = [o.lower().replace(' ', '_') for o in ORGANS]

CATEGORY_COLORS = {
    'C0 Confidence': '#FF9999',
    'CF Probability': '#FFCC99',
    'Key-Organ Anatomy': '#88BBDD',
    'Key-Organ Ratios': '#88BBDD',
    'Demographics': "#63BD978C",
    'Other-Organ Anatomy': "#88BBDD7E",
    'Key-Organ Anatomy (CF delta)': '#88BBDD',
    'Key-Organ Ratios (CF delta)': '#88BBDD',
    'Demographics (CF delta)': '#63BD978C',
    'Other-Organ Anatomy (CF delta)': "#88BBDD7E",
    'Margin/Uncategorized': '#CCCCCC'
}

# ======================
# ----- FEATURE BUILDER -
# ======================
def build_feature_matrix(pred_df, disease, config):
    """Build feature matrix X and feature names for a given config."""
    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
    diff_cols = [c for c in pred_df.columns if c.startswith('delta_')]
    emb_cols  = [c for c in pred_df.columns if c.startswith('emb_')]
    attr_cols = [c for c in pred_df.columns if c not in meta_cols and not c.startswith('delta_') and not c.startswith('emb_')]

    prob = pred_df[[f'{disease}_prob']].values
    attr = pred_df[attr_cols].values
    diff = pred_df[diff_cols].values
    emb  = pred_df[emb_cols].values
    cf_prob = pred_df[['cf_prob']].values

    # Map config → selected features
    if config == 'B1':
        X = prob; names = [f'{disease}_prob']
    elif config == 'B2':
        X = attr; names = attr_cols
    elif config == 'B3':
        X = emb; names = emb_cols
    elif config == 'B4':
        X = np.hstack([prob, attr]); names = [f'{disease}_prob'] + attr_cols
    elif config == 'B5':
        X = np.hstack([prob, emb]); names = [f'{disease}_prob'] + emb_cols
    elif config == 'M1':
        X = diff; names = diff_cols
    elif config == 'M2':
        X = np.hstack([prob, diff]); names = [f'{disease}_prob'] + diff_cols
    elif config == 'M3':
        X = np.hstack([prob, diff, attr]); names = [f'{disease}_prob'] + diff_cols + attr_cols
    elif config == 'M4':
        X = np.hstack([prob, diff, emb]); names = [f'{disease}_prob'] + diff_cols + emb_cols
    elif config == 'M5':
        X = np.hstack([prob, diff, attr, emb]); names = [f'{disease}_prob'] + diff_cols + attr_cols + emb_cols
    elif config == 'M6':
        X = np.hstack([prob, diff, attr, cf_prob]); names = [f'{disease}_prob'] + diff_cols + attr_cols + ['cf_prob']
    else:
        raise ValueError(f'Unknown config: {config}')
    return X, names

# ======================
# ----- IMPORTANCE -----
# ======================
def compute_fold_importance(fold_idx, pred_df, disease, config, model_type, cf_count, cache_dir):
    """Compute or load cached permutation importance for a fold/config."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{config}_cf{cf_count}_fold_{fold_idx}_importance.pkl')
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    X, feature_names = build_feature_matrix(pred_df, disease, config)
    model_path = os.path.join(RESULTS_DIR,
                              f'C2_sim_cf/{disease}/cv_results/cf_{cf_count}/models/{model_type}/{model_type}_{config}_fold{fold_idx}.pkl')
    saved = joblib.load(model_path)
    model = saved['model']
    scaler = saved.get('scaler', None)
    if scaler:
        X = scaler.transform(X.astype(np.float32))

    y = pred_df['correct'].values
    r = permutation_importance(model, X, y, n_repeats=20, random_state=42, scoring = 'roc_auc', n_jobs=-1)
    fold_df = pd.DataFrame({'feature': feature_names,
                            'importance': r.importances_mean,
                            'std': r.importances_std,
                            'fold': fold_idx})
    fold_df.to_pickle(cache_path)
    return fold_df

# ======================
# ----- CATEGORIZATION --
# ======================
def categorize_feature(feature, disease):
    if feature == f'{disease}_prob':
        return 'C0 Confidence'
    if feature == 'cf_prob':
        return 'CF Probability'
    if feature in DEMO_COLS:
        return 'Demographics'
    if feature in [f'delta_{c}' for c in DEMO_COLS]:
        return 'Demographics (CF delta)'
    if feature in KEY_ORGAN_RATIOS or 'ratio' in feature:
        return 'Key-Organ Ratios'
    if feature in [f'delta_{c}' for c in KEY_ORGAN_RATIOS]:
        return 'Key-Organ Ratios (CF delta)'
    stripped = feature.replace('delta_', '')
    for organ in KEY_ORGAN_STRUCTURES:
        if stripped.startswith(organ):
            return 'Key-Organ Anatomy (CF delta)' if feature.startswith('delta_') else 'Key-Organ Anatomy'
    for organ in OTHER_ORGAN_STRUCTURES:
        if stripped.startswith(organ):
            return 'Other-Organ Anatomy (CF delta)' if feature.startswith('delta_') else 'Other-Organ Anatomy'
    print(f"Warning: Feature '{feature}' did not match any category")
    return 'Margin/Uncategorized'

def categorize_organ(feature):
    stripped = feature.replace('delta_', '').lower()
    stripped = stripped.replace(' ', '_')

    for organ in NORMALIZED_ORGANS:
        if organ in stripped:
            return organ
    return 'Other (Non-organ)'

# ======================
# ----- AGGREGATION ----
# ======================
def aggregate_importances(df_list, group_fn):
    combined = pd.concat(df_list, ignore_index=True)
    # Mean importance per feature across folds [cite: 507]
    feat_mean = combined.groupby('feature')['importance'].mean().reset_index()
    # Assign category
    feat_mean['group'] = feat_mean['feature'].apply(group_fn)
    # Aggregate per category
    agg = feat_mean.groupby('group')['importance'].agg(['mean','sum','count']).reset_index()
    return agg.sort_values('sum', ascending=False)

# ======================
# ----- PLOTTING -------
# ======================
def plot_group_importance(df, title, color_map=None, save_path=None, metric='mean'):
    df_sorted = df.sort_values(metric, ascending=True)
    colors = [color_map.get(g, '#999999') if color_map else '#3399CC' for g in df_sorted['group']]
    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.barh(df_sorted['group'], df_sorted[metric], color=colors, edgecolor='white', linewidth=0.8)
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f'n={int(row["count"])}', va='center')
    ax.set_title(title)
    ax.set_xlabel(f'Total Importance ({metric})')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_top_n_features(combined, disease, model_type, config, cf_count, cache_dir, n=30):
    """Plot top N individual features by mean importance, excluding C0 prob and CF prob and margin."""
    combined_no_prob = combined[combined['feature'] != f'{disease}_prob']
    combined_no_prob = combined_no_prob[combined_no_prob['feature'] != f'cf_prob']
    combined_no_prob = combined_no_prob[combined_no_prob['feature'] != f'margin']
    combined_no_prob = combined_no_prob[combined_no_prob['feature'] != f'delta_margin']
    top_n = (combined_no_prob
             .groupby('feature')['importance']
             .mean()
             .sort_values(ascending=False)
             .head(n))

    fig, ax = plt.subplots(figsize=(8, 10))
    top_n.sort_values().plot(kind='barh', ax=ax, color='#3399CC')
    ax.set_xlabel('Mean Permutation Importance (AUC drop)')
    ax.set_title(f'{disease.capitalize()} - Top {n} features excl. C0 and CF probs\n({model_type} {config} CF={cf_count})')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(cache_dir, f'{disease}_{config}_cf{cf_count}_top{n}_no_prob.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top-{n} plot to {save_path}")
    
PILLAR_COLORS = {
    'C0 Probability': '#FF9999',       # Light Red/Coral
    'CF Probability': '#FFCC99',       # Light Orange
    'Raw Attributes': '#88BBDD',       # Light Blue
    'Delta Attributes (ΔA)': '#3399CC' # Deep Blue
}

def plot_simplified_comparison(all_importances, disease, save_path=None):
    """
    Collapses categories into 4 high-level pillars with fixed color mapping.
    """
    combined = pd.concat(all_importances, ignore_index=True)
    # Mean importance per feature across folds
    feat_mean = combined.groupby('feature')['importance'].mean().reset_index()

    def high_level_map(feature):
        if feature == f'{disease}_prob':
            return 'C0 Probability'
        if feature == 'cf_prob':
            return 'CF Probability'
        if feature.startswith('delta_'):
            return 'Delta Attributes (ΔA)'
        return 'Raw Attributes'

    feat_mean['pillar'] = feat_mean['feature'].apply(high_level_map)
    
    # Aggregate importance by summing the mean drops
    pillar_agg = feat_mean.groupby('pillar')['importance'].sum().reset_index()
    pillar_agg = pillar_agg.sort_values('importance', ascending=True)

    # Create the color list based on the sorted pillar names
    current_colors = [PILLAR_COLORS[p] for p in pillar_agg['pillar']]

    plt.figure(figsize=(10, 6))
    plt.barh(pillar_agg['pillar'], pillar_agg['importance'], color=current_colors)
    
    plt.title(f'C2 Simplified Feature Pillars - {disease.capitalize()}')
    plt.xlabel('Total Importance (Sum of Mean AUROC Drops)')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ======================
# ----- MAIN -----------
# ======================
def main(disease, config, model_type, cf_count):
    cache_dir = os.path.join(RESULTS_DIR, 'C2_importances_cache')
    all_preds = []
    print("Starting feature importance computation...")
    for fold in range(N_FOLDS):
        print(f"Processing fold {fold+1}/{N_FOLDS}...")
        fold_path = os.path.join(RESULTS_DIR,
                                 f'C2_sim_cf/{disease}/cv_results/cf_{cf_count}/fold_{fold}_predictions.csv')
        all_preds.append(pd.read_csv(fold_path))
    
    selected_cols = [col for col in all_preds[0].columns if not any(m in col for m in ['MLP', 'LR', 'RF'])]
    all_preds = [df[selected_cols] for df in all_preds]
    print("Number of selected columns for importance computation:", len(all_preds[0].columns))
    
    # Compute importances
    all_importances = [compute_fold_importance(f, pred, disease, config, model_type, cf_count, cache_dir)
                       for f, pred in enumerate(all_preds)]
    
    print("Combining importances across folds...")
    combined_path = os.path.join(cache_dir, f'{disease}_{config}_cf{cf_count}_all_importances.pkl')
    combined = pd.concat(all_importances, ignore_index=True)
    combined.to_pickle(combined_path)    
    print(f"Saved combined importances to {combined_path}")

    # Plot top N features excluding C0 prob
    plot_top_n_features(combined, disease, model_type, config, cf_count, cache_dir)

    # Aggregate & plot by category
    print("Aggregating importances by category...")
    category_importance = aggregate_importances(all_importances, lambda f: categorize_feature(f, disease))
    cat_plot_path = os.path.join(cache_dir, f'{disease}_{config}_cf{cf_count}_category_importance.png')
    print("Plotting category importance...")
    plot_group_importance(category_importance, f'{disease.capitalize()} - Category Importance', CATEGORY_COLORS, cat_plot_path)

    # Aggregate & plot by organ
    print("Aggregating importances by organ...")
    organ_importance = aggregate_importances(all_importances, categorize_organ)
    organ_colors = {org: ('#88BBDD' if org in KEY_ORGAN_STRUCTURES else '#CCCCCC') for org in organ_importance['group']}
    print("Plotting organ importance...")
    organ_plot_path = os.path.join(cache_dir, f'{disease}_{config}_cf{cf_count}_organ_importance.png')
    plot_group_importance(organ_importance, f'{disease.capitalize()} - Organ Importance', organ_colors, organ_plot_path)

    # New Simplified Plot
    print("Generating simplified pillar comparison...")
    simplified_plot_path = os.path.join(cache_dir, f'{disease}_{config}_cf{cf_count}_pillar_comparison.png')
    plot_simplified_comparison(all_importances, disease, simplified_plot_path)

# ======================
# ----- ENTRY POINT ----
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, default='effusion')
    parser.add_argument('--config', type=str, default= 'M6', choices=CONFIGS)
    parser.add_argument('--model_type', type=str, default='MLP', choices=['MLP','LR','RF'])
    parser.add_argument('--cf_count', type=int, default=1, help='Number of counterfactuals (k)')
    args = parser.parse_args()
    main(args.disease, args.config, args.model_type, args.cf_count)