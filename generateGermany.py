from networkx import display
import pandas as pd
import numpy as np
import os
import sys
import torch 
import plotly.io as pio
# import kaleido
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality, get_column_plot, get_column_pair_plot

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA (GPU): {torch.cuda.get_device_name(0)}')
    torch.cuda.empty_cache()

else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available. Using CPU.')

ctab_gan_plus_path = r"./CTAB-GAN-Plus"

sys.path.append(ctab_gan_plus_path)

try:
    from model.ctabgan import CTABGAN
except ImportError:
    print(f"Error: Could not find CTAB-GAN-Plus code or CTABGAN class.")
    print(f"Please check if '{ctab_gan_plus_path}' path and 'model/ctabgan.py' file are correct.")
    sys.exit(1)

original_data_path = r"./german_travel_survey/personen_germany.csv"
original_data = pd.read_csv(original_data_path, encoding='utf-8', engine='python')
output_base_path = r"./german_travel_survey/personen_synth_germany.csv"


try:
    print(f"Original data load complete: {original_data_path} (data type: {original_data.shape})")
except FileNotFoundError:
    print(f"Error: Original data file not found. Check path: {original_data_path}")
    sys.exit(1)

all_columns = original_data.columns.tolist()
categorical_columns = [col for col in all_columns if col not in ['age']]
integer_columns = ['age']

print(f"number of categorical columns: ({len(categorical_columns)}): {categorical_columns}")
print(f"number of integer columns ({len(integer_columns)}): {integer_columns}")


import gc 
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

data_folder = os.path.dirname(original_data_path)

try:
    original_data = pd.read_csv(original_data_path, encoding='utf-8', engine='python')
    print(f"Original data load complete: {original_data_path} (shape: {original_data.shape})\n")
except FileNotFoundError:
    print(f"Error: Original data file not found. Check path: {original_data_path}")
    sys.exit(1)


def objective(trial):
    
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
    epochs = trial.suggest_int('epochs', 500, 1500, step=100) 
    lr_g = trial.suggest_loguniform('lr_g', 1e-4, 1e-3)
    lr_d = trial.suggest_loguniform('lr_d', 1e-4, 1e-3)
    lr_c = trial.suggest_loguniform('lr_c', 1e-4, 1e-3)
    lambda_gp = trial.suggest_loguniform('lambda_gp', 1.0, 30.0) 
    gumbel_tau = trial.suggest_uniform('gumbel_tau', 0.1, 1.0)

    print(f"\nTrial {trial.number}: Starting CTABGAN model initialization...")
    
    synthesizer = CTABGAN(
        raw_csv_path=original_data_path,
        test_ratio=0.00,
        categorical_columns=categorical_columns,
        log_columns=[],
        mixed_columns={},
        general_columns=[],
        non_categorical_columns=[],
        integer_columns=integer_columns,
        problem_type={None: None},
        
        batch_size=batch_size, 
        epochs=epochs,      
        lr_g=lr_g,
        lr_d=lr_d,
        lr_c=lr_c,
        lambda_gp=lambda_gp,
        gumbel_tau=gumbel_tau,

        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        classifier_dropout_rate=0.5,
        leaky_relu_slope_classifier=0.2,
        beta1_g=0.5, beta2_g=0.9,
        beta1_d=0.5, beta2_d=0.9,
        beta1_c=0.5, beta2_c=0.9,
        eps_common=1e-8,
        l2scale_common=1e-5,
        leaky_relu_slope=0.2,
        gen_leaky_relu_slope=0.2,
        disc_max_conv_blocks=3,
        gen_max_conv_blocks=3,
        lambda_cond_loss=1.0,
        lambda_info_loss=1.0,
        lambda_aux_classifier_loss=1.0,
        ci_discriminator_steps=5,
        verbose=False 
    ) 
    print("CTABGAN model initialization complete. Starting training...")

    try:
        synthesizer.fit(trial=trial) 
        synthetic_data = synthesizer.generate_samples()
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(original_data) 
        quality_report = evaluate_quality(original_data, synthetic_data, metadata)
        score = quality_report.get_score()
        print(f"Trial {trial.number} Overall Quality Score: {score}")
        
        return score
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Trial {trial.number} Failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 
        raise optuna.exceptions.TrialPruned(f"Error during training or evaluation: {e}")
    finally:
        if 'synthesizer' in locals() and synthesizer is not None:
            del synthesizer
        if 'synthetic_data' in locals() and synthetic_data is not None:
            del synthetic_data
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

study = optuna.create_study(direction='maximize', study_name='ctabgan_hyperparameter_tuning')
study.optimize(objective, n_trials=10)

print("\nHyperparameter tuning complete.")
print(f"Best hyperparameters: {study.best_params}")
print(f"Best Overall Quality Score: {study.best_value}")

best_params = study.best_params

print("\nRetraining CTABGAN model with best hyperparameters...")
synthesizer_best = CTABGAN(
    raw_csv_path=original_data_path,
    test_ratio=0.00,
    categorical_columns=categorical_columns,
    log_columns=[],
    mixed_columns={},
    general_columns=[],
    non_categorical_columns=[],
    integer_columns=integer_columns,
    problem_type={None: None},
    
    batch_size=best_params.get('batch_size'), 
    epochs=best_params.get('epochs'),      
    lr_g=best_params.get('lr_g'),
    lr_d=best_params.get('lr_d'),
    lr_c=best_params.get('lr_c'),
    lambda_gp=best_params.get('lambda_gp'),
    gumbel_tau=best_params.get('gumbel_tau'),

    class_dim=(256, 256, 256, 256),
    random_dim=100,
    num_channels=64,
    classifier_dropout_rate=0.5,
    leaky_relu_slope_classifier=0.2,
    beta1_g=0.5, beta2_g=0.9,
    beta1_d=0.5, beta2_d=0.9,
    beta1_c=0.5, beta2_c=0.9,
    eps_common=1e-8,
    l2scale_common=1e-5,
    leaky_relu_slope=0.2,
    gen_leaky_relu_slope=0.2,
    disc_max_conv_blocks=3,
    gen_max_conv_blocks=3,
    lambda_cond_loss=1.0,
    lambda_info_loss=1.0,
    lambda_aux_classifier_loss=1.0,
    ci_discriminator_steps=5,
    verbose=True 
) 
synthesizer_best.fit()

final_synthetic_data = synthesizer_best.generate_samples()
final_synthetic_data_path = os.path.join(data_folder, "travel_survey_generated.csv")
final_synthetic_data.to_csv(final_synthetic_data_path, index=False, encoding='utf-8-sig')
print(f"Final synthetic data generated with best hyperparameters saved to: {final_synthetic_data_path}")

# Save the trained model
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = os.path.join(data_folder, f"ctabgan_model_{timestamp}.pth")
synthesizer_best.save(model_save_path)
print(f"Trained CTABGAN model saved to: {model_save_path}")

# ============================================================
# Data Quality Analysis and Visualization
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

def analyze_and_visualize_data_quality(original_data, synthetic_data, save_dir="outputs/optimal_ctabgan+"):
    """Compare and visualize distribution of original and synthetic data"""
    
    print("📊 Starting Data Quality Analysis and Visualization!")
    print("=" * 60)
    
    # plt.rcParams['font.family'] = 'Malgun Gothic' # Removed Korean font setting
    # plt.rcParams['axes.unicode_minus'] = False  # Removed Korean font setting 
    
    save_dir = Path(save_dir)
    viz_dir = save_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("📈 Basic Statistics Comparison:")
    print("-" * 40)
    
    stats_comparison = {}
    
    print("🔢 Continuous Variable (age):")
    age_stats = {
        'Mean': [original_data['age'].mean(), synthetic_data['age'].mean()],
        'Std': [original_data['age'].std(), synthetic_data['age'].std()],
        'Min': [original_data['age'].min(), synthetic_data['age'].min()],
        'Max': [original_data['age'].max(), synthetic_data['age'].max()],
        'Median': [original_data['age'].median(), synthetic_data['age'].median()]
    }
    
    age_df = pd.DataFrame(age_stats, index=['Original', 'Synthetic'])
    print(age_df.round(2))
    stats_comparison['age'] = age_df

    all_categorical_columns = ['sex', 'housetype', 'driver_license', 'drive_regularly', 'commute_to_fixed_workplace', 'occupation', 
                               'home_office_days', 'car_group', 'bicycle_group', 'other_group', 'mode_choice']
    
    print("\n📊 Categorical Variable Distribution Comparison:")
    print("-" * 40)
    
    for col in all_categorical_columns:
        print(f"\n🏷️ {col} Distribution:")
        
        orig_dist = original_data[col].value_counts(normalize=True).sort_index()
        synth_dist = synthetic_data[col].value_counts(normalize=True).sort_index()
        
        all_cats = sorted(set(orig_dist.index) | set(synth_dist.index))
        
        comparison_data = []
        for cat in all_cats:
            orig_pct = orig_dist.get(cat, 0) * 100
            synth_pct = synth_dist.get(cat, 0) * 100
            diff = abs(orig_pct - synth_pct)
            comparison_data.append({
                'Category': cat,
                'Original(%)': orig_pct,
                'Synthetic(%)': synth_pct,
                'Difference(%)': diff
            })
            print(f"       {cat}: Original {orig_pct:.1f}% vs Synthetic {synth_pct:.1f}% (Diff: {diff:.1f}%)")
        
        stats_comparison[col] = pd.DataFrame(comparison_data)
    
    print(f"\n🎨 Generating Visualizations... (Saved to: {viz_dir})")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10)) 
    fig.suptitle('Age Distribution Comparison', fontsize=16, fontweight='bold')
    
    min_age = min(original_data['age'].min(), synthetic_data['age'].min())
    max_age = max(original_data['age'].max(), synthetic_data['age'].max())
    bins = np.linspace(min_age, max_age, 20) 

    orig_binned_counts, _ = np.histogram(original_data['age'], bins=bins, density=True)
    synth_binned_counts, _ = np.histogram(synthetic_data['age'], bins=bins, density=True)
    
    bin_labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins)-1)]
    x = np.arange(len(bin_labels))
    width = 0.35

    ax = axes[0,0]
    ax.bar(x - width/2, orig_binned_counts, width, label='Original', alpha=0.8, color='blue')
    ax.bar(x + width/2, synth_binned_counts, width, label='Synthetic', alpha=0.8, color='red')
    ax.set_title('Age Binned Distribution')
    ax.set_xlabel('Age Bins')
    ax.set_ylabel('Density')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    box_data = [original_data['age'], synthetic_data['age']]
    axes[0,1].boxplot(box_data, labels=['Original', 'Synthetic'])
    axes[0,1].set_title('Box Plot Comparison')
    axes[0,1].set_ylabel('Age')
    axes[0,1].grid(True, alpha=0.3)
    
    stats.probplot(original_data['age'], dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot - Original')
    axes[1,0].grid(True, alpha=0.3)
    
    stats.probplot(synthetic_data['age'], dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot - Synthetic')
    axes[1,1].grid(True, alpha=0.3)

    fig.delaxes(axes[0,2]) 
    fig.delaxes(axes[1,2])
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'age_distribution_comparison_5.png', dpi=300, bbox_inches='tight')
    plt.close()


   

    other_categorical_columns = [col for col in all_categorical_columns if col not in ['home_province', 'home_administrative']]
    
    n_other_categorical = len(other_categorical_columns)
    n_cols = 3
    n_rows = (n_other_categorical + n_cols - 1) // n_cols
    
    if n_other_categorical > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        
        if n_rows == 1 and n_cols > 1: 
            axes = axes.reshape(1, -1)
        elif n_rows > 1 and n_cols == 1:
            axes = axes.reshape(-1, 1)
        elif n_rows == 1 and n_cols == 1: 
            axes = np.array([[axes]]) 

        for idx, col in enumerate(other_categorical_columns):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row, col_idx] 
            
            orig_dist = original_data[col].value_counts(normalize=True).sort_index()
            synth_dist = synthetic_data[col].value_counts(normalize=True).sort_index()
            
            all_cats = sorted(set(orig_dist.index) | set(synth_dist.index))
            orig_values = [orig_dist.get(cat, 0) for cat in all_cats]
            synth_values = [synth_dist.get(cat, 0) for cat in all_cats]
            
            x = np.arange(len(all_cats))
            width = 0.35
            
            ax.bar(x - width/2, orig_values, width, label='Original', alpha=0.8, color='blue')
            ax.bar(x + width/2, synth_values, width, label='Synthetic', alpha=0.8, color='red')
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel('Categories')
            ax.set_ylabel('Proportion')
            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_other_categorical, n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            fig.delaxes(axes[row, col_idx])

        plt.tight_layout()
        plt.savefig(viz_dir / 'other_categorical_distribution_comparison_5.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No other categorical columns to visualize (excluding home_province, home_administrative).")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    def encode_for_correlation(df):
        df_encoded = df.copy()
        le_dict = {}
        for col in all_categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                le_dict[col] = le
            else:
                print(f"Warning: '{col}' column not found in DataFrame during correlation encoding.")
        return df_encoded, le_dict
    
    orig_encoded, _ = encode_for_correlation(original_data)
    synth_encoded, _ = encode_for_correlation(synthetic_data)
    
    orig_corr = orig_encoded.corr(numeric_only=True)
    synth_corr = synth_encoded.corr(numeric_only=True)
    
    sns.heatmap(orig_corr, annot=True, cmap='coolwarm', center=0, ax=ax1, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax1.set_title('Original Data Correlation Matrix')
    
    sns.heatmap(synth_corr, annot=True, cmap='coolwarm', center=0, ax=ax2,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax2.set_title('Synthetic Data Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'correlation_comparison_5.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Calculating Statistical Similarity Scores:")
    print("-" * 40)
    
    similarity_scores = {}
    
    from scipy.stats import wasserstein_distance
    age_wd = wasserstein_distance(original_data['age'], synthetic_data['age'])
    age_similarity = 1 / (1 + age_wd)
    similarity_scores['age'] = age_similarity
    print(f"🔢 Age Similarity Score: {age_similarity:.4f}")

    categorical_similarities = []
    for col in all_categorical_columns:
        if col in original_data.columns and col in synthetic_data.columns:
            orig_dist = original_data[col].value_counts(normalize=True).sort_index()
            synth_dist = synthetic_data[col].value_counts(normalize=True).sort_index()
            
            all_cats = sorted(set(orig_dist.index) | set(synth_dist.index))
            
            temp_df = pd.DataFrame(index=all_cats)
            temp_df['orig'] = orig_dist.reindex(all_cats, fill_value=0)
            temp_df['synth'] = synth_dist.reindex(all_cats, fill_value=0)

            similarity = 1 - np.abs(temp_df['orig'] - temp_df['synth']).mean()
        else:
            similarity = 0.0 
            print(f"Warning: '{col}' column not found in DataFrame during similarity calculation.")
            
        similarity_scores[col] = similarity
        categorical_similarities.append(similarity)
        print(f"🏷️ {col} Similarity Score: {similarity:.4f}")
    
    overall_similarity = np.mean(list(similarity_scores.values()))
    print(f"\n🎯 Overall Average Similarity Score: {overall_similarity:.4f}")
    
    summary_report = f"""
Data Quality Analysis Report
========================

📊 Data Size:
- Original Data: {len(original_data):,} rows
- Synthetic Data: {len(synthetic_data):,} rows

📈 Similarity Scores:
- Age: {similarity_scores['age']:.4f}
- Categorical Variable Average: {np.mean(categorical_similarities):.4f}
- Overall Average: {overall_similarity:.4f}

📋 Scores by Categorical Variable:
"""
    
    for col in all_categorical_columns: 
        if col in similarity_scores:
            summary_report += f"- {col}: {similarity_scores[col]:.4f}\n"
        else:
            summary_report += f"- {col}: No Score (Column Missing)\n"
    
    summary_report += f"""
🎨 Generated Visualizations:
- age_distribution_comparison_5.png
- home_administrative_distribution_Daejeon.png
- home_administrative_distribution_Sejong.png

- other_categorical_distribution_comparison_5.png
- correlation_comparison_5.png

📁 Save Location: {viz_dir}
"""
    
    with open(viz_dir / 'quality_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print("\n" + "=" * 60)
    print("✅ Data Quality Analysis and Visualization Complete!")
    print(f"📁 Results Saved to: {viz_dir}")
    print(f"📊 Overall Similarity Score: {overall_similarity:.4f}")
    print("=" * 60)
    
    return similarity_scores, stats_comparison

def run_complete_analysis():
    
    print("📂 Loading Data...")
    
    original_data = pd.read_csv(r"./german_travel_survey/personen_germany.csv")
    synthetic_data = pd.read_csv(r"./german_travel_survey/personen_synth_germany.csv")
    
    similarity_scores, stats_comparison = analyze_and_visualize_data_quality(
        original_data, 
        synthetic_data
    )
    
    return similarity_scores, stats_comparison

if __name__ == "__main__":
    similarity_scores, stats_comparison = run_complete_analysis()

from sklearn.preprocessing import MinMaxScaler
from dython import nominal
from scipy.stats import wasserstein_distance
from scipy.spatial import distance

def stat_sim(real_path, fake_path, cat_cols=None):
    real = pd.read_csv(real_path)
    fake = pd.read_csv(fake_path)

    if cat_cols:
        categorical_columns = cat_cols[0] 
        for col in categorical_columns:
            if col in real.columns:
                real[col] = real[col].astype('category')
            if col in fake.columns:
                fake[col] = fake[col].astype('category')

    real_assoc = nominal.associations(real, nominal_columns=cat_cols[0], mark_columns=False)
    fake_assoc = nominal.associations(fake, nominal_columns=cat_cols[0], mark_columns=False)

    real_corr = real_assoc['corr']
    fake_corr = fake_assoc['corr']

    keys = sorted(real_corr.columns)
    real_matrix = real_corr.loc[keys, keys].to_numpy()
    fake_matrix = fake_corr.loc[keys, keys].to_numpy()
    corr_dist = np.linalg.norm(real_matrix - fake_matrix)

    cat_stat = [] 
    num_stat = []

    for column in real.columns:
        if column in categorical_columns: 
            real_pdf = (real[column].value_counts() / real[column].value_counts().sum())
            fake_pdf = (fake[column].value_counts() / fake[column].value_counts().sum())
            categories = sorted(set(real[column].unique()).union(set(fake[column].unique())))
            real_pdf_values = [real_pdf.get(cat, 0) for cat in categories]
            fake_pdf_values = [fake_pdf.get(cat, 0) for cat in categories]
            cat_stat.append(distance.jensenshannon(real_pdf_values, fake_pdf_values, 2.0))
        else: 
            scaler = MinMaxScaler()
            real_col_data = real[column].dropna().values.reshape(-1, 1)
            fake_col_data = fake[column].dropna().values.reshape(-1, 1)

            if len(real_col_data) > 0 and len(fake_col_data) > 0: 
                scaler.fit(real_col_data)
                l1 = scaler.transform(real_col_data).flatten()
                l2 = scaler.transform(fake_col_data).flatten()
                num_stat.append(wasserstein_distance(l1, l2))
            else:
                print(f"Warning: Column '{column}' contains no valid data for numerical statistics after dropping NaNs.")


    avg_wd = np.mean(num_stat) if num_stat else 0
    avg_jsd = np.mean(cat_stat) if cat_stat else 0

    return [avg_wd, avg_jsd, corr_dist]


real_path = r"./german_travel_survey/personen_germany.csv"
fake_path = r"./german_travel_survey/personen_synth_germany.csv"

cat_cols = [['sex', 'housetype', 'driver_license', 'drive_regularly', 'commute_to_fixed_workplace',
             'occupation', 'home_office_days', 'car_group', 'bicycle_group', 'other_group', 'mode_choice']]

avg_wd, avg_jsd, diff_corr = stat_sim(real_path, fake_path, cat_cols=cat_cols)

result = pd.DataFrame(
    [[f"{avg_jsd:.3f}", f"{avg_wd:.3f}", f"{diff_corr:.2f}"]],
    columns=[
        "Avg JSD", "Avg WD", "Diff. corr."
    ],
    index=["Result"]
)

print(result)