import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_results(binary_df, multi_df, target_digit=2):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    digit_list = [2, 5, 8]

    # ---- Binary Experiment ----
    binary_grouped = binary_df.groupby('reward_value_digit2').agg({
        'selection_rate_digit2': ['mean', 'std'],
        'reward_ratio': 'mean'
    }).reset_index()
    binary_grouped.columns = ['reward_value_digit2', 'selection_rate_mean', 'selection_rate_std', 'reward_ratio']

    axs[0, 0].errorbar(range(len(binary_grouped)), binary_grouped['selection_rate_mean'],
                       yerr=binary_grouped['selection_rate_std'], fmt='o-', capsize=5)
    axs[0, 0].set_xticks(range(len(binary_grouped)))
    axs[0, 0].set_xticklabels(binary_grouped['reward_value_digit2'])
    axs[0, 0].set_title('Experiment 1: Reward vs Selection (2 Digits)')
    axs[0, 0].set_xlabel('Reward Value Digit 2')
    axs[0, 0].set_ylabel('Selection Rate')
    axs[0, 0].grid(True)

    response_ratios = binary_grouped['selection_rate_mean'] / (1 - binary_grouped['selection_rate_mean'])
    axs[1, 0].loglog(binary_grouped['reward_ratio'], response_ratios, 'o-')
    ideal_line = np.linspace(binary_grouped['reward_ratio'].min(), binary_grouped['reward_ratio'].max(), 100)
    axs[1, 0].loglog(ideal_line, ideal_line, 'k--', label='Matching Law')
    axs[1, 0].set_title('Experiment 1: Matching Law Log-Log')
    axs[1, 0].set_xlabel('Reward Ratio')
    axs[1, 0].set_ylabel('Response Ratio')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # ---- Multi-Digit Experiment ----
    multi_grouped = multi_df.groupby(f'reward_value_digit_{target_digit}').agg({
        f'selection_rate_digit_{target_digit}': ['mean', 'std']
    }).reset_index()

    x_vals = range(len(multi_grouped))
    y_vals = multi_grouped[(f'selection_rate_digit_{target_digit}', 'mean')]
    y_errs = multi_grouped[(f'selection_rate_digit_{target_digit}', 'std')]

    axs[0, 1].errorbar(x_vals, y_vals, yerr=y_errs, fmt='o-', capsize=5, label=f'Digit {target_digit}')
    axs[0, 1].set_xticks(x_vals)
    axs[0, 1].set_xticklabels(multi_grouped[f'reward_value_digit_{target_digit}'])
    axs[0, 1].set_title('Experiment 2: Reward vs Selection (3 Digits)')
    axs[0, 1].set_xlabel(f'Reward Value Digit {target_digit}')
    axs[0, 1].set_ylabel('Selection Rate')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Log-log plot for all reward/response ratios
    for digit in [5, 8]:
        rr_col = f'reward_ratio_{target_digit}vs{digit}'
        resp_col = f'response_ratio_{target_digit}vs{digit}'
        if rr_col in multi_df.columns and resp_col in multi_df.columns:
            avg_rr = multi_df.groupby(f'reward_value_digit_{target_digit}')[rr_col].mean()
            avg_resp = multi_df.groupby(f'reward_value_digit_{target_digit}')[resp_col].mean()
            axs[1, 1].loglog(avg_rr, avg_resp, 'o-', label=f'{target_digit} vs {digit}')

    if not multi_df.empty:
        min_r = min([multi_df[f'reward_ratio_{target_digit}vs{d}'].min() for d in [5, 8]])
        max_r = max([multi_df[f'reward_ratio_{target_digit}vs{d}'].max() for d in [5, 8]])
        ideal = np.logspace(np.log10(max(min_r, 0.1)), np.log10(max_r), 100)
        axs[1, 1].loglog(ideal, ideal, 'k--', label='Matching Law')

    axs[1, 1].set_title('Experiment 2: Matching Law Log-Log')
    axs[1, 1].set_xlabel('Reward Ratio')
    axs[1, 1].set_ylabel('Response Ratio')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('results/combined_experiments_comparison.png')
    return fig, axs


if __name__ == "__main__":
    binary_df = pd.read_csv('matching_law_results.csv')
    multi_df = pd.read_csv('multi_digit_matching_law_results.csv')
    plot_combined_results(binary_df, multi_df)
    plt.show()
