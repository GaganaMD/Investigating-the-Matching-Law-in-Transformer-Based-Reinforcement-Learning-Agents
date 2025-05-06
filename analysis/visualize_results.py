import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_df):
    grouped = results_df.groupby('reward_value_digit2').agg({
        'selection_rate_digit2': ['mean', 'std'],
        'reward_ratio': 'mean',
        'total_reward': 'mean'
    }).reset_index()

    grouped.columns = ['reward_value_digit2', 'selection_rate_mean', 'selection_rate_std',
                       'reward_ratio', 'avg_total_reward']

    plt.figure(figsize=(12, 8))

    # Plot 1
    plt.subplot(2, 1, 1)
    plt.errorbar(range(len(grouped)), grouped['selection_rate_mean'],
                 yerr=grouped['selection_rate_std'], fmt='o-', capsize=5)
    plt.xticks(range(len(grouped)), grouped['reward_value_digit2'])
    plt.xlabel('Reward Value for Digit 2')
    plt.ylabel('Selection Rate')
    plt.title('Reward Amount vs Selection Rate')
    plt.grid(True)

    # Plot 2
    plt.subplot(2, 1, 2)
    response_ratios = grouped['selection_rate_mean'] / (1 - grouped['selection_rate_mean'])
    plt.loglog(grouped['reward_ratio'], response_ratios, 'o-')
    ideal_line = np.linspace(grouped['reward_ratio'].min(), grouped['reward_ratio'].max(), 100)
    plt.loglog(ideal_line, ideal_line, 'k--', label='Ideal Matching')
    plt.xlabel('Reward Ratio (2 vs 5)')
    plt.ylabel('Response Ratio')
    plt.title('Matching Law Log-Log Plot')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/matching_law_results.png')
    plt.show()


if __name__ == "__main__":
    results = pd.read_csv('matching_law_results.csv')
    print("Reward values:", sorted(results['reward_value_digit2'].unique()))
    plot_results(results)
