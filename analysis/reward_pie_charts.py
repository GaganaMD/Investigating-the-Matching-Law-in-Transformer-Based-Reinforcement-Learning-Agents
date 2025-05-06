import pandas as pd
import matplotlib.pyplot as plt

def create_reward_pie_charts(binary_df, multi_df, reward_levels=[1, 2, 4], fixed_binary=1, fixed_multi=1):
    fig, axes = plt.subplots(2, len(reward_levels), figsize=(18, 10))
    fig.suptitle('Selection Proportions Across Reward Levels')

    # Binary
    for i, r in enumerate(reward_levels):
        ax = axes[0, i]
        row = binary_df[binary_df['reward_value_digit2'] == r]
        if not row.empty:
            sel_2 = row['selection_rate_digit2'].mean()
            sel_5 = 1 - sel_2
            ax.pie([sel_2, sel_5], labels=['Digit 2', 'Digit 5'], autopct='%1.1f%%',
                   colors=['#3498db', '#e74c3c'])
            ratio = row['reward_ratio'].mean()
            ax.set_title(f'Binary: R2={r}, R5={fixed_binary}, Ratio={ratio:.2f}')
        else:
            ax.axis('off')

    # Multi
    for i, r in enumerate(reward_levels):
        ax = axes[1, i]
        row = multi_df[multi_df['reward_value_digit_2'] == r]
        if not row.empty:
            sel_2 = row['selection_rate_digit_2'].mean()
            sel_5 = row['selection_rate_digit_5'].mean()
            sel_8 = row['selection_rate_digit_8'].mean()
            ax.pie([sel_2, sel_5, sel_8], labels=['Digit 2', 'Digit 5', 'Digit 8'],
                   autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#2ecc71'])
            ratio_25 = row['reward_ratio_2vs5'].mean()
            ratio_28 = row['reward_ratio_2vs8'].mean()
            ax.set_title(f'Multi: R2={r}, R5/R8={fixed_multi}\nR2vs5={ratio_25:.2f}, R2vs8={ratio_28:.2f}')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('results/reward_pie_charts.png')
    return fig, axes


if __name__ == "__main__":
    binary_df = pd.read_csv('matching_law_results.csv')
    multi_df = pd.read_csv('multi_digit_matching_law_results.csv')
    create_reward_pie_charts(binary_df, multi_df)
    plt.show()
