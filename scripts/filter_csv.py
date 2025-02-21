import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

df = pd.read_csv('outputs/mesh_analysis_results.csv')
result = df.groupby('Triangle Count')['Processing Time (s)'].max().reset_index()

result_filtered = result[
    (result['Triangle Count'] >= 9000) &
    (result['Processing Time (s)'] <= 200)
]

result_filtered = result_filtered.sort_values('Triangle Count')

# Function to detect outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

result_no_outliers = remove_outliers(result_filtered, 'Processing Time (s)')

pearson_corr, pearson_p = stats.pearsonr(result_no_outliers['Triangle Count'],
                                        result_no_outliers['Processing Time (s)'])
spearman_corr, spearman_p = stats.spearmanr(result_no_outliers['Triangle Count'],
                                           result_no_outliers['Processing Time (s)'])

print(f"Original data points: {len(result)}")
print(f"Data points after triangle count and time filtering: {len(result_filtered)}")
print(f"Data points after removing outliers: {len(result_no_outliers)}")
print(f"Total points removed: {len(result) - len(result_no_outliers)}")
print(f"Pearson correlation coefficient (without outliers): {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
print(f"Spearman correlation coefficient (without outliers): {spearman_corr:.3f} (p-value: {spearman_p:.3f})")

plt.figure(figsize=(12, 8))

# Plot non-outliers with color gradient
scatter = plt.scatter(result_no_outliers['Triangle Count'],
                     result_no_outliers['Processing Time (s)'],
                     c=result_no_outliers['Triangle Count'],
                     cmap='viridis',
                     alpha=0.9)
# do cmap if you want colors

# Plot outliers in gray
outliers = result_filtered[~result_filtered.index.isin(result_no_outliers.index)]
# if len(outliers) > 0:
    # plt.scatter(outliers['Triangle Count'], outliers['Processing Time (s)'], cmap='viridis',
      #         alpha=0.4, label='Outliers')

# Add correlation info to plot
plt.text(0.05, 0.9,
         f'Pearson r: {pearson_corr:.3f}\nSpearman ρ: {spearman_corr:.3f}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# Add labels and title
plt.xlabel('Triangle Count')
plt.ylabel('Maximum Processing Time (seconds)')
plt.title('Maximum Processing Time vs Triangle Count')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Automatically adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('processing_times_plot_gradient.png')
plt.close()

# Save cleaned data to CSV
result_no_outliers.to_csv('max_times_by_triangle_count_filtered.csv', index=False)