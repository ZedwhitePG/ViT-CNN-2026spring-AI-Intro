import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if not os.path.exists('results_summary.csv'):
        print("未找到 results_summary.csv，请确认文件是否存在！")
        return
    
    os.makedirs('figures', exist_ok=True)
    
    df = pd.read_csv('results_summary.csv')
    
    # Accuracy trend
    plt.figure(figsize=(8, 6))
    for model_name in sorted(df['model'].unique()):
        subset = df[df['model'] == model_name].sort_values('data_ratio')
        plt.plot(subset['data_ratio'] * 100, subset['best_val_acc'], marker='o', label=model_name.upper(), linewidth=2, markersize=8)
    
    plt.title('Validation Accuracy under Different Data Scales', fontsize=14)
    plt.xlabel('Data Ratio (%)', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.xticks([10, 20, 100])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('figures/data_scale_trend_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("生成图表: figures/data_scale_trend_accuracy.png")

    # F1 trend
    plt.figure(figsize=(8, 6))
    for model_name in sorted(df['model'].unique()):
        subset = df[df['model'] == model_name].sort_values('data_ratio')
        plt.plot(subset['data_ratio'] * 100, subset['best_val_f1'], marker='s', label=model_name.upper(), linewidth=2, markersize=8)
    
    plt.title('Validation F1 Score under Different Data Scales', fontsize=14)
    plt.xlabel('Data Ratio (%)', fontsize=12)
    plt.ylabel('Validation F1 Score', fontsize=12)
    plt.xticks([10, 20, 100])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('figures/data_scale_trend_f1.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("生成图表: figures/data_scale_trend_f1.png")

if __name__ == '__main__':
    main()
