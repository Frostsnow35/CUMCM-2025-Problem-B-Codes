#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import json
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('results', exist_ok=True)

def load_data():
    df10 = pd.read_csv('results/附件3_10度.csv')
    df15 = pd.read_csv('results/附件4_15度.csv')
    
    df10.columns = ['wavenumber', 'reflectance']
    df15.columns = ['wavenumber', 'reflectance']
    
    print(f"10°数据: {len(df10)}点")
    print(f"15°数据: {len(df15)}点")
    print(f"波数范围: {df10['wavenumber'].min():.1f}-{df10['wavenumber'].max():.1f} cm⁻¹")
    
    return df10, df15

def analyze_patterns(df10, df15):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(df10['wavenumber'], df10['reflectance'], 'b-', linewidth=1, label='10°')
    axes[0, 0].plot(df15['wavenumber'], df15['reflectance'], 'r-', linewidth=1, label='15°')
    axes[0, 0].set_xlabel('波数 (cm⁻¹)')
    axes[0, 0].set_ylabel('反射率 (%)')
    axes[0, 0].set_title('光谱对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    smooth10 = savgol_filter(df10['reflectance'], 21, 3)
    smooth15 = savgol_filter(df15['reflectance'], 21, 3)
    
    axes[0, 1].plot(df10['wavenumber'], df10['reflectance'], 'b-', alpha=0.3, label='10°原始')
    axes[0, 1].plot(df10['wavenumber'], smooth10, 'b-', linewidth=2, label='10°平滑')
    axes[0, 1].plot(df15['wavenumber'], df15['reflectance'], 'r-', alpha=0.3, label='15°原始')
    axes[0, 1].plot(df15['wavenumber'], smooth15, 'r-', linewidth=2, label='15°平滑')
    axes[0, 1].set_xlabel('波数 (cm⁻¹)')
    axes[0, 1].set_ylabel('反射率 (%)')
    axes[0, 1].set_title('数据平滑')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    peaks10, _ = find_peaks(smooth10, height=0.1, distance=50, prominence=0.05)
    peaks15, _ = find_peaks(smooth15, height=0.1, distance=50, prominence=0.05)
    
    axes[0, 2].plot(df10['wavenumber'], smooth10, 'b-', linewidth=1, label='10°平滑')
    axes[0, 2].scatter(df10['wavenumber'].iloc[peaks10], smooth10[peaks10], 
                color='red', s=50, marker='o', label=f'10°峰位 ({len(peaks10)}个)')
    axes[0, 2].plot(df15['wavenumber'], smooth15, 'r-', linewidth=1, label='15°平滑')
    axes[0, 2].scatter(df15['wavenumber'].iloc[peaks15], smooth15[peaks15], 
                color='red', s=50, marker='s', label=f'15°峰位 ({len(peaks15)}个)')
    axes[0, 2].set_xlabel('波数 (cm⁻¹)')
    axes[0, 2].set_ylabel('反射率 (%)')
    axes[0, 2].set_title('峰位检测')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    if len(peaks10) > 1:
        spacing10 = np.diff(df10['wavenumber'].iloc[peaks10])
        axes[1, 0].hist(spacing10, bins=15, alpha=0.7, color='blue', 
                 label=f'10°间距 ({np.mean(spacing10):.1f})')
    if len(peaks15) > 1:
        spacing15 = np.diff(df15['wavenumber'].iloc[peaks15])
        axes[1, 0].hist(spacing15, bins=15, alpha=0.7, color='red', 
                 label=f'15°间距 ({np.mean(spacing15):.1f})')
    axes[1, 0].set_xlabel('峰位间距 (cm⁻¹)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('峰位间距分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(df10['reflectance'], bins=50, alpha=0.7, color='blue', label='10°反射率')
    axes[1, 1].hist(df15['reflectance'], bins=50, alpha=0.7, color='red', label='15°反射率')
    axes[1, 1].set_xlabel('反射率 (%)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('反射率分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    r10_mean = np.mean(df10['reflectance'])
    r10_std = np.std(df10['reflectance'])
    r15_mean = np.mean(df15['reflectance'])
    r15_std = np.std(df15['reflectance'])
    
    cond10 = r10_mean > 10
    cond15 = r15_mean > 10
    
    stats_text = f"""
    干涉条件分析:
    
    10°:
    平均反射率: {r10_mean:.2f}%
    标准差: {r10_std:.2f}%
    干涉条件: {'满足' if cond10 else '不满足'}
    
    15°:
    平均反射率: {r15_mean:.2f}%
    标准差: {r15_std:.2f}%
    干涉条件: {'满足' if cond15 else '不满足'}
    
    结论:
    {'存在干涉现象' if cond10 and cond15 else '干涉现象不明显'}
    """
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('干涉条件分析')
    
    plt.tight_layout()
    plt.savefig('results/interference_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return peaks10, peaks15, smooth10, smooth15

def calc_thickness(df10, df15, peaks10, peaks15):
    n_si = 3.4
    theta10 = np.radians(10)
    theta15 = np.radians(15)
    
    results = {}
    
    if len(peaks10) > 1:
        spacing10 = np.diff(df10['wavenumber'].iloc[peaks10])
        avg_spacing10 = np.mean(spacing10)
        
        lambda_avg10 = 1 / (df10['wavenumber'].iloc[peaks10].mean()) * 10000
        thickness10 = lambda_avg10 / (2 * n_si * np.cos(theta10))
        
        results['10度'] = {
            'thickness': thickness10,
            'spacing': avg_spacing10,
            'lambda': lambda_avg10,
            'peaks': len(peaks10)
        }
    
    if len(peaks15) > 1:
        spacing15 = np.diff(df15['wavenumber'].iloc[peaks15])
        avg_spacing15 = np.mean(spacing15)
        
        lambda_avg15 = 1 / (df15['wavenumber'].iloc[peaks15].mean()) * 10000
        thickness15 = lambda_avg15 / (2 * n_si * np.cos(theta15))
        
        results['15度'] = {
            'thickness': thickness15,
            'spacing': avg_spacing15,
            'lambda': lambda_avg15,
            'peaks': len(peaks15)
        }
    
    return results

def plot_thickness(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    angles = list(results.keys())
    thicknesses = [results[angle]['thickness'] for angle in angles]
    
    bars = axes[0, 0].bar(angles, thicknesses, color=['blue', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel('厚度 (μm)')
    axes[0, 0].set_title('厚度计算结果')
    axes[0, 0].grid(True)
    
    for bar, t in zip(bars, thicknesses):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{t:.2f} μm', ha='center', va='bottom')
    
    spacings = [results[angle]['spacing'] for angle in angles]
    
    bars2 = axes[0, 1].bar(angles, spacings, color=['blue', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('平均间距 (cm⁻¹)')
    axes[0, 1].set_title('峰位间距对比')
    axes[0, 1].grid(True)
    
    for bar, s in zip(bars2, spacings):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{s:.1f}', ha='center', va='bottom')
    
    lambdas = [results[angle]['lambda'] for angle in angles]
    
    bars3 = axes[1, 0].bar(angles, lambdas, color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('平均波长 (μm)')
    axes[1, 0].set_title('波长对比')
    axes[1, 0].grid(True)
    
    for bar, l in zip(bars3, lambdas):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{l:.1f}', ha='center', va='bottom')
    
    if len(thicknesses) == 2:
        avg_t = np.mean(thicknesses)
        diff_t = abs(thicknesses[0] - thicknesses[1])
        consistency = 1 - diff_t / avg_t
        
        stats_text = f"""
        厚度计算结果:
        
        10°: {thicknesses[0]:.2f} μm
        15°: {thicknesses[1]:.2f} μm
        
        平均: {avg_t:.2f} μm
        差异: {diff_t:.2f} μm
        一致性: {consistency:.3f}
        
        峰位数量:
        10°: {results['10度']['peaks']}个
        15°: {results['15度']['peaks']}个
        
        结论:
        {'结果一致性好' if consistency > 0.95 else '结果存在差异'}
        """
    else:
        stats_text = f"""
        厚度计算结果:
        
        可用角度: {len(thicknesses)}个
        平均厚度: {np.mean(thicknesses):.2f} μm
        
        峰位数量:
        {angles[0]}: {results[angles[0]]['peaks']}个
        """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('结果统计')
    
    plt.tight_layout()
    plt.savefig('results/thickness_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("开始干涉分析...")
    
    df10, df15 = load_data()
    
    peaks10, peaks15, smooth10, smooth15 = analyze_patterns(df10, df15)
    
    results = calc_thickness(df10, df15, peaks10, peaks15)
    
    plot_thickness(results)
    
    with open('results/interference_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("分析完成")

if __name__ == "__main__":
    main()