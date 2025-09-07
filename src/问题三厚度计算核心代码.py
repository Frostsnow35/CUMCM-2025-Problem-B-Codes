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

def load_and_analyze():
    df10 = pd.read_csv('results/附件3_10度.csv')
    df15 = pd.read_csv('results/附件4_15度.csv')
    
    df10.columns = ['wavenumber', 'reflectance']
    df15.columns = ['wavenumber', 'reflectance']
    
    print(f"数据范围: {df10['wavenumber'].min():.1f}-{df10['wavenumber'].max():.1f} cm⁻¹")
    print(f"10度反射率: {df10['reflectance'].min():.1f}-{df10['reflectance'].max():.1f}%")
    print(f"15度反射率: {df15['reflectance'].min():.1f}-{df15['reflectance'].max():.1f}%")
    
    r_range10 = df10['reflectance'].max() - df10['reflectance'].min()
    r_range15 = df15['reflectance'].max() - df15['reflectance'].min()
    
    print(f"10度变化范围: {r_range10:.1f}%")
    print(f"15度变化范围: {r_range15:.1f}%")
    
    smooth10 = savgol_filter(df10['reflectance'], 21, 3)
    smooth15 = savgol_filter(df15['reflectance'], 21, 3)
    
    peaks10, _ = find_peaks(smooth10, height=0.1, distance=50, prominence=0.05)
    valleys10, _ = find_peaks(-smooth10, height=0.1, distance=50, prominence=0.05)
    
    peaks15, _ = find_peaks(smooth15, height=0.1, distance=50, prominence=0.05)
    valleys15, _ = find_peaks(-smooth15, height=0.1, distance=50, prominence=0.05)
    
    print(f"10度: {len(peaks10)}峰, {len(valleys10)}谷")
    print(f"15度: {len(peaks15)}峰, {len(valleys15)}谷")
    
    return df10, df15, peaks10, valleys10, peaks15, valleys15

def calc_thickness(df10, df15, peaks10, valleys10, peaks15, valleys15):
    print("\n厚度计算:")
    
    n_si = 3.4
    cos10 = np.cos(np.radians(10))
    cos15 = np.cos(np.radians(15))
    
    if len(peaks10) > 1:
        sp10 = np.diff(df10['wavenumber'].iloc[peaks10])
        avg_sp10 = np.mean(sp10)
        print(f"10度峰间距: {avg_sp10:.1f} cm⁻¹")
        
        t10_1 = 1 / (2 * n_si * cos10 * avg_sp10) * 10000
        print(f"方法1-10度厚度: {t10_1:.2f} μm")
    
    if len(peaks15) > 1:
        sp15 = np.diff(df15['wavenumber'].iloc[peaks15])
        avg_sp15 = np.mean(sp15)
        print(f"15度峰间距: {avg_sp15:.1f} cm⁻¹")
        
        t15_1 = 1 / (2 * n_si * cos15 * avg_sp15) * 10000
        print(f"方法1-15度厚度: {t15_1:.2f} μm")
    
    if len(valleys10) > 1:
        sp_v10 = np.diff(df10['wavenumber'].iloc[valleys10])
        avg_sp_v10 = np.mean(sp_v10)
        print(f"10度谷间距: {avg_sp_v10:.1f} cm⁻¹")
        
        t10_2 = 1 / (2 * n_si * cos10 * avg_sp_v10) * 10000
        print(f"方法2-10度厚度: {t10_2:.2f} μm")
    
    if len(valleys15) > 1:
        sp_v15 = np.diff(df15['wavenumber'].iloc[valleys15])
        avg_sp_v15 = np.mean(sp_v15)
        print(f"15度谷间距: {avg_sp_v15:.1f} cm⁻¹")
        
        t15_2 = 1 / (2 * n_si * cos15 * avg_sp_v15) * 10000
        print(f"方法2-15度厚度: {t15_2:.2f} μm")
    
    print(f"\n干涉级次分析:")
    
    if len(peaks10) >= 3:
        w10 = df10['wavenumber'].iloc[peaks10[:3]]
        λ10 = 1 / w10 * 10000
        
        print(f"10度波长: {λ10.values}")
        
        for m in range(1, 6):
            t10_m = m * np.mean(λ10) / (2 * n_si * cos10)
            print(f"10度厚度(m={m}): {t10_m:.2f} μm")
    
    if len(peaks15) >= 3:
        w15 = df15['wavenumber'].iloc[peaks15[:3]]
        λ15 = 1 / w15 * 10000
        
        print(f"15度波长: {λ15.values}")
        
        for m in range(1, 6):
            t15_m = m * np.mean(λ15) / (2 * n_si * cos15)
            print(f"15度厚度(m={m}): {t15_m:.2f} μm")
    
    std10 = np.std(df10['reflectance'])
    std15 = np.std(df15['reflectance'])
    
    print(f"\n反射率标准差: 10度={std10:.2f}%, 15度={std15:.2f}%")
    
    if std10 > 5 and std15 > 5:
        print("存在明显干涉现象")
    else:
        print("干涉现象不明显")

def create_plot(df10, df15, peaks10, valleys10, peaks15, valleys15):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    smooth10 = savgol_filter(df10['reflectance'], 21, 3)
    smooth15 = savgol_filter(df15['reflectance'], 21, 3)
    
    axes[0, 0].plot(df10['wavenumber'], df10['reflectance'], 'b-', linewidth=1, label='10°')
    axes[0, 0].plot(df15['wavenumber'], df15['reflectance'], 'r-', linewidth=1, label='15°')
    axes[0, 0].set_xlabel('波数 (cm⁻¹)')
    axes[0, 0].set_ylabel('反射率 (%)')
    axes[0, 0].set_title('光谱对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(df10['wavenumber'], smooth10, 'b-', linewidth=1, label='10°平滑')
    axes[0, 1].scatter(df10['wavenumber'].iloc[peaks10], smooth10[peaks10], 
                color='red', s=50, marker='o', label=f'10°峰 ({len(peaks10)}个)')
    axes[0, 1].scatter(df10['wavenumber'].iloc[valleys10], smooth10[valleys10], 
                color='green', s=50, marker='v', label=f'10°谷 ({len(valleys10)}个)')
    
    axes[0, 1].plot(df15['wavenumber'], smooth15, 'r-', linewidth=1, label='15°平滑')
    axes[0, 1].scatter(df15['wavenumber'].iloc[peaks15], smooth15[peaks15], 
                color='red', s=50, marker='s', label=f'15°峰 ({len(peaks15)}个)')
    axes[0, 1].scatter(df15['wavenumber'].iloc[valleys15], smooth15[valleys15], 
                color='green', s=50, marker='^', label=f'15°谷 ({len(valleys15)}个)')
    
    axes[0, 1].set_xlabel('波数 (cm⁻¹)')
    axes[0, 1].set_ylabel('反射率 (%)')
    axes[0, 1].set_title('峰谷检测')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    if len(peaks10) > 1:
        sp10 = np.diff(df10['wavenumber'].iloc[peaks10])
        axes[1, 0].plot(range(len(sp10)), sp10, 'bo-', label='10°峰间距')
    
    if len(peaks15) > 1:
        sp15 = np.diff(df15['wavenumber'].iloc[peaks15])
        axes[1, 0].plot(range(len(sp15)), sp15, 'ro-', label='15°峰间距')
    
    axes[1, 0].set_xlabel('峰位序号')
    axes[1, 0].set_ylabel('间距 (cm⁻¹)')
    axes[1, 0].set_title('峰间距分析')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    analysis_text = ""
    
    axes[1, 1].text(0.1, 0.9, analysis_text, transform=axes[1, 1].transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('分析结果')
    
    plt.tight_layout()
    plt.savefig('results/thickness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("开始分析...")
    
    df10, df15, peaks10, valleys10, peaks15, valleys15 = load_and_analyze()
    
    calc_thickness(df10, df15, peaks10, valleys10, peaks15, valleys15)
    
    create_plot(df10, df15, peaks10, valleys10, peaks15, valleys15)
    
    print("\n分析完成")

if __name__ == "__main__":
    main()