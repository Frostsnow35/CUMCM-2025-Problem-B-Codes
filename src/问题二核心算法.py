#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks, savgol_filter
import json
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('results', exist_ok=True)

def find_peaks_adv(w, r, method='multi'):
    smoothed = savgol_filter(r, 21, 3)
    
    if method == 'multi':
        std_val = np.std(smoothed)
        thresholds = [std_val * 0.1, std_val * 0.3, std_val * 0.5]
        
        all_peaks = []
        for t in thresholds:
            p, _ = find_peaks(smoothed, prominence=t, distance=15)
            all_peaks.extend(p)
        
        peaks = np.unique(all_peaks)
        
    elif method == 'adaptive':
        window = 200
        local_thresh = []
        
        for i in range(0, len(smoothed), window//2):
            end = min(i + window, len(smoothed))
            local_std = np.std(smoothed[i:end])
            local_thresh.extend([local_std * 0.3] * (end - i))
        
        p_found = []
        for idx in range(1, len(smoothed)-1):
            if (smoothed[idx] > smoothed[idx-1] and 
                smoothed[idx] > smoothed[idx+1] and 
                smoothed[idx] > local_thresh[idx]):
                p_found.append(idx)
        
        peaks = np.array(p_found)
        
    else:
        prom = np.std(smoothed) * 0.4
        p, _ = find_peaks(smoothed, prominence=prom, distance=20)
        peaks = p
    
    quality = []
    for p_idx in peaks:
        start = max(0, p_idx-10)
        end = min(len(smoothed), p_idx+10)
        prom_val = smoothed[p_idx] - np.mean(smoothed[start:end])
        quality.append(prom_val)
    
    if len(peaks) > 8:
        q_thresh = np.percentile(quality, 70)
        good_peaks = peaks[quality >= q_thresh]
    else:
        good_peaks = peaks
    
    v_prom = np.std(smoothed) * 0.3
    v, _ = find_peaks(-smoothed, prominence=v_prom*0.8, distance=15)
    
    return good_peaks, v, w[good_peaks], w[v], smoothed

def calc_spacing(peak_pos, method='weighted'):
    if len(peak_pos) < 3:
        return np.nan, np.nan, np.array([])
    
    spacings = []
    
    if method == 'weighted':
        weights = []
        
        for i in range(len(peak_pos)):
            for j in range(i+1, len(peak_pos)):
                sp = peak_pos[j] - peak_pos[i]
                if sp > 0:
                    w = 1.0 / (j - i)
                    spacings.append(sp)
                    weights.append(w)
        
        if not spacings:
            return np.nan, np.nan, np.array([])
        
        spacings = np.array(spacings)
        weights = np.array(weights)
        
        sorted_idx = np.argsort(spacings)
        sorted_sp = spacings[sorted_idx]
        sorted_w = weights[sorted_idx]
        
        cum_w = np.cumsum(sorted_w)
        med_idx = np.searchsorted(cum_w, cum_w[-1] / 2)
        opt_spacing = sorted_sp[med_idx]
        
        mad_val = np.sqrt(np.average((spacings - opt_spacing)**2, weights=weights))
        
    elif method == 'frequency':
        for i in range(len(peak_pos)):
            for j in range(i+1, len(peak_pos)):
                sp = peak_pos[j] - peak_pos[i]
                if sp > 0:
                    spacings.append(sp)
        
        if not spacings:
            return np.nan, np.nan, np.array([])
        
        hist, bins = np.histogram(spacings, bins=50)
        max_idx = np.argmax(hist)
        opt_spacing = (bins[max_idx] + bins[max_idx + 1]) / 2
        
        mad_val = np.median(np.abs(np.array(spacings) - opt_spacing))
        
    else:
        for i in range(len(peak_pos)):
            for j in range(i+1, len(peak_pos)):
                sp = peak_pos[j] - peak_pos[i]
                if sp > 0:
                    spacings.append(sp)
        
        if not spacings:
            return np.nan, np.nan, np.array([])
        
        spacings = np.array(spacings)
        opt_spacing = np.median(spacings)
        mad_val = np.median(np.abs(spacings - opt_spacing))
    
    return opt_spacing, mad_val, np.array(spacings)

def n_disp(n0, w, B=0.01, C=1000):
    return n0 + B * w**2 / (w**2 - C)

def thickness_disp(n0, B, C, dv, theta, ref_w=1000):
    n_ref = n_disp(n0, ref_w, B, C)
    theta_r = np.arcsin(np.sin(np.radians(theta)) / n_ref)
    d = 1 / (2 * n_ref * dv * np.cos(theta_r))
    return d * 10000

def optimize_params(dv10, dv15):
    def obj_func(params):
        n0, B, C = params
        
        d10 = thickness_disp(n0, B, C, dv10, 10)
        d15 = thickness_disp(n0, B, C, dv15, 15)
        
        diff = abs(d10 - d15)
        penalty = 0
        
        n_ref = n_disp(n0, 1000, B, C)
        if n_ref < 2.0 or n_ref > 3.5:
            penalty += 1000 * abs(n_ref - 2.75)
        
        if B < 0 or B > 0.05:
            penalty += 1000 * abs(B - 0.01)
        
        if C < 800 or C > 1500:
            penalty += 1000 * abs(C - 1000)
        
        return diff + penalty
    
    bounds = [(2.0, 3.5), (0.0, 0.05), (800, 1500)]
    
    result = differential_evolution(obj_func, bounds, seed=42, maxiter=1000, popsize=15)
    
    if result.success:
        n0, B, C = result.x
        d10 = thickness_disp(n0, B, C, dv10, 10)
        d15 = thickness_disp(n0, B, C, dv15, 15)
        
        consistency = abs(d10 - d15) / ((d10 + d15) / 2)
        
        return {
            'n0': n0, 'B': B, 'C': C,
            'd_10': d10, 'd_15': d15,
            'consistency': consistency,
            'success': True
        }
    
    return {'success': False}

def analyze_all(s10, R10, s15, R15):
    p10, v10, ps10, vs10, Rs10 = find_peaks_adv(s10, R10, 'multi')
    p15, v15, ps15, vs15, Rs15 = find_peaks_adv(s15, R15, 'multi')
    
    if len(ps10) < 3 or len(ps15) < 3:
        p10, v10, ps10, vs10, Rs10 = find_peaks_adv(s10, R10, 'conservative')
        p15, v15, ps15, vs15, Rs15 = find_peaks_adv(s15, R15, 'conservative')
    
    sp10, mad10, sps10 = calc_spacing(ps10, 'weighted')
    sp15, mad15, sps15 = calc_spacing(ps15, 'weighted')
    
    if np.isnan(sp10) or np.isnan(sp15):
        sp10, mad10, sps10 = calc_spacing(ps10, 'robust')
        sp15, mad15, sps15 = calc_spacing(ps15, 'robust')
    
    c10 = np.cos(np.radians(10))
    c15 = np.cos(np.radians(15))
    th_ratio = c10 / c15
    exp_ratio = sp10 / sp15
    ratio_err = abs(th_ratio - exp_ratio) / th_ratio * 100
    
    opt_result = optimize_params(sp10, sp15)
    
    if not opt_result['success']:
        n_vals = np.linspace(2.5, 3.5, 200)
        best_cons = float('inf')
        best_res = None
        
        for n in n_vals:
            d10 = thickness_disp(n, 0, 1000, sp10, 10)
            d15 = thickness_disp(n, 0, 1000, sp15, 15)
            cons = abs(d10 - d15) / ((d10 + d15) / 2)
            
            if cons < best_cons:
                best_cons = cons
                best_res = {
                    'n0': n, 'B': 0, 'C': 1000,
                    'd_10': d10, 'd_15': d15,
                    'consistency': cons
                }
        
        opt_result = best_res
    
    return {
        'sp10': sp10,
        'sp15': sp15,
        'ratio_err': ratio_err,
        'opt': opt_result,
        'p10': len(p10),
        'p15': len(p15),
        'ps10': ps10,
        'ps15': ps15
    }

def plot_results(s10, R10, s15, R15, results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    p10, _, ps10, _, Rs10 = find_peaks_adv(s10, R10, 'multi')
    p15, _, ps15, _, Rs15 = find_peaks_adv(s15, R15, 'multi')
    
    axes[0,0].plot(s10, R10*100, 'b-', alpha=0.3)
    axes[0,0].plot(s10, Rs10*100, 'b-', linewidth=2)
    axes[0,0].plot(ps10, Rs10[p10]*100, 'ro', markersize=8)
    axes[0,0].set_xlabel('波数 (cm⁻¹)')
    axes[0,0].set_ylabel('反射率 (%)')
    axes[0,0].set_title('10° 峰位检测')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(s15, R15*100, 'r-', alpha=0.3)
    axes[0,1].plot(s15, Rs15*100, 'r-', linewidth=2)
    axes[0,1].plot(ps15, Rs15[p15]*100, 'ro', markersize=8)
    axes[0,1].set_xlabel('波数 (cm⁻¹)')
    axes[0,1].set_ylabel('反射率 (%)')
    axes[0,1].set_title('15° 峰位检测')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].hist(results['ps10'], bins=20, alpha=0.7, color='blue')
    axes[0,2].hist(results['ps15'], bins=20, alpha=0.7, color='red')
    axes[0,2].set_xlabel('波数 (cm⁻¹)')
    axes[0,2].set_ylabel('频次')
    axes[0,2].set_title('峰位分布')
    axes[0,2].grid(True, alpha=0.3)
    
    n_vals = np.linspace(2.0, 3.5, 100)
    t10_vals = []
    t15_vals = []
    
    for n in n_vals:
        d10 = thickness_disp(n, 0, 1000, results['sp10'], 10)
        d15 = thickness_disp(n, 0, 1000, results['sp15'], 15)
        t10_vals.append(d10)
        t15_vals.append(d15)
    
    axes[1,0].plot(n_vals, t10_vals, 'b-', linewidth=2)
    axes[1,0].plot(n_vals, t15_vals, 'r-', linewidth=2)
    axes[1,0].axvline(results['opt']['n0'], color='g', linestyle='--', alpha=0.7)
    axes[1,0].set_xlabel('折射率 n')
    axes[1,0].set_ylabel('厚度 (μm)')
    axes[1,0].set_title('厚度变化')
    axes[1,0].grid(True, alpha=0.3)
    
    angles = ['10°', '15°']
    thicknesses = [results['opt']['d_10'], results['opt']['d_15']]
    
    bars = axes[1,1].bar(angles, thicknesses, color=['blue', 'red'], alpha=0.7)
    axes[1,1].set_ylabel('厚度 (μm)')
    axes[1,1].set_title('厚度估计')
    axes[1,1].grid(True, alpha=0.3)
    
    for bar, t in zip(bars, thicknesses):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{t:.2f} μm', ha='center', va='bottom')
    
    cons_data = [results['opt']['consistency'] * 1000]
    axes[1,2].bar(['一致性'], cons_data, color='green', alpha=0.7)
    axes[1,2].set_ylabel('一致性 (‰)')
    axes[1,2].set_title('一致性指标')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].text(0, cons_data[0] + 0.1, f'{cons_data[0]:.2f}‰', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        df10 = pd.read_csv('results/clean_附件1.csv')
        df15 = pd.read_csv('results/clean_附件2.csv')
        
        s10 = df10['wavenumber_cm-1'].values
        R10 = df10['reflectance_detrended_%'].values / 100
        
        s15 = df15['wavenumber_cm-1'].values
        R15 = df15['reflectance_detrended_%'].values / 100
        
    except:
        return
    
    results = analyze_all(s10, R10, s15, R15)
    
    plot_results(s10, R10, s15, R15, results)
    
    final_results = {
        'spacing': {
            'sp10': float(results['sp10']),
            'sp15': float(results['sp15']),
            'ratio_err': float(results['ratio_err'])
        },
        'thickness': {
            'n0': float(results['opt']['n0']),
            'B': float(results['opt']['B']),
            'C': float(results['opt']['C']),
            't10': float(results['opt']['d_10']),
            't15': float(results['opt']['d_极']),
            'avg_t': float((results['opt']['d_10'] + results['opt']['d_15']) / 2),
            'consistency': float(results['opt']['consistency'])
        },
        'peaks': {
            'p10': int(results['p10']),
            'p15': int(results['p15']),
            'diff': abs(results['p10'] - results['p15'])
        }
    }
    
    with open('results/results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"峰数: 10°={results['p10']}, 15°={results['p15']}")
    print(f"间距: 10°={results['sp10']:.2f}, 15°={results['sp15']:.2f}")
    print(f"误差: {results['ratio_err']:.2f}%")
    print(f"n0: {results['opt']['n0']:.4f}, B: {results['opt']['B']:.6f}, C: {results['opt']['C']:.1f}")
    print(f"厚度: 10°={results['opt']['d_10']:.2f}, 15°={results['opt']['d_15']:.2f}")
    print(f"平均: {(results['opt']['d_10'] + results['opt']['d_15'])/2:.2f}")
    print(f"一致性: {results['opt']['consistency']:.6f}")

if __name__ == "__main__":
    main()