#!/usr/bin/env python3
"""
反折錯修正成功率評估指標
提供直觀的像素點修正統計，比RMSE更能反映實際修正效果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional

def compute_dealiasing_success_metrics(raw_vel: np.ndarray,
                                     pred_vel: np.ndarray,
                                     gt_vel: np.ndarray,
                                     nyq_val: float,
                                     threshold_factor: float = 0.5,
                                     use_fold_based: bool = True) -> Dict:
    """
    計算反折錯修正成功率指標

    Parameters:
    -----------
    raw_vel : np.ndarray (H,W)
        原始速度場
    pred_vel : np.ndarray (H,W)
        預測速度場
    gt_vel : np.ndarray (H,W)
        真實速度場 (ground truth)
    nyq_val : float
        Nyquist速度
    threshold_factor : float, default=0.5
        判斷需要修正的閾值係數 (相對於nyq_val)
        例如: 0.5 表示差異超過0.5×nyq_val的像素需要修正
    use_fold_based : bool, default=True
        是否使用基於fold的嚴格判斷（推薦True）
        True: 成功 = fold判斷完全正確（無容忍度）
        False: 成功 = 誤差小於閾值（舊版，不推薦）
    
    Returns:
    --------
    dict : 包含詳細修正統計的字典
    {
        'total_valid_pixels': int,           # 總有效像素數
        'need_correction_pixels': int,       # 需要修正的像素數
        'corrected_pixels': int,             # 成功修正的像素數
        'correction_success_rate': float,    # 修正成功率 (0-1)
        'overcorrection_pixels': int,        # 過度修正的像素數
        'undercorrection_pixels': int,       # 修正不足的像素數
        'perfect_pixels': int,               # 原本就正確的像素數
        'improvement_ratio': float,          # 整體改進比例
        'detailed_stats': dict              # 詳細統計
    }
    """
    
    # 創建有效像素掩碼
    valid_mask = (~np.isnan(raw_vel) & 
                  ~np.isnan(pred_vel) & 
                  ~np.isnan(gt_vel))
    
    if not np.any(valid_mask):
        return _empty_metrics()
    
    # 提取有效像素
    raw_valid = raw_vel[valid_mask]
    pred_valid = pred_vel[valid_mask]
    gt_valid = gt_vel[valid_mask]
    
    total_valid = len(raw_valid)
    
    # 設定閾值
    correction_threshold = threshold_factor * nyq_val

    # 1. 識別需要修正的像素 (RAW與GT差異較大)
    raw_error = np.abs(raw_valid - gt_valid)
    need_correction_mask = raw_error > correction_threshold
    need_correction_pixels = np.sum(need_correction_mask)

    # 2. 計算fold（折錯週期數）
    # fold = round((vel - raw) / (2 × Vnyq))
    pred_fold = np.round((pred_valid - raw_valid) / (2 * nyq_val))
    gt_fold = np.round((gt_valid - raw_valid) / (2 * nyq_val))

    # 3. 評估預測修正效果
    pred_error = np.abs(pred_valid - gt_valid)

    # 4. 分類像素點
    if use_fold_based:
        # ✅ 正確方法：基於fold判斷（無容忍度）
        # 成功修正: 原本需要修正且fold判斷完全正確
        fold_correct_mask = (pred_fold == gt_fold)
        successful_correction_mask = need_correction_mask & fold_correct_mask
        corrected_pixels = np.sum(successful_correction_mask)
    else:
        # ❌ 舊方法：基於誤差容忍度（不推薦）
        success_threshold = 0.3 * nyq_val
        successful_correction_mask = (need_correction_mask &
                                      (pred_error <= success_threshold))
        corrected_pixels = np.sum(successful_correction_mask)
    
    # 過度修正和修正不足的判斷
    if use_fold_based:
        # 基於fold判斷
        fold_wrong_mask = (pred_fold != gt_fold)

        # 過度修正: 原本不需要修正但fold判斷錯誤
        overcorrection_mask = ~need_correction_mask & fold_wrong_mask
        overcorrection_pixels = np.sum(overcorrection_mask)

        # 修正不足: 需要修正但fold判斷錯誤
        undercorrection_mask = need_correction_mask & fold_wrong_mask
        undercorrection_pixels = np.sum(undercorrection_mask)

        # 本來就好的像素
        perfect_mask = ~need_correction_mask & fold_correct_mask
        perfect_pixels = np.sum(perfect_mask)
    else:
        # 基於誤差容忍度（舊方法）
        success_threshold = 0.3 * nyq_val

        overcorrection_mask = (~need_correction_mask &
                               (pred_error > raw_error) &
                               (pred_error > success_threshold))
        overcorrection_pixels = np.sum(overcorrection_mask)

        undercorrection_mask = (need_correction_mask &
                                (pred_error > success_threshold) &
                                (pred_error >= raw_error))
        undercorrection_pixels = np.sum(undercorrection_mask)

        perfect_mask = (~need_correction_mask &
                        (pred_error <= success_threshold))
        perfect_pixels = np.sum(perfect_mask)
    
    # 4. 計算成功率指標
    correction_success_rate = (corrected_pixels / need_correction_pixels
                              if need_correction_pixels > 0 else 1.0)

    # Clean pixels (原本不需修正的像素)
    clean_pixels = total_valid - need_correction_pixels

    # False Positive Rate: 原本正確的像素被錯誤修正的比例
    false_positive_rate = (overcorrection_pixels / clean_pixels
                           if clean_pixels > 0 else 0.0)

    # False Negative Rate: 需要修正但沒修正好的比例 (= 1 - SR)
    false_negative_rate = (undercorrection_pixels / need_correction_pixels
                           if need_correction_pixels > 0 else 0.0)

    # Precision: 所有被修正的像素中，真正需要修正的比例
    total_corrected_attempt = corrected_pixels + overcorrection_pixels
    precision = (corrected_pixels / total_corrected_attempt
                 if total_corrected_attempt > 0 else 1.0)

    # 整體改進比例 (減少誤差的像素佔比)
    improvement_mask = pred_error < raw_error
    improvement_pixels = np.sum(improvement_mask)
    improvement_ratio = improvement_pixels / total_valid

    # 5. 詳細統計
    detailed_stats = {
        'correction_threshold': correction_threshold,
        'nyquist_velocity': nyq_val,
        'use_fold_based': use_fold_based,
        'raw_error_mean': np.mean(raw_error),
        'raw_error_std': np.std(raw_error),
        'pred_error_mean': np.mean(pred_error),
        'pred_error_std': np.std(pred_error),
        'error_reduction_mean': np.mean(raw_error - pred_error),
        'aliased_pixels_ratio': need_correction_pixels / total_valid,
        'successful_repair_ratio': corrected_pixels / total_valid,
        'overcorrection_ratio': overcorrection_pixels / total_valid,
        'undercorrection_ratio': undercorrection_pixels / total_valid,
        'fold_accuracy': np.sum(pred_fold == gt_fold) / total_valid  # 整體fold準確率
    }

    return {
        'total_valid_pixels': total_valid,
        'need_correction_pixels': need_correction_pixels,
        'clean_pixels': clean_pixels,
        'corrected_pixels': corrected_pixels,
        'correction_success_rate': correction_success_rate,  # Recall on aliased pixels
        'false_positive_rate': false_positive_rate,          # Over-correction rate on clean pixels
        'false_negative_rate': false_negative_rate,          # Miss rate on aliased pixels
        'precision': precision,                              # Of all corrections, fraction that were needed
        'overcorrection_pixels': overcorrection_pixels,
        'undercorrection_pixels': undercorrection_pixels,
        'perfect_pixels': perfect_pixels,
        'improvement_pixels': improvement_pixels,
        'improvement_ratio': improvement_ratio,
        'detailed_stats': detailed_stats
    }

def _empty_metrics() -> Dict:
    """返回空的指標字典"""
    return {
        'total_valid_pixels': 0,
        'need_correction_pixels': 0,
        'clean_pixels': 0,
        'corrected_pixels': 0,
        'correction_success_rate': 0.0,
        'false_positive_rate': 0.0,
        'false_negative_rate': 0.0,
        'precision': 1.0,
        'overcorrection_pixels': 0,
        'undercorrection_pixels': 0,
        'perfect_pixels': 0,
        'improvement_pixels': 0,
        'improvement_ratio': 0.0,
        'detailed_stats': {}
    }

def create_correction_success_map(raw_vel: np.ndarray,
                                pred_vel: np.ndarray,
                                gt_vel: np.ndarray,
                                nyq_val: float,
                                threshold_factor: float = 0.5,
                                correction_tolerance: float = 0.3) -> np.ndarray:
    """
    創建修正成功分類圖
    
    Returns:
    --------
    np.ndarray : 分類標籤圖
        0: 無效像素 (NaN)
        1: 本來就正確 (perfect)
        2: 成功修正 (successful correction)
        3: 修正不足 (under-correction)
        4: 過度修正 (over-correction)
    """
    
    # 初始化分類圖
    correction_map = np.zeros(raw_vel.shape, dtype=np.int32)
    
    # 有效像素掩碼
    valid_mask = (~np.isnan(raw_vel) & 
                  ~np.isnan(pred_vel) & 
                  ~np.isnan(gt_vel))
    
    if not np.any(valid_mask):
        return correction_map
    
    # 設定閾值
    correction_threshold = threshold_factor * nyq_val
    success_threshold = correction_tolerance * nyq_val
    
    # 計算誤差
    raw_error = np.abs(raw_vel - gt_vel)
    pred_error = np.abs(pred_vel - gt_vel)
    
    # 需要修正的像素
    need_correction = raw_error > correction_threshold
    
    # 分類
    # 1: 本來就正確
    perfect = valid_mask & (~need_correction) & (pred_error <= success_threshold)
    correction_map[perfect] = 1
    
    # 2: 成功修正
    successful = valid_mask & need_correction & (pred_error <= success_threshold)
    correction_map[successful] = 2
    
    # 3: 修正不足
    under_corrected = (valid_mask & need_correction & 
                      (pred_error > success_threshold) & 
                      (pred_error >= raw_error))
    correction_map[under_corrected] = 3
    
    # 4: 過度修正
    over_corrected = (valid_mask & (~need_correction) & 
                     (pred_error > raw_error) & 
                     (pred_error > success_threshold))
    correction_map[over_corrected] = 4
    
    return correction_map

def visualize_correction_success(raw_vel: np.ndarray,
                               pred_vel: np.ndarray,
                               gt_vel: np.ndarray,
                               nyq_val: float,
                               save_path: Optional[str] = None,
                               title: str = "Dealiasing Correction Analysis"):
    """
    可視化修正成功分析
    """
    
    # 計算指標
    metrics = compute_dealiasing_success_metrics(raw_vel, pred_vel, gt_vel, nyq_val)
    correction_map = create_correction_success_map(raw_vel, pred_vel, gt_vel, nyq_val)
    
    # 創建圖像
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{title}\nCorrection Success Rate: {metrics["correction_success_rate"]:.1%}', 
                fontsize=16)
    
    # 速度場對比
    vmin, vmax = -80, 80
    
    im1 = axes[0,0].imshow(raw_vel, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0,0].set_title('Raw Velocity')
    axes[0,0].set_xlabel('Range')
    axes[0,0].set_ylabel('Azimuth')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    
    im2 = axes[0,1].imshow(pred_vel, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0,1].set_title('Predicted Velocity')
    axes[0,1].set_xlabel('Range')
    axes[0,1].set_ylabel('Azimuth')
    plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    
    im3 = axes[0,2].imshow(gt_vel, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0,2].set_title('Ground Truth')
    axes[0,2].set_xlabel('Range')
    axes[0,2].set_ylabel('Azimuth')
    plt.colorbar(im3, ax=axes[0,2], shrink=0.8)
    
    # 修正成功分類圖
    correction_colors = ['black', 'green', 'blue', 'orange', 'red']
    correction_labels = ['Invalid', 'Perfect', 'Success', 'Under-corrected', 'Over-corrected']
    
    im4 = axes[1,0].imshow(correction_map, cmap=plt.cm.colors.ListedColormap(correction_colors), 
                          vmin=0, vmax=4, aspect='auto')
    axes[1,0].set_title('Correction Classification')
    axes[1,0].set_xlabel('Range')
    axes[1,0].set_ylabel('Azimuth')
    
    # 自定義colorbar
    cbar4 = plt.colorbar(im4, ax=axes[1,0], shrink=0.8, ticks=range(5))
    cbar4.ax.set_yticklabels(correction_labels)
    
    # 誤差對比
    raw_error = np.abs(raw_vel - gt_vel)
    pred_error = np.abs(pred_vel - gt_vel)
    
    im5 = axes[1,1].imshow(raw_error, cmap='Reds', vmin=0, vmax=30, aspect='auto')
    axes[1,1].set_title('Raw Error (|RAW - GT|)')
    axes[1,1].set_xlabel('Range')
    axes[1,1].set_ylabel('Azimuth')
    plt.colorbar(im5, ax=axes[1,1], shrink=0.8)
    
    im6 = axes[1,2].imshow(pred_error, cmap='Reds', vmin=0, vmax=30, aspect='auto')
    axes[1,2].set_title('Prediction Error (|PRED - GT|)')
    axes[1,2].set_xlabel('Range')
    axes[1,2].set_ylabel('Azimuth')
    plt.colorbar(im6, ax=axes[1,2], shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 修正成功分析圖已保存: {save_path}")
    
    return fig, metrics

def print_correction_summary(metrics: Dict, title: str = "Dealiasing Correction Summary"):
    """
    打印修正成功率總結
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    print(f"📊 總體統計:")
    print(f"   有效像素總數: {metrics['total_valid_pixels']:,}")
    print(f"   需要修正像素: {metrics['need_correction_pixels']:,} ({metrics['detailed_stats']['aliased_pixels_ratio']:.1%})")
    print(f"   本來就正確像素: {metrics['perfect_pixels']:,}")
    
    print(f"\n🎯 修正效果:")
    print(f"   成功修正: {metrics['corrected_pixels']:,}")
    print(f"   修正成功率 (Recall): {metrics['correction_success_rate']:.1%}")
    print(f"   Precision:           {metrics['precision']:.1%}")
    print(f"   整體改進比例: {metrics['improvement_ratio']:.1%}")

    print(f"\n⚠️  問題統計:")
    print(f"   修正不足 (FNR): {metrics['undercorrection_pixels']:,}  ({metrics['false_negative_rate']:.1%})")
    print(f"   過度修正 (FPR): {metrics['overcorrection_pixels']:,}  ({metrics['false_positive_rate']:.1%})")
    
    print(f"\n📈 誤差統計:")
    stats = metrics['detailed_stats']
    print(f"   RAW平均誤差: {stats['raw_error_mean']:.2f} m/s")
    print(f"   預測平均誤差: {stats['pred_error_mean']:.2f} m/s")
    print(f"   平均誤差減少: {stats['error_reduction_mean']:.2f} m/s")
    
    print(f"\n⚙️  閾值設定:")
    print(f"   需修正閾值: {stats['correction_threshold']:.1f} m/s")
    print(f"   成功容忍度: {stats['success_threshold']:.1f} m/s")
    print(f"   Nyquist速度: {stats['nyquist_velocity']:.1f} m/s")
    
    print(f"{'='*50}\n")

# 使用示例函數
def analyze_dealiasing_performance(raw_vel, pred_vel, gt_vel, nyq_val, 
                                 title="Dealiasing Performance Analysis",
                                 save_dir=None):
    """
    完整的反折錯性能分析
    
    Parameters:
    -----------
    raw_vel, pred_vel, gt_vel : np.ndarray
        速度場數據
    nyq_val : float
        Nyquist速度
    title : str
        分析標題
    save_dir : str, optional
        保存目錄
    
    Returns:
    --------
    dict : 完整的分析結果
    """
    
    # 計算指標
    metrics = compute_dealiasing_success_metrics(raw_vel, pred_vel, gt_vel, nyq_val)
    
    # 打印總結
    print_correction_summary(metrics, title)
    
    # 可視化
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "correction_analysis.png")
    else:
        save_path = None
    
    fig, _ = visualize_correction_success(raw_vel, pred_vel, gt_vel, nyq_val, 
                                        save_path=save_path, title=title)
    
    return metrics, fig