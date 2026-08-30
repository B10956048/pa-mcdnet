#!/usr/bin/env python3
"""
時間平滑後處理工具（方案D）
利用前一時刻的預測結果平滑當前預測
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json


def load_csv_as_matrix(csv_path):
    """載入CSV並轉換為極座標矩陣"""
    df = pd.read_csv(csv_path)
    nyquist = df['Nyquist'].iloc[0]
    azimuths = sorted(df['Azimuth'].unique())
    ranges = sorted(df['Range'].unique())
    n_az, n_range = len(azimuths), len(ranges)
    velocity_matrix = np.full((n_az, n_range), np.nan)

    for _, row in df.iterrows():
        az_idx = azimuths.index(row['Azimuth'])
        r_idx = ranges.index(row['Range'])
        if row['Value'] != -999.0:
            velocity_matrix[az_idx, r_idx] = row['Value']

    return velocity_matrix, nyquist, azimuths, ranges


def temporal_smooth(vel_current, vel_previous, alpha=0.5):
    """
    時間平滑

    Parameters:
        vel_current: 當前時刻預測 (n_az, n_range)
        vel_previous: 前一時刻預測 (n_az, n_range)
        alpha: 平滑參數 (0-1)
               - alpha=1: 完全信任當前
               - alpha=0: 完全信任歷史
               - alpha=0.5: 平衡（推薦）

    Returns:
        vel_smoothed: 平滑後的預測
    """
    # 只在雙方都有效的位置進行平滑
    valid_mask = ~np.isnan(vel_current) & ~np.isnan(vel_previous)

    vel_smoothed = vel_current.copy()
    vel_smoothed[valid_mask] = (alpha * vel_current[valid_mask] +
                                (1 - alpha) * vel_previous[valid_mask])

    return vel_smoothed


def adaptive_temporal_smooth(vel_current, vel_previous, alpha_base=0.5):
    """
    自適應時間平滑：根據局部變化程度動態調整α

    在變化大的區域用較大的α（信任當前）
    在變化小的區域用較小的α（信任歷史）
    """
    valid_mask = ~np.isnan(vel_current) & ~np.isnan(vel_previous)

    # 計算局部變化幅度
    local_change = np.abs(vel_current - vel_previous)

    # 動態α：變化大→α大，變化小→α小
    # 使用sigmoid函數平滑過渡
    change_threshold = 3.0  # 3 m/s 作為參考點
    alpha_map = np.full_like(vel_current, alpha_base)

    # 在有效區域計算動態α
    # 變化小於閾值：α減小（更信任歷史）
    # 變化大於閾值：α增大（更信任當前）
    alpha_map[valid_mask] = alpha_base + 0.3 * np.tanh(
        (local_change[valid_mask] - change_threshold) / change_threshold
    )
    # 限制在 [0.2, 0.8] 範圍
    alpha_map = np.clip(alpha_map, 0.2, 0.8)

    # 平滑
    vel_smoothed = vel_current.copy()
    vel_smoothed[valid_mask] = (alpha_map[valid_mask] * vel_current[valid_mask] +
                                (1 - alpha_map[valid_mask]) * vel_previous[valid_mask])

    return vel_smoothed, alpha_map


def compute_metrics(pred, gt):
    """計算評估指標"""
    mask = ~np.isnan(pred) & ~np.isnan(gt)
    if not np.any(mask):
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan}

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    mae = np.mean(np.abs(pred_valid - gt_valid))
    corr = np.corrcoef(pred_valid, gt_valid)[0, 1] if len(pred_valid) > 1 else 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "corr": float(corr),
        "valid_pixels": int(np.sum(mask))
    }


def test_temporal_smoothing():
    """
    測試時間平滑效果
    使用真實連續時間點數據
    """
    print("="*80)
    print("測試時間平滑效果")
    print("="*80)

    # 測試數據：RCSL 2021-09-12 連續3個時間點
    data_dir = Path("/mnt/e/DealData/ALL")

    # 假設我們有這三個時間點的預測結果和GT
    # t0: 00:04 (作為歷史)
    # t1: 00:12 (當前，要優化的)
    # t2: 00:19 (未來)

    files = {
        't0_raw': data_dir / "RCSL.20210912.0004.bvel_raw.01.csv",
        't0_gt':  data_dir / "RCSL.20210912.0004.bvel_vdaqc.01.csv",
        't1_raw': data_dir / "RCSL.20210912.0012.bvel_raw.01.csv",
        't1_gt':  data_dir / "RCSL.20210912.0012.bvel_vdaqc.01.csv",
    }

    # 檢查文件存在
    for key, path in files.items():
        if not path.exists():
            print(f"❌ 文件不存在: {path}")
            return

    print("\n載入數據...")
    # 載入數據
    vel_t0_gt, _ = load_csv_as_matrix(files['t0_gt'])[:2]    # t0的GT（作為"完美預測"的代理）
    vel_t0_raw, _ = load_csv_as_matrix(files['t0_raw'])[:2]  # t0的RAW
    vel_t1_gt, _ = load_csv_as_matrix(files['t1_gt'])[:2]    # t1的真實值
    vel_t1_raw, _ = load_csv_as_matrix(files['t1_raw'])[:2]  # t1的原始觀測

    print(f"  t0 RAW形狀: {vel_t0_raw.shape}")
    print(f"  t0 GT形狀: {vel_t0_gt.shape}")
    print(f"  t1 RAW形狀: {vel_t1_raw.shape}")
    print(f"  t1 GT形狀: {vel_t1_gt.shape}")

    # 情境測試：
    # 假設 t0 已經處理好（用GT），t1 是RAW（待處理）
    # 這模擬了「前一時刻已經用模型處理過，當前時刻是新的RAW輸入」
    vel_t0_pred = vel_t0_gt.copy()  # t0的處理結果（用GT模擬好的預測）
    vel_t1_pred_initial = vel_t1_raw.copy()  # t1的初步狀態（RAW）

    # 計算baseline（不平滑）
    baseline_metrics = compute_metrics(vel_t1_pred_initial, vel_t1_gt)
    print(f"\nBaseline (RAW, 無平滑):")
    print(f"  RMSE={baseline_metrics['rmse']:.3f}, "
          f"MAE={baseline_metrics['mae']:.3f}, Corr={baseline_metrics['corr']:.4f}")

    print("\n測試不同α參數...")
    print("-"*80)

    # 測試不同α值
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = []

    for alpha in alphas:
        # 應用時間平滑
        vel_t1_smoothed = temporal_smooth(vel_t1_pred_initial, vel_t0_pred, alpha)

        # 計算指標
        metrics = compute_metrics(vel_t1_smoothed, vel_t1_gt)

        results.append({
            'alpha': alpha,
            **metrics
        })

        print(f"α={alpha:.1f}: RMSE={metrics['rmse']:.3f}, "
              f"MAE={metrics['mae']:.3f}, Corr={metrics['corr']:.4f}")

    # 找出最佳α
    best_idx = np.argmin([r['rmse'] for r in results])
    best_alpha = results[best_idx]['alpha']
    print(f"\n最佳α: {best_alpha} (RMSE={results[best_idx]['rmse']:.3f})")

    # 測試自適應平滑
    print("\n測試自適應平滑...")
    vel_t1_adaptive, alpha_map = adaptive_temporal_smooth(
        vel_t1_pred_initial, vel_t0_pred, alpha_base=0.5
    )
    metrics_adaptive = compute_metrics(vel_t1_adaptive, vel_t1_gt)
    print(f"自適應: RMSE={metrics_adaptive['rmse']:.3f}, "
          f"MAE={metrics_adaptive['mae']:.3f}, Corr={metrics_adaptive['corr']:.4f}")
    print(f"  α範圍: [{np.nanmin(alpha_map):.2f}, {np.nanmax(alpha_map):.2f}]")

    # 保存結果
    output_dir = Path("results/temporal_smoothing")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_summary = {
        "test_case": "RCSL.20210912 t0→t1",
        "fixed_alpha_results": results,
        "adaptive_result": {
            "alpha_base": 0.5,
            "alpha_min": float(np.nanmin(alpha_map)),
            "alpha_max": float(np.nanmax(alpha_map)),
            **metrics_adaptive
        },
        "best_alpha": float(best_alpha)
    }

    with open(output_dir / "temporal_smoothing_test.json", 'w') as f:
        json.dump(result_summary, f, indent=2)

    print(f"\n✅ 結果已保存到: {output_dir}")
    print("="*80)

    return result_summary


if __name__ == "__main__":
    test_temporal_smoothing()
