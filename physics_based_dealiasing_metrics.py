"""
物理基礎反折錯指標
為mixed_patch_test_model.py提供缺少的物理指標函數
"""
import numpy as np
import pandas as pd
from dealiasing_success_metrics import (
    compute_dealiasing_success_metrics,
    print_correction_summary,
    visualize_correction_success,
    analyze_dealiasing_performance
)

def compute_physics_based_dealiasing_metrics(raw_vel, model_vel, gt_vel, nyquist_velocity,
                                           correction_threshold=0.5, correction_tolerance=0.3,
                                           output_prefix="physics", **kwargs):
    """
    計算基於物理的反折錯指標
    這是一個wrapper函數，使用現有的dealiasing_success_metrics

    Args:
        raw_vel: 原始速度場
        model_vel: 模型預測速度場
        gt_vel: 真值速度場
        nyquist_velocity: Nyquist速度
        correction_tolerance: 修正容忍度
    """
    print(f"🔬 計算物理基礎反折錯指標 (閾值={correction_threshold}, 容忍度={correction_tolerance})")

    # 直接使用現有的指標計算函數 - 修正參數名稱
    try:
        metrics = compute_dealiasing_success_metrics(
            raw_vel=raw_vel,
            pred_vel=model_vel,  # 使用正確的參數名
            gt_vel=gt_vel,
            nyq_val=nyquist_velocity,  # 使用正確的參數名
            threshold_factor=correction_threshold,  # 使用正確的參數名
            correction_tolerance=correction_tolerance
        )
        return metrics
    except Exception as e:
        print(f"⚠️ 物理基礎指標計算失敗: {e}")
        return {
            'correction_success_rate': 0.0,
            'false_correction_rate': 0.0,
            'total_corrections': 0,
            'successful_corrections': 0,
            'failed_corrections': 0,
            'need_correction_pixels': 0  # 添加缺少的鍵
        }

def print_physics_based_summary(metrics_dict, **kwargs):
    """
    打印物理基礎指標摘要
    """
    print("🔬 物理基礎反折錯指標摘要")
    print("=" * 50)

    for branch_name, metrics in metrics_dict.items():
        if isinstance(metrics, dict) and 'correction_success_rate' in metrics:
            success_rate = metrics['correction_success_rate'] * 100
            print(f"{branch_name:>20}: 成功率 {success_rate:.1f}%")

    print("=" * 50)

def visualize_physics_based_correction(raw_vel, gt_vel, model_outputs,
                                     correction_threshold=0.5, correction_tolerance=0.3,
                                     save_path=None, **kwargs):
    """
    可視化物理基礎修正結果
    使用現有的可視化函數
    """
    print("🎨 生成物理基礎修正可視化...")

    # 選擇一個主要分支進行可視化
    main_branch = None
    for branch_name, output_vel in model_outputs.items():
        if 'physics' in branch_name.lower() or 'constraint' in branch_name.lower():
            main_branch = branch_name
            break

    if not main_branch:
        main_branch = list(model_outputs.keys())[0] if model_outputs else None

    if main_branch:
        try:
            visualize_correction_success(
                raw_vel=raw_vel,
                gt_vel=gt_vel,
                model_vel=model_outputs[main_branch],
                correction_threshold=correction_threshold,
                correction_tolerance=correction_tolerance,
                save_path=save_path
            )
        except Exception as e:
            print(f"⚠️ 可視化生成失敗: {e}")
    else:
        print("⚠️ 沒有可用的分支進行可視化")

def compare_threshold_vs_physics_methods(raw_vel, gt_vel, threshold_output, physics_output,
                                       correction_threshold=0.5, correction_tolerance=0.3,
                                       **kwargs):
    """
    比較閾值方法與物理方法
    """
    print("⚖️ 比較閾值方法 vs 物理方法")

    methods = {
        'threshold_method': threshold_output,
        'physics_method': physics_output
    }

    comparison_results = compute_physics_based_dealiasing_metrics(
        raw_vel=raw_vel,
        gt_vel=gt_vel,
        model_outputs=methods,
        correction_threshold=correction_threshold,
        correction_tolerance=correction_tolerance,
        output_prefix="comparison"
    )

    print_physics_based_summary(comparison_results)

    return comparison_results

def analyze_physics_based_performance(metrics_dict, **kwargs):
    """
    分析物理基礎性能
    使用現有的性能分析函數
    """
    print("📊 物理基礎性能分析")

    # 調用現有的性能分析
    try:
        analyze_dealiasing_performance(metrics_dict)
    except Exception as e:
        print(f"⚠️ 性能分析失敗: {e}")

        # 簡化版分析
        print("📊 簡化性能分析:")
        best_branch = None
        best_score = 0

        for branch_name, metrics in metrics_dict.items():
            if isinstance(metrics, dict) and 'correction_success_rate' in metrics:
                score = metrics['correction_success_rate']
                if score > best_score:
                    best_score = score
                    best_branch = branch_name

        if best_branch:
            print(f"🏆 最佳分支: {best_branch} (成功率: {best_score*100:.1f}%)")
        else:
            print("⚠️ 無法確定最佳分支")