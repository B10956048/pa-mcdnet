#!/usr/bin/env python3
"""
manual_weight_transfer.py — 產生部署權重 best_model_manual.h5

於 Stage 2 遷移學習(微調)完成後執行:重建模型架構、載入該微調模型的
best_model.h5 權重後再另存一次,使較新版本的 TensorFlow 也能載入
(避開子類別模型的層數不符錯誤)。Stage 1 的 best_model.h5 可直接載入,不需轉。

用法:
  python manual_weight_transfer.py \
      --input  results/stage2/best_model.h5 \
      --output results/stage2/best_model_manual.h5
  # --output 省略時,預設為輸入檔同目錄下的 best_model_manual.h5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制 CPU，不需要 GPU
import argparse
import fix_typing

import tensorflow as tf
import h5py
import numpy as np
from unet_model.dealias_mulit_v2_physics import VelocityDealiaser
from unet_model.feature_extraction_mulit_v2 import (
    create_downsampler,
    create_upsampler_cls,
    create_upsampler_reg
)

parser = argparse.ArgumentParser(
    description='Re-save fine-tuned weights as best_model_manual.h5')
parser.add_argument('--input', required=True,
                    help='來源 best_model.h5 (Stage 2 微調輸出)')
parser.add_argument('--output', default=None,
                    help='輸出路徑;省略時為輸入檔同目錄下的 best_model_manual.h5')
args = parser.parse_args()

transfer_path = args.input
output_path = args.output or os.path.join(
    os.path.dirname(transfer_path), 'best_model_manual.h5')

print("="*80)
print("手動轉移權重")
print("="*80)

print(f"\n輸入: {transfer_path}")
print(f"輸出: {output_path}")

# 步驟 1: 創建模型
print("\n步驟 1: 創建模型架構...")
extractor = create_downsampler(input_channels=1, start_neurons=32)
upsampler_cls = create_upsampler_cls(n_inputs=1, start_neurons=32, classes=6)
upsampler_reg = create_upsampler_reg(n_inputs=1, start_neurons=32)
model = VelocityDealiaser(extractor, upsampler_cls, upsampler_reg)

# 初始化
dummy_input = {
    'vel': tf.zeros((1, 1, 128, 128, 1), dtype=tf.float32),
    'nyq': tf.ones((1, 1), dtype=tf.float32)
}
_ = model(dummy_input, training=False)
print(f"✅ 模型已初始化，共 {len(model.weights)} 個權重")

# 步驟 2: 手動讀取權重並設置
print("\n步驟 2: 手動讀取並設置權重...")

def find_weight_in_h5(h5_file, weight_name):
    """在 H5 文件中查找權重"""
    # 移除 :0 後綴
    clean_name = weight_name.replace(':0', '')

    # 遞迴搜索所有可能的路徑
    def search_recursive(group, path=""):
        for key in group.keys():
            item = group[key]
            current_path = f"{path}/{key}" if path else key

            if isinstance(item, h5py.Dataset):
                # 匹配條件：路徑結尾匹配權重名稱
                if current_path.endswith(clean_name) or current_path.endswith(weight_name):
                    return item[()]

                # 也檢查是否部分匹配（去掉頂層組名）
                parts = clean_name.split('/')
                path_parts = current_path.split('/')
                if len(path_parts) >= len(parts):
                    if path_parts[-len(parts):] == parts:
                        return item[()]

            elif isinstance(item, h5py.Group):
                result = search_recursive(item, current_path)
                if result is not None:
                    return result

        return None

    return search_recursive(h5_file)

matched = 0
failed = 0

with h5py.File(transfer_path, 'r') as h5f:
    for weight in model.weights:
        weight_name = weight.name

        weight_data = find_weight_in_h5(h5f, weight_name)

        if weight_data is not None:
            if weight_data.shape == tuple(weight.shape):
                weight.assign(weight_data)
                matched += 1
            else:
                print(f"   ⚠️  形狀不匹配: {weight_name}")
                print(f"      期望: {weight.shape}, 實際: {weight_data.shape}")
                failed += 1
        else:
            print(f"   ❌ 找不到: {weight_name}")
            failed += 1

print(f"\n結果: 成功 {matched}/{len(model.weights)}, 失敗 {failed}")

if failed > 0:
    print(f"\n❌ 有 {failed} 個權重無法設置，中止保存")
    exit(1)

# 步驟 3: 保存
print(f"\n步驟 3: 保存新權重...")
model.save_weights(output_path, save_format='h5')
print(f"✅ 已保存: {output_path}")

# 步驟 4: 驗證
print(f"\n步驟 4: 驗證新權重...")
test_model = VelocityDealiaser(
    create_downsampler(input_channels=1, start_neurons=32),
    create_upsampler_cls(n_inputs=1, start_neurons=32, classes=6),
    create_upsampler_reg(n_inputs=1, start_neurons=32)
)
_ = test_model(dummy_input, training=False)

try:
    test_model.load_weights(output_path)
    print("✅ 新權重可以正常載入")
except Exception as e:
    print(f"❌ 新權重載入失敗: {e}")
    exit(1)

print("\n" + "="*80)
print("✅ 轉移完成！")
print(f"請使用: {output_path}")
print("="*80)
