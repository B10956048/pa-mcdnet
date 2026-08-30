'''
🎛️ 使用方式

  1. 不使用數據增強 (預設)

  python dealiasing_pre_patch_v2.py --mode reorganized
  # 或明確指定
  python dealiasing_pre_patch_v2.py --mode reorganized --enable_augmentation=False

  2. 啟用數據增強 + 自動強度

  # 輕度增強 (2倍數據)
  python dealiasing_pre_patch_v2.py --mode reorganized --enable_augmentation --augment_strength light

  # 中度增強 (3倍數據)
  python dealiasing_pre_patch_v2.py --mode reorganized --enable_augmentation --augment_strength medium

  # 重度增強 (6倍數據)
  python dealiasing_pre_patch_v2.py --mode reorganized --enable_augmentation --augment_strength heavy

  3. 手動指定增強數量

  # 自定義增強變體數量
  python dealiasing_pre_patch_v2.py --mode reorganized --augment_variations 3

  4. 智能自動增強 🤖

  # 自動根據數據量選擇最佳增強程度
  python dealiasing_pre_patch_v2.py --mode reorganized --auto_augment

  基本使用 (預設啟用方位角旋轉)

  python dealiasing_pre_patch_v2.py --mode reorganized

  只使用原有矩陣增強

  python dealiasing_pre_patch_v2.py --mode reorganized \
      --enable_augmentation \
      --augment_variations=2 \
      --enable_azimuth_rotation=False

  只使用新的方位角旋轉增強

  python dealiasing_pre_patch_v2.py --mode reorganized \
      --enable_azimuth_rotation \
      --max_azimuth_variations=3 \
      --enable_augmentation=False

'''
import os
import glob
import numpy as np
import pandas as pd
import random
import h5py
from tqdm import tqdm
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json

# 設置隨機種子
SEED = 46
random.seed(SEED)
np.random.seed(SEED)

###############################################################################
# 讀取 & 處理 CSV 相關函式
###############################################################################
def load_csv_as_matrix(csv_path: str):
    """載入CSV文件為矩陣，優化版本"""
    data = pd.read_csv(csv_path)
    data['Value'] = data['Value'].replace(-999.0, np.nan)
    data = data.sort_values(by=['Azimuth', 'Range']).reset_index(drop=True)
    azimuths = data['Azimuth'].unique()
    ranges = data['Range'].unique()
    n_az = len(azimuths)
    n_rad = len(ranges)
    
    # 使用更高效的方法創建網格
    velocity_grid = np.full((n_az, n_rad), np.nan)
    
    # 創建索引字典以加速查找
    az_indices = {az: i for i, az in enumerate(azimuths)}
    rad_indices = {r: i for i, r in enumerate(ranges)}
    
    # 使用numpy高效操作
    az_idx = np.array([az_indices[az] for az in data['Azimuth']])
    rad_idx = np.array([rad_indices[r] for r in data['Range']])
    
    velocity_grid[az_idx, rad_idx] = data['Value'].values
    nyquist = data['Nyquist'].unique()
    
    return velocity_grid, nyquist

def get_all_valid_files(data_folder, pattern='*bvel_raw.*.csv', exclude_folder=None):
    """獲取所有有效的文件對，支援排除特定資料夾"""
    # 尋找所有符合模式的文件
    raw_files = sorted(glob.glob(os.path.join(data_folder, '**', pattern), recursive=True))
    
    # 準備排除檔案清單
    excluded_files = set()
    if exclude_folder:
        print(f"🚫 排除資料夾: {exclude_folder}")
        if os.path.exists(exclude_folder):
            # 掃描排除資料夾中的所有CSV檔案
            exclude_pattern = os.path.join(exclude_folder, "**", "*.csv")
            exclude_csv_files = glob.glob(exclude_pattern, recursive=True)
            
            for csv_file in exclude_csv_files:
                # 提取檔案基本名稱（去除路徑和擴展名）
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                excluded_files.add(base_name)
            
            print(f"   發現 {len(excluded_files)} 個要排除的檔案")
            if len(excluded_files) <= 10:
                print(f"   排除檔案: {sorted(list(excluded_files))}")
            else:
                print(f"   排除檔案範例: {sorted(list(excluded_files))[:5]}...")
        else:
            print(f"   ⚠️ 排除資料夾不存在，將跳過排除功能")
    
    valid_files = []
    excluded_count = 0
    
    # 檢查是否存在對應的真實值文件
    for rf in raw_files:
        # 檢查是否需要排除
        if exclude_folder:
            base_name = os.path.splitext(os.path.basename(rf))[0]
            if base_name in excluded_files:
                excluded_count += 1
                continue
        
        gt_f = rf.replace('bvel_raw', 'bvel_vdaqc')
        if os.path.exists(gt_f):
            valid_files.append((rf, gt_f))
    
    if exclude_folder and excluded_count > 0:
        print(f"   實際排除檔案: {excluded_count}")
    
    return valid_files

def get_reorganized_files(reorganized_data_path, split='train', aliased_ratio=0.7, exclude_folder=None):
    """
    從temporal_reorganized_data讀取已分割和分類的資料
    
    Parameters:
    -----------
    reorganized_data_path : str
        temporal_reorganized_data資料夾路徑
    split : str
        數據分割 ('train', 'val', 'test')
    aliased_ratio : float
        摺錯patch的比例 (0.0-1.0)
    exclude_folder : str or None
        要排除的資料夾路徑 (防止資料洩漏)，如 "path/to/test_cases"
    
    Returns:
    --------
    mixed_files : list
        混合的檔案對列表 [(raw_file, gt_file, patch_type), ...]
    """
    split_path = os.path.join(reorganized_data_path, split)
    
    # 準備排除檔案清單
    excluded_files = set()
    if exclude_folder:
        print(f"🚫 排除資料夾: {exclude_folder}")
        if os.path.exists(exclude_folder):
            # 掃描排除資料夾中的所有CSV檔案
            exclude_pattern = os.path.join(exclude_folder, "**", "*.csv")
            exclude_csv_files = glob.glob(exclude_pattern, recursive=True)
            
            for csv_file in exclude_csv_files:
                # 提取檔案基本名稱（去除路徑和擴展名）
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                excluded_files.add(base_name)
            
            print(f"   發現 {len(excluded_files)} 個要排除的檔案")
            if len(excluded_files) <= 10:
                print(f"   排除檔案: {sorted(list(excluded_files))}")
            else:
                print(f"   排除檔案範例: {sorted(list(excluded_files))[:5]}...")
        else:
            print(f"   ⚠️ 排除資料夾不存在，將跳過排除功能")
    
    # 讀取aliased資料
    aliased_path = os.path.join(split_path, 'aliased')
    aliased_raw_pattern = os.path.join(aliased_path, 'raw', '*.csv')
    aliased_raw_files = sorted(glob.glob(aliased_raw_pattern))
    aliased_files = []
    excluded_count = 0
    
    for rf in aliased_raw_files:
        # 從raw檔案路徑推導gt檔案路徑
        rf_basename = os.path.basename(rf)
        base_name = os.path.splitext(rf_basename)[0]
        
        # 檢查是否需要排除
        if exclude_folder and base_name in excluded_files:
            excluded_count += 1
            continue
            
        gt_basename = rf_basename.replace('bvel_raw', 'bvel_vdaqc')
        gt_f = os.path.join(aliased_path, 'gt', gt_basename)
        
        if os.path.exists(gt_f):
            aliased_files.append((rf, gt_f, 'aliased'))
    
    # 讀取clean資料
    clean_path = os.path.join(split_path, 'clean')
    clean_raw_pattern = os.path.join(clean_path, 'raw', '*.csv')
    clean_raw_files = sorted(glob.glob(clean_raw_pattern))
    clean_files = []
    
    for rf in clean_raw_files:
        rf_basename = os.path.basename(rf)
        base_name = os.path.splitext(rf_basename)[0]
        
        # 檢查是否需要排除
        if exclude_folder and base_name in excluded_files:
            excluded_count += 1
            continue
            
        gt_basename = rf_basename.replace('bvel_raw', 'bvel_vdaqc')
        gt_f = os.path.join(clean_path, 'gt', gt_basename)
        
        if os.path.exists(gt_f):
            clean_files.append((rf, gt_f, 'clean'))
    
    # 計算需要採樣的數量
    total_aliased = len(aliased_files)
    total_clean = len(clean_files)
    
    if aliased_ratio >= 1.0:
        # 全部使用aliased
        selected_aliased = aliased_files
        selected_clean = []
    elif aliased_ratio <= 0.0:
        # 全部使用clean
        selected_aliased = []
        selected_clean = clean_files
    else:
        # 混合採樣
        target_aliased = int(total_aliased * aliased_ratio)
        target_clean = int(total_clean * (1.0 - aliased_ratio))
        
        # 確保不超過可用數量
        actual_aliased = min(target_aliased, total_aliased)
        actual_clean = min(target_clean, total_clean)
        
        # 隨機採樣
        random.shuffle(aliased_files)
        random.shuffle(clean_files)
        
        selected_aliased = aliased_files[:actual_aliased]
        selected_clean = clean_files[:actual_clean]
    
    # 合併並打亂順序
    mixed_files = selected_aliased + selected_clean
    random.shuffle(mixed_files)
    
    print(f"📊 {split}集檔案採樣結果:")
    print(f"   摺錯檔案: {len(selected_aliased)} / {total_aliased}")
    print(f"   乾淨檔案: {len(selected_clean)} / {total_clean}")
    if exclude_folder and excluded_count > 0:
        print(f"   排除檔案: {excluded_count}")
    print(f"   總計: {len(mixed_files)} 檔案對")
    print(f"   實際比例: {len(selected_aliased)/(len(mixed_files)+1e-8):.1%} aliased")
    
    return mixed_files

def auto_zero_pad(data_2d, layers=4, fill_value=None):
    """自動填充為2^n大小"""
    factor = 2 ** layers
    H, W = data_2d.shape
    remainder_h = H % factor
    remainder_w = W % factor
    pad_h = (factor - remainder_h) if remainder_h != 0 else 0
    pad_w = (factor - remainder_w) if remainder_w != 0 else 0
    if fill_value is None:
        fill_value = np.nan
    padded = np.pad(data_2d, ((0, pad_h), (0, pad_w)), 
                    mode='constant', constant_values=fill_value)
    return padded, pad_h, pad_w

###############################################################################
# 資料增強方法：隨機翻轉與旋轉
###############################################################################
def azimuth_rotation_augment(df, rotation_angle):
    """
    方位角旋轉資料增強
    在原始CSV資料階段進行，保持雷達資料的物理特性
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含 ['Azimuth', 'Range', 'Nyquist', 'Velocity'] 的雷達資料
    rotation_angle : float
        旋轉角度（度）
    
    Returns:
    --------
    df_rotated : pd.DataFrame
        方位角旋轉後的資料框
    """
    df_rotated = df.copy()
    df_rotated['Azimuth'] = (df_rotated['Azimuth'] + rotation_angle) % 360
    return df_rotated

def create_azimuth_rotated_file(original_csv_path, rotation_angle, temp_dir="temp_rotated"):
    """
    創建方位角旋轉的臨時CSV文件
    
    Parameters:
    -----------
    original_csv_path : str
        原始CSV文件路徑
    rotation_angle : float
        旋轉角度
    temp_dir : str
        臨時目錄
        
    Returns:
    --------
    rotated_csv_path : str
        旋轉後的臨時CSV文件路徑
    """
    # 確保臨時目錄存在
    os.makedirs(temp_dir, exist_ok=True)
    
    # 載入原始數據
    df = pd.read_csv(original_csv_path)
    
    # 執行方位角旋轉
    df_rotated = azimuth_rotation_augment(df, rotation_angle)
    
    # 生成臨時文件路徑
    base_name = os.path.splitext(os.path.basename(original_csv_path))[0]
    rotated_filename = f"{base_name}_rot{rotation_angle}.csv"
    rotated_csv_path = os.path.join(temp_dir, rotated_filename)
    
    # 保存旋轉後的數據
    df_rotated.to_csv(rotated_csv_path, index=False)
    
    return rotated_csv_path

def augment_sample(raw, gt, label):
    """對 (H,W) or (H,W,1) 的影像做同步增強。"""
    # 隨機水平翻轉
    if np.random.rand() < 0.5:
        raw = np.fliplr(raw)
        gt = np.fliplr(gt)
        label = np.fliplr(label)
    # 隨機垂直翻轉
    if np.random.rand() < 0.5:
        raw = np.flipud(raw)
        gt = np.flipud(gt)
        label = np.flipud(label)
    # 隨機旋轉 0, 90, 180, 270 度
    k = np.random.randint(0, 4)
    raw = np.rot90(raw, k)
    gt = np.rot90(gt, k)
    label = np.rot90(label, k)
    return raw, gt, label

###############################################################################
# 計算 alias label - 新版，標籤0和3互換
###############################################################################
def compute_alias_label_swapped(raw_2d, gt_2d, nyq):
    """
    計算標籤（0和3互換版本）
    
    - 標籤 0: 無效像素或超過 ±2 cycle (原標籤3)
    - 標籤 1: -2 cycle 的 alias
    - 標籤 2: -1 cycle 的 alias
    - 標籤 3: 沒有 alias (n=0) (原標籤0)
    - 標籤 4: +1 cycle 的 alias
    - 標籤 5: +2 cycle 的 alias
    """
    H, W = raw_2d.shape
    label_2d = np.full((H,W), 0, dtype=np.int32)  # 默認為0 (無效)
    valid_mask = ~np.isnan(raw_2d) & ~np.isnan(gt_2d) & (abs(nyq)>1e-6)
    if not np.any(valid_mask):
        return label_2d
    
    diff = gt_2d - raw_2d
    n_float = diff/(2*nyq)
    n_rounded = np.round(n_float)
    n_rounded[~valid_mask] = 999
    
    # 使用向量化操作代替循環
    label_2d = np.zeros_like(n_rounded, dtype=np.int32)
    
    # 一次性設置所有標籤
    label_2d[(n_rounded == -2)] = 1
    label_2d[(n_rounded == -1)] = 2
    label_2d[(n_rounded == 0)] = 3
    label_2d[(n_rounded == 1)] = 4
    label_2d[(n_rounded == 2)] = 5
    # 其他值(包括999)保持為0
    
    return label_2d

###############################################################################
# 智能patch類型判斷
###############################################################################
def determine_patch_type_from_data(vel_data, gt_data, aliased_ratio, random_state=None):
    """
    基於原始數據智能判斷patch類型，實現mixed patch策略
    
    Parameters:
    -----------
    vel_data : np.ndarray
        原始velocity數據 (H, W)
    gt_data : np.ndarray  
        ground truth數據 (H, W)
    aliased_ratio : float
        期望的aliased patch比例 (0.0-1.0)
    random_state : np.random.RandomState, optional
        隨機狀態，用於可重現的結果
        
    Returns:
    --------
    str : 'aliased' 或 'clean'
        判斷的patch類型
    """
    if random_state is None:
        random_state = np.random
    
    # 計算RAW和GT的差異程度
    valid_mask = (~np.isnan(vel_data)) & (~np.isnan(gt_data))
    if not np.any(valid_mask):
        # 如果沒有有效數據，隨機判斷
        return 'aliased' if random_state.random() < aliased_ratio else 'clean'
    
    # 計算有效區域的速度差異
    vel_valid = vel_data[valid_mask] 
    gt_valid = gt_data[valid_mask]
    
    # 使用多個指標判斷是否為aliased
    # 1. 平均絕對差異
    mean_abs_diff = np.mean(np.abs(gt_valid - vel_valid))
    
    # 2. 相關係數  
    correlation = np.corrcoef(vel_valid.flatten(), gt_valid.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
        
    # 3. 值域差異
    vel_range = np.ptp(vel_valid)  # peak to peak
    gt_range = np.ptp(gt_valid)
    range_diff_ratio = abs(gt_range - vel_range) / max(vel_range, gt_range, 1e-6)
    
    # 綜合判斷：如果差異大且相關性低，更可能是aliased
    aliasing_score = (mean_abs_diff / 10.0) + (1.0 - correlation) + (range_diff_ratio * 0.5)
    aliasing_score = np.clip(aliasing_score, 0.0, 1.0)
    
    # 結合數據特徵和期望比例進行概率判斷
    # 如果數據看起來很aliased，增加被判為aliased的概率
    adjusted_aliased_prob = aliased_ratio * 0.7 + aliasing_score * 0.3
    
    return 'aliased' if random_state.random() < adjusted_aliased_prob else 'clean'

def assign_patch_types_to_files(file_pairs, aliased_ratio, random_seed=SEED):
    """
    為文件對分配patch類型，確保整體比例符合aliased_ratio
    
    Parameters:
    -----------
    file_pairs : list of tuples
        (raw_file, gt_file) 元組列表
    aliased_ratio : float
        期望的aliased patch比例
    random_seed : int
        隨機種子
        
    Returns:
    --------
    list of tuples : (raw_file, gt_file, patch_type)
        包含patch類型的文件元組列表
    """
    random_state = np.random.RandomState(random_seed)
    
    total_files = len(file_pairs)
    target_aliased_count = int(total_files * aliased_ratio)
    target_clean_count = total_files - target_aliased_count
    
    # 為每個文件pair快速預判類型（基於文件名或簡單採樣）
    file_scores = []
    for i, (raw_f, gt_f) in enumerate(file_pairs):
        # 這裡可以加入基於文件名的啟發式判斷
        # 或者讀取文件的一小部分進行快速判斷
        # 暫時使用隨機+權重的方式
        score = random_state.random()
        file_scores.append((i, score))
    
    # 按分數排序，分數高的優先分配為aliased
    file_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 分配類型
    result_files = []
    aliased_assigned = 0
    clean_assigned = 0
    
    for i, score in file_scores:
        raw_f, gt_f = file_pairs[i]
        
        if aliased_assigned < target_aliased_count:
            patch_type = 'aliased'
            aliased_assigned += 1
        else:
            patch_type = 'clean' 
            clean_assigned += 1
            
        result_files.append((raw_f, gt_f, patch_type))
    
    print(f"   📊 分配結果: {aliased_assigned} aliased, {clean_assigned} clean")
    return result_files

###############################################################################
# 從文件提取patch - 使用新的標籤系統（0和3互換）
###############################################################################
def extract_mixed_patches_from_file(file_tuple, layers=4, patch_size=128, 
                                    num_patches_per_file=5, min_alias_pixels=10, 
                                    augment_variations=1, enable_azimuth_rotation=True,
                                    azimuth_rotation_angles=None, max_azimuth_variations=2):
    """
    從單個文件中提取混合patch，支援aliased和clean兩種類型
    使用新的標籤系統（0和3互換）
    
    Parameters:
    -----------
    file_tuple : tuple
        (raw_file_path, ground_truth_file_path, patch_type)元組
        patch_type: 'aliased' 或 'clean'
    """
    if len(file_tuple) == 3:
        raw_f, gt_f, patch_type = file_tuple
    else:
        # 向後相容舊格式
        raw_f, gt_f = file_tuple
        patch_type = 'aliased'  # 預設為aliased
    
    # 記錄文件來源
    source_info = {
        'raw_file': os.path.basename(raw_f),
        'gt_file': os.path.basename(gt_f)
    }
    
    try:
        # 準備資料版本清單（原始 + 方位角旋轉增強）
        data_versions = []
        temp_files_to_cleanup = []
        
        # 添加原始版本
        data_versions.append(('original', raw_f, gt_f))
        
        # 添加方位角旋轉版本
        if enable_azimuth_rotation and max_azimuth_variations > 0:
            # 使用提供的角度列表或預設值
            if azimuth_rotation_angles is None:
                rotation_angles = [45, 90, 135, 180, 225, 270, 315]
            else:
                rotation_angles = azimuth_rotation_angles
            
            selected_angles = np.random.choice(rotation_angles, 
                                             min(max_azimuth_variations, len(rotation_angles)), 
                                             replace=False)
            
            temp_dir = "temp_azimuth_rotated"
            for angle in selected_angles:
                try:
                    rotated_raw = create_azimuth_rotated_file(raw_f, angle, temp_dir)
                    rotated_gt = create_azimuth_rotated_file(gt_f, angle, temp_dir)
                    data_versions.append((f'azimuth_rot_{angle}deg', rotated_raw, rotated_gt))
                    temp_files_to_cleanup.extend([rotated_raw, rotated_gt])
                except Exception as e:
                    print(f"⚠️ 方位角旋轉失敗 {angle}°: {e}")
        
        all_patches = []
        
        # 處理每個資料版本
        for version_name, current_raw_f, current_gt_f in data_versions:
            try:
                data_vel, data_nyq = load_csv_as_matrix(current_raw_f)  # (H,W)
                gt_vel, _ = load_csv_as_matrix(current_gt_f)
                nyq_val = np.nanmean(data_nyq)
                
                # 計算 label，使用新的互換標籤函數
                label_2d = compute_alias_label_swapped(data_vel, gt_vel, nyq_val)
                if np.all(label_2d == 0):  # 標籤0現在表示無效
                    # 全部是無效像素，跳過這個版本
                    print(f"⚠️ {version_name}: 全部為無效像素，跳過")
                    continue
        
                # zero-pad
                data_vel_pad, _, _ = auto_zero_pad(data_vel, layers=layers, fill_value=np.nan)
                gt_vel_pad, _, _ = auto_zero_pad(gt_vel, layers=layers, fill_value=np.nan)
                label_pad, _, _ = auto_zero_pad(label_2d, layers=layers, fill_value=0)  # 填充0，表示無效
                H_pad, W_pad = data_vel_pad.shape
                
                version_patches = []
                extracted_count = 0
                attempts = 0
                max_attempts = num_patches_per_file * 10
        
                while extracted_count < num_patches_per_file and attempts < max_attempts:
                    attempts += 1
                    if H_pad < patch_size or W_pad < patch_size:
                        # 影像比 patch_size 還小, 就整張當 patch
                        top, left = 0, 0
                        ph, pw = H_pad, W_pad
                    else:
                        top = np.random.randint(0, H_pad - patch_size + 1)
                        left = np.random.randint(0, W_pad - patch_size + 1)
                        ph, pw = patch_size, patch_size
                    
                    patch_vel = data_vel_pad[top:top+ph, left:left+pw]
                    patch_gtvel = gt_vel_pad[top:top+ph, left:left+pw]
                    patch_label = label_pad[top:top+ph, left:left+pw]
            
                    # 根據patch_type進行不同的篩選策略
                    alias_mask = np.isin(patch_label, [1,2,4,5])
                    alias_count = np.sum(alias_mask)
                    clean_mask = (patch_label == 3)  # 標籤3為無摺錯
                    clean_count = np.sum(clean_mask)
                    
                    if patch_type == 'aliased':
                        # 對於摺錯patch，需要足夠的alias pixels
                        if alias_count < min_alias_pixels:
                            continue
                    elif patch_type == 'clean':
                        # 對於乾淨patch，需要大部分像素是乾淨的
                        total_valid_pixels = np.sum(patch_label > 0)  # 排除標籤0(無效像素)
                        if total_valid_pixels == 0:
                            continue
                        clean_ratio = clean_count / total_valid_pixels
                        if clean_ratio < 0.7:  # 至少70%是乾淨像素
                            continue
                    
                    # 根據patch類型設定不同的目標
                    if patch_type == 'aliased':
                        # 摺錯patch：目標是GT修復後的結果
                        target_vel = patch_gtvel[..., None].astype(np.float32)
                    elif patch_type == 'clean':
                        # 乾淨patch：目標是保持原始RAW (identity mapping)
                        target_vel = patch_vel[..., None].astype(np.float32)
                    else:
                        # 預設行為
                        target_vel = patch_gtvel[..., None].astype(np.float32)
                    
                    # 添加版本信息到source_info
                    version_source_info = source_info.copy()
                    version_source_info['augment_type'] = version_name
                    
                    # 加入原始patch（每個版本只做一次矩陣增強，方位角增強已在CSV階段完成）
                    patch_dict = {
                        'vel': patch_vel[..., None].astype(np.float32),      # (H,W,1) RAW輸入
                        'nyq': np.array([nyq_val], dtype=np.float32),        # (1,) Nyquist
                        'alias_label': patch_label.astype(np.int32),         # (H,W) 標籤
                        'gt_vel': patch_gtvel[..., None].astype(np.float32), # (H,W,1) GT參考
                        'target_vel': target_vel,                            # (H,W,1) 實際訓練目標
                        'patch_type': patch_type,                            # str patch類型標記
                        'source_info': version_source_info                   # dict 來源信息
                    }
                    version_patches.append(patch_dict)
                    
                    # 如果是原始版本，才進行矩陣增強（避免重複增強）
                    if version_name == 'original' and augment_variations > 0:
                        for aug_idx in range(augment_variations):
                            aug_vel, aug_gtvel, aug_label = augment_sample(
                                patch_vel.copy(), patch_gtvel.copy(), patch_label.copy()
                            )
                            
                            # 對增強數據也設定相應的target
                            if patch_type == 'aliased':
                                aug_target_vel = aug_gtvel[..., None].astype(np.float32)
                            elif patch_type == 'clean':
                                aug_target_vel = aug_vel[..., None].astype(np.float32)
                            else:
                                aug_target_vel = aug_gtvel[..., None].astype(np.float32)
                            
                            # 添加矩陣增強信息
                            matrix_source_info = source_info.copy()
                            matrix_source_info['augment_type'] = f'matrix_aug_{aug_idx}'
                            
                            aug_patch = {
                                'vel': aug_vel[..., None].astype(np.float32),
                                'nyq': np.array([nyq_val], dtype=np.float32),
                                'alias_label': aug_label.astype(np.int32),
                                'gt_vel': aug_gtvel[..., None].astype(np.float32),
                                'target_vel': aug_target_vel,
                                'patch_type': patch_type,
                                'source_info': matrix_source_info
                            }
                            version_patches.append(aug_patch)
                    
                    extracted_count += 1
                
                # 將當前版本的patches加入總列表
                all_patches.extend(version_patches)
                print(f"✅ {version_name}: 提取了 {len(version_patches)} 個patches")
                
            except Exception as e:
                print(f"⚠️ 處理版本 {version_name} 失敗: {e}")
                continue
        
        # 清理臨時文件
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"⚠️ 清理臨時文件失敗 {temp_file}: {e}")
        
        # 清理臨時目錄（如果為空）
        temp_dir = "temp_azimuth_rotated"
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"⚠️ 清理臨時目錄失敗: {e}")
        
        return all_patches
    
    except Exception as e:
        print(f"處理文件 {raw_f} 時發生錯誤: {e}")
        return []

###############################################################################
# 智能數據分割 - 考慮雷達站和時間
###############################################################################
def stratified_split_files(all_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=SEED):
    """
    考慮雷達站和時間的智能分割
    
    Parameters:
    -----------
    all_files : list
        所有有效文件對 (raw_file, gt_file)
    train_ratio, val_ratio, test_ratio : float
        訓練、驗證和測試集的比例（和應該為1）
    
    Returns:
    --------
    train_files, val_files, test_files : list
        分割後的文件列表
    """
    # 設置隨機種子
    random.seed(random_seed)
    
    # 驗證比例之和是否為1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必須為1"
    
    # 按雷達站分組
    site_files = {}
    for raw_f, gt_f in all_files:
        # 提取雷達站信息 (格式: RCWF.20210912.2352...)
        filename = os.path.basename(raw_f)
        parts = filename.split('.')
        
        if len(parts) >= 2:
            site = parts[0]  # RCWF, RCKT, etc.
            
            if site not in site_files:
                site_files[site] = []
            
            site_files[site].append((raw_f, gt_f))
    
    # 分配訓練、驗證和測試文件
    train_files = []
    val_files = []
    test_files = []
    
    # 對每個雷達站的文件進行分割
    for site, files in site_files.items():
        # 按時間排序
        files.sort()
        
        # 計算每個集合的數量
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        # n_test = n_files - n_train - n_val
        
        # 隨機打亂但保持種子固定，確保可重複性
        random.shuffle(files)
        
        # 分配文件
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train+n_val])
        test_files.extend(files[n_train+n_val:])
    
    print(f"數據分割完成:")
    print(f"  訓練集: {len(train_files)} 文件")
    print(f"  驗證集: {len(val_files)} 文件")
    print(f"  測試集: {len(test_files)} 文件")
    
    return train_files, val_files, test_files

###############################################################################
# 並行處理函數
###############################################################################
def process_file_batch(batch_files, **kwargs):
    """處理一批文件 (支援混合patch)"""
    all_patches = []
    for file_tuple in batch_files:
        patches = extract_mixed_patches_from_file(file_tuple, **kwargs)
        all_patches.extend(patches)
    return all_patches

###############################################################################
# 生成並保存混合patches - 使用temporal_reorganized_data
###############################################################################
def generate_mixed_patches_from_reorganized_data(reorganized_data_path, output_h5_path,
                                                 patch_size=128, num_patches_per_file=5,
                                                 min_alias_pixels=10, augment_variations=2,
                                                 aliased_ratio=0.7, random_seed=SEED, 
                                                 num_workers=None, exclude_folder=None,
                                                 enable_azimuth_rotation=True,
                                                 azimuth_rotation_angles=None,
                                                 max_azimuth_variations=2):
    """
    從temporal_reorganized_data生成混合patch並保存到H5文件
    
    Parameters:
    -----------
    reorganized_data_path : str
        temporal_reorganized_data資料夾路徑
    output_h5_path : str
        輸出H5文件路徑
    aliased_ratio : float
        摺錯patch的比例 (0.0-1.0)
    exclude_folder : str or None
        要排除的資料夾路徑，防止資料洩漏 (如測試案例資料夾)
    """
    # 設置隨機種子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 如果未指定工作進程數，使用CPU核心數減1
    if num_workers is None:
        num_workers = min(8, max(1, multiprocessing.cpu_count() - 1))
    
    print(f"🚀 開始生成混合patch (aliased: {aliased_ratio:.1%})")
    print(f"📁 數據來源: {reorganized_data_path}")
    print(f"💾 輸出路徑: {output_h5_path}")
    if enable_azimuth_rotation:
        print(f"🔄 方位角旋轉增強已啟用，角度: {azimuth_rotation_angles}, 最大變體數: {max_azimuth_variations}")
    
    # 獲取各分割的混合檔案
    train_files = get_reorganized_files(reorganized_data_path, 'train', aliased_ratio, exclude_folder)
    val_files = get_reorganized_files(reorganized_data_path, 'val', aliased_ratio, exclude_folder)
    test_files = get_reorganized_files(reorganized_data_path, 'test', aliased_ratio, exclude_folder)
    
    # 創建H5文件和目錄
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)
    
    # 提取參數
    extract_kwargs = {
        'layers': 4,
        'patch_size': patch_size,
        'num_patches_per_file': num_patches_per_file,
        'min_alias_pixels': min_alias_pixels,
        'augment_variations': augment_variations,
        'enable_azimuth_rotation': enable_azimuth_rotation,
        'azimuth_rotation_angles': azimuth_rotation_angles,
        'max_azimuth_variations': max_azimuth_variations
    }
    
    with h5py.File(output_h5_path, 'w') as h5f:
        # 創建訓練、驗證和測試數據集
        train_group = h5f.create_group('train')
        val_group = h5f.create_group('val')
        test_group = h5f.create_group('test')
        
        print(f"🔧 使用 {num_workers} 個工作進程並行處理...")
        
        # 處理各數據集
        for split_name, files, group in [('train', train_files, train_group), 
                                       ('val', val_files, val_group), 
                                       ('test', test_files, test_group)]:
            
            if not files:
                print(f"⚠️  {split_name}集沒有文件，跳過")
                continue
                
            print(f"🔄 處理{split_name}集 ({len(files)}個檔案)...")
            
            # 分批處理
            batch_size = max(1, min(10, len(files) // num_workers))
            batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
            
            count = 0
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_batch = {
                    executor.submit(process_file_batch, batch, **extract_kwargs): batch
                    for batch in batches
                }
                
                for future in tqdm(as_completed(future_to_batch), total=len(batches), 
                                 desc=f"處理{split_name}批次"):
                    try:
                        patches = future.result()
                        
                        # 保存patch到H5文件 - 支援混合patch格式
                        for p in patches:
                            patch_group = group.create_group(f'patch_{count}')
                            patch_group.create_dataset('vel', data=p['vel'])
                            patch_group.create_dataset('nyq', data=p['nyq'])
                            patch_group.create_dataset('alias_label', data=p['alias_label'])
                            patch_group.create_dataset('gt_vel', data=p['gt_vel'])
                            patch_group.create_dataset('target_vel', data=p['target_vel'])  # 新增
                            
                            # 保存patch類型為屬性
                            patch_group.attrs['patch_type'] = p['patch_type']
                            
                            # 保存來源信息 - 修正格式以支援洩漏檢測
                            # 使用raw_file作為主要來源標識
                            source_file = p['source_info'].get('raw_file', 'unknown')
                            patch_group.create_dataset('source', data=source_file.encode('utf-8'))
                            
                            # 同時保持原有詳細資訊
                            source_group = patch_group.create_group('source_detail')
                            for key, value in p['source_info'].items():
                                source_group.attrs[key] = value
                            
                            count += 1
                            
                    except Exception as e:
                        print(f"❌ 處理批次時發生錯誤: {e}")
            
            print(f"✅ 已處理 {count} 個{split_name}patches")
        
        # 保存元數據
        metadata = h5f.create_group('metadata')
        metadata.attrs['patch_size'] = patch_size
        metadata.attrs['min_alias_pixels'] = min_alias_pixels
        metadata.attrs['aliased_ratio'] = aliased_ratio
        metadata.attrs['created_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['data_source'] = 'temporal_reorganized_data'
        metadata.attrs['mixed_patch_version'] = 'v1.0'
    
    print(f"🎉 混合patch生成完成!")
    print(f"💾 數據已保存到 {output_h5_path}")
    
    return output_h5_path

###############################################################################
# 生成並保存patches - 並行優化版本 (舊版本，保留相容性)
###############################################################################
def generate_and_save_patches_parallel(data_folder, output_h5_path, 
                                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                     patch_size=128, num_patches_per_file=5, 
                                     min_alias_pixels=10, augment_variations=2,
                                     aliased_ratio=0.7, random_seed=SEED, num_workers=None,
                                     exclude_folder=None,         
                                     enable_azimuth_rotation=True,
                                     azimuth_rotation_angles=None,
                                     max_azimuth_variations=2):
    """
    並行版本：從所有文件生成混合patch並保存到H5文件
    
    Parameters:
    -----------
    data_folder : str
        原始數據文件夾
    output_h5_path : str
        輸出H5文件路徑
    train_ratio, val_ratio, test_ratio : float
        數據分割比例
    aliased_ratio : float
        摺錯patch的比例 (0.0-1.0)
    num_workers : int, optional
        並行處理的工作進程數，預設為None（使用CPU核心數）
    """
    # 設置隨機種子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 如果未指定工作進程數，使用CPU核心數減1
    if num_workers is None:
        num_workers = min(8, max(1, multiprocessing.cpu_count() - 1))
    
    # 獲取所有有效文件
    print(f"查找所有有效文件...")
    all_files = get_all_valid_files(data_folder, exclude_folder=exclude_folder)
    print(f"找到 {len(all_files)} 個有效文件對")
    
    # 分割數據
    train_files, val_files, test_files = stratified_split_files(
        all_files, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # 🎯 為每個分割分配patch類型 (新增功能)
    print(f"🎯 分配混合patch類型 (aliased比例: {aliased_ratio:.1%})...")
    train_files = assign_patch_types_to_files(train_files, aliased_ratio, random_seed)
    val_files = assign_patch_types_to_files(val_files, aliased_ratio, random_seed + 1) 
    test_files = assign_patch_types_to_files(test_files, aliased_ratio, random_seed + 2)
    
    # 保存分割信息 (包含patch類型)
    split_info = {
        'train_files': [(os.path.basename(f[0]), f[2]) for f in train_files],  # (filename, patch_type)
        'val_files': [(os.path.basename(f[0]), f[2]) for f in val_files], 
        'test_files': [(os.path.basename(f[0]), f[2]) for f in test_files],
        'aliased_ratio': aliased_ratio,
        'mixed_patch_version': 'v2.0_original_mode'
    }
    
    split_dir = os.path.dirname(output_h5_path)
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'data_split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # 創建H5文件
    with h5py.File(output_h5_path, 'w') as h5f:
        # 創建訓練、驗證和測試數據集
        train_group = h5f.create_group('train')
        val_group = h5f.create_group('val')
        test_group = h5f.create_group('test')
        
        # 處理訓練文件 - 並行版本
        print(f"使用 {num_workers} 個工作進程並行處理...")
        
        # 提取參數
        extract_kwargs = {
            'layers': 4,
            'patch_size': patch_size,
            'num_patches_per_file': num_patches_per_file,
            'min_alias_pixels': min_alias_pixels,
            'augment_variations': augment_variations,
            'enable_azimuth_rotation': enable_azimuth_rotation,
        'azimuth_rotation_angles': azimuth_rotation_angles,
        'max_azimuth_variations': max_azimuth_variations
        }
        
        # 將文件分成批次以實現更好的進度跟蹤
        batch_size = max(1, min(20, len(train_files) // num_workers))
        train_batches = [train_files[i:i+batch_size] for i in range(0, len(train_files), batch_size)]
        
        train_count = 0
        print(f"處理 {len(train_batches)} 批訓練文件...")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有批次任務
            future_to_batch = {
                executor.submit(process_file_batch, batch, **extract_kwargs): batch
                for batch in train_batches
            }
            
            # 處理完成的任務
            for future in tqdm(as_completed(future_to_batch), total=len(train_batches)):
                try:
                    patches = future.result()
                    
                    # 保存patch到H5文件
                    for p in patches:
                        patch_group = train_group.create_group(f'patch_{train_count}')
                        patch_group.create_dataset('vel', data=p['vel'])
                        patch_group.create_dataset('nyq', data=p['nyq'])
                        patch_group.create_dataset('alias_label', data=p['alias_label'])
                        patch_group.create_dataset('gt_vel', data=p['gt_vel'])
                        patch_group.create_dataset('target_vel', data=p['target_vel'])  # 🎯 新增
                        
                        # 保存patch類型
                        patch_group.attrs['patch_type'] = p.get('patch_type', 'aliased')  # 🎯 新增
                        
                        # 保存來源信息 - 修正格式以支援洩漏檢測
                        source_file = p['source_info'].get('raw_file', 'unknown')
                        patch_group.create_dataset('source', data=source_file.encode('utf-8'))
                        
                        # 同時保持原有詳細資訊
                        source_group = patch_group.create_group('source_detail')
                        for key, value in p['source_info'].items():
                            source_group.attrs[key] = value
                        
                        train_count += 1
                        
                except Exception as e:
                    print(f"處理批次時發生錯誤: {e}")
        
        print(f"已處理 {train_count} 個訓練patches")
        
        # 處理驗證文件 - 使用相同的並行方法
        val_batches = [val_files[i:i+batch_size] for i in range(0, len(val_files), batch_size)]
        val_count = 0
        print(f"處理 {len(val_batches)} 批驗證文件...")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {
                executor.submit(process_file_batch, batch, **extract_kwargs): batch
                for batch in val_batches
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(val_batches)):
                try:
                    patches = future.result()
                    
                    for p in patches:
                        patch_group = val_group.create_group(f'patch_{val_count}')
                        patch_group.create_dataset('vel', data=p['vel'])
                        patch_group.create_dataset('nyq', data=p['nyq'])
                        patch_group.create_dataset('alias_label', data=p['alias_label'])
                        patch_group.create_dataset('gt_vel', data=p['gt_vel'])
                        patch_group.create_dataset('target_vel', data=p['target_vel'])  # 🎯 新增
                        
                        # 保存patch類型
                        patch_group.attrs['patch_type'] = p.get('patch_type', 'aliased')  # 🎯 新增
                        
                        # 保存來源信息 - 修正格式以支援洩漏檢測
                        source_file = p['source_info'].get('raw_file', 'unknown')
                        patch_group.create_dataset('source', data=source_file.encode('utf-8'))
                        
                        # 同時保持原有詳細資訊
                        source_group = patch_group.create_group('source_detail')
                        for key, value in p['source_info'].items():
                            source_group.attrs[key] = value
                        
                        val_count += 1
                
                except Exception as e:
                    print(f"處理批次時發生錯誤: {e}")
        
        print(f"已處理 {val_count} 個驗證patches")
        
        # 處理測試文件 - 使用相同的並行方法
        test_batches = [test_files[i:i+batch_size] for i in range(0, len(test_files), batch_size)]
        test_count = 0
        print(f"處理 {len(test_batches)} 批測試文件...")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {
                executor.submit(process_file_batch, batch, **extract_kwargs): batch
                for batch in test_batches
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(test_batches)):
                try:
                    patches = future.result()
                    
                    for p in patches:
                        patch_group = test_group.create_group(f'patch_{test_count}')
                        patch_group.create_dataset('vel', data=p['vel'])
                        patch_group.create_dataset('nyq', data=p['nyq'])
                        patch_group.create_dataset('alias_label', data=p['alias_label'])
                        patch_group.create_dataset('gt_vel', data=p['gt_vel'])
                        patch_group.create_dataset('target_vel', data=p['target_vel'])  # 🎯 新增
                        
                        # 保存patch類型
                        patch_group.attrs['patch_type'] = p.get('patch_type', 'aliased')  # 🎯 新增
                        
                        # 保存來源信息 - 修正格式以支援洩漏檢測
                        source_file = p['source_info'].get('raw_file', 'unknown')
                        patch_group.create_dataset('source', data=source_file.encode('utf-8'))
                        
                        # 同時保持原有詳細資訊
                        source_group = patch_group.create_group('source_detail')
                        for key, value in p['source_info'].items():
                            source_group.attrs[key] = value
                        
                        test_count += 1
                
                except Exception as e:
                    print(f"處理批次時發生錯誤: {e}")
        
        print(f"已處理 {test_count} 個測試patches")
        
        # 保存元數據
        metadata = h5f.create_group('metadata')
        metadata.attrs['patch_size'] = patch_size
        metadata.attrs['train_patches'] = train_count
        metadata.attrs['val_patches'] = val_count
        metadata.attrs['test_patches'] = test_count
        metadata.attrs['min_alias_pixels'] = min_alias_pixels
        metadata.attrs['aliased_ratio'] = aliased_ratio  # 🎯 新增：混合patch比例
        metadata.attrs['created_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['data_source'] = 'original_files_mixed'  # 🎯 標記為original模式mixed patch
        metadata.attrs['label_system'] = "swapped_0_3"  # 標記使用了互換標籤系統
        metadata.attrs['mixed_patch_version'] = 'v2.0_original'  # 🎯 版本標記
        
        # 保存分割信息
        splits = metadata.create_group('splits')
        splits.attrs['train_ratio'] = train_ratio
        splits.attrs['val_ratio'] = val_ratio
        splits.attrs['test_ratio'] = test_ratio
        splits.attrs['num_train_files'] = len(train_files)
        splits.attrs['num_val_files'] = len(val_files)
        splits.attrs['num_test_files'] = len(test_files)
    
    print(f"Patch生成完成!")
    print(f"總共生成:")
    print(f"  - {train_count} 個訓練patch")
    print(f"  - {val_count} 個驗證patch")
    print(f"  - {test_count} 個測試patch")
    print(f"數據已保存到 {output_h5_path}")
    
    return output_h5_path, train_count, val_count, test_count

###############################################################################
# 檔名生成函數
###############################################################################
def generate_output_filename(args):
    """根據參數生成輸出檔案路徑"""
    import datetime
    
    # 如果直接指定了完整路徑，直接使用
    if args.output_file:
        return args.output_file
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 開始構建檔名
    if args.auto_name:
        # 自動產生描述性檔名
        filename_parts = ['mixed_patches']
        
        # 添加模式信息
        if args.mode == 'reorganized':
            filename_parts.append('reorganized')
        else:
            filename_parts.append('original')
        
        # 添加aliased比例信息 (兩種模式都支持)
        if hasattr(args, 'aliased_ratio'):
            ratio_str = f"aliased{int(args.aliased_ratio*100)}p"
            filename_parts.append(ratio_str)
        
        # 添加patch信息
        if args.patch_size != 128:
            filename_parts.append(f"patch{args.patch_size}")
        
        if args.num_patches_per_file != 10:
            filename_parts.append(f"n{args.num_patches_per_file}")
        
        # 添加增強信息 (這個變量在主程序中會被處理，這裡先使用原始值)
        if hasattr(args, 'enable_augmentation') and args.enable_augmentation:
            if args.augment_variations > 0:
                filename_parts.append(f"aug{args.augment_variations}")
            elif hasattr(args, 'augment_strength'):
                filename_parts.append(f"aug{args.augment_strength}")
        elif hasattr(args, 'auto_augment') and args.auto_augment:
            filename_parts.append('autoaug')
        
        base_name = '_'.join(filename_parts)
        
    elif args.output_name:
        # 使用指定的檔名
        base_name = args.output_name
    else:
        # 使用預設檔名
        base_name = 'mixed_radar_patches'
    
    # 添加前綴
    if args.name_prefix:
        base_name = args.name_prefix + '_' + base_name
    
    # 添加後綴
    if args.name_suffix:
        base_name = base_name + '_' + args.name_suffix
    
    # 添加時間戳
    if args.include_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = base_name + '_' + timestamp
    
    # 組合完整路徑
    filename = base_name + '.h5'
    full_path = os.path.join(args.output_dir, filename)
    
    return full_path

###############################################################################
# 主函數 
###############################################################################
def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='生成並保存雷達速度混合patches')
    
    # 選擇數據來源模式
    parser.add_argument('--mode', type=str, choices=['reorganized', 'original'], default='reorganized',
                        help='數據來源模式: reorganized=使用temporal_reorganized_data, original=使用原始資料夾')
    
    # temporal_reorganized_data 模式參數
    parser.add_argument('--reorganized_data_path', type=str, 
                        default='temporal_reorganized_data',
                        help='temporal_reorganized_data資料夾路徑')
    parser.add_argument('--aliased_ratio', type=float, default=0.7,
                        help='摺錯patch的比例 (0.0-1.0)')
    parser.add_argument('--exclude_folder', type=str, default=None,
                        help='要排除的資料夾路徑，防止資料洩漏 (如測試案例資料夾)')
    
    # 原始模式參數 (向後相容)
    parser.add_argument('--data_folder', type=str, default='E:/DealData/ALL',
                        help='原始數據文件夾路徑 (僅original模式)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='訓練集比例 (僅original模式)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='驗證集比例 (僅original模式)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='測試集比例 (僅original模式)')
    
    # 輸出文件命名參數
    parser.add_argument('--output_file', type=str, default=None,
                        help='輸出H5文件完整路徑 (若指定則忽略其他命名選項)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='輸出文件名 (不含路徑和副檔名)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='輸出目錄 (預設: data)')
    parser.add_argument('--name_prefix', type=str, default='',
                        help='文件名前綴')
    parser.add_argument('--name_suffix', type=str, default='',
                        help='文件名後綴')
    parser.add_argument('--auto_name', action='store_true', default=False,
                        help='自動產生描述性檔名')
    parser.add_argument('--include_timestamp', action='store_true', default=False,
                        help='在檔名中包含時間戳')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='patch大小')
    parser.add_argument('--num_patches_per_file', type=int, default=10,
                        help='每個文件生成的patch數量')
    parser.add_argument('--min_alias_pixels', type=int, default=10,
                        help='每個patch中最小alias像素數量')
    # 數據增強相關參數
    parser.add_argument('--enable_augmentation', action='store_true', default=False,
                        help='是否啟用數據增強 (預設: 關閉)')
    parser.add_argument('--augment_variations', type=int, default=0,
                        help='每個原始patch的數據增強變體數量 (預設: 0, 即不增強)')
    parser.add_argument('--augment_strength', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='數據增強強度: light(1x), medium(2-3x), heavy(4-6x)')
    parser.add_argument('--auto_augment', action='store_true', default=False,
                        help='自動根據數據量決定增強程度')
    
    # 方位角旋轉增強參數
    parser.add_argument('--enable_azimuth_rotation', action='store_true', default=True,
                        help='是否啟用方位角旋轉增強 (預設: 啟用)')
    parser.add_argument('--disable_azimuth_rotation', action='store_true', default=False,
                        help='停用方位角旋轉增強')
    parser.add_argument('--azimuth_rotation_angles', type=str, default='45,90,135,180,225,270,315',
                        help='方位角旋轉角度列表，以逗號分隔 (預設: 45,90,135,180,225,270,315)')
    parser.add_argument('--max_azimuth_variations', type=int, default=2,
                        help='每個patch最多使用的方位角旋轉變體數量 (預設: 2)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='並行處理的工作進程數，預設為CPU核心數-1')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='隨機種子')
    
    args = parser.parse_args()
    
    # 先處理數據增強參數邏輯
    augment_variations = args.augment_variations
    
    if args.enable_augmentation and augment_variations == 0:
        # 如果啟用增強但沒有指定變體數量，使用預設值
        if args.augment_strength == 'light':
            augment_variations = 1      # 總共2倍數據
        elif args.augment_strength == 'medium':
            augment_variations = 2      # 總共3倍數據  
        elif args.augment_strength == 'heavy':
            augment_variations = 5      # 總共6倍數據
    elif not args.enable_augmentation:
        # 如果未啟用增強，強制設為0
        augment_variations = 0
    
    # 自動增強邏輯 (覆蓋其他設定)
    if args.auto_augment:
        print("🤖 啟用自動增強模式，正在分析數據量...")
        
        if args.mode == 'reorganized':
            # 預先分析檔案數量來決定增強程度
            try:
                total_files = 0
                for split in ['train', 'val']:
                    for patch_type in ['aliased', 'clean']:
                        raw_path = os.path.join(args.reorganized_data_path, split, patch_type, 'raw')
                        if os.path.exists(raw_path):
                            files = glob.glob(os.path.join(raw_path, '*.csv'))
                            total_files += len(files)
                
                # 根據檔案數量自動決定增強程度
                estimated_patches = total_files * args.num_patches_per_file
                
                if estimated_patches < 5000:
                    augment_variations = 5      # 重度增強
                    strength_msg = "重度 (數據量較少)"
                elif estimated_patches < 15000:
                    augment_variations = 2      # 中度增強
                    strength_msg = "中度 (數據量適中)"
                else:
                    augment_variations = 1      # 輕度增強
                    strength_msg = "輕度 (數據量充足)"
                
                print(f"   預估patch數量: {estimated_patches}")
                print(f"   自動選擇增強強度: {strength_msg}")
                
            except Exception as e:
                print(f"   ⚠️  自動分析失敗，使用預設中等增強: {e}")
                augment_variations = 2
        else:
            # 原始模式使用預設值
            augment_variations = 2
    
    # 處理方位角旋轉參數
    # 處理互斥的啟用/停用選項
    if args.disable_azimuth_rotation:
        args.enable_azimuth_rotation = False
    
    azimuth_rotation_angles = None
    if args.enable_azimuth_rotation:
        # 解析方位角列表
        try:
            azimuth_rotation_angles = [int(angle.strip()) for angle in args.azimuth_rotation_angles.split(',')]
        except ValueError:
            print("⚠️ 方位角參數格式錯誤，使用預設值")
            azimuth_rotation_angles = [45, 90, 135, 180, 225, 270, 315]
    
    # 處理輸出檔名邏輯 (在增強參數處理後)
    # 為了讓自動檔名能正確反映增強信息，暫時將處理後的值賦給args
    original_augment_variations = args.augment_variations
    args.augment_variations = augment_variations
    output_file_path = generate_output_filename(args)
    args.augment_variations = original_augment_variations  # 恢復原值
    
    print(f"🚀 混合patch生成程式 (模式: {args.mode})")
    print(f"📝 輸出檔案: {output_file_path}")
    print(f"🎨 矩陣增強: {'啟用' if augment_variations > 0 else '停用'}")
    if augment_variations > 0:
        print(f"   矩陣增強變體數量: {augment_variations} (總共 {augment_variations + 1}倍數據)")
    print(f"🔄 方位角旋轉增強: {'啟用' if args.enable_azimuth_rotation else '停用'}")
    if args.enable_azimuth_rotation:
        print(f"   支援角度: {azimuth_rotation_angles}")
        print(f"   最大變體數: {args.max_azimuth_variations}")
    
    if args.mode == 'reorganized':
        # 使用temporal_reorganized_data
        print(f"📁 使用重組資料: {args.reorganized_data_path}")
        print(f"📊 摺錯比例: {args.aliased_ratio:.1%}")
        
        output_path = generate_mixed_patches_from_reorganized_data(
            reorganized_data_path=args.reorganized_data_path,
            output_h5_path=output_file_path,
            patch_size=args.patch_size,
            num_patches_per_file=args.num_patches_per_file,
            min_alias_pixels=args.min_alias_pixels,
            augment_variations=augment_variations,  # 使用處理後的值
            aliased_ratio=args.aliased_ratio,
            random_seed=args.seed,
            num_workers=args.num_workers,
            exclude_folder=args.exclude_folder,
            enable_azimuth_rotation=args.enable_azimuth_rotation,
            azimuth_rotation_angles=azimuth_rotation_angles,
            max_azimuth_variations=args.max_azimuth_variations
        )
        
    else:
        # 使用原始模式 (現已支持混合patch！)
        print(f"📁 使用原始資料夾: {args.data_folder}")
        print(f"📊 摺錯比例: {args.aliased_ratio:.1%}")  # 🎯 新增顯示
        if args.exclude_folder:
            print(f"排除資料夾: {args.exclude_folder}")
        assert args.train_ratio + args.val_ratio + args.test_ratio <= 1.0, "數據分割比例之和必須不超過1"
        
        output_path, train_count, val_count, test_count = generate_and_save_patches_parallel(
            data_folder=args.data_folder,
            output_h5_path=output_file_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            patch_size=args.patch_size,
            num_patches_per_file=args.num_patches_per_file,
            min_alias_pixels=args.min_alias_pixels,
            augment_variations=augment_variations,  # 使用處理後的值
            aliased_ratio=args.aliased_ratio,  # 傳遞混合patch比例
            random_seed=args.seed,
            num_workers=args.num_workers,
            exclude_folder=args.exclude_folder,  # 新增: 排除資料夾支援
            enable_azimuth_rotation=args.enable_azimuth_rotation,  # 新增: 方位角旋轉控制
            azimuth_rotation_angles=azimuth_rotation_angles,       # 新增: 方位角參數
            max_azimuth_variations=args.max_azimuth_variations     # 新增: 最大變體數
        )
    
    print(f"✅ 成功生成並保存patch到 {output_path}")

if __name__ == "__main__":
    # 測量執行時間
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"總執行時間: {(end_time - start_time)/60:.2f} 分鐘")