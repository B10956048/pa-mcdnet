#!/usr/bin/env python3
"""
混合Patch模型測試程式
適配mixed_patch_train_v2.py訓練的模型，支持地理位置可視化

使用範例:

  # 使用物理原理方法 (預設)
  python mixed_patch_test_model.py --raw_csv data.csv --gt_csv gt.csv --model_path model.h5

  # 使用傳統閾值方法
  python mixed_patch_test_model.py --raw_csv data.csv --gt_csv gt.csv --model_path model.h5 --use_threshold_method

  # 調整容忍度
  python mixed_patch_test_model.py --raw_csv data.csv --gt_csv gt.csv --model_path model.h5 --correction_tolerance 0.2

  # 包含地理位置可視化 (自動尋找.gz檔案)
  python mixed_patch_test_model.py --raw_csv temporal_reorganized_data/test/aliased/raw/RCGI.20210911.1020.bvel_raw.01.csv --gt_csv temporal_reorganized_data/test/aliased/gt/RCGI.20210911.1020.bvel_vdaqc.01.csv --model_path results/mixed_patch_exp/models/mixed_patch_exp_best.h5

  # 明確指定.gz檔案和台灣地圖
  python mixed_patch_test_model.py --raw_csv data.csv --gt_csv gt.csv --model_path model.h5 --raw_gz data.gz --gt_gz gt.gz --shape_path mapdata201805310314/COUNTY_MOI_1070516

  # 完整範例
  python mixed_patch_test_model.py --raw_csv temporal_reorganized_data/test/aliased/raw/RCGI.20210911.1020.bvel_raw.01.csv --gt_csv temporal_reorganized_data/test/aliased/gt/RCGI.20210911.1020.bvel_vdaqc.01.csv --model_path results/mixed_patch_exp/models/mixed_patch_exp_best.h5 --shape_path mapdata201805310314/COUNTY_MOI_1070516 --use_physics_method --correction_tolerance 0.25
"""

# 重要：必須在所有其他導入之前執行修復
import fix_typing  # 這行必須放在最前面

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
import struct
import gzip

# 使用與訓練一致的模型版本
from unet_model.dealias_mulit_v2 import VelocityDealiaser, make_velocity_mask
from unet_model.feature_extraction_mulit_v2 import (
    create_downsampler,
    create_upsampler_cls,
    create_upsampler_reg
)
# 導入反折錯成功率指標 (支持閾值和物理方法)
from dealiasing_success_metrics import (
    compute_dealiasing_success_metrics,
    print_correction_summary,
    visualize_correction_success,
    analyze_dealiasing_performance
)
from physics_based_dealiasing_metrics import (
    compute_physics_based_dealiasing_metrics,
    print_physics_based_summary,
    visualize_physics_based_correction,
    compare_threshold_vs_physics_methods
)

# 導入梯度檢測工具
import sys
sys.path.append(str(Path(__file__).parent / "simplified_quality_assessor"))
from gradient_checker import compute_gradients_with_periodicity

# 導入時間平滑工具
from temporal_smoothing import temporal_smooth

# 地理可視化相關
try:
    from pyart.graph import RadarMapDisplayBasemap
    from pyart.config import get_metadata
    from pyart.core.radar import Radar
    from mpl_toolkits.basemap import Basemap
    GEO_VIZ_AVAILABLE = True
    print("[OK] 地理可視化模塊已載入 (PyART + Basemap)")
except ImportError as e:
    GEO_VIZ_AVAILABLE = False
    print(f"[WARNING]  地理可視化模塊未安裝: {e}")
    print("   將使用標準matplotlib可視化")

# 設置隨機種子
SEED = 46
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 可視化設置
levels = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
colors = [
    "#000078", "#0000ce", "#0063ff", "#31ffff", 
    "#31ff63", "#00ce00", "#009c00", "#cecece", 
    "#636363",
    "#ffff31", "#ffce31", "#ff9c00", "#ff6363", 
    "#ff0000", "#ce0000", "#8d005c"
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
norm = mcolors.BoundaryNorm(levels, cmap.N)

# 地理可視化常數
degree_per_km = 1 / 110

class RadarGzReader:
    """讀取雷達.gz檔案獲取地理資訊 - 完全複製 dealiasing_test_model.py 的 radar 類別"""
    
    def __init__(self, fname):
        if '.gz' in fname:
            import gzip
            f = gzip.open(fname).read()
        else:
            f = open(fname, "rb").read()
            
        header_end = 160
        header = struct.unpack('<16s36i', f[0:header_end])
        info = np.array(header[1:-1])
        
        if (len(str(header[0], 'utf-8')) == 12):
            self.name = str(header[0], 'utf-8')[0:16:4]
        else:
            self.name = str(header[0][0:4], 'utf-8')
            
        self.h_scale = info[0]
        info_flt = info / self.h_scale
        self.radar_elev = info_flt[1]
        self.rlat = info_flt[2]  # 雷達站緯度
        self.rlon = info_flt[3]  # 雷達站經度
        self.yyyy = int(info_flt[4])
        self.mm = int(info_flt[5])
        self.dd = int(info_flt[6])
        self.hh = int(info_flt[7])
        self.mn = int(info_flt[8])
        self.ss = int(info_flt[9])
        self.nyquist = info_flt[10]
        self.nivcp = int(info_flt[11])
        self.itit = info[12]
        self.theta = info_flt[13]  # 仰角
        self.nray = int(info_flt[14])  # 徑向數
        self.ngate = int(info_flt[15])  # 距離閘數
        self.azm_start = info_flt[16]
        self.azm_sp = 360.0 / self.nray
        self.gate_start = info_flt[18]
        self.gate_sp = info_flt[19]
        self.var_scale = info_flt[20]
        self.var_miss = info[21]
        self.rf_miss = -44400
        
        # 讀取實際數據
        n = self.nray * self.ngate
        data = struct.unpack('<' + str(n) + 'i', f[header_end:])
        self.data = np.reshape(np.array(data) / self.var_scale,
                               (self.nray, self.ngate), order='C')

def find_corresponding_gz_file(csv_path: str) -> str:
    """
    根據CSV檔案路徑尋找對應的.gz檔案
    
    例如: RCGI.20210911.1020.bvel_raw.01.csv 
    對應: RCGI.20210911.1020.bvel_raw.01.gz
    """
    csv_path = Path(csv_path)
    
    # 在同一目錄查找
    gz_file = csv_path.with_suffix('.gz')
    if gz_file.exists():
        return str(gz_file)
    
    # 在相鄰目錄查找 (考慮資料夾結構)
    possible_dirs = [
        csv_path.parent,  # 同目錄
        csv_path.parent.parent,  # 上一層
        csv_path.parent / 'gz_files',  # gz_files子目錄
    ]
    
    for search_dir in possible_dirs:
        if search_dir.exists():
            gz_pattern = csv_path.stem + '.gz'
            for gz_file in search_dir.rglob(gz_pattern):
                return str(gz_file)
    
    return None

def alias_to_velocity(raw_vel, alias_logits, nyq_val):
    """
    將分類logits轉換為速度修正值 - 複製自 dealiasing_test_model.py
    
    raw_vel : shape=(H,W) float
    alias_logits: shape=(H,W,6)
    nyq_val: float
    """
    cat_map = np.argmax(alias_logits, axis=-1)  # (H,W)
    print('cat_map', cat_map)
    correction_map = np.zeros_like(cat_map, dtype=np.float32)
    correction_map[cat_map==1] = -4 * nyq_val
    correction_map[cat_map==2] = -2 * nyq_val
    correction_map[cat_map==4] =  2 * nyq_val
    correction_map[cat_map==5] =  4 * nyq_val
    dealiased_vel = raw_vel + correction_map
    return dealiased_vel

def apply_physics_based_correction(raw_vel, regression_output, nyq_val):
    """
    對 regression 輸出應用物理公式後處理
    使用反折錯物理規律：V = Vobs + 2n×Va
    
    Parameters:
    -----------
    raw_vel : np.ndarray (H,W)
        原始觀測速度 Vobs
    regression_output : np.ndarray (H,W)  
        模型regression輸出 Vref
    nyq_val : float
        Nyquist速度 Va
    
    Returns:
    --------
    corrected_vel : np.ndarray (H,W)
        物理修正後的速度
    """
    # 計算折疊次數 n = round((Vref - Vobs) / (2×Va))
    if abs(nyq_val) < 1e-6:  # 避免除以零
        return regression_output
    
    n_2d = np.rint((regression_output - raw_vel) / (2.0 * nyq_val))
    
    # 應用反折錯公式 V = Vobs + 2×n×Va
    corrected_vel = raw_vel + 2.0 * n_2d * nyq_val
    
    return corrected_vel

def read_cwb_radar_sweep(fname):
    """直接複製 dealiasing_test_model.py 的函數"""
    cwb_radar_object = RadarGzReader(fname)
    ngates = cwb_radar_object.ngate
    rays_per_sweep = cwb_radar_object.nray
    nsweeps = 1
    nrays = rays_per_sweep * nsweeps
    distances = np.linspace(
        cwb_radar_object.gate_start/1000,
        cwb_radar_object.gate_start/1000 + cwb_radar_object.gate_sp/1000 * (ngates),
        ngates
    )
    time = get_metadata('time')
    _range = get_metadata('range')
    latitude = get_metadata('latitude')
    longitude = get_metadata('longitude')
    altitude = get_metadata('altitude')
    sweep_number = get_metadata('sweep_number')
    sweep_mode = get_metadata('sweep_mode')
    fixed_angle = get_metadata('fixed_angle')
    sweep_start_ray_index = get_metadata('sweep_start_ray_index')
    sweep_end_ray_index = get_metadata('sweep_end_ray_index')
    azimuth = get_metadata('azimuth')
    elevation = get_metadata('elevation')

    if ("ref_qc" in fname):
        varname = 'corrected_reflectivity'
    elif ("ref_raw" in fname):
        varname = 'reflectivity'
    elif ("vel" in fname):
        varname = 'velocity'
    else:
        varname = 'unknown_field'

    data_dict = get_metadata(varname)
    data_dict['data'] = np.array(cwb_radar_object.data, dtype='float32')
    fields = {varname: data_dict}
    scan_type = 'ppi'
    metadata = {'instrument_name': cwb_radar_object.name}
    time['data'] = np.arange(nrays, dtype='float64')
    time['units'] = 'seconds since ' + f"{cwb_radar_object.yyyy:04d}-{cwb_radar_object.mm:02d}-{cwb_radar_object.dd:02d}T{cwb_radar_object.hh:02d}:{cwb_radar_object.mn:02d}:{cwb_radar_object.ss:02d}Z"
    _range['data'] = np.linspace(cwb_radar_object.gate_start,
                                 cwb_radar_object.gate_start + cwb_radar_object.gate_sp * (ngates-1),
                                 ngates).astype('float32')
    latitude['data'] = np.array([cwb_radar_object.rlat], dtype='float64')
    longitude['data'] = np.array([cwb_radar_object.rlon], dtype='float64')
    altitude['data'] = np.array([cwb_radar_object.radar_elev], dtype='float64')
    sweep_number['data'] = np.arange(nsweeps, dtype='int32')
    sweep_mode['data'] = np.array(['azimuth_surveillance'] * nsweeps)
    fixed_angle['data'] = np.array([cwb_radar_object.theta] * nsweeps, dtype='float32')
    sweep_start_ray_index['data'] = np.array([0], dtype='int32')
    sweep_end_ray_index['data'] = np.array([nrays-1], dtype='int32')
    azimuth['data'] = np.linspace(cwb_radar_object.azm_start,
                                  cwb_radar_object.azm_start + cwb_radar_object.azm_sp * (nrays-1),
                                  nrays, dtype='float32')
    elevation['data'] = np.array([cwb_radar_object.theta] * nrays, dtype='float32')
    nyq = cwb_radar_object.nyquist

    radar_data = Radar(
        time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle,
        sweep_start_ray_index, sweep_end_ray_index,
        azimuth, elevation,
        instrument_parameters=None
    )
    return radar_data, distances, nyq

def plot_taiwan_basemap(lon: float, lat: float, radius_km: np.ndarray, shape_path: str = None):
    """
    繪製台灣底圖
    
    Parameters:
    -----------
    lon, lat : float
        雷達站經緯度
    radius_km : np.ndarray
        距離範圍 (km)
    shape_path : str, optional
        台灣地圖shapefile路徑
    """
    if not GEO_VIZ_AVAILABLE:
        return None
        
    r_deg = radius_km[-1] * degree_per_km
    
    m = Basemap(projection='merc', resolution='i', fix_aspect=True,
                llcrnrlon=lon - r_deg,
                llcrnrlat=lat - r_deg,
                urcrnrlon=lon + r_deg,
                urcrnrlat=lat + r_deg,
                lat_ts=lat)
    
    # 如果有shapefile就載入台灣縣界
    if shape_path and os.path.exists(shape_path):
        try:
            m.readshapefile(shape_path, linewidth=0.25, drawbounds=True, name='Taiwan')
        except:
            print(f"[WARNING]  無法載入shapefile: {shape_path}")
    
    m.drawcoastlines(linewidth=1)
    return m

def build_mixed_patch_model_for_inference(use_physics_model=False):
    """
    構建與混合patch訓練一致的模型
    使用與mixed_patch_train_v2.py相同的參數

    Parameters:
    -----------
    use_physics_model : bool
        是否使用物理約束模型版本
    """
    if use_physics_model:
        print("[FIX] 構建物理約束推理模型...")
        try:
            from unet_model.dealias_mulit_v2_physics import VelocityDealiaser as PhysicsVelocityDealiaser

            # 與物理約束訓練腳本保持一致的參數
            extractor = create_downsampler(input_channels=1, start_neurons=32)
            up_cls = create_upsampler_cls(n_inputs=1, start_neurons=32, classes=6)
            up_reg = create_upsampler_reg(n_inputs=1, start_neurons=32)

            model = PhysicsVelocityDealiaser(extractor, up_cls, up_reg)
            print("[OK] 物理約束模型構建完成")
            return model
        except ImportError:
            print("[ERROR] 物理約束模型未找到，回退到標準模型")
            use_physics_model = False

    if not use_physics_model:
        print("[FIX] 構建混合patch推理模型...")

        # 與訓練腳本保持一致的參數
        extractor = create_downsampler(input_channels=1, start_neurons=32)
        up_cls = create_upsampler_cls(n_inputs=1, start_neurons=32, classes=6)  # 修正參數名
        up_reg = create_upsampler_reg(n_inputs=1, start_neurons=32)             # 移除n_outputs

        model = VelocityDealiaser(extractor, up_cls, up_reg)
        print("[OK] 模型構建完成")
        return model

def sort_azimuths_circular(azimuths):
    """
    對方位角進行周期性排序，處理跨越0度邊界的情況

    雷達掃描可能從任意角度開始，例如:
    - 從350度開始: [350, 351, ..., 359, 0, 1, ..., 349]
    - 從-10度開始: [-10, -9, ..., 0, 1, ..., 349] (等同於 [350, 351, ..., 359, 0, ..., 349])

    Args:
        azimuths: 方位角陣列

    Returns:
        sorted_azimuths: 按掃描順序排序的方位角（保持周期連續性）
    """
    azimuths = np.array(azimuths)

    # 正規化到 [0, 360) 範圍
    azimuths_normalized = azimuths % 360.0

    # 找到最小方位角作為起始點
    min_az = azimuths_normalized.min()

    # 檢查是否跨越0度邊界
    # 如果最大值和最小值差距接近360度，表示跨越邊界
    az_span = azimuths_normalized.max() - min_az

    if az_span > 180:
        # 跨越0度邊界的情況
        # 找到間隔最大的位置（掃描起點）
        sorted_az = np.sort(azimuths_normalized)
        gaps = np.diff(sorted_az)

        # 找到最大間隔（這是掃描的起點/終點斷裂處）
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            # 將陣列從斷裂處重新排列
            sorted_azimuths = np.concatenate([sorted_az[max_gap_idx+1:], sorted_az[:max_gap_idx+1]])
        else:
            sorted_azimuths = sorted_az
    else:
        # 未跨越邊界，直接排序
        sorted_azimuths = np.sort(azimuths_normalized)

    return sorted_azimuths


def load_csv_as_matrix(file_path: str):
    """從CSV加載雷達數據為矩陣格式"""
    print(f"[FOLDER] 載入CSV: {file_path}")
    data = pd.read_csv(file_path)
    data['Value'] = data['Value'].replace(-999.0, np.nan)

    # [FIX] 使用原始方位角順序（不排序），保持與 .gz 檔案一致
    # 這樣地理可視化時，方位角和速度值才能正確對應
    azimuths = data['Azimuth'].unique()  # 保持 CSV 中的原始順序
    ranges = sorted(data['Range'].unique())
    n_az = len(azimuths)
    n_rad = len(ranges)

    velocity_grid = np.full((n_az, n_rad), np.nan, dtype=np.float32)

    # 建立方位角到索引的映射（使用正規化值）
    azimuth_to_idx = {(az % 360.0): idx for idx, az in enumerate(azimuths)}

    for _, row in data.iterrows():
        # 正規化方位角到 [0, 360)
        az_normalized = row['Azimuth'] % 360.0

        # 找到最接近的方位角索引（考慮數值精度誤差）
        az_idx = azimuth_to_idx.get(az_normalized)
        if az_idx is None:
            # 如果找不到精確匹配，找最接近的
            azimuths_normalized = np.array([az % 360.0 for az in azimuths])
            az_idx = np.argmin(np.abs(azimuths_normalized - az_normalized))

        rad_idx = np.where(ranges == row['Range'])[0][0]
        velocity_grid[az_idx, rad_idx] = row['Value']

    nyquist = data['Nyquist'].unique()
    print(f"   形狀: {velocity_grid.shape}, Nyquist: {np.nanmean(nyquist):.2f}")
    return velocity_grid, nyquist

def auto_zero_pad(data_2d, layers=4, fill_value=0):
    """自動padding以符合模型輸入要求"""
    factor = 2 ** layers
    H, W = data_2d.shape
    pad_h = (factor - (H % factor)) if (H % factor) != 0 else 0
    pad_w = (factor - (W % factor)) if (W % factor) != 0 else 0
    padded = np.pad(data_2d, ((0, pad_h), (0, pad_w)),
                    mode='constant', constant_values=fill_value)
    return padded, pad_h, pad_w

def alias_to_velocity_corrected(raw_vel, alias_logits, nyq_val):
    """
    修正的別名轉速度函數
    與dealias_mulit_v2.py中的邏輯保持一致
    """
    cat_map = np.argmax(alias_logits, axis=-1)  # (H,W)
    correction_map = np.zeros_like(cat_map, dtype=np.float32)
    
    # 修正的別名修正係數 - 與dealias_mulit_v2.py一致
    correction_map[cat_map==1] = -2.0 * nyq_val  # -2 cycle
    correction_map[cat_map==2] = -1.0 * nyq_val  # -1 cycle  
    correction_map[cat_map==4] =  1.0 * nyq_val  # +1 cycle
    correction_map[cat_map==5] =  2.0 * nyq_val  # +2 cycle
    # cat_map==0 或 3 => 0 (不修正)
    
    dealiased_vel = raw_vel + correction_map
    return dealiased_vel

def compute_metrics(pred, gt, raw=None, nyq_val=None, name="", 
                   correction_threshold=0.5, correction_tolerance=0.3, use_physics_method=True):
    """
    計算完整評估指標 (包含傳統指標和反折錯成功率指標)
    
    Parameters:
    -----------
    pred : np.ndarray
        預測速度
    gt : np.ndarray
        真實速度
    raw : np.ndarray, optional
        原始速度 (用於計算反折錯成功率)
    nyq_val : float, optional
        Nyquist速度 (用於計算反折錯成功率)
    name : str
        指標名稱
    correction_threshold : float
        反折錯修正閾值係數 (相對Nyquist) - 僅在use_physics_method=False時使用
    correction_tolerance : float
        修正成功容忍度係數 (相對Nyquist)
    use_physics_method : bool, default=True
        是否使用物理原理方法 (compute_alias_label_swapped) 而非閾值方法
    """
    mask = ~np.isnan(pred) & ~np.isnan(gt)
    if raw is not None:
        mask = mask & ~np.isnan(raw)
    if not np.any(mask):
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan, "valid_pixels": 0}
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    # 傳統指標
    # RMSE
    mse = np.mean((pred_valid - gt_valid) ** 2)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(pred_valid - gt_valid))
    
    # 相關係數
    if len(pred_valid) > 1:
        corr = np.corrcoef(pred_valid, gt_valid)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    
    metrics = {
        "rmse": rmse,
        "mae": mae, 
        "corr": corr,
        "valid_pixels": np.sum(mask)
    }
    
    # 反折錯成功率指標 (如果提供了raw和nyq_val)
    if raw is not None and nyq_val is not None:
        # 統一使用基於 fold 的嚴格判斷方法
        # (use_physics_method 參數保留用於其他地方，但這裡都使用相同的計算方法)
        success_metrics = compute_dealiasing_success_metrics(
            raw, pred, gt, nyq_val,
            threshold_factor=correction_threshold,
            use_fold_based=True  # 使用基於fold的嚴格判斷
        )

        if use_physics_method:
            method_name = "physics_based"
        else:
            method_name = "threshold_based"
        
        # 將反折錯指標加入主要指標
        metrics.update({
            "correction_success_rate": success_metrics["correction_success_rate"],
            "false_positive_rate": success_metrics.get("false_positive_rate", None),
            "false_negative_rate": success_metrics.get("false_negative_rate", None),
            "precision": success_metrics.get("precision", None),
            "need_correction_pixels": success_metrics["need_correction_pixels"],
            "clean_pixels": success_metrics.get("clean_pixels", 0),
            "corrected_pixels": success_metrics["corrected_pixels"],
            "improvement_ratio": success_metrics["improvement_ratio"],
            "overcorrection_pixels": success_metrics["overcorrection_pixels"],
            "undercorrection_pixels": success_metrics["undercorrection_pixels"],
            "aliased_pixels_ratio": success_metrics["detailed_stats"].get("aliased_pixels_ratio", 0),
            "dealiasing_metrics": success_metrics,  # 保存完整的反折錯指標
            "evaluation_method": method_name  # 記錄使用的方法
        })
        
        # 如果使用物理方法，還要添加特有的指標
        if use_physics_method:
            metrics.update({
                "no_alias_pixels": success_metrics.get("no_alias_pixels", 0),
                "maintained_good_pixels": success_metrics.get("maintained_good_pixels", 0)
            })
    
    return metrics

def test_mixed_patch_model(model, raw_csv, gt_csv, model_path, result_dir="results/test_inference",
                          correction_threshold=0.5, correction_tolerance=0.3, use_physics_method=True,
                          raw_gz=None, gt_gz=None, shape_path=None, use_physics_model=False,
                          enable_temporal_smoothing=False, previous_prediction=None, temporal_alpha=0.3):
    """
    測試混合patch模型

    Parameters:
    -----------
    model : tf.keras.Model
        混合patch模型
    raw_csv : str
        原始速度CSV路徑
    gt_csv : str
        真實速度CSV路徑
    model_path : str
        訓練好的模型權重路徑
    result_dir : str
        結果輸出目錄
    correction_threshold : float
        反折錯修正閾值係數 (相對Nyquist) - 僅在use_physics_method=False時使用
    correction_tolerance : float
        修正成功容忍度係數 (相對Nyquist)
    use_physics_method : bool, default=True
        是否使用物理原理方法 (compute_alias_label_swapped) 評估反折錯成功率
    raw_gz : str, optional
        原始數據對應的.gz檔案路徑 (用於獲取地理資訊)
    gt_gz : str, optional
        GT數據對應的.gz檔案路徑
    shape_path : str, optional
        台灣地圖shapefile路徑 (如: mapdata201805310314/COUNTY_MOI_1070516)
    enable_temporal_smoothing : bool, default=False
        是否啟用時間平滑後處理
    previous_prediction : str, optional
        前一時刻的預測結果CSV路徑 (啟用時間平滑時需要)
    temporal_alpha : float, default=0.3
        時間平滑參數 (0-1)，α越大越信任當前預測，α越小越信任歷史
    """
    # [NEW] 判斷是否需要保存結果
    save_results = (result_dir is not None)

    if save_results:
        os.makedirs(result_dir, exist_ok=True)

    print(f"[TEST] 開始測試推理...")
    print(f"   模型權重: {model_path}")
    if save_results:
        print(f"   輸出目錄: {result_dir}")
    else:
        print(f"   模式: 僅推理（不保存文件）")
    
    # 1. 載入數據
    raw_vel_2d, nyq_val_raw_array = load_csv_as_matrix(raw_csv)
    gt_vel_2d, nyq_val_gt_array = load_csv_as_matrix(gt_csv)
    
    nyq_val = np.nanmean(nyq_val_raw_array)
    print(f"[INFO] 數據資訊: RAW形狀={raw_vel_2d.shape}, Nyquist={nyq_val:.2f}")
    
    # 2. 數據預處理
    raw_padded, pad_h, pad_w = auto_zero_pad(raw_vel_2d, layers=4, fill_value=0)
    print(f"   Padding後形狀: {raw_padded.shape}")
    
    # 3. 準備模型輸入 (5D張量)
    vel_5d = raw_padded[None, None, :, :, None].astype(np.float32)  # (1,1,H,W,1)
    nyq_2d = np.array([[nyq_val]], dtype=np.float32)               # (1,1)
    
    # 4. 初始化模型 (dummy forward)
    print("[RELOAD] 初始化模型...")
    dummy_out = model({'vel': vel_5d, 'nyq': nyq_2d}, training=False)
    
    # 5. 載入訓練好的權重
    print("[LOAD] 載入模型權重...")
    model.load_weights(model_path)
    print("[OK] 權重載入完成")
    
    # 6. 模型推理
    print("[START] 執行推理...")
    out = model({'vel': vel_5d, 'nyq': nyq_2d}, training=False)

    alias_mask_4d = out['alias_mask']       # (1,H,W,6)
    dealiased_4d = out['dealiased_vel']     # (1,H,W,1)
    velocity_residual_4d = out['velocity_residual']  # (1,H,W,1)

    # 檢查是否為物理約束模型 (有額外輸出)
    physics_dealiased_4d = None
    fold_probs_4d = None
    soft_classification_4d = None
    confidence_guided_4d = None
    alias_confidence_4d = None

    print(f"[SEARCH] 模型輸出鍵: {list(out.keys())}")
    print(f"[SEARCH] use_physics_model: {use_physics_model}")

    if 'physics_dealiased_vel' in out:
        physics_dealiased_4d = out['physics_dealiased_vel']  # (1,H,W,1)
        fold_probs_4d = out['fold_probs']  # (1,H,W,5)
        print("[OK] 檢測到物理約束模型輸出")
    else:
        print("[ERROR] 未檢測到物理約束模型輸出，使用標準3分支顯示")

    # [NEW] 檢查軟分類和置信度導向輸出
    if 'soft_classification_vel' in out:
        soft_classification_4d = out['soft_classification_vel']  # (1,H,W,1)
        print("[OK] 檢測到軟分類輸出")

    if 'confidence_guided_vel' in out:
        confidence_guided_4d = out['confidence_guided_vel']  # (1,H,W,1)
        alias_confidence_4d = out['alias_confidence']  # (1,H,W,1)
        print("[OK] 檢測到置信度導向輸出")
    else:
        print("[ERROR] 未檢測到置信度導向輸出")

    # 7. 提取結果並移除padding
    alias_mask_3d = alias_mask_4d[0].numpy()           # (H,W,6)
    dealiased_cls_optimized_padded = dealiased_4d[0,:,:,0].numpy()  # (H,W) - 軟分類訓練優化後的硬分類
    residual_padded = velocity_residual_4d[0,:,:,0].numpy()  # (H,W)

    # [FIX] 修正：使用分類logits計算去別名速度（傳統硬分類）
    dealiased_cls_padded = alias_to_velocity(raw_padded, alias_mask_3d, nyq_val)

    # 移除padding還原原始尺寸
    H_orig, W_orig = raw_vel_2d.shape
    dealiased_cls_optimized = dealiased_cls_optimized_padded[:H_orig, :W_orig]  # 軟分類訓練優化後的硬分類
    dealiased_cls = dealiased_cls_padded[:H_orig, :W_orig]      # 傳統硬分類分支輸出
    residual = residual_padded[:H_orig, :W_orig]                # 回歸殘差

    # 處理物理約束模型的額外輸出
    physics_dealiased = None
    fold_probs = None
    if physics_dealiased_4d is not None:
        physics_dealiased_padded = physics_dealiased_4d[0,:,:,0].numpy()  # (H,W)
        physics_dealiased = physics_dealiased_padded[:H_orig, :W_orig]
        print(f"[FIX] 物理約束分支設置完成: shape={physics_dealiased.shape}")
        print(f"[FIX] 範圍: [{np.nanmin(physics_dealiased):.1f}, {np.nanmax(physics_dealiased):.1f}] m/s")

    if fold_probs_4d is not None:
        fold_probs_padded = fold_probs_4d[0].numpy()  # (H,W,5)
        fold_probs = fold_probs_padded[:H_orig, :W_orig, :]

    # [NEW] 處理軟分類和置信度導向輸出
    soft_classification = None
    confidence_guided = None
    alias_confidence = None

    if soft_classification_4d is not None:
        soft_classification_padded = soft_classification_4d[0,:,:,0].numpy()  # (H,W)
        soft_classification = soft_classification_padded[:H_orig, :W_orig]
        print(f"[NEW] 軟分類分支設置完成: shape={soft_classification.shape}")
        print(f"[NEW] 範圍: [{np.nanmin(soft_classification):.1f}, {np.nanmax(soft_classification):.1f}] m/s")

    if confidence_guided_4d is not None:
        confidence_guided_padded = confidence_guided_4d[0,:,:,0].numpy()  # (H,W)
        confidence_guided = confidence_guided_padded[:H_orig, :W_orig]

        alias_confidence_padded = alias_confidence_4d[0,:,:,0].numpy()  # (H,W)
        alias_confidence = alias_confidence_padded[:H_orig, :W_orig]

        print(f"[TARGET] 置信度導向分支設置完成: shape={confidence_guided.shape}")
        print(f"[TARGET] 範圍: [{np.nanmin(confidence_guided):.1f}, {np.nanmax(confidence_guided):.1f}] m/s")
        print(f"[TARGET] 置信度範圍: [{np.nanmin(alias_confidence):.3f}, {np.nanmax(alias_confidence):.3f}]")

    # [TIMEOUT] 時間平滑後處理 (方案D)
    temporal_smoothed_results = {}
    if enable_temporal_smoothing and previous_prediction and os.path.exists(previous_prediction):
        print(f"\n[TIMEOUT]  應用時間平滑 (α={temporal_alpha})...")
        print(f"   前一時刻預測: {previous_prediction}")

        # 載入前一時刻的預測結果
        vel_prev, _ = load_csv_as_matrix(previous_prediction)

        # 檢查形狀是否匹配
        if vel_prev.shape != raw_vel_2d.shape:
            print(f"   [WARNING]  形狀不匹配: 當前{raw_vel_2d.shape} vs 前一時刻{vel_prev.shape}，跳過時間平滑")
        else:
            # 對每個分支應用時間平滑
            temporal_smoothed_results['optimized_classification'] = temporal_smooth(
                dealiased_cls_optimized, vel_prev, temporal_alpha
            )
            temporal_smoothed_results['traditional_classification'] = temporal_smooth(
                dealiased_cls, vel_prev, temporal_alpha
            )

            if physics_dealiased is not None:
                temporal_smoothed_results['physics_constrained'] = temporal_smooth(
                    physics_dealiased, vel_prev, temporal_alpha
                )

            if soft_classification is not None:
                temporal_smoothed_results['soft_classification'] = temporal_smooth(
                    soft_classification, vel_prev, temporal_alpha
                )

            if confidence_guided is not None:
                temporal_smoothed_results['confidence_guided'] = temporal_smooth(
                    confidence_guided, vel_prev, temporal_alpha
                )

            print(f"   [OK] 時間平滑完成，共處理 {len(temporal_smoothed_results)} 個分支")
    elif enable_temporal_smoothing:
        print(f"\n[WARNING]  時間平滑已啟用但前一時刻預測不可用，跳過")

    # [FIX] 計算純回歸分支結果（原始觀測 + 回歸殘差，不含分類校正）
    print("[FIX] 計算純回歸分支結果...")
    pure_regression = raw_vel_2d + residual  # 僅回歸殘差，不含分類校正

    # [FIX] 重新命名：dealiased_cls_optimized 是軟分類訓練優化後的硬分類
    # 計算優化分類與純回歸的差異
    classification_correction = dealiased_cls_optimized - pure_regression
    correction_magnitude = np.nanmean(np.abs(classification_correction))
    print(f"   優化分類校正平均幅度: {correction_magnitude:.3f} m/s")
    
    # 檢查結果有效性
    print(f"   Pure regression範圍: [{np.nanmin(pure_regression):.1f}, {np.nanmax(pure_regression):.1f}] m/s")
    print(f"   Optimized classification範圍: [{np.nanmin(dealiased_cls_optimized):.1f}, {np.nanmax(dealiased_cls_optimized):.1f}] m/s")
    print(f"   Traditional classification範圍: [{np.nanmin(dealiased_cls):.1f}, {np.nanmax(dealiased_cls):.1f}] m/s")
    
    
    # [ALERT] GT品質檢查：如果GT與物理修正結果差異很大，可能GT有問題
    print("\n[ALERT] GT品質分析 (針對你提到的GT錯誤問題):")
    gt_vs_physics_reg = np.abs(gt_vel_2d - dealiased_cls_optimized)
    gt_vs_physics_cls = np.abs(gt_vel_2d - dealiased_cls)
    
    # 找出與物理修正結果差異較大的區域
    suspicious_threshold = 2.0 * nyq_val  # 超過2倍Nyquist的差異視為可疑
    suspicious_reg_pixels = np.sum(gt_vs_physics_reg > suspicious_threshold)
    suspicious_cls_pixels = np.sum(gt_vs_physics_cls > suspicious_threshold)
    
    total_pixels = np.sum(~np.isnan(gt_vel_2d))
    
    print(f"   GT vs Physics-corrected Regression 可疑像素: {suspicious_reg_pixels:,} ({suspicious_reg_pixels/total_pixels:.1%})")
    print(f"   GT vs Classification 可疑像素: {suspicious_cls_pixels:,} ({suspicious_cls_pixels/total_pixels:.1%})")
    
    if suspicious_reg_pixels > total_pixels * 0.1:  # 超過10%的像素可疑
        print("   [WARNING]  警告：GT可能包含較多錯誤，建議人工檢查")
    elif suspicious_reg_pixels < total_pixels * 0.02:  # 少於2%的像素可疑
        print("   [OK] GT品質良好")
    else:
        print("   [*] GT品質一般，存在少量可疑區域")

    # 7.5 強制物理約束（消除數值精度誤差）
    # 重新計算 GT 和所有預測分支，確保使用相同的 nyquist 值和計算方式
    # 這樣當 fold 完全一致時，RMSE 會是精確的 0
    print(f"\n[FIX] 應用物理約束 (RAW + fold × 2Vn)，消除數值精度誤差...")

    # [FIX] GT 的 valid mask（RAW 和 GT 都有效）
    valid_gt_mask = ~np.isnan(raw_vel_2d) & ~np.isnan(gt_vel_2d)

    # 計算 GT 的 fold
    gt_fold = np.zeros_like(gt_vel_2d)
    gt_fold[valid_gt_mask] = np.round((gt_vel_2d[valid_gt_mask] - raw_vel_2d[valid_gt_mask]) / (2 * nyq_val))

    # 使用相同的方式重新計算 GT（確保數值完全一致）
    gt_vel_2d_constrained = np.full_like(gt_vel_2d, np.nan)
    gt_vel_2d_constrained[valid_gt_mask] = raw_vel_2d[valid_gt_mask] + gt_fold[valid_gt_mask] * (2.0 * nyq_val)

    # 對所有預測分支應用物理約束
    def apply_physics_constraint(pred_vel, name=""):
        """
        對預測速度場應用物理約束

        重要：預測不受 GT 限制，只要 RAW 和 pred 有效即可
        這樣實務應用時（沒有 GT）模型仍能輸出完整結果
        """
        if pred_vel is None:
            return None

        pred_constrained = np.full_like(pred_vel, np.nan)
        pred_fold = np.zeros_like(pred_vel)

        # [FIX] 預測的 valid mask（只看 RAW 和 pred，不受 GT 限制）
        # 實務上使用時沒有 GT，模型仍要給出所有 RAW 有效點的預測
        valid_pred_mask = ~np.isnan(raw_vel_2d) & ~np.isnan(pred_vel)
        pred_fold[valid_pred_mask] = np.round((pred_vel[valid_pred_mask] - raw_vel_2d[valid_pred_mask]) / (2 * nyq_val))
        pred_constrained[valid_pred_mask] = raw_vel_2d[valid_pred_mask] + pred_fold[valid_pred_mask] * (2.0 * nyq_val)

        # NaN 遮罩已經在初始化時設定好了

        # 計算改變的像素數（用於驗證）
        if np.any(valid_pred_mask):
            diff = np.abs(pred_constrained[valid_pred_mask] - pred_vel[valid_pred_mask])
            changed_pixels = np.sum(diff > 1e-10)
            if changed_pixels > 0:
                max_diff = np.max(diff)
                print(f"   {name}: 修正 {changed_pixels} 個像素的數值精度誤差 (最大差異: {max_diff:.6f} m/s)")

        return pred_constrained

    # 應用物理約束到所有分支
    gt_vel_2d = gt_vel_2d_constrained  # 替換 GT
    dealiased_cls_optimized = apply_physics_constraint(dealiased_cls_optimized, "Optimized Classification")
    dealiased_cls = apply_physics_constraint(dealiased_cls, "Traditional Classification")

    if physics_dealiased is not None:
        physics_dealiased = apply_physics_constraint(physics_dealiased, "Physics Constrained")

    if soft_classification is not None:
        soft_classification = apply_physics_constraint(soft_classification, "Soft Classification")

    if confidence_guided is not None:
        confidence_guided = apply_physics_constraint(confidence_guided, "Confidence Guided")

    # 時間平滑結果也需要應用約束
    if len(temporal_smoothed_results) > 0:
        for branch_name in list(temporal_smoothed_results.keys()):
            temporal_smoothed_results[branch_name] = apply_physics_constraint(
                temporal_smoothed_results[branch_name],
                f"Temporal {branch_name}"
            )

    print(f"   [OK] 物理約束已應用到所有分支")

    # 8. 計算評估指標 (包含新的反折錯成功率指標)
    method_name = "物理原理方法" if use_physics_method else f"閾值方法 (係數: {correction_threshold})"
    print(f"\n[INFO] 計算評估指標 ({method_name}, 容忍度: {correction_tolerance})...")
    raw_metrics = compute_metrics(raw_vel_2d, gt_vel_2d, raw=None, nyq_val=None, name="RAW")

    # [FIX] 重新命名：使用正確的變數名
    optimized_cls_metrics = compute_metrics(dealiased_cls_optimized, gt_vel_2d, raw=raw_vel_2d, nyq_val=nyq_val, name="Optimized_Classification",
                                           correction_threshold=correction_threshold, correction_tolerance=correction_tolerance,
                                           use_physics_method=use_physics_method)
    cls_metrics = compute_metrics(dealiased_cls, gt_vel_2d, raw=raw_vel_2d, nyq_val=nyq_val, name="Traditional_Classification",
                                 correction_threshold=correction_threshold, correction_tolerance=correction_tolerance,
                                 use_physics_method=use_physics_method)

    # 如果是物理約束模型，也計算物理分支的指標
    physics_metrics = None
    if physics_dealiased is not None:
        physics_metrics = compute_metrics(physics_dealiased, gt_vel_2d, raw=raw_vel_2d, nyq_val=nyq_val, name="Physics",
                                         correction_threshold=correction_threshold, correction_tolerance=correction_tolerance,
                                         use_physics_method=use_physics_method)

    # [NEW] 計算軟分類和置信度導向分支的指標
    soft_classification_metrics = None
    confidence_guided_metrics = None

    if soft_classification is not None:
        soft_classification_metrics = compute_metrics(soft_classification, gt_vel_2d, raw=raw_vel_2d, nyq_val=nyq_val, name="Soft_Classification",
                                                     correction_threshold=correction_threshold, correction_tolerance=correction_tolerance,
                                                     use_physics_method=use_physics_method)

    if confidence_guided is not None:
        confidence_guided_metrics = compute_metrics(confidence_guided, gt_vel_2d, raw=raw_vel_2d, nyq_val=nyq_val, name="Confidence_Guided",
                                                   correction_threshold=correction_threshold, correction_tolerance=correction_tolerance,
                                                   use_physics_method=use_physics_method)

    # [TIMEOUT] 計算時間平滑後的指標
    temporal_smoothed_metrics = {}
    if len(temporal_smoothed_results) > 0:
        print(f"\n[TIMEOUT]  計算時間平滑後的評估指標...")
        for branch_name, smoothed_pred in temporal_smoothed_results.items():
            temporal_smoothed_metrics[branch_name] = compute_metrics(
                smoothed_pred, gt_vel_2d, raw=raw_vel_2d, nyq_val=nyq_val,
                name=f"Temporal_{branch_name}",
                correction_threshold=correction_threshold,
                correction_tolerance=correction_tolerance,
                use_physics_method=use_physics_method
            )

    print("\n[STATS] 傳統評估指標:")
    print(f"   RAW vs GT               : RMSE={raw_metrics['rmse']:.3f}, MAE={raw_metrics['mae']:.3f}, Corr={raw_metrics['corr']:.3f}")
    print(f"   Optimized Classification: RMSE={optimized_cls_metrics['rmse']:.3f}, MAE={optimized_cls_metrics['mae']:.3f}, Corr={optimized_cls_metrics['corr']:.3f}")
    print(f"   Traditional Classification: RMSE={cls_metrics['rmse']:.3f}, MAE={cls_metrics['mae']:.3f}, Corr={cls_metrics['corr']:.3f}")

    if soft_classification_metrics is not None:
        print(f"   Soft Classification     : RMSE={soft_classification_metrics['rmse']:.3f}, MAE={soft_classification_metrics['mae']:.3f}, Corr={soft_classification_metrics['corr']:.3f}")

    if confidence_guided_metrics is not None:
        print(f"[TARGET] Confidence-Guided      : RMSE={confidence_guided_metrics['rmse']:.3f}, MAE={confidence_guided_metrics['mae']:.3f}, Corr={confidence_guided_metrics['corr']:.3f}")

    if physics_metrics is not None:
        print(f"   Physics Branch          : RMSE={physics_metrics['rmse']:.3f}, MAE={physics_metrics['mae']:.3f}, Corr={physics_metrics['corr']:.3f}")

    # [TIMEOUT] 打印時間平滑後的指標及改善情況
    if len(temporal_smoothed_metrics) > 0:
        print(f"\n[TIMEOUT]  時間平滑後的評估指標 (α={temporal_alpha}):")

        # 優化分類分支
        if 'optimized_classification' in temporal_smoothed_metrics:
            ts_metrics = temporal_smoothed_metrics['optimized_classification']
            orig_rmse = optimized_cls_metrics['rmse']
            improvement = orig_rmse - ts_metrics['rmse']
            improvement_pct = (improvement / orig_rmse) * 100
            print(f"   Optimized Classification (Temporal): RMSE={ts_metrics['rmse']:.3f}, MAE={ts_metrics['mae']:.3f}, Corr={ts_metrics['corr']:.3f}")
            print(f"      改善: {improvement:+.3f} m/s ({improvement_pct:+.1f}%)")

        # 傳統分類分支
        if 'traditional_classification' in temporal_smoothed_metrics:
            ts_metrics = temporal_smoothed_metrics['traditional_classification']
            orig_rmse = cls_metrics['rmse']
            improvement = orig_rmse - ts_metrics['rmse']
            improvement_pct = (improvement / orig_rmse) * 100
            print(f"   Traditional Classification (Temporal): RMSE={ts_metrics['rmse']:.3f}, MAE={ts_metrics['mae']:.3f}, Corr={ts_metrics['corr']:.3f}")
            print(f"      改善: {improvement:+.3f} m/s ({improvement_pct:+.1f}%)")

        # 軟分類分支
        if 'soft_classification' in temporal_smoothed_metrics and soft_classification_metrics:
            ts_metrics = temporal_smoothed_metrics['soft_classification']
            orig_rmse = soft_classification_metrics['rmse']
            improvement = orig_rmse - ts_metrics['rmse']
            improvement_pct = (improvement / orig_rmse) * 100
            print(f"   Soft Classification (Temporal): RMSE={ts_metrics['rmse']:.3f}, MAE={ts_metrics['mae']:.3f}, Corr={ts_metrics['corr']:.3f}")
            print(f"      改善: {improvement:+.3f} m/s ({improvement_pct:+.1f}%)")

        # 置信度導向分支
        if 'confidence_guided' in temporal_smoothed_metrics and confidence_guided_metrics:
            ts_metrics = temporal_smoothed_metrics['confidence_guided']
            orig_rmse = confidence_guided_metrics['rmse']
            improvement = orig_rmse - ts_metrics['rmse']
            improvement_pct = (improvement / orig_rmse) * 100
            print(f"[TARGET] Confidence-Guided (Temporal): RMSE={ts_metrics['rmse']:.3f}, MAE={ts_metrics['mae']:.3f}, Corr={ts_metrics['corr']:.3f}")
            print(f"      改善: {improvement:+.3f} m/s ({improvement_pct:+.1f}%)")

        # 物理約束分支
        if 'physics_constrained' in temporal_smoothed_metrics and physics_metrics:
            ts_metrics = temporal_smoothed_metrics['physics_constrained']
            orig_rmse = physics_metrics['rmse']
            improvement = orig_rmse - ts_metrics['rmse']
            improvement_pct = (improvement / orig_rmse) * 100
            print(f"   Physics Branch (Temporal): RMSE={ts_metrics['rmse']:.3f}, MAE={ts_metrics['mae']:.3f}, Corr={ts_metrics['corr']:.3f}")
            print(f"      改善: {improvement:+.3f} m/s ({improvement_pct:+.1f}%)")

    # 打印反折錯成功率指標
    if 'correction_success_rate' in optimized_cls_metrics:
        method_name = optimized_cls_metrics.get('evaluation_method', 'unknown')
        print(f"\n[TARGET] 反折錯修正成功率指標 ({method_name}):")
        print("  【優化分類輸出】")
        print(f"    需要修正像素: {optimized_cls_metrics['need_correction_pixels']:,} ({optimized_cls_metrics['aliased_pixels_ratio']:.1%})")
        if use_physics_method:
            print(f"    無alias像素: {optimized_cls_metrics.get('no_alias_pixels', 0):,}")
        print(f"    成功修正像素: {optimized_cls_metrics['corrected_pixels']:,}")
        print(f"    修正成功率: {optimized_cls_metrics['correction_success_rate']:.1%}")
        print(f"    整體改進率: {optimized_cls_metrics['improvement_ratio']:.1%}")
        print(f"    過度修正: {optimized_cls_metrics['overcorrection_pixels']:,}, 修正不足: {optimized_cls_metrics['undercorrection_pixels']:,}")
        if use_physics_method:
            print(f"    保持良好: {optimized_cls_metrics.get('maintained_good_pixels', 0):,}")
        
    if 'correction_success_rate' in cls_metrics:
        print("  【分類輸出】")  
        print(f"    需要修正像素: {cls_metrics['need_correction_pixels']:,} ({cls_metrics['aliased_pixels_ratio']:.1%})")
        if use_physics_method:
            print(f"    無alias像素: {cls_metrics.get('no_alias_pixels', 0):,}")
        print(f"    成功修正像素: {cls_metrics['corrected_pixels']:,}")
        print(f"    修正成功率: {cls_metrics['correction_success_rate']:.1%}")
        print(f"    整體改進率: {cls_metrics['improvement_ratio']:.1%}")
        print(f"    過度修正: {cls_metrics['overcorrection_pixels']:,}, 修正不足: {cls_metrics['undercorrection_pixels']:,}")
        if use_physics_method:
            print(f"    保持良好: {cls_metrics.get('maintained_good_pixels', 0):,}")

    # [NEW] 置信度導向分支的成功率指標
    if confidence_guided_metrics is not None and 'correction_success_rate' in confidence_guided_metrics:
        print("  【[TARGET] 置信度導向輸出】")
        print(f"    需要修正像素: {confidence_guided_metrics['need_correction_pixels']:,} ({confidence_guided_metrics['aliased_pixels_ratio']:.1%})")
        if use_physics_method:
            print(f"    無alias像素: {confidence_guided_metrics.get('no_alias_pixels', 0):,}")
        print(f"    成功修正像素: {confidence_guided_metrics['corrected_pixels']:,}")
        print(f"    修正成功率: {confidence_guided_metrics['correction_success_rate']:.1%}")
        print(f"    整體改進率: {confidence_guided_metrics['improvement_ratio']:.1%}")
        print(f"    過度修正: {confidence_guided_metrics['overcorrection_pixels']:,}, 修正不足: {confidence_guided_metrics['undercorrection_pixels']:,}")
        if use_physics_method:
            print(f"    保持良好: {confidence_guided_metrics.get('maintained_good_pixels', 0):,}")

    # 軟分類分支的成功率指標
    if soft_classification_metrics is not None and 'correction_success_rate' in soft_classification_metrics:
        print("  【軟分類輸出】")
        print(f"    需要修正像素: {soft_classification_metrics['need_correction_pixels']:,} ({soft_classification_metrics['aliased_pixels_ratio']:.1%})")
        if use_physics_method:
            print(f"    無alias像素: {soft_classification_metrics.get('no_alias_pixels', 0):,}")
        print(f"    成功修正像素: {soft_classification_metrics['corrected_pixels']:,}")
        print(f"    修正成功率: {soft_classification_metrics['correction_success_rate']:.1%}")
        print(f"    整體改進率: {soft_classification_metrics['improvement_ratio']:.1%}")
        print(f"    過度修正: {soft_classification_metrics['overcorrection_pixels']:,}, 修正不足: {soft_classification_metrics['undercorrection_pixels']:,}")
        if use_physics_method:
            print(f"    保持良好: {soft_classification_metrics.get('maintained_good_pixels', 0):,}")

    # 詳細的反折錯成功分析 (使用優化分類輸出)
    if 'dealiasing_metrics' in optimized_cls_metrics:
        print("\n[LIST] 詳細反折錯分析 (優化分類輸出):")
        if use_physics_method:
            print_physics_based_summary(optimized_cls_metrics['dealiasing_metrics'],
                                      title="Mixed Patch Model - Physics-based Analysis")
        else:
            print_correction_summary(optimized_cls_metrics['dealiasing_metrics'],
                                    title="Mixed Patch Model - Threshold-based Analysis")

    # 8.5 梯度檢測（檢查預測結果的空間平滑性）
    print("\n[SEARCH] 梯度檢測分析 (檢測異常跳躍):")
    gradient_results = {}

    # 對每個預測分支進行梯度檢測
    branches_to_check = {
        "raw": raw_vel_2d,
        "optimized_classification": dealiased_cls_optimized,
        "traditional_classification": dealiased_cls,
        "gt": gt_vel_2d
    }

    # 添加其他分支（如果存在）
    if physics_dealiased is not None:
        branches_to_check["physics_constrained"] = physics_dealiased
    if soft_classification is not None:
        branches_to_check["soft_classification"] = soft_classification
    if confidence_guided is not None:
        branches_to_check["confidence_guided"] = confidence_guided

    for branch_name, velocity_field in branches_to_check.items():
        try:
            all_gradients, grad_stats = compute_gradients_with_periodicity(velocity_field, nyq_val)
            gradient_results[branch_name] = grad_stats

            print(f"\n  【{branch_name.upper()}】")
            print(f"    有效梯度對數: {grad_stats['total_valid_pairs']:,}")
            print(f"    平均梯度: {grad_stats['mean_gradient']:.6f} fold")
            print(f"    P95梯度: {grad_stats['p95']:.6f} fold")
            print(f"    P99梯度: {grad_stats['p99']:.6f} fold")
            print(f"    最大梯度: {grad_stats['max_gradient']:.6f} fold")
            print(f"    正常 (< 0.5 fold): {grad_stats['jumps_0_to_0.5']:,} ({grad_stats['normal_ratio']*100:.2f}%)")
            print(f"    異常 (0.5-1.0 fold): {grad_stats['jumps_0.5_to_1.0']:,} ({(grad_stats['jumps_0.5_to_1.0']/grad_stats['total_valid_pairs'])*100:.2f}%)")
            print(f"    嚴重 (≥ 1.0 fold): {grad_stats['jumps_1.0_to_1.5'] + grad_stats['jumps_above_1.5']:,} ({grad_stats['severe_anomaly_ratio']*100:.2f}%)")
            print(f"    總異常比例: {grad_stats['anomaly_ratio']*100:.2f}%")

        except Exception as e:
            print(f"\n  【{branch_name.upper()}】")
            print(f"    [WARNING]  梯度計算失敗: {e}")
            gradient_results[branch_name] = None

    # 9. 保存可視化結果 (包含反折錯成功分析)
    if save_results:
        save_visualizations(raw_vel_2d, dealiased_cls_optimized, dealiased_cls, gt_vel_2d, residual,
                           raw_csv, result_dir, nyq_val=nyq_val, use_physics_method=use_physics_method,
                           raw_gz=raw_gz, gt_gz=gt_gz, shape_path=shape_path,
                           gt_csv=gt_csv,  # [FIX] 傳入 GT CSV
                           pure_regression=pure_regression,
                           physics_dealiased=physics_dealiased, fold_probs=fold_probs,
                           # [NEW] 新增參數
                           soft_classification=soft_classification,
                           confidence_guided=confidence_guided,
                           alias_confidence=alias_confidence,
                           # [TIMEOUT] 時間平滑結果
                           temporal_smoothed_results=temporal_smoothed_results,
                           temporal_alpha=temporal_alpha)

    # 10. 保存評估結果
    results = {
        "raw_csv": raw_csv,
        "gt_csv": gt_csv,
        "model_path": model_path,
        "data_shape": raw_vel_2d.shape,
        "nyquist_velocity": float(nyq_val),
        "use_physics_model": use_physics_model,
        "metrics": {
            "raw": raw_metrics,
            "optimized_classification": optimized_cls_metrics,  # [FIX] 軟分類訓練優化後的硬分類
            "traditional_classification": cls_metrics           # [FIX] 傳統硬分類（通過logits重新計算）
        },
        # [SEARCH] 添加梯度檢測結果
        "gradient_analysis": gradient_results
    }

    # 如果有物理分支指標，也保存
    if physics_metrics is not None:
        results["metrics"]["physics_constrained"] = physics_metrics

    # [NEW] 如果有軟分類和置信度導向指標，也保存
    if soft_classification_metrics is not None:
        results["metrics"]["soft_classification"] = soft_classification_metrics

    if confidence_guided_metrics is not None:
        results["metrics"]["confidence_guided"] = confidence_guided_metrics

    # [TIMEOUT] 如果有時間平滑指標，也保存
    if len(temporal_smoothed_metrics) > 0:
        results["temporal_smoothing"] = {
            "enabled": True,
            "alpha": temporal_alpha,
            "previous_prediction": previous_prediction,
            "metrics": temporal_smoothed_metrics
        }

    import json
    if save_results:
        results_path = os.path.join(result_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)

    # [TIMEOUT] 保存時間平滑後的預測結果到CSV (供下一時刻使用)
    if save_results and len(temporal_smoothed_results) > 0:
        temporal_dir = os.path.join(result_dir, "temporal_smoothed")
        os.makedirs(temporal_dir, exist_ok=True)

        # 載入原始CSV以獲取座標資訊
        df_raw = pd.read_csv(raw_csv)
        azimuths_raw = df_raw['Azimuth'].unique()
        azimuths = sort_azimuths_circular(azimuths_raw)
        ranges = sorted(df_raw['Range'].unique())

        for branch_name, smoothed_pred in temporal_smoothed_results.items():
            # 將矩陣轉換回CSV格式
            rows = []
            for i, az in enumerate(azimuths):
                for j, rng in enumerate(ranges):
                    value = smoothed_pred[i, j]
                    if np.isnan(value):
                        value = -999.0
                    rows.append({
                        'Azimuth': az,
                        'Range': rng,
                        'Nyquist': nyq_val,
                        'Value': value
                    })

            df_output = pd.DataFrame(rows)
            output_csv = os.path.join(temporal_dir, f"{os.path.basename(raw_csv).replace('.csv', '')}_{branch_name}_temporal.csv")
            df_output.to_csv(output_csv, index=False)
            print(f"   [TIMEOUT]  已保存時間平滑結果: {output_csv}")

    if save_results:
        print(f"[FOLDER] 結果已保存到: {result_dir}")
    else:
        print(f"[OK] 推理完成（未保存文件）")

    # [NEW] 添加預測結果到返回值，方便輕量級推理提取
    results['predictions'] = {
        'optimized_classification': dealiased_cls_optimized,
        'traditional_classification': dealiased_cls,
        'pure_regression': pure_regression
    }
    if physics_dealiased is not None:
        results['predictions']['physics_constrained'] = physics_dealiased
    if soft_classification is not None:
        results['predictions']['soft_classification'] = soft_classification
    if confidence_guided is not None:
        results['predictions']['confidence_guided'] = confidence_guided

    return results

def save_visualizations(raw_vel, dealiased_cls_optimized, dealiased_cls, gt_vel, residual, raw_csv, result_dir,
                       nyq_val=None, use_physics_method=True, raw_gz=None, gt_gz=None, shape_path=None,
                       gt_csv=None,  # [FIX] 新增 GT CSV 參數
                       pure_regression=None, physics_dealiased=None, fold_probs=None,
                       # [NEW] 新增參數
                       soft_classification=None, confidence_guided=None, alias_confidence=None,
                       # [TIMEOUT] 時間平滑參數
                       temporal_smoothed_results=None, temporal_alpha=None):
    """保存可視化圖像 (包含反折錯成功分析和地理位置圖)"""
    basename = os.path.basename(raw_csv).replace('.csv', '')

    # 設置colormap和normalization
    cmap = 'RdYlBu_r'
    vmin, vmax = np.nanmin([raw_vel, gt_vel]), np.nanmax([raw_vel, gt_vel])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # 嘗試自動尋找.gz檔案 (如果沒有提供)
    if raw_gz is None:
        raw_gz = find_corresponding_gz_file(raw_csv)
        if raw_gz:
            print(f"[SEARCH] 自動找到對應的GZ檔案: {raw_gz}")
    
    if gt_gz is None:
        gt_gz = find_corresponding_gz_file(raw_csv.replace('_raw.', '_vdaqc.'))
        if gt_gz:
            print(f"[SEARCH] 自動找到對應的GT GZ檔案: {gt_gz}")
    
    # 1. 保存基本比較圖 (矩陣視圖) - 支援物理約束模型
    print(f"[IMG]️  選擇可視化布局: physics_dealiased={'是' if physics_dealiased is not None else '否'}")
    if physics_dealiased is not None:
        # 3x3 佈局：物理約束模型版本
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f'Physics-Constrained Model Results: {basename}', fontsize=16)

        # 第一行：Raw, GT, Classification
        im1 = axes[0,0].imshow(raw_vel, cmap=cmap, norm=norm, aspect='auto')
        axes[0,0].set_title('Raw Velocity')
        axes[0,0].set_xlabel('Range'); axes[0,0].set_ylabel('Azimuth')

        im2 = axes[0,1].imshow(gt_vel, cmap=cmap, norm=norm, aspect='auto')
        axes[0,1].set_title('Ground Truth')
        axes[0,1].set_xlabel('Range'); axes[0,1].set_ylabel('Azimuth')

        im3 = axes[0,2].imshow(dealiased_cls, cmap=cmap, norm=norm, aspect='auto')
        axes[0,2].set_title('Dealiased (Classification)')
        axes[0,2].set_xlabel('Range'); axes[0,2].set_ylabel('Azimuth')

        # 第二行：物理約束結果, Fold機率分布, 分類vs物理差異
        im4 = axes[1,0].imshow(physics_dealiased, cmap=cmap, norm=norm, aspect='auto')
        axes[1,0].set_title('Dealiased (Physics-Constrained)')
        axes[1,0].set_xlabel('Range'); axes[1,0].set_ylabel('Azimuth')

        # 顯示最可能的fold number
        if fold_probs is not None:
            fold_prediction = np.argmax(fold_probs, axis=-1) - 2  # 轉換為[-2,-1,0,1,2]
            im5 = axes[1,1].imshow(fold_prediction, cmap='RdYlBu', vmin=-2, vmax=2, aspect='auto')
            axes[1,1].set_title('Predicted Fold Number')
        else:
            # [NEW] 如果有置信度導向分支，顯示它；否則顯示優化分類
            if confidence_guided is not None:
                im5 = axes[1,1].imshow(confidence_guided, cmap=cmap, norm=norm, aspect='auto')
                axes[1,1].set_title('[TARGET] Confidence-Guided')
            else:
                im5 = axes[1,1].imshow(dealiased_cls_optimized, cmap=cmap, norm=norm, aspect='auto')
                axes[1,1].set_title('Dealiased (Mixed Branch)')
        axes[1,1].set_xlabel('Range'); axes[1,1].set_ylabel('Azimuth')

        # 分類分支與物理分支的差異
        branch_diff = physics_dealiased - dealiased_cls
        im6 = axes[1,2].imshow(branch_diff, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
        axes[1,2].set_title('Physics - Classification')
        axes[1,2].set_xlabel('Range'); axes[1,2].set_ylabel('Azimuth')

        # 第三行：Residual, GT差異 (分類), GT差異 (物理)
        im7 = axes[2,0].imshow(residual, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
        axes[2,0].set_title('Velocity Residual')
        axes[2,0].set_xlabel('Range'); axes[2,0].set_ylabel('Azimuth')

        diff_cls = gt_vel - dealiased_cls
        im8 = axes[2,1].imshow(diff_cls, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
        axes[2,1].set_title('GT - Classification')
        axes[2,1].set_xlabel('Range'); axes[2,1].set_ylabel('Azimuth')

        diff_physics = gt_vel - physics_dealiased
        im9 = axes[2,2].imshow(diff_physics, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
        axes[2,2].set_title('GT - Physics-Constrained')
        axes[2,2].set_xlabel('Range'); axes[2,2].set_ylabel('Azimuth')

        # 添加colorbar
        plt.colorbar(im1, ax=axes[0,0], shrink=0.6)
        plt.colorbar(im2, ax=axes[0,1], shrink=0.6)
        plt.colorbar(im3, ax=axes[0,2], shrink=0.6)
        plt.colorbar(im4, ax=axes[1,0], shrink=0.6)
        plt.colorbar(im5, ax=axes[1,1], shrink=0.6)
        plt.colorbar(im6, ax=axes[1,2], shrink=0.6)
        plt.colorbar(im7, ax=axes[2,0], shrink=0.6)
        plt.colorbar(im8, ax=axes[2,1], shrink=0.6)
        plt.colorbar(im9, ax=axes[2,2], shrink=0.6)

    elif pure_regression is not None:
        # 3x3 佈局：包含原始regression輸出
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f'Mixed Patch Model Results (with Physics Post-processing): {basename}', fontsize=16)
        
        # 第一行：Raw, GT, Classification
        im1 = axes[0,0].imshow(raw_vel, cmap=cmap, norm=norm, aspect='auto')
        axes[0,0].set_title('Raw Velocity')
        axes[0,0].set_xlabel('Range'); axes[0,0].set_ylabel('Azimuth')
        
        im2 = axes[0,1].imshow(gt_vel, cmap=cmap, norm=norm, aspect='auto')
        axes[0,1].set_title('Ground Truth')
        axes[0,1].set_xlabel('Range'); axes[0,1].set_ylabel('Azimuth')
        
        im3 = axes[0,2].imshow(dealiased_cls, cmap=cmap, norm=norm, aspect='auto')
        axes[0,2].set_title('Dealiased (Classification)')
        axes[0,2].set_xlabel('Range'); axes[0,2].set_ylabel('Azimuth')
        
        # 第二行：純回歸, 混合分支(分類+回歸), 分類校正效果
        im4 = axes[1,0].imshow(pure_regression, cmap=cmap, norm=norm, aspect='auto')
        axes[1,0].set_title('Pure Regression (Raw + Residual)')
        axes[1,0].set_xlabel('Range'); axes[1,0].set_ylabel('Azimuth')

        im5 = axes[1,1].imshow(dealiased_cls_optimized, cmap=cmap, norm=norm, aspect='auto')
        axes[1,1].set_title('Mixed Branch (Classification + Regression)')
        axes[1,1].set_xlabel('Range'); axes[1,1].set_ylabel('Azimuth')

        classification_correction = dealiased_cls_optimized - pure_regression
        im6 = axes[1,2].imshow(classification_correction, cmap='RdBu_r', vmin=-40, vmax=40, aspect='auto')
        axes[1,2].set_title('Classification Correction Effect')
        axes[1,2].set_xlabel('Range'); axes[1,2].set_ylabel('Azimuth')
        
        # 第三行：Residual, GT差異 (原始), GT差異 (修正後)
        im7 = axes[2,0].imshow(residual, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
        axes[2,0].set_title('Velocity Residual')
        axes[2,0].set_xlabel('Range'); axes[2,0].set_ylabel('Azimuth')
        
        diff_pure = gt_vel - pure_regression
        im8 = axes[2,1].imshow(diff_pure, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
        axes[2,1].set_title('GT - Pure Regression')
        axes[2,1].set_xlabel('Range'); axes[2,1].set_ylabel('Azimuth')
        
        diff_mixed = gt_vel - dealiased_cls_optimized
        im9 = axes[2,2].imshow(diff_mixed, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
        axes[2,2].set_title('GT - Mixed Branch')
        axes[2,2].set_xlabel('Range'); axes[2,2].set_ylabel('Azimuth')
        
        # 添加colorbar
        plt.colorbar(im1, ax=axes[0,0], shrink=0.6)
        plt.colorbar(im2, ax=axes[0,1], shrink=0.6)
        plt.colorbar(im3, ax=axes[0,2], shrink=0.6)
        plt.colorbar(im4, ax=axes[1,0], shrink=0.6)
        plt.colorbar(im5, ax=axes[1,1], shrink=0.6)
        plt.colorbar(im6, ax=axes[1,2], shrink=0.6)
        plt.colorbar(im7, ax=axes[2,0], shrink=0.6)
        plt.colorbar(im8, ax=axes[2,1], shrink=0.6)
        plt.colorbar(im9, ax=axes[2,2], shrink=0.6)
        
    else:
        # 原始 2x3 佈局 (向後兼容)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Mixed Patch Model Results: {basename}', fontsize=16)
        
        # Raw
        im1 = axes[0,0].imshow(raw_vel, cmap=cmap, norm=norm, aspect='auto')
        axes[0,0].set_title('Raw Velocity')
        axes[0,0].set_xlabel('Range'); axes[0,0].set_ylabel('Azimuth')
        
        # GT
        im2 = axes[0,1].imshow(gt_vel, cmap=cmap, norm=norm, aspect='auto')
        axes[0,1].set_title('Ground Truth')
        axes[0,1].set_xlabel('Range'); axes[0,1].set_ylabel('Azimuth')
        
        # Regression
        im3 = axes[0,2].imshow(dealiased_cls_optimized, cmap=cmap, norm=norm, aspect='auto')
        axes[0,2].set_title('Dealiased (Mixed Branch)')
        axes[0,2].set_xlabel('Range'); axes[0,2].set_ylabel('Azimuth')
        
        # Classification
        im4 = axes[1,0].imshow(dealiased_cls, cmap=cmap, norm=norm, aspect='auto')
        axes[1,0].set_title('Dealiased (Classification)')
        axes[1,0].set_xlabel('Range'); axes[1,0].set_ylabel('Azimuth')
        
        # Residual
        im5 = axes[1,1].imshow(residual, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
        axes[1,1].set_title('Velocity Residual')
        axes[1,1].set_xlabel('Range'); axes[1,1].set_ylabel('Azimuth')
        
        # Difference (GT - Dealiased_reg)
        diff = gt_vel - dealiased_cls_optimized
        im6 = axes[1,2].imshow(diff, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
        axes[1,2].set_title('GT - Mixed Branch')
        axes[1,2].set_xlabel('Range'); axes[1,2].set_ylabel('Azimuth')
        
        # 添加colorbar
        plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
        plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
        plt.colorbar(im3, ax=axes[0,2], shrink=0.8)
        plt.colorbar(im4, ax=axes[1,0], shrink=0.8)
        plt.colorbar(im5, ax=axes[1,1], shrink=0.8)
        plt.colorbar(im6, ax=axes[1,2], shrink=0.8)
    
    plt.tight_layout()
    
    # 保存基本比較圖
    comparison_path = os.path.join(result_dir, f"{basename}_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 基本比較圖已保存: {comparison_path}")
    
    # 2. 保存反折錯成功分析圖 (如果提供了Nyquist速度)
    if nyq_val is not None:
        try:
            method_suffix = "physics" if use_physics_method else "threshold"
            
            if use_physics_method:
                # 使用物理原理方法的可視化
                # 回歸輸出
                success_analysis_path = os.path.join(result_dir, f"{basename}_dealiasing_success_{method_suffix}.png")
                fig_success, metrics_reg = visualize_physics_based_correction(
                    raw_vel, dealiased_cls_optimized, gt_vel, nyq_val,
                    save_path=success_analysis_path,
                    title=f"Physics-based Dealiasing Analysis - {basename} (Regression)"
                )
                plt.close()
                
                # 分類輸出
                success_analysis_cls_path = os.path.join(result_dir, f"{basename}_dealiasing_success_cls_{method_suffix}.png")
                fig_success_cls, metrics_cls = visualize_physics_based_correction(
                    raw_vel, dealiased_cls, gt_vel, nyq_val,
                    save_path=success_analysis_cls_path,
                    title=f"Physics-based Dealiasing Analysis - {basename} (Classification)"
                )
                plt.close()
                
            else:
                # 使用傳統閾值方法的可視化
                # 回歸輸出
                success_analysis_path = os.path.join(result_dir, f"{basename}_dealiasing_success_{method_suffix}.png")
                fig_success, metrics_reg = visualize_correction_success(
                    raw_vel, dealiased_cls_optimized, gt_vel, nyq_val,
                    save_path=success_analysis_path,
                    title=f"Threshold-based Dealiasing Analysis - {basename} (Regression)"
                )
                plt.close()
                
                # 分類輸出  
                success_analysis_cls_path = os.path.join(result_dir, f"{basename}_dealiasing_success_cls_{method_suffix}.png")
                fig_success_cls, metrics_cls = visualize_correction_success(
                    raw_vel, dealiased_cls, gt_vel, nyq_val,
                    save_path=success_analysis_cls_path,
                    title=f"Threshold-based Dealiasing Analysis - {basename} (Classification)"
                )
                plt.close()
            
            print(f"[TARGET] 反折錯成功分析圖已保存 ({method_suffix} method):")
            print(f"   回歸輸出: {success_analysis_path}")
            print(f"   分類輸出: {success_analysis_cls_path}")
            
        except Exception as e:
            print(f"[WARNING]  保存反折錯成功分析圖時出錯: {e}")
    
    else:
        print("[WARNING]  未提供Nyquist速度，跳過反折錯成功分析圖")
    
    # 3. 保存地理位置可視化 (如果提供了.gz檔案)
    if raw_gz and GEO_VIZ_AVAILABLE:
        try:
            print("[GEO] 創建地理位置可視化...")
            save_geographical_visualizations(raw_vel, dealiased_cls_optimized, dealiased_cls, gt_vel,
                                           raw_csv, raw_gz, result_dir, shape_path,
                                           gt_csv_path=gt_csv,  # [FIX] 傳入 GT CSV
                                           gt_gz_path=gt_gz,    # [FIX] 傳入 GT .gz
                                           pure_regression=pure_regression,
                                           physics_dealiased=physics_dealiased,
                                           # [NEW] 新增置信度導向參數
                                           soft_classification=soft_classification,
                                           confidence_guided=confidence_guided,
                                           alias_confidence=alias_confidence,
                                           # [TIMEOUT] 時間平滑參數
                                           temporal_smoothed_results=temporal_smoothed_results,
                                           temporal_alpha=temporal_alpha)
        except Exception as e:
            print(f"[WARNING]  地理可視化失敗: {e}")
            import traceback
            traceback.print_exc()
    elif not GEO_VIZ_AVAILABLE:
        print("[WARNING]  地理可視化模塊未安裝，跳過地理圖")
    else:
        print("[WARNING]  未提供GZ檔案，跳過地理圖")

def save_geographical_visualizations(raw_vel, dealiased_cls_optimized, dealiased_cls, gt_vel,
                                   csv_path, gz_path, result_dir, shape_path=None,
                                   gt_csv_path=None, gt_gz_path=None,  # [FIX] 新增 GT 獨立檔案路徑
                                   pure_regression=None, physics_dealiased=None,
                                   # [NEW] 新增置信度導向參數
                                   soft_classification=None, confidence_guided=None, alias_confidence=None,
                                   # [TIMEOUT] 時間平滑參數
                                   temporal_smoothed_results=None, temporal_alpha=None):
    """保存地理位置可視化圖像 - GT 使用獨立的 CSV 和 .gz 檔案"""
    basename = os.path.basename(csv_path).replace('.csv', '')
    
    try:
        # 1) 從 .gz 檔拿雷達資訊 (經緯度、nyquist...等)，如 dealiasing_test_model.py
        print("   讀取雷達 .gz 檔案...")
        raw_radar_data, raw_radius_km, raw_nyq = read_cwb_radar_sweep(gz_path)
        lat = raw_radar_data.latitude['data'][0]
        lon = raw_radar_data.longitude['data'][0]

        print(f"   雷達站: {raw_radar_data.metadata['instrument_name']}")
        print(f"   位置: {lat:.3f}°N, {lon:.3f}°E")
        print(f"   距離範圍: {raw_radius_km[0]:.1f} - {raw_radius_km[-1]:.1f} km")

        # 2) [FIX] 從 CSV 讀取實際的方位角順序，替換 .gz 假設的線性方位角
        print("   從 CSV 讀取實際方位角...")
        csv_data = pd.read_csv(csv_path)
        actual_azimuths = csv_data['Azimuth'].unique()  # 保持原始順序

        # 更新 radar object 的方位角數據
        azimuth_metadata = get_metadata('azimuth')
        azimuth_metadata['data'] = np.array(actual_azimuths, dtype='float32')
        raw_radar_data.azimuth = azimuth_metadata

        print(f"   實際方位角範圍: [{actual_azimuths.min():.1f}° - {actual_azimuths.max():.1f}°]")
        print(f"   方位角數量: {len(actual_azimuths)}")

        # 檢查數據形狀
        nRay, nGate = raw_vel.shape
        print(f"   CSV數據形狀: {raw_vel.shape}")

        if nRay != len(actual_azimuths):
            print(f"   [WARNING]  警告: CSV 矩陣行數 ({nRay}) 與方位角數量 ({len(actual_azimuths)}) 不一致！")
        
        # 3) Radar 物件更新 + 畫圖 (與 dealiasing_test_model.py 相同)
        raw_field = get_metadata('velocity')
        raw_field['data'] = raw_vel
        raw_radar_data.fields['velocity'] = raw_field

        dealiased_cls_optimized_field = get_metadata('velocity')
        dealiased_cls_optimized_field['data'] = dealiased_cls_optimized
        raw_radar_data.fields['dealiased_velocity_optimized'] = dealiased_cls_optimized_field
        
        dealiased_cls_field = get_metadata('velocity')
        dealiased_cls_field['data'] = dealiased_cls
        raw_radar_data.fields['dealiased_velocity_cls'] = dealiased_cls_field

        gt_field = get_metadata('velocity')
        gt_field['data'] = gt_vel
        raw_radar_data.fields['gt_velocity'] = gt_field
        
        # 如果有原始regression輸出，也添加到radar物件中
        if pure_regression is not None:
            pure_regression_field = get_metadata('velocity')
            pure_regression_field['data'] = pure_regression
            raw_radar_data.fields['dealiased_velocity_pure_reg'] = pure_regression_field

        # 如果有物理約束輸出，也添加到radar物件中
        if physics_dealiased is not None:
            physics_field = get_metadata('velocity')
            physics_field['data'] = physics_dealiased
            raw_radar_data.fields['dealiased_velocity_physics'] = physics_field

        # [NEW] 如果有軟分類輸出，也添加到radar物件中
        if soft_classification is not None:
            soft_field = get_metadata('velocity')
            soft_field['data'] = soft_classification
            raw_radar_data.fields['dealiased_velocity_soft'] = soft_field

        # [NEW] 如果有置信度導向輸出，也添加到radar物件中
        if confidence_guided is not None:
            confidence_field = get_metadata('velocity')
            confidence_field['data'] = confidence_guided
            raw_radar_data.fields['dealiased_velocity_confidence'] = confidence_field

        # [NEW] 如果有置信度數據，也添加到radar物件中
        if alias_confidence is not None:
            alias_conf_field = get_metadata('velocity')
            alias_conf_field['data'] = alias_confidence
            raw_radar_data.fields['alias_confidence'] = alias_conf_field

        # 4) 畫圖 (Raw)
        print("   繪製 Raw Velocity...")
        m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
        display = RadarMapDisplayBasemap(raw_radar_data)
        display.plot_ppi_map(
            'velocity', sweep=0, resolution='h', vmin=-80, vmax=80,
            cmap=cmap, norm=norm,
            min_lon=lon-1, max_lon=lon+1,
            min_lat=lat-1, max_lat=lat+1,
            mask_outside=True, projection='aeqd', basemap=m
        )
        display.plot_range_rings([raw_radius_km[-1]])
        raw_img = os.path.join(result_dir, f"{basename}_geo_raw.png")
        plt.savefig(raw_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Raw Velocity: {raw_img}")

        # 5) 畫圖 (Dealiased Regression)
        print("   繪製 Dealiased (Optimized Classification)...")
        m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
        display = RadarMapDisplayBasemap(raw_radar_data)
        display.plot_ppi_map(
            'dealiased_velocity_optimized', sweep=0, resolution='h', vmin=-80, vmax=80,
            cmap=cmap, norm=norm,
            min_lon=lon-1, max_lon=lon+1,
            min_lat=lat-1, max_lat=lat+1,
            mask_outside=True, projection='aeqd', basemap=m
        )
        display.plot_range_rings([raw_radius_km[-1]])
        dealiased_optimized_img = os.path.join(result_dir, f"{basename}_geo_dealiased_classification_optimized.png")
        plt.savefig(dealiased_optimized_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Dealiased (Optimized Classification): {dealiased_optimized_img}")

        # 6) 畫圖 (Dealiased Classification)
        print("   繪製 Dealiased (Classification)...")
        m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
        display = RadarMapDisplayBasemap(raw_radar_data)
        display.plot_ppi_map(
            'dealiased_velocity_cls', sweep=0, resolution='h', vmin=-80, vmax=80,
            cmap=cmap, norm=norm,
            min_lon=lon-1, max_lon=lon+1,
            min_lat=lat-1, max_lat=lat+1,
            mask_outside=True, projection='aeqd', basemap=m
        )
        display.plot_range_rings([raw_radius_km[-1]])
        dealiased_cls_img = os.path.join(result_dir, f"{basename}_geo_dealiased_classification.png")
        plt.savefig(dealiased_cls_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Dealiased (Classification): {dealiased_cls_img}")

        # 7) 畫圖 (GT) - [FIX] 使用獨立的 GT .gz 檔案
        if gt_vel is not None:
            if gt_gz_path and os.path.exists(gt_gz_path):
                print("   繪製 Ground Truth（使用獨立的 GT .gz 檔案）...")

                # 讀取 GT 專屬的 .gz 檔案
                gt_radar_data, gt_radius_km, gt_nyq = read_cwb_radar_sweep(gt_gz_path)
                gt_lat = gt_radar_data.latitude['data'][0]
                gt_lon = gt_radar_data.longitude['data'][0]

                # 從 GT CSV 讀取實際方位角
                if gt_csv_path and os.path.exists(gt_csv_path):
                    gt_csv_data = pd.read_csv(gt_csv_path)
                    gt_actual_azimuths = gt_csv_data['Azimuth'].unique()

                    # 更新 GT radar object 的方位角
                    gt_azimuth_metadata = get_metadata('azimuth')
                    gt_azimuth_metadata['data'] = np.array(gt_actual_azimuths, dtype='float32')
                    gt_radar_data.azimuth = gt_azimuth_metadata

                # 設置 GT 速度場
                gt_field = get_metadata('velocity')
                gt_field['data'] = gt_vel
                gt_radar_data.fields['velocity'] = gt_field

                # 繪製 GT
                m = plot_taiwan_basemap(gt_lon, gt_lat, gt_radius_km, shape_path)
                display = RadarMapDisplayBasemap(gt_radar_data)
                display.plot_ppi_map(
                    'velocity', sweep=0, resolution='h', vmin=-80, vmax=80,
                    cmap=cmap, norm=norm,
                    min_lon=gt_lon-1, max_lon=gt_lon+1,
                    min_lat=gt_lat-1, max_lat=gt_lat+1,
                    mask_outside=True, projection='aeqd', basemap=m
                )
                display.plot_range_rings([gt_radius_km[-1]])
                gt_img = os.path.join(result_dir, f"{basename}_geo_gt.png")
                plt.savefig(gt_img, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   [OK] Ground Truth: {gt_img}")
            else:
                print("   [WARNING]  未提供 GT .gz 檔案，使用 RAW 的 radar object 繪製 GT...")
                # 降級方案：使用 RAW 的 radar object（保持向後兼容）
                m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
                display = RadarMapDisplayBasemap(raw_radar_data)
                display.plot_ppi_map(
                    'gt_velocity', sweep=0, resolution='h', vmin=-80, vmax=80,
                    cmap=cmap, norm=norm,
                    min_lon=lon-1, max_lon=lon+1,
                    min_lat=lat-1, max_lat=lat+1,
                    mask_outside=True, projection='aeqd', basemap=m
                )
                display.plot_range_rings([raw_radius_km[-1]])
                gt_img = os.path.join(result_dir, f"{basename}_geo_gt.png")
                plt.savefig(gt_img, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   [OK] Ground Truth: {gt_img}")
        else:
            print("   [SKIP]  跳過 Ground Truth（未提供）")
        
        # 8) 如果有原始regression輸出，也畫出來
        if pure_regression is not None:
            print("   繪製 Dealiased (Pure Regression)...")
            m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
            display = RadarMapDisplayBasemap(raw_radar_data)
            display.plot_ppi_map(
                'dealiased_velocity_pure_reg', sweep=0, resolution='h', vmin=-80, vmax=80,
                cmap=cmap, norm=norm,
                min_lon=lon-1, max_lon=lon+1,
                min_lat=lat-1, max_lat=lat+1,
                mask_outside=True, projection='aeqd', basemap=m
            )
            display.plot_range_rings([raw_radius_km[-1]])
            pure_regression_img = os.path.join(result_dir, f"{basename}_geo_pure_regression.png")
            plt.savefig(pure_regression_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Dealiased (Pure Regression): {pure_regression_img}")

        # 9) 如果有物理約束輸出，也畫出來
        if physics_dealiased is not None:
            print("   繪製 Dealiased (Physics-Constrained)...")
            m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
            display = RadarMapDisplayBasemap(raw_radar_data)
            display.plot_ppi_map(
                'dealiased_velocity_physics', sweep=0, resolution='h', vmin=-80, vmax=80,
                cmap=cmap, norm=norm,
                min_lon=lon-1, max_lon=lon+1,
                min_lat=lat-1, max_lat=lat+1,
                mask_outside=True, projection='aeqd', basemap=m
            )
            display.plot_range_rings([raw_radius_km[-1]])
            physics_img = os.path.join(result_dir, f"{basename}_geo_physics_constrained.png")
            plt.savefig(physics_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Dealiased (Physics-Constrained): {physics_img}")

        # [NEW] 如果有軟分類輸出，也畫出來
        if soft_classification is not None:
            print("   繪製 Dealiased (Soft Classification)...")
            m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
            display = RadarMapDisplayBasemap(raw_radar_data)
            display.plot_ppi_map(
                'dealiased_velocity_soft', sweep=0, resolution='h', vmin=-80, vmax=80,
                cmap=cmap, norm=norm,
                min_lon=lon-1, max_lon=lon+1,
                min_lat=lat-1, max_lat=lat+1,
                mask_outside=True, projection='aeqd', basemap=m
            )
            display.plot_range_rings([raw_radius_km[-1]])
            soft_img = os.path.join(result_dir, f"{basename}_geo_soft_classification.png")
            plt.savefig(soft_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Dealiased (Soft Classification): {soft_img}")

        # [NEW] 如果有置信度導向輸出，也畫出來
        if confidence_guided is not None:
            print("   [TARGET] 繪製 Dealiased (Confidence-Guided)...")
            m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
            display = RadarMapDisplayBasemap(raw_radar_data)
            display.plot_ppi_map(
                'dealiased_velocity_confidence', sweep=0, resolution='h', vmin=-80, vmax=80,
                cmap=cmap, norm=norm,
                min_lon=lon-1, max_lon=lon+1,
                min_lat=lat-1, max_lat=lat+1,
                mask_outside=True, projection='aeqd', basemap=m
            )
            display.plot_range_rings([raw_radius_km[-1]])
            confidence_img = os.path.join(result_dir, f"{basename}_geo_confidence_guided.png")
            plt.savefig(confidence_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [TARGET] [OK] Dealiased (Confidence-Guided): {confidence_img}")

        # [NEW] 如果有置信度數據，也畫出來
        if alias_confidence is not None:
            print("   [INFO] 繪製 Alias Confidence...")
            m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
            display = RadarMapDisplayBasemap(raw_radar_data)
            display.plot_ppi_map(
                'alias_confidence', sweep=0, resolution='h', vmin=0, vmax=1,
                cmap='viridis', norm=None,
                min_lon=lon-1, max_lon=lon+1,
                min_lat=lat-1, max_lat=lat+1,
                mask_outside=True, projection='aeqd', basemap=m
            )
            display.plot_range_rings([raw_radius_km[-1]])
            alias_conf_img = os.path.join(result_dir, f"{basename}_geo_alias_confidence.png")
            plt.savefig(alias_conf_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [INFO] [OK] Alias Confidence: {alias_conf_img}")

        # [TIMEOUT] 時間平滑結果的地理可視化
        if temporal_smoothed_results is not None and len(temporal_smoothed_results) > 0:
            print(f"\n   [TIMEOUT] 繪製時間平滑結果 (α={temporal_alpha})...")

            for branch_name, smoothed_vel in temporal_smoothed_results.items():
                # 將時間平滑結果添加到 radar 資料對象
                field_name = f'temporal_{branch_name}'
                temporal_field = get_metadata('velocity')
                temporal_field['data'] = smoothed_vel
                raw_radar_data.fields[field_name] = temporal_field

                # 繪製地理圖
                print(f"      繪製 {branch_name} (Temporal Smoothed)...")
                m = plot_taiwan_basemap(lon, lat, raw_radius_km, shape_path)
                display = RadarMapDisplayBasemap(raw_radar_data)
                display.plot_ppi_map(
                    field_name, sweep=0, resolution='h', vmin=-80, vmax=80,
                    cmap=cmap, norm=norm,
                    min_lon=lon-1, max_lon=lon+1,
                    min_lat=lat-1, max_lat=lat+1,
                    mask_outside=True, projection='aeqd', basemap=m
                )
                display.plot_range_rings([raw_radius_km[-1]])
                temporal_img = os.path.join(result_dir, f"{basename}_geo_{branch_name}_temporal_alpha{temporal_alpha}.png")
                plt.savefig(temporal_img, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"      [OK] {branch_name} (Temporal): {temporal_img}")

        print(f"[GEO] 地理可視化完成")
        
    except Exception as e:
        print(f"[ERROR] 地理可視化失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='混合patch模型測試 (支持物理原理和閾值方法的反折錯成功率分析 + 地理位置可視化 + 物理約束模型)')
    parser.add_argument('--raw_csv', type=str, required=True,
                        help='原始速度CSV文件路徑')
    parser.add_argument('--gt_csv', type=str, required=True,
                        help='真實速度CSV文件路徑')
    parser.add_argument('--model_path', type=str, required=True,
                        help='訓練好的模型權重路徑 (.h5)')

    # 物理約束模型選項
    parser.add_argument('--use_physics_model', action='store_true',
                        help='使用物理約束模型 (dealias_mulit_v2_physics) 而非標準模型')
    parser.add_argument('--result_dir', type=str, default='results/test_inference',
                        help='結果輸出目錄')
    parser.add_argument('--correction_threshold', type=float, default=0.5,
                        help='反折錯修正閾值係數 (相對Nyquist), 預設0.5 - 僅在使用閾值方法時使用')
    parser.add_argument('--correction_tolerance', type=float, default=0.3,
                        help='修正成功容忍度係數 (相對Nyquist), 預設0.3')
    parser.add_argument('--use_physics_method', action='store_true', default=True,
                        help='使用物理原理方法 (compute_alias_label_swapped) 評估反折錯成功率 (預設: True)')
    parser.add_argument('--use_threshold_method', action='store_true',
                        help='使用閾值方法而非物理原理方法評估反折錯成功率')
    
    # 地理可視化相關參數
    parser.add_argument('--raw_gz', type=str, default=None,
                        help='原始數據對應的.gz檔案路徑 (用於地理可視化，若未提供會自動尋找)')
    parser.add_argument('--gt_gz', type=str, default=None,
                        help='GT數據對應的.gz檔案路徑 (用於地理可視化，若未提供會自動尋找)')
    parser.add_argument('--shape_path', type=str, default=None,
                        help='台灣地圖shapefile路徑 (如: mapdata201805310314/COUNTY_MOI_1070516)')

    # [NEW] 時間平滑相關參數
    parser.add_argument('--enable_temporal_smoothing', action='store_true',
                        help='啟用時間平滑（使用前一時刻預測結果）')
    parser.add_argument('--previous_prediction', type=str, default=None,
                        help='前一時刻的預測結果CSV路徑（啟用時間平滑時需要）')
    parser.add_argument('--temporal_alpha', type=float, default=0.3,
                        help='時間平滑參數 α (0-1)，預設0.3。α越大越信任當前預測，α越小越信任歷史')

    args = parser.parse_args()
    
    # 處理方法選擇邏輯
    use_physics_method = not args.use_threshold_method  # 如果明確指定用閾值方法，則不用物理方法
    
    # 檢查文件存在
    for file_path in [args.raw_csv, args.gt_csv, args.model_path]:
        if not os.path.exists(file_path):
            print(f"[ERROR] 文件不存在: {file_path}")
            return
    
    method_name = "物理原理方法 (compute_alias_label_swapped)" if use_physics_method else "閾值方法"
    model_type = "物理約束模型" if args.use_physics_model else "標準模型"
    print("=== 混合Patch模型推理測試 ===")
    print(f"模型類型: {model_type}")
    print(f"反折錯評估方法: {method_name}")
    if not use_physics_method:
        print(f"閾值係數: {args.correction_threshold}")
    print(f"容忍度係數: {args.correction_tolerance}")

    # 構建模型
    model = build_mixed_patch_model_for_inference(use_physics_model=args.use_physics_model)
    
    # 檢查地理可視化相關檔案
    if args.raw_gz and not os.path.exists(args.raw_gz):
        print(f"[WARNING]  指定的RAW GZ檔案不存在: {args.raw_gz}")
        args.raw_gz = None
        
    if args.gt_gz and not os.path.exists(args.gt_gz):
        print(f"[WARNING]  指定的GT GZ檔案不存在: {args.gt_gz}")
        args.gt_gz = None
        
    if args.shape_path and not os.path.exists(args.shape_path):
        print(f"[WARNING]  指定的shapefile路徑不存在: {args.shape_path}")
        args.shape_path = None
    
    # 顯示地理可視化狀態
    if GEO_VIZ_AVAILABLE:
        print(f"[GEO] 地理可視化: {'已啟用' if args.raw_gz or args.gt_gz else '將自動尋找.gz檔案'}")
        if args.shape_path:
            print(f"   台灣地圖: {args.shape_path}")
    else:
        print("[WARNING]  地理可視化: PyART/Basemap未安裝，將跳過地理圖")

    # [NEW] 檢查時間平滑設置
    if args.enable_temporal_smoothing:
        if args.previous_prediction and os.path.exists(args.previous_prediction):
            print(f"[TIMEOUT]  時間平滑: 已啟用 (α={args.temporal_alpha})")
            print(f"   前一時刻預測: {args.previous_prediction}")
        else:
            print(f"[WARNING]  時間平滑: 已啟用但未提供有效的前一時刻預測，將停用")
            args.enable_temporal_smoothing = False

    # 執行測試
    results = test_mixed_patch_model(
        model=model,
        raw_csv=args.raw_csv,
        gt_csv=args.gt_csv,
        model_path=args.model_path,
        result_dir=args.result_dir,
        correction_threshold=args.correction_threshold,
        correction_tolerance=args.correction_tolerance,
        use_physics_method=use_physics_method,
        raw_gz=args.raw_gz,
        gt_gz=args.gt_gz,
        shape_path=args.shape_path,
        use_physics_model=args.use_physics_model,
        # [NEW] 時間平滑參數
        enable_temporal_smoothing=args.enable_temporal_smoothing,
        previous_prediction=args.previous_prediction,
        temporal_alpha=args.temporal_alpha
    )
    
    print("[OK] 測試完成!")

if __name__ == "__main__":
    main()