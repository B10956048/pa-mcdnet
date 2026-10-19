# PA-MCDNet：物理約束雷達都卜勒速度反折錯

[English](README.md) | **繁體中文**

> **AIES 投稿論文之重現典藏。** 本 repository 於 Zenodo 典藏，正式標題為
> *PA-MCDNet: code, trained weights and evaluation-set definitions for
> physics-aware Doppler radar velocity dealiasing*。
> 重現論文結果請使用並引用此典藏版本：**https://doi.org/10.5281/zenodo.23479449**。
> 涵蓋所有版本的 Concept DOI：https://doi.org/10.5281/zenodo.22167560。
>
> 軟體版本：**v1.0.1**（Git tag `v1.0.1`）。Zenodo 頁面上的「Version」是典藏紀錄的版次，不是軟體版號。
> 逐步重現說明見 [REPRODUCE.md](REPRODUCE.md)。

用於氣象雷達資料**都卜勒速度反折錯**的深度學習模型。當實際風速超過奈奎斯特速度時，觀測到的徑向速度會發生折返；PA-MCDNet 將其還原為連續且符合物理的速度場。本模型定位為現行作業化反折錯演算法（如 VDAQC / SFDA）的輔助與交叉驗證工具，並非取代。

## 方法

- **架構**：U-Net，共用編碼器搭配**雙解碼器**（折返分類 + 殘差回歸），另含一個軟分類分支。
- **物理約束損失**：`L = 2.0·L_cls + 0.1·L_reg + 0.5·L_soft + 10.0·L_spatial`，其中 `L_spatial` 為空間物理一致性約束（即「PA」項）。
- **兩階段遷移學習**：Stage 1 於 NWP 模擬雷達場預訓練；Stage 2 於真實雷達觀測微調。

```mermaid
flowchart LR
    A["原始徑向速度<br/>+ 奈奎斯特"] --> E[共用編碼器]
    E --> Dc[分類解碼器]
    E --> Dr[回歸解碼器]
    Dc --> C[折返類別]
    Dc --> S[軟分類]
    Dr --> R[殘差]
    C --> M["合併 +<br/>物理約束"]
    R --> M
    S --> M
    M --> O["反折後<br/>速度場"]
```

<p align="center"><img src="docs/images/architecture.png" alt="PA-MCDNet 架構" width="860"></p>

<p align="center"><em>完整架構:共用 U-Net 編碼器接上分類解碼器（hard 與 soft 分支）與殘差回歸解碼器。分類頭同時驅動可微分的訓練路徑（softmax 期望 → 軟分類與空間平滑損失）與離散的推論路徑（argmax → 反折後速度）。</em></p>

## 成效

於 200 個真實觀測案例的鎖定測試集：

| 指標 | 數值 |
|---|---|
| 颱風反折錯成功率 | 約 96% |
| 整體成功率（含颮線） | 約 88% |
| 誤修率 | 0.39% |
| 速度均方根誤差 | 2.75 m/s |
| 推論時間 | 每張全圖低於 60 毫秒 |

## 流程

```mermaid
flowchart LR
    N[NWP 模擬] --> P1[nwp_to_patches.py] --> S1["Stage 1<br/>預訓練"]
    OB[真實觀測] --> P2[realobs_to_patches.py] --> S2["Stage 2<br/>微調"]
    S1 --> S2 --> MW[manual_weight_transfer.py] --> W[best_model_manual.h5]
    W --> INF["test_nwp_comparison.py<br/>推論與評估"]
```

## 專案結構

```
.
├── nwp_to_patches.py            # 前處理：NWP parquet → 訓練 patch
├── realobs_to_patches.py        # 前處理：真實觀測 gz → 微調 patch
├── convert_h5_to_tfrecord.py    # h5 → 分片 tfrecord（多張 GPU 必要）
├── transfer_learning_complete.py# 訓練：Stage 1（從零）與 Stage 2（微調）
├── manual_weight_transfer.py    # Stage 2 後產生 best_model_manual.h5
├── test_nwp_comparison.py       # 推論與評估（主要進入點）
├── verify_weights.py            # 不需 CWA 資料即可檢查公開權重
├── unet_model/                  # 模型定義（編碼器／解碼器／層）
├── mapdata201805310314/         # 地理可視化用的台灣 shapefile
├── data/locked_*_test_set.json  # 鎖定測試集定義
├── docs/USAGE.md                # 每支程式的完整參數說明
├── REPRODUCE.md                 # 論文重現說明
├── CITATION.cff                 # 引用資訊
└── ... （其他相依模組與工具，見下方「程式一覽」）
```

大型訓練資料、訓練好的模型權重、以及執行輸出**不納入版本控制**（見 `.gitignore`）；取得方式見下方說明。

## 程式一覽

每支程式的完整參數見 [docs/USAGE.md](docs/USAGE.md)。

| 程式 | 角色 |
|---|---|
| `nwp_to_patches.py` | 前處理 —— NWP parquet → 訓練 patch |
| `realobs_to_patches.py` | 前處理 —— 真實觀測 gz → 微調 patch |
| `realobs_append_patches.py` | 於既有 patch 集追加事件 |
| `realobs_to_fullsweep.py` | 建立全掃描（full-sweep）資料集變體 |
| `convert_h5_to_tfrecord.py` | h5 → 分片 tfrecord（多張 GPU） |
| `transfer_learning_complete.py` | 訓練 —— Stage 1（從零）與 Stage 2（微調） |
| `manual_weight_transfer.py` | Stage 2 後產生可部署的 `best_model_manual.h5` |
| `test_nwp_comparison.py` | 推論與評估（主要進入點） |
| `verify_weights.py` | 不需 CWA 資料即可檢查公開權重（MD5、載入、合成場測試） |
| `build_locked_test_sets.py` | 建立固定（鎖定）測試集 |
| `scan_realobs_alias.py` | 掃描真實觀測的折返比例並快取 |
| `unet_model/` | 模型定義（編碼器、解碼器、層） |
| 相依模組 | `mixed_patch_*`、`improved_physics_constraints.py`、`dealiasing_pre_patch_v2.py`、`dealiasing_success_metrics.py`、`batch_test_nwp.py`、`temporal_smoothing.py`、`physics_based_dealiasing_metrics.py`、`fix_typing.py` —— 由上列程式匯入，不直接執行 |

## 安裝

需要 **Python 3.10** 與 **TensorFlow 2.10**（內含 Keras 2.10）。訓練建議用 GPU；推論在 CPU 也可執行。

```bash
conda create -n pyart2 python=3.10 -y
conda activate pyart2

# 僅 GPU 使用者 —— TensorFlow 2.10 需要 CUDA 11.2 與 cuDNN 8.1，例如：
#   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# 核心套件（版本鎖定為實測環境）
pip install tensorflow==2.10.0 \
            numpy==1.26.4 scipy==1.15.3 pandas==2.3.2 h5py==3.14.0 \
            pyarrow==22.0.0 matplotlib==3.8.4 seaborn==0.13.2 tqdm==4.67.1 \
            scikit-image==0.25.1 scikit-learn==1.6.1 opencv-python==4.11.0.86
```

地理可視化為選用，另需 `pyart` 與 `basemap`：

```bash
pip install arm_pyart basemap basemap-data pyproj==3.6.1 pillow
```

未安裝這些仍可推論，只是不輸出地理圖。`basemap` 在部分系統較難安裝（需系統的 `geos` 與 `proj` 函式庫）。

## 快速開始（推論）

先取得模型權重（見 [模型權重](#模型權重)），再對一個裝有 PPI 檔案的資料夾執行推論：

```bash
python test_nwp_comparison.py \
  --model_path weights/pa_mcdnet_M6_headline.h5 \
  --inference_input path/to/ppi_folder \
  --output_dir infer_out \
  --use_physics_model \
  --enable_geo_viz --shape_path mapdata201805310314/COUNTY_MOI_1070516 \
  --downsample_720 --save_fields
```

輸出落在 `infer_out/`：反折後速度場（`_fields/*.npz`）與地理圖（`*_geo_*.png`）。`--inference_input` 可接受單一 `.gz` 檔或整個資料夾；奈奎斯特速度自檔頭讀取。

> **範例資料。** 此處不附雷達原始檔：CWA 觀測、作業參考產品、以及 NWP 模擬場皆受中央氣象署資料政策規範，作者無法轉散布（見資料可用性聲明）。不需任何 CWA 資料即可用 `python verify_weights.py --model_path weights/pa_mcdnet_M6_headline.h5` 檢查公開權重；若要對雷達資料執行，請將 `--inference_input` 指向你自己的 PPI `.gz` 檔。此處的 `data/locked_*_test_set.json` 已列明用到哪些案例。

## 範例

**極端折錯 —— 小犬颱風（RCHL，單張掃描 54.4% 折錯）。** 由左至右：原始（折錯）速度場、PA-MCDNet 修正、作業化參考真值。模型在此完整還原颱風速度偶極，成功率 99.98%。

![反折錯範例 —— 小犬颱風](docs/images/example_koinu_extreme.png)

**兩階段遷移 —— 蘇拉颱風（RCKT）。** 由左至右：原始速度場、僅 NWP 預訓練（未微調）、PA-MCDNet、作業化參考真值。僅 NWP 預訓練已能還原偶極但殘留顆粒雜訊；經真實觀測微調後得到平滑、與參考一致的速度場。

![兩階段效果 —— 蘇拉颱風](docs/images/example_saola_twostage.png)

執行快速開始即可對每個案例重現這類反折前後對照：`infer_out/` 內的 `*_geo_raw.png`（反折前）與 `*_geo_ours.png`（反折後）即可看出折返被消除、速度場恢復連續。

## 前處理

將原始雷達／NWP 檔轉為下一節訓練所用的 TFRecord。完整參數見 [docs/USAGE.md](docs/USAGE.md)。

```bash
# NWP → patch → tfrecord（Stage 1 輸入）
python nwp_to_patches.py --nwp_root <NWP parquet 根目錄> \
  --output_h5 data/nwp_patches.h5 \
  --target_stations RCCG RCWF RCHL RCKT RCGI --aliased_ratio 0.9
python convert_h5_to_tfrecord.py --h5_path data/nwp_patches.h5 \
  --output_dir data/nwp_tfrecord --splits train,val --num_shards 32 --compression GZIP

# 真實觀測 → patch → tfrecord（Stage 2 輸入）
python realobs_to_patches.py --realobs_root <typhoonnew 根目錄> \
  --output_h5 data/realobs_patches.h5 \
  --train_cases 2021_Chanthu 2023_DOKSURI 2023_HAIKUI
python convert_h5_to_tfrecord.py --h5_path data/realobs_patches.h5 \
  --output_dir data/realobs_tfrecord --splits train,val --num_shards 32 --compression GZIP
```

## 訓練（兩階段）

多張 GPU 訓練需要 TFRecord 輸入（單一 h5 無法在多張 GPU 間分片）。先轉檔，再訓練。

```bash
# Stage 1 —— 於 NWP 預訓練（從零開始）
python transfer_learning_complete.py \
  --nwp_h5 data/nwp_tfrecord --output_dir results/stage1 \
  --freeze_ratio 0.0 --learning_rate 5e-5 --batch_size 2048 --epochs 200 --patience 5 \
  --lambda_cls 2.0 --lambda_reg 0.1 --lambda_soft 0.5 --lambda_spatial 10.0 \
  --lambda_physics 0.0 --lambda_confidence 0.0

# Stage 2 —— 於真實觀測微調
python transfer_learning_complete.py \
  --pretrained_model results/stage1/best_model.h5 \
  --nwp_h5 data/realobs_tfrecord --output_dir results/stage2 \
  --freeze_ratio 0.3 --learning_rate 5e-5 --batch_size 2048 --epochs 100 --patience 5 \
  --lambda_cls 2.0 --lambda_reg 0.1 --lambda_soft 0.5 --lambda_spatial 10.0
```

注意：損失權重的預設值與論文不同（例如 `--lambda_spatial` 預設為 50），務必如上明確帶齊。Stage 2 完成後，執行 `manual_weight_transfer.py` 產生部署用的 `best_model_manual.h5`（微調後的 `best_model.h5` 在較新版本的 TensorFlow 可能無法直接載入）。每個參數見 [docs/USAGE.md](docs/USAGE.md)。

## 模型權重

訓練好的權重因檔案體積未納入版本控制。論文所報告的最終 PA-MCDNet 權重典藏於 Zenodo（https://doi.org/10.5281/zenodo.23479449），檔名 `pa_mcdnet_M6_headline.h5`（約 82 MB，MD5 `1b1685ec72c7a765a6f957afb518a551`）。請存為 `weights/pa_mcdnet_M6_headline.h5`，並將此路徑傳給 `--model_path`。自行訓練的權重會產生為 `best_model_manual.h5`（見訓練一節）。

## 資料

<p align="center"><img src="docs/images/radar_network.png" alt="中央氣象署作業化都卜勒雷達網" width="430"></p>

<p align="center"><em>供訓練與評估之中央氣象署 10 座作業化都卜勒雷達網（4 座 S 波段、6 座 C 波段）。</em></p>

訓練與評估資料為中央氣象署 S 與 C 波段雷達網的觀測（原始徑向速度 `bvel_raw` 與參考真值 `bvel_sfda` / `bvel_vdaqc`）以及對應的 NWP 模擬場。這些資料未在此重散布；前處理程式會將原始雷達／NWP 檔轉為訓練 patch。`data/locked_*_test_set.json` 記錄哪些案例組成固定測試集（其中的檔案路徑為佔位，需指向你自己的資料）。

## 第三方比較模型

評估時與 **UNet-VDA**（MIT Lincoln Laboratory，MIT 授權）比較。本專案**未**內含它。評估模式預設自 `unet-vda-main/` 載入，缺少時會停止：

```bash
git clone https://github.com/mit-ll/unet-vda.git unet-vda-main
```

純推論與 `verify_weights.py` 不需要它。

## 授權

各部分適用不同條款：

| 項目 | 授權 |
|---|---|
| 原始碼（`*.py`、`unet_model/`） | MIT —— 見 [LICENSE](LICENSE) |
| 訓練好的模型權重（`pa_mcdnet_M6_headline.h5`，於 Zenodo） | CC BY 4.0。訓練權重之公開已獲中央氣象署同意。 |
| 鎖定測試集定義（`data/locked_*_test_set.json`） | CC BY 4.0。內容為由 CWA 資料衍生之案例識別資訊與摘要統計（雷達站、時間、仰角、Nyquist 速度、折錯比例），經中央氣象署同意公開；不含雷達觀測值。 |
| `docs/images/` 內的圖 | © 作者；案例圖由 CWA 資料衍生，經中央氣象署同意展示。不適用 MIT 或 CC BY 4.0；與正式刊出論文相同的圖，依 AMS 著作權聲明。 |
| `mapdata201805310314/`（縣市界線） | 第三方資料，來源為內政部國土測繪中心，依政府資料開放授權條款第 1 版。 |
| CWA 雷達觀測、作業參考產品、NWP 模擬場 | **未包含**，亦不適用此處任何授權；可向中央氣象署正式申請。 |

## 引用

使用本軟體請引用典藏版本（https://doi.org/10.5281/zenodo.23479449）；引用資訊見 [CITATION.cff](CITATION.cff)。對應論文為 Lin, Zhang, Fang, and Chang, *Physics-Aware Deep Learning for Radar Doppler Velocity Dealiasing Using NWP Simulations and Observations*（投稿至 *Artificial Intelligence for the Earth Systems*）。
