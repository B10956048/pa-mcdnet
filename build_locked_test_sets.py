#!/usr/bin/env python3
"""
build_locked_test_sets.py
─────────────────────────────────────────────────────────────────────────────
產生**鎖定**的 NWP 跟 Real Obs 測試集 JSON，供所有 phase 實驗共用。

設計原則：
  1. **不偏向 clean / aliased**：按 alias_bin 等比例分層
  2. **所有 station 都包含**：按 station 分層
  3. **所有 elevation 都涵蓋**：按 elev_group 分層
  4. **Reproducible**：固定 seed
  5. **No leakage**：排除已進訓練 H5 的 case
  6. **每 cell 至少採樣**：保證 cross-tab 不出現 0

輸出：
  data/locked_nwp_test_set.json    (~100 cases)
  data/locked_realobs_test_set.json (~160 cases)
  data/locked_test_distribution.md   (報告)

使用：
  python build_locked_test_sets.py [--nwp_n 100] [--realobs_n 160] [--seed 46]
"""

import argparse
import gzip
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================================
# 共用：分層 binning
# ============================================================================
def alias_bin_of(ratio):
    """
    alias_ratio → bin name (2 bins: clean / aliased)
    簡化分箱，避免 high alias 在 real obs 過於稀缺造成的不均勻。
    JSON 仍保留 alias_ratio 原值供後續 fine-grained 分析。
    """
    if ratio < 0.01:
        return 'clean'
    else:
        return 'aliased'


def elev_group_of(elev_str):
    """elevation '01'~'15' → low/mid/high"""
    e = int(elev_str)
    if e <= 4:
        return 'low'
    elif e <= 8:
        return 'mid'
    else:
        return 'high'


def stratified_sample(candidates, key_fn, n_target, seed=46):
    """
    按 key_fn 分組，每組均勻抽，總共 n_target 個。
    返回 (sampled_list, group_counts_actual)
    """
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for c in candidates:
        groups[key_fn(c)].append(c)

    n_groups = len(groups)
    if n_groups == 0:
        return [], {}

    per_group = max(1, n_target // n_groups)

    sampled = []
    actual = {}
    for key in sorted(groups.keys()):
        pool = groups[key]
        rng.shuffle(pool)
        take = min(per_group, len(pool))
        sampled.extend(pool[:take])
        actual[key] = take

    # 若總數仍 < n_target，從剩餘 candidates 補（不重複）
    used_ids = {id(c) for c in sampled}
    extra = [c for c in candidates if id(c) not in used_ids]
    rng.shuffle(extra)
    deficit = n_target - len(sampled)
    if deficit > 0:
        sampled.extend(extra[:deficit])

    return sampled[:n_target], actual


# ============================================================================
# 讀 .gz radar
# ============================================================================
def read_gz_radar(gz_path):
    with gzip.open(gz_path, 'rb') as gf:
        f = gf.read()
    header = struct.unpack('<16s36i', f[:160])
    info = np.array(header[1:-1])
    info_flt = info / info[0]
    nray, ngate = int(info_flt[14]), int(info_flt[15])
    nyq = float(info_flt[10])
    raw = np.array(struct.unpack('<' + str(nray*ngate) + 'i', f[160:]))
    valid = (raw != -990) & (raw != -9990)
    data = (raw / info_flt[20]).reshape(nray, ngate).astype(np.float32)
    data[~valid.reshape(nray, ngate)] = np.nan
    azm_start = float(info_flt[16])
    return data, nyq, azm_start


def compute_alias_ratio(raw_vel, gt_vel, nyq):
    """alias_ratio = |round((gt - raw) / 2*nyq)| != 0 的有效 pixel 比例"""
    mask = ~np.isnan(raw_vel) & ~np.isnan(gt_vel)
    if mask.sum() == 0:
        return 0.0
    fold = np.round((gt_vel[mask] - raw_vel[mask]) / (2 * nyq))
    return float((fold != 0).sum() / mask.sum())


def align_raw_to_gt(raw, raw_az_start, gt_az_start, gt_naz):
    """720 raw → 360 GT 對齊（azimuth nearest neighbor）"""
    raw_naz = raw.shape[0]
    if raw_naz == gt_naz:
        return raw
    raw_sp = 360.0 / raw_naz
    gt_sp = 360.0 / gt_naz
    raw_az = (raw_az_start + np.arange(raw_naz) * raw_sp) % 360.0
    gt_az = (gt_az_start + np.arange(gt_naz) * gt_sp) % 360.0
    diff = np.abs(gt_az[:, None] - raw_az[None, :])
    circ = np.minimum(diff, 360.0 - diff)
    idx = np.argmin(circ, axis=1)
    return raw[idx]


# ============================================================================
# NWP test set builder
# ============================================================================
def build_nwp_test_set_from_h5(h5_path, n_target, seed=46, split='test'):
    """
    從 NWP patch H5 抓**唯一個案** (= unique nwp_dir + source_file)，
    按 (alias_bin × station) 分層採樣。

    為什麼不分 elev_group:
        實測這 H5 內 source_file 全部 elev01，無 mid/high 仰角。
        分層維度只有 alias_bin × station (4 × 9 = 36 cells)。

    Returns: list of cleaned cases
    """
    print(f"\n{'='*70}")
    print(f"Building NWP test set from H5 (target {n_target} cases)")
    print(f"  source: {h5_path}")
    print(f"  split : {split}")
    print(f"{'='*70}")

    if not Path(h5_path).exists():
        print(f"❌ H5 不存在: {h5_path}")
        return []

    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    import h5py

    with h5py.File(h5_path, 'r') as f:
        if split not in f:
            print(f"❌ split '{split}' 不存在於 H5")
            return []
        g = f[split]
        n_patches = len(g['source_file'])
        print(f"\n讀 {n_patches} 個 patches 的 metadata...")

        # 向量化讀 metadata
        def _decode_arr(name):
            raw = g[name][:]
            if raw.dtype.kind == 'S':
                return np.char.decode(raw, 'utf-8')
            return np.array([s.decode() if isinstance(s, bytes) else str(s) for s in raw])

        nwp_dirs = _decode_arr('nwp_dir')
        sources  = _decode_arr('source_file')
        stations = _decode_arr('station')
        patch_types = _decode_arr('patch_type')
        alias_ratios = g['alias_ratio'][:]
        nyqs = g['nyq'][:].squeeze() if g['nyq'][:].ndim > 1 else g['nyq'][:]

    # 唯一個案 = (nwp_dir, source_file)
    case_keys = np.array([f'{d}__{s}' for d, s in zip(nwp_dirs, sources)])
    unique_cases = np.unique(case_keys)
    print(f"唯一個案數: {len(unique_cases)}")

    # 聚合 metadata per case
    candidates = []
    for ck in unique_cases:
        mask = case_keys == ck
        # 取 representative
        idx0 = np.where(mask)[0][0]
        source = sources[idx0]
        # 從 source_file 抽仰角（e.g. RCWF.20241030.2210.bvel_raw.01.parquet → 01）
        clean_name = source.replace('.parquet', '')
        parts = clean_name.rsplit('.', 1)
        elev = parts[-1] if parts[-1].isdigit() else '01'

        candidates.append({
            'nwp_dir': nwp_dirs[idx0],
            'file': clean_name,  # without .parquet
            'station': stations[idx0],
            'elevation': elev,
            'elev_group': elev_group_of(elev),
            'alias_ratio': float(alias_ratios[mask].mean()),  # mean over patches
            'alias_ratio_max': float(alias_ratios[mask].max()),
            'nyquist': float(nyqs[idx0]),
            'n_patches': int(mask.sum()),
            'n_aliased_patches': int((patch_types[mask] == 'aliased').sum()),
        })

    # 補 alias_bin
    for c in candidates:
        c['alias_bin'] = alias_bin_of(c['alias_ratio'])

    # 分層 sample by (alias_bin, station)（不分 elev_group：全 elev01）
    def strata_key(c):
        return f"{c['alias_bin']}__{c['station']}"

    sampled, group_counts = stratified_sample(candidates, strata_key, n_target, seed=seed)

    cleaned = []
    for c in sampled:
        cleaned.append({
            'nwp_dir': c['nwp_dir'],
            'file': c['file'],
            'station': c['station'],
            'elevation': c['elevation'],
            'elev_group': c['elev_group'],
            'alias_ratio': round(c['alias_ratio'], 4),
            'alias_ratio_max': round(c['alias_ratio_max'], 4),
            'alias_bin': c['alias_bin'],
            'nyquist': round(c['nyquist'], 2),
            'n_patches': c['n_patches'],
            'n_aliased_patches': c['n_aliased_patches'],
            '_strata': strata_key(c),
        })
    return cleaned


# ============================================================================
# Real Obs test set builder
# ============================================================================
def list_realobs_pairs(realobs_root):
    """掃 realobs_root/EVENT/polar/binary/STATION/ 找 raw+GT pairs"""
    root = Path(realobs_root)
    pairs = []
    GT_SUFFIXES = ['bvel_vdaqc', 'bvel_sfda', 'bvel_da']

    for event_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        event_name = event_dir.name
        is_squall = 'SQUALL' in event_name.upper()
        binary = event_dir / 'polar' / 'binary'
        station_dirs = []
        if binary.exists():
            station_dirs = [(d.name, d) for d in sorted(binary.iterdir()) if d.is_dir()]
        else:
            # flat format
            flat = list(event_dir.glob('*.gz'))
            if flat:
                stations = set(f.name.split('.')[0] for f in flat)
                station_dirs = [(st, event_dir) for st in sorted(stations)]

        for station, scan_dir in station_dirs:
            for raw_gz in sorted(scan_dir.glob(f'{station}.*.bvel_raw.*.gz')):
                gt_gz = None
                for suf in GT_SUFFIXES:
                    cand = Path(str(raw_gz).replace('bvel_raw', suf))
                    if cand.exists():
                        gt_gz = cand
                        break
                if gt_gz is None:
                    continue
                stem = raw_gz.stem  # RCXX.YYYYMMDD.HHMM.bvel_raw.NN
                parts = stem.split('.')
                elev = parts[-1] if parts[-1].isdigit() else '01'
                pairs.append({
                    'event': event_name,
                    'event_type': 'squall' if is_squall else 'typhoon',
                    'station': station,
                    'date': parts[1] if len(parts) > 1 else '',
                    'time': parts[2] if len(parts) > 2 else '',
                    'elevation': elev,
                    'elev_group': elev_group_of(elev),
                    'raw': str(raw_gz),
                    'gt': str(gt_gz),
                    'filename': stem,
                })
    return pairs


def build_realobs_test_set_from_cache(cache_path, n_target, seed=46,
                                        min_valid_pixels=100):
    """
    從 scan_realobs_alias.py 產生的 cache 讀，per-event 配額均勻採樣。

    策略：
      1. **過濾 valid_pixels < min_valid_pixels**（避免測試時被 [SKIP] 砍）
      2. 每 event 配額 = n_target / n_events（強制 squall 不被淹）
      3. 每 event 內 50/50 clean/aliased（強制平衡）
      4. 每 subset 按 (station × elev_group) 分層採樣（保多樣性）
      5. 真不夠就降額 + 補另一 bin（誠實達標）

    需求：先跑 `python scan_realobs_alias.py` 產生 cache JSON。
    """
    print(f"\n{'='*70}")
    print(f"Building Real Obs test set FROM CACHE (target {n_target} cases)")
    print(f"  cache: {cache_path}")
    print(f"  min valid pixels: {min_valid_pixels} (避免 test 端被 [SKIP])")
    print(f"{'='*70}")

    if not Path(cache_path).exists():
        print(f"❌ Cache 不存在: {cache_path}")
        print(f"   請先跑: python scan_realobs_alias.py")
        return []

    with open(cache_path, encoding='utf-8') as f:
        all_results = json.load(f)
    print(f"\n載入 cache: {len(all_results)} pairs")

    # 過濾錯誤
    valid = [r for r in all_results if r.get('alias_ratio', -1) >= 0]
    print(f"無 error: {len(valid)}")

    # 🔑 過濾 valid_pixels 太少（會被 test 端的 [SKIP] 砍掉）
    before_vp = len(valid)
    valid = [r for r in valid if r.get('valid_pixels', 0) >= min_valid_pixels]
    print(f"valid_pixels >= {min_valid_pixels}: {len(valid)} (砍掉 {before_vp - len(valid)})")

    # 按 event 分組
    by_event = defaultdict(list)
    for r in valid:
        by_event[r['event']].append(r)

    events = sorted(by_event.keys())
    n_events = len(events)
    per_event_n = n_target // n_events
    print(f"\n📊 Per-event 配額: {per_event_n} (n_target={n_target} / n_events={n_events})")

    rng = np.random.RandomState(seed)
    all_sampled = []

    for event in events:
        event_cases = by_event[event]
        clean = [r for r in event_cases if r['alias_ratio'] < 0.01]
        aliased = [r for r in event_cases if r['alias_ratio'] >= 0.01]

        # Per event 目標：50/50 clean/aliased
        n_clean_target = per_event_n // 2
        n_aliased_target = per_event_n - n_clean_target

        print(f"\n  {event}:")
        print(f"    可用 clean={len(clean)}, aliased={len(aliased)}")
        print(f"    目標 clean={n_clean_target}, aliased={n_aliased_target}")

        # 在每 subset 內按 (station × elev_group) 多樣性抽
        def diverse_sample(pool, n):
            """按 (station, elev_group) 分層，每 cell 抽至少 1，剩餘隨機補"""
            if len(pool) <= n:
                return list(pool)
            groups = defaultdict(list)
            for r in pool:
                groups[(r['station'], r['elev_group'])].append(r)
            sampled = []
            for k in sorted(groups.keys()):
                items = list(groups[k])
                rng.shuffle(items)
                sampled.append(items[0])  # 每 cell 至少 1
                groups[k] = items[1:]  # 剩下備補
            # 不夠補：把所有剩餘混一起隨機
            rest = []
            for v in groups.values():
                rest.extend(v)
            rng.shuffle(rest)
            sampled.extend(rest[:n - len(sampled)])
            return sampled[:n]

        def diverse_aliased_sample(pool, n):
            """
            針對 Real Obs aliased：優先採高 alias_ratio cases，剩餘按多樣性補
            策略：
              1. 一半 quota 從 alias_ratio 前 30% 採（challenging cases）
              2. 另一半從剩餘按 (station × elev_group) 多樣性採
            這樣 KOINU/SAOLA 的 mid alias 一定進來，但仍保跨站多樣性。
            """
            if len(pool) <= n:
                return list(pool)
            sorted_pool = sorted(pool, key=lambda r: -r['alias_ratio'])  # 降序
            n_top = n // 2
            top_candidates = sorted_pool[:max(n_top * 2, n_top + 5)]  # 前 N 個 high alias 池
            rng.shuffle(top_candidates)
            high_picks = top_candidates[:n_top]
            # 剩餘按 station × elev_group 多樣性
            used_ids = {id(r) for r in high_picks}
            rest_pool = [r for r in pool if id(r) not in used_ids]
            rest_picks = diverse_sample(rest_pool, n - n_top)
            return high_picks + rest_picks

        sampled_clean = diverse_sample(clean, n_clean_target)
        sampled_aliased = diverse_aliased_sample(aliased, n_aliased_target)

        # 不夠就補另一 bin
        deficit_clean = n_clean_target - len(sampled_clean)
        deficit_aliased = n_aliased_target - len(sampled_aliased)
        if deficit_aliased > 0:
            # 從剩餘 clean 補
            used_clean = {r['raw'] for r in sampled_clean}
            extra = [r for r in clean if r['raw'] not in used_clean]
            rng.shuffle(extra)
            sampled_clean.extend(extra[:deficit_aliased])
            print(f"    ⚠️  aliased 不足，補 {deficit_aliased} 個 clean")
        if deficit_clean > 0:
            used_aliased = {r['raw'] for r in sampled_aliased}
            extra = [r for r in aliased if r['raw'] not in used_aliased]
            rng.shuffle(extra)
            sampled_aliased.extend(extra[:deficit_clean])
            print(f"    ⚠️  clean 不足，補 {deficit_clean} 個 aliased")

        actual = sampled_clean + sampled_aliased
        print(f"    實際採樣: {len(actual)} ({len(sampled_clean)} clean + {len(sampled_aliased)} aliased)")
        all_sampled.extend(actual)

    # 清理 output schema
    cleaned = []
    for c in all_sampled:
        cleaned.append({
            'event': c['event'],
            'event_type': c.get('event_type', 'unknown'),
            'station': c['station'],
            'date': c['date'],
            'time': c['time'],
            'elevation': c['elevation'],
            'elev_group': c['elev_group'],
            'alias_ratio': round(c['alias_ratio'], 4),
            'alias_bin': alias_bin_of(c['alias_ratio']),
            'nyquist': round(c.get('nyquist', 0), 2),
            'valid_pixels': int(c.get('valid_pixels', 0)),
            'raw': c['raw'],
            'gt': c['gt'],
            'filename': c['filename'],
            '_strata': f"{c['event']}__{c['station']}__{c['elev_group']}",
        })
    print(f"\n✅ 共採樣 {len(cleaned)} cases")
    return cleaned


def build_realobs_test_set(realobs_root, n_target, seed=46,
                             per_cell_pool=8):
    """
    1. 掃所有 (raw, GT) pairs
    2. 先按 (event × station × elev_group) 分層，每格抽 per_cell_pool 個（候選池）
    3. 對候選池逐個讀 .gz 算 alias_ratio
    4. 再按 (event × station × elev_group × alias_bin) 分層採樣到 n_target
    """
    print(f"\n{'='*70}")
    print(f"Building Real Obs test set (target {n_target} cases)")
    print(f"  root: {realobs_root}")
    print(f"{'='*70}")

    all_pairs = list_realobs_pairs(realobs_root)
    print(f"\n候選 raw+GT pairs 總數: {len(all_pairs)}")

    # Step 1: 第一輪採樣（按 event × station × elev_group）
    def pool_key(p):
        return f"{p['event']}__{p['station']}__{p['elev_group']}"

    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for p in all_pairs:
        groups[pool_key(p)].append(p)

    pool = []
    for key in sorted(groups.keys()):
        items = groups[key]
        rng.shuffle(items)
        pool.extend(items[:per_cell_pool])
    print(f"第一輪候選池: {len(pool)} pairs")

    # Step 2: 對候選池算 alias_ratio
    print(f"\n計算 alias_ratio (讀 .gz 配對)...")
    for p in tqdm(pool, desc='compute alias'):
        try:
            raw, nyq, raw_az = read_gz_radar(p['raw'])
            gt, _, gt_az = read_gz_radar(p['gt'])
            # 720+360 對齊
            if raw.shape[0] != gt.shape[0]:
                raw = align_raw_to_gt(raw, raw_az, gt_az, gt.shape[0])
            # ngate 對齊
            if raw.shape == gt.shape:
                pass
            elif raw.shape[0] == gt.shape[0] and raw.shape[1] != gt.shape[1]:
                n = min(raw.shape[1], gt.shape[1])
                raw, gt = raw[:, :n], gt[:, :n]
            if raw.shape != gt.shape:
                p['alias_ratio'] = -1.0
                continue
            p['alias_ratio'] = compute_alias_ratio(raw, gt, nyq)
            p['nyquist'] = nyq
            p['alias_bin'] = alias_bin_of(p['alias_ratio'])
        except Exception as e:
            print(f"  [ERR] {p['raw']}: {e}")
            p['alias_ratio'] = -1.0
            p['alias_bin'] = 'error'

    # 過濾 error
    valid_pool = [p for p in pool if p.get('alias_ratio', -1) >= 0]
    print(f"\n計算成功: {len(valid_pool)}/{len(pool)}")

    # Step 3: 4 維分層採樣
    def strata_key(p):
        return f"{p['event']}__{p['station']}__{p['elev_group']}__{p['alias_bin']}"

    sampled, group_counts = stratified_sample(valid_pool, strata_key, n_target, seed=seed)

    cleaned = []
    for p in sampled:
        cleaned.append({
            'event': p['event'],
            'event_type': p['event_type'],
            'station': p['station'],
            'date': p['date'],
            'time': p['time'],
            'elevation': p['elevation'],
            'elev_group': p['elev_group'],
            'alias_ratio': round(p['alias_ratio'], 4),
            'alias_bin': p['alias_bin'],
            'nyquist': round(p['nyquist'], 2),
            'raw': p['raw'],
            'gt': p['gt'],
            'filename': p['filename'],
            '_strata': strata_key(p),
        })
    return cleaned


# ============================================================================
# Report 產生器
# ============================================================================
def generate_report(nwp_set, realobs_set, report_path):
    from datetime import datetime
    lines = ['# Locked Test Sets — Distribution Report\n']
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # NWP
    lines.append(f'\n## NWP Test Set ({len(nwp_set)} cases)\n')
    lines.append('\n### By alias_bin\n')
    from collections import Counter
    bins = Counter(c['alias_bin'] for c in nwp_set)
    for k in ['clean', 'aliased']:
        lines.append(f'  - {k}: {bins.get(k, 0)} ({100*bins.get(k,0)/len(nwp_set) if nwp_set else 0:.1f}%)')
    lines.append('\n### By station\n')
    for st, n in sorted(Counter(c['station'] for c in nwp_set).items()):
        lines.append(f'  - {st}: {n}')
    lines.append('\n### By elev_group\n')
    for eg, n in sorted(Counter(c['elev_group'] for c in nwp_set).items()):
        lines.append(f'  - {eg}: {n}')
    lines.append('\n### Cross: alias_bin × station\n')
    cross = defaultdict(int)
    for c in nwp_set:
        cross[(c['alias_bin'], c['station'])] += 1
    stations = sorted(set(c['station'] for c in nwp_set))
    lines.append('| | ' + ' | '.join(stations) + ' |')
    lines.append('|---|' + '|'.join(['---']*len(stations)) + '|')
    for b in ['clean','aliased']:
        row = [b] + [str(cross.get((b,s), 0)) for s in stations]
        lines.append('| ' + ' | '.join(row) + ' |')

    # Real Obs
    lines.append(f'\n\n## Real Obs Test Set ({len(realobs_set)} cases)\n')
    lines.append('\n### By event\n')
    for ev, n in sorted(Counter(c['event'] for c in realobs_set).items()):
        lines.append(f'  - {ev}: {n}')
    lines.append('\n### By event_type\n')
    for et, n in sorted(Counter(c['event_type'] for c in realobs_set).items()):
        lines.append(f'  - {et}: {n}')
    lines.append('\n### By alias_bin\n')
    bins2 = Counter(c['alias_bin'] for c in realobs_set)
    for k in ['clean', 'aliased']:
        lines.append(f'  - {k}: {bins2.get(k, 0)} ({100*bins2.get(k,0)/len(realobs_set) if realobs_set else 0:.1f}%)')
    lines.append('\n### By station\n')
    for st, n in sorted(Counter(c['station'] for c in realobs_set).items()):
        lines.append(f'  - {st}: {n}')
    lines.append('\n### By elev_group\n')
    for eg, n in sorted(Counter(c['elev_group'] for c in realobs_set).items()):
        lines.append(f'  - {eg}: {n}')
    lines.append('\n### Cross: event × alias_bin\n')
    cross2 = defaultdict(int)
    for c in realobs_set:
        cross2[(c['event'], c['alias_bin'])] += 1
    events = sorted(set(c['event'] for c in realobs_set))
    lines.append('| event | clean | aliased |')
    lines.append('|---|---|---|')
    for e in events:
        row = [e] + [str(cross2.get((e,b), 0)) for b in ['clean','aliased']]
        lines.append('| ' + ' | '.join(row) + ' |')

    Path(report_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n📝 Report 寫入 {report_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nwp_n', type=int, default=100, help='NWP test set target size')
    parser.add_argument('--realobs_n', type=int, default=160, help='Real Obs test set target size')
    parser.add_argument('--realobs_root', type=str, default='/path/to/typhoonnew/test',
                        help='Real obs root (含 KOINU/SAOLA/SQUALL events)')
    parser.add_argument('--realobs_cache', type=str,
                        default='data/realobs_alias_cache.json',
                        help='Real obs alias cache (由 scan_realobs_alias.py 產生)')
    parser.add_argument('--use_cache', action='store_true', default=True,
                        help='優先用 cache，若不存在 fallback 到 on-the-fly 計算')
    parser.add_argument('--min_valid_pixels', type=int, default=100,
                        help='Real Obs 採樣時過濾 valid_pixels 下限 '
                             '(避免測試時被 test_nwp_comparison.py 的 [SKIP] 砍)。'
                             '預設 100（跟 test 端閾值相同）。'
                             '想嚴格一點可設 1000+')
    parser.add_argument('--nwp_h5', type=str,
                        default='data/nwp_balanced_patches_no_rccg_v2_alias50.h5',
                        help='NWP patch H5 (含 test split 的 metadata)')
    parser.add_argument('--nwp_h5_split', type=str, default='test',
                        help='從 H5 哪個 split 抽（預設 test）')
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--per_cell_pool', type=int, default=8,
                        help='Real Obs 第一輪每 cell 抽多少進 alias_ratio 計算池')
    parser.add_argument('--out_dir', type=str, default='data')
    args = parser.parse_args()

    # Windows path 處理
    if sys.platform.startswith(('linux', 'darwin')):
        rr = args.realobs_root.replace('\\', '/')
        if rr[:3].lower() in ('h:/', 'c:/', 'd:/', 'e:/'):
            rr = f'/mnt/{rr[0].lower()}/' + rr[3:]
        args.realobs_root = rr

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # NWP (從 H5 抓)
    nwp_set = build_nwp_test_set_from_h5(args.nwp_h5, args.nwp_n,
                                          seed=args.seed, split=args.nwp_h5_split)
    nwp_path = out_dir / 'locked_nwp_test_set.json'
    with open(nwp_path, 'w') as f:
        json.dump(nwp_set, f, indent=2, ensure_ascii=False)
    print(f"\n✅ NWP test set 寫入 {nwp_path} ({len(nwp_set)} cases)")

    # Real Obs：優先用 cache，否則 fallback 到舊 on-the-fly
    if args.use_cache and Path(args.realobs_cache).exists():
        realobs_set = build_realobs_test_set_from_cache(
            args.realobs_cache, args.realobs_n, seed=args.seed,
            min_valid_pixels=args.min_valid_pixels)
    else:
        print(f"\n⚠️  Cache 不存在，fallback 到 on-the-fly 計算（慢）")
        print(f"   建議先跑: python scan_realobs_alias.py")
        realobs_set = build_realobs_test_set(args.realobs_root, args.realobs_n,
                                              seed=args.seed,
                                              per_cell_pool=args.per_cell_pool)
    realobs_path = out_dir / 'locked_realobs_test_set.json'
    with open(realobs_path, 'w') as f:
        json.dump(realobs_set, f, indent=2, ensure_ascii=False)
    print(f"✅ Real Obs test set 寫入 {realobs_path} ({len(realobs_set)} cases)")

    # Report
    report_path = out_dir / 'locked_test_distribution.md'
    generate_report(nwp_set, realobs_set, report_path)

    print(f"\n{'='*70}")
    print(f"完成！輸出檔案:")
    print(f"  {nwp_path}")
    print(f"  {realobs_path}")
    print(f"  {report_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
