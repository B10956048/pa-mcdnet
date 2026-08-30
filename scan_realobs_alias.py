#!/usr/bin/env python3
"""
scan_realobs_alias.py
─────────────────────────────────────────────────────────────────────────────
一次性 scan 所有 real obs (raw, gt) pairs，計算 alias_ratio，cache 到 JSON。
之後 build_locked_test_sets.py 直接讀 cache 智能採樣，不用重算。

特性：
  - 多 process 平行（Windows / Linux 都支援）
  - 增量儲存（每 N pairs 寫一次 cache，中斷可續跑）
  - 跳過已 cache 的 case
  - 錯誤容忍（單檔失敗不會中斷整個 scan）

使用：
  python scan_realobs_alias.py
  python scan_realobs_alias.py --workers 8 --save_every 1000
  python scan_realobs_alias.py --realobs_root /path/to/typhoonnew/test
"""

import argparse
import gzip
import json
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================================
# 1. .gz 讀取 + alias_ratio 計算（從 build_locked_test_sets 複製）
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


def align_raw_to_gt(raw, raw_az_start, gt_az_start, gt_naz):
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


def compute_alias_ratio(raw_vel, gt_vel, nyq):
    mask = ~np.isnan(raw_vel) & ~np.isnan(gt_vel)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return 0.0, 0
    fold = np.round((gt_vel[mask] - raw_vel[mask]) / (2 * nyq))
    n_aliased = int((fold != 0).sum())
    return n_aliased / n_valid, n_valid


# ============================================================================
# 2. Pair worker（單 pair 計算，給 multiprocessing 用）
# ============================================================================
def process_pair(pair_info):
    """
    處理單一 pair，回傳完整 metadata。
    pair_info: dict 含 raw, gt, event, station, date, time, elevation, elev_group
    """
    try:
        raw, nyq_r, raw_az = read_gz_radar(pair_info['raw'])
        gt, nyq_g, gt_az = read_gz_radar(pair_info['gt'])

        # 720→360 azimuth-aware 對齊
        if raw.shape[0] != gt.shape[0]:
            raw = align_raw_to_gt(raw, raw_az, gt_az, gt.shape[0])
        # ngate 裁切
        if raw.shape == gt.shape:
            pass
        elif raw.shape[0] == gt.shape[0] and raw.shape[1] != gt.shape[1]:
            n = min(raw.shape[1], gt.shape[1])
            raw, gt = raw[:, :n], gt[:, :n]
        if raw.shape != gt.shape:
            pair_info['alias_ratio'] = -1.0
            pair_info['valid_pixels'] = 0
            pair_info['error'] = 'shape_mismatch'
            return pair_info

        alias_ratio, valid = compute_alias_ratio(raw, gt, nyq_r)
        pair_info['alias_ratio'] = float(alias_ratio)
        pair_info['valid_pixels'] = int(valid)
        pair_info['nyquist'] = float(nyq_r)
        pair_info['raw_shape'] = list(raw.shape)
        return pair_info
    except Exception as e:
        pair_info['alias_ratio'] = -1.0
        pair_info['valid_pixels'] = 0
        pair_info['error'] = str(e)[:200]
        return pair_info


# ============================================================================
# 3. 列出所有 pairs（從 realobs_root）
# ============================================================================
def elev_group_of(elev_str):
    e = int(elev_str)
    if e <= 4:
        return 'low'
    elif e <= 8:
        return 'mid'
    else:
        return 'high'


def list_all_pairs(realobs_root):
    """掃 realobs_root/EVENT/polar/binary/STATION/ 列出所有 raw+GT pairs"""
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
                stem = raw_gz.stem
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


# ============================================================================
# 4. 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Scan all realobs pairs and cache alias_ratio')
    parser.add_argument('--realobs_root', type=str, default='/path/to/typhoonnew/test',
                        help='Real obs root directory')
    parser.add_argument('--output', type=str, default='data/realobs_alias_cache.json',
                        help='Output cache JSON path')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of multiprocessing workers (4-8 推薦)')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save cache every N pairs (incremental save)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from existing cache (skip already-computed pairs)')
    args = parser.parse_args()

    # Windows path 處理
    if sys.platform.startswith(('linux', 'darwin')):
        rr = args.realobs_root.replace('\\', '/')
        if rr[:3].lower() in ('h:/', 'c:/', 'd:/', 'e:/'):
            rr = f'/mnt/{rr[0].lower()}/' + rr[3:]
        args.realobs_root = rr

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"Real Obs Alias Scanner")
    print(f"{'='*70}")
    print(f"  realobs_root : {args.realobs_root}")
    print(f"  output cache : {args.output}")
    print(f"  workers      : {args.workers}")
    print(f"  save_every   : {args.save_every} pairs")
    print(f"{'='*70}\n")

    # Step 1: 列出所有 pairs
    print(f"📂 掃描 (raw, GT) pairs...")
    t0 = time.time()
    all_pairs = list_all_pairs(args.realobs_root)
    print(f"   找到 {len(all_pairs)} pairs (耗時 {time.time()-t0:.1f}s)")

    # Step 2: Resume 載入既有 cache
    cached_results = {}  # key: raw_path → result dict
    if args.resume and output_path.exists():
        print(f"\n🔄 載入既有 cache: {output_path}")
        with open(output_path, encoding='utf-8') as f:
            existing = json.load(f)
        for r in existing:
            cached_results[r['raw']] = r
        print(f"   已 cache: {len(cached_results)} pairs")

    # 過濾未 cache 的
    pending = [p for p in all_pairs if p['raw'] not in cached_results]
    print(f"   待計算: {len(pending)} pairs")

    if len(pending) == 0:
        print("\n✅ 全部已 cache，無需計算")
        return

    # Step 3: Multi-process 計算
    print(f"\n⚙️  Compute alias_ratio (workers={args.workers})...")

    from multiprocessing import Pool
    n_done = 0
    t_compute = time.time()

    def save_cache(results_dict):
        out = list(results_dict.values())
        # 排序穩定性（按 raw path）
        out.sort(key=lambda r: r['raw'])
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    with Pool(processes=args.workers) as pool:
        try:
            with tqdm(total=len(pending), desc='compute alias', unit='pair') as pbar:
                for result in pool.imap_unordered(process_pair, pending, chunksize=8):
                    cached_results[result['raw']] = result
                    n_done += 1
                    pbar.update(1)

                    # 增量儲存
                    if n_done % args.save_every == 0:
                        save_cache(cached_results)
                        pbar.set_postfix_str(f"saved {len(cached_results)} total")
        except KeyboardInterrupt:
            print("\n⚠️  中斷！儲存 current cache...")
            save_cache(cached_results)
            print(f"   已存 {len(cached_results)} pairs 到 {output_path}")
            print(f"   下次跑時自動 resume")
            sys.exit(1)

    # 最終儲存
    save_cache(cached_results)
    elapsed = time.time() - t_compute
    print(f"\n✅ 全部完成！")
    print(f"   總耗時 (compute only): {elapsed/60:.1f} 分鐘")
    print(f"   平均速度: {len(pending)/elapsed:.1f} pairs/s")
    print(f"   cache: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")

    # 統計
    errors = sum(1 for r in cached_results.values() if r.get('alias_ratio', -1) < 0)
    print(f"\n📊 結果統計:")
    print(f"   總 pairs       : {len(cached_results)}")
    print(f"   成功計算       : {len(cached_results) - errors}")
    print(f"   失敗（shape/error）: {errors}")

    valid_results = [r for r in cached_results.values() if r.get('alias_ratio', -1) >= 0]
    if valid_results:
        # alias 分布
        from collections import Counter

        def bin_of(r):
            if r < 0.01: return 'clean'
            elif r < 0.30: return 'low'
            elif r < 0.70: return 'mid'
            else: return 'high'

        bins = Counter(bin_of(r['alias_ratio']) for r in valid_results)
        print(f"\n   alias_ratio 4-bin 分布:")
        for k in ['clean', 'low', 'mid', 'high']:
            n = bins.get(k, 0)
            print(f"     {k:7}: {n:>7d} ({100*n/len(valid_results):.1f}%)")

        # 按 event
        evt = Counter(r['event'] for r in valid_results)
        print(f"\n   按 event:")
        for k, n in sorted(evt.items()):
            print(f"     {k}: {n}")

        # 重要：每 event × bin
        evt_bin = defaultdict(int)
        for r in valid_results:
            evt_bin[(r['event'], bin_of(r['alias_ratio']))] += 1
        events = sorted(set(r['event'] for r in valid_results))
        print(f"\n   每 event 各 bin:")
        print(f"   {'event':30} {'clean':>8} {'low':>8} {'mid':>8} {'high':>8}")
        for e in events:
            row = [str(evt_bin.get((e, b), 0)) for b in ['clean','low','mid','high']]
            print(f"   {e:30} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")


if __name__ == '__main__':
    main()
