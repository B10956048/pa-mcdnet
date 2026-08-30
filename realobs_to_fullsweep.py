# -*- coding: utf-8 -*-
"""
realobs_to_fullsweep.py — 真實觀測 full-sweep fine-tune 資料集建置

從 3 個颱風訓練事件(璨樹/杜蘇芮/海葵)的 (raw, GT) gz 配對抽樣 full sweeps,
輸出與 nwp_to_patches --sweep_only 相同 schema 的 sweeps h5,
供 train_fullsweep.py --real_pairs fine-tune 使用(鏡射 M4 patch Stage-2 之資料來源)。

抽樣設計:
  - per-event 抽 target_per_event 個 sweep,含折錯 sweep 優先(上限 aliased_frac)
  - 含折錯判定:|GT - raw| > 0.5*V_nyq 之像素數 >= alias_px_min
  - 720-ray raw 以 azimuth-aware 對齊至 360-ray GT(重用 realobs_to_patches 實作)
  - split:per-sweep Bernoulli(val_ratio),事件內切分(與 patch FT 同哲學)

用法(WSL 或 Windows 皆可,路徑自動轉換):
  python realobs_to_fullsweep.py \
      --realobs_root /path/to/typhoonnew \
      --output_h5 data/realobs_fullsweep_3ty_sweeps.h5 \
      --target_per_event 1400 --aliased_frac 0.5 --seed 46
"""
import os
import sys
import time
import argparse
import random
from collections import defaultdict

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realobs_to_patches import (  # noqa: E402
    read_gz_radar, find_realobs_pairs, align_raw_to_gt_by_azimuth)


def build(args):
    rng = random.Random(args.seed)
    events = args.events
    print(f"事件: {events}")
    pairs = find_realobs_pairs(args.realobs_root, events, elevation=args.elevation)
    by_event = defaultdict(list)
    for p in pairs:
        by_event[p['case_name']].append(p)
    for e in events:
        rng.shuffle(by_event[e])
        print(f"  {e}: {len(by_event[e])} 對候選")

    tgt = args.target_per_event
    tgt_al = int(tgt * args.aliased_frac)
    tgt_cl = tgt - tgt_al
    max_reads_per_event = int(tgt * args.max_read_factor)

    os.makedirs(os.path.dirname(args.output_h5) or '.', exist_ok=True)
    t0 = time.time()
    stats = defaultdict(lambda: defaultdict(int))
    n_saved = 0

    with h5py.File(args.output_h5, 'w') as swf:
        for ev in events:
            got = {'aliased': 0, 'clean': 0}
            reads = 0
            for p in by_event[ev]:
                if got['aliased'] >= tgt_al and got['clean'] >= tgt_cl:
                    break
                if reads >= max_reads_per_event:
                    print(f"  [{ev}] 達讀取上限 {max_reads_per_event},"
                          f" aliased={got['aliased']}/{tgt_al} clean={got['clean']}/{tgt_cl}")
                    break
                reads += 1
                try:
                    raw, nyq, rmeta = read_gz_radar(p['raw'])
                    gt, _, gmeta = read_gz_radar(p['gt'])
                except Exception as exc:
                    stats[ev]['read_fail'] += 1
                    continue
                # 720-ray raw -> 360-ray GT 方位對齊
                if raw.shape[0] != gt.shape[0]:
                    if raw.shape[0] == 2 * gt.shape[0]:
                        raw = align_raw_to_gt_by_azimuth(
                            raw, rmeta['azm_start'], rmeta['azm_sp'],
                            gmeta['azm_start'], gmeta['azm_sp'], gt.shape[0])
                    else:
                        stats[ev]['shape_skip'] += 1
                        continue
                if raw.shape[1] != gt.shape[1]:
                    w = min(raw.shape[1], gt.shape[1])
                    raw, gt = raw[:, :w], gt[:, :w]

                valid = ~np.isnan(raw) & ~np.isnan(gt)
                nv = int(valid.sum())
                if nv < args.min_valid or not np.isfinite(nyq) or nyq <= 0:
                    stats[ev]['sparse_skip'] += 1
                    continue
                aliased_px = int((np.abs(gt[valid] - raw[valid]) > 0.5 * nyq).sum())
                is_al = aliased_px >= args.alias_px_min
                kind = 'aliased' if is_al else 'clean'
                if got[kind] >= (tgt_al if is_al else tgt_cl):
                    continue

                split = 'val' if rng.random() < args.val_ratio else 'train'
                sg = swf.create_group(f'sweep_{n_saved}')
                sg.create_dataset('da_vel', data=gt.astype(np.float32), compression='lzf')
                sg.create_dataset('raw_vel', data=raw.astype(np.float32), compression='lzf')
                sg.attrs['nyq'] = float(nyq)
                sg.attrs['station'] = p['station']
                sg.attrs['timestamp'] = f"{p['date']}.{p['time']}"
                sg.attrs['nwp_dir'] = ev
                sg.attrs['split'] = split
                sg.attrs['shape_az'] = gt.shape[0]
                sg.attrs['shape_range'] = gt.shape[1]
                sg.attrs['elev'] = p.get('elev', '')
                sg.attrs['aliased_px'] = aliased_px
                sg.attrs['is_aliased'] = bool(is_al)
                sg.attrs['src_raw'] = str(p['raw'])
                n_saved += 1
                got[kind] += 1
                stats[ev][f'{split}_{kind}'] += 1
                stats[ev][p['station']] += 1

                if n_saved % 200 == 0:
                    dt = time.time() - t0
                    print(f"  [{ev}] saved={n_saved} reads={reads} "
                          f"al={got['aliased']}/{tgt_al} cl={got['clean']}/{tgt_cl} "
                          f"({dt/60:.1f} min)", flush=True)

            print(f"[{ev}] 完成: aliased={got['aliased']}/{tgt_al} clean={got['clean']}/{tgt_cl} "
                  f"reads={reads}")

        swf.attrs['num_sweeps'] = n_saved
        swf.attrs['created_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        swf.attrs['events'] = ','.join(events)
        swf.attrs['target_per_event'] = tgt
        swf.attrs['aliased_frac'] = args.aliased_frac
        swf.attrs['alias_px_min'] = args.alias_px_min
        swf.attrs['min_valid'] = args.min_valid
        swf.attrs['val_ratio'] = args.val_ratio
        swf.attrs['seed'] = args.seed
        swf.attrs['purpose'] = 'realobs_fullsweep_finetune_real_pairs'

    print(f"\n✅ 共 {n_saved} sweeps -> {args.output_h5} "
          f"({os.path.getsize(args.output_h5)/1e9:.2f} GB, {(time.time()-t0)/60:.1f} min)")
    for ev in events:
        d = dict(stats[ev])
        print(f"  {ev}: " + ', '.join(f'{k}={v}' for k, v in sorted(d.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--realobs_root', type=str, default='/path/to/typhoonnew')
    ap.add_argument('--events', nargs='+',
                    default=['2021_Chanthu', '2023_DOKSURI', '2023_HAIKUI'])
    ap.add_argument('--output_h5', type=str,
                    default='data/realobs_fullsweep_3ty_sweeps.h5')
    ap.add_argument('--elevation', type=str, default='all')
    ap.add_argument('--target_per_event', type=int, default=1400)
    ap.add_argument('--aliased_frac', type=float, default=0.5,
                    help='含折錯 sweep 目標比例(不足時以 clean 補滿的上限邏輯)')
    ap.add_argument('--alias_px_min', type=int, default=50,
                    help='sweep 判定為含折錯之最少折錯像素數')
    ap.add_argument('--min_valid', type=int, default=800,
                    help='raw/GT 皆有效之最少像素數')
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--max_read_factor', type=float, default=8.0,
                    help='每事件最多讀取 target*factor 個檔(避免掃全庫)')
    ap.add_argument('--seed', type=int, default=46)
    build(ap.parse_args())


if __name__ == '__main__':
    main()
