"""Validate and show how to organize HOSS dataset for Hoss-ReID project.

Usage:
    python tools/validate_hoss_dataset.py --root /path/to/HOSS

Expectations (from code in `datasets/hoss.py`):
- Root folder contains a folder named `HOSS` (or pass that folder as --root).
- Inside HOSS, there should be three subfolders:
    - `bounding_box_train`
    - `query`
    - `bounding_box_test`
- Images are TIFF files with filenames where the person id (pid) is the first token
  separated by underscore, and the modality is indicated by the final token:
      <pid>_..._RGB.tif  or  <pid>_..._SAR.tif

Examples of valid names:
  0001_1_RGB.tif
  0001_2_SAR.tif
  12_cam3_RGB.tif

This script checks these constraints and prints summary information and
suggested mv commands for common mismatches (non-conforming extensions or
modality labels).
"""

import argparse
import os
import re
from collections import defaultdict

FILENAME_RE = re.compile(r"^(?P<pid>\d+)_.*_(?P<modality>(RGB|SAR))\.tif$", re.IGNORECASE)


def scan_folder(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.tif')]
    info = []
    for f in files:
        m = FILENAME_RE.match(f)
        if m:
            pid = int(m.group('pid'))
            modality = m.group('modality').upper()
            info.append((f, pid, modality))
        else:
            info.append((f, None, None))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='HOSS', help='Path to HOSS folder (default: HOSS)')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"ERROR: root folder '{root}' not found.")
        return

    expected_subs = ['bounding_box_train', 'query', 'bounding_box_test']
    for s in expected_subs:
        p = os.path.join(root, s)
        if not os.path.isdir(p):
            print(f"WARNING: expected subfolder '{s}' not found under {root}")

    print('\nScanning folders (showing first 20 mismatches per folder)...')
    overall_pairs = {}
    for s in expected_subs:
        folder = os.path.join(root, s)
        if not os.path.isdir(folder):
            continue
        info = scan_folder(folder)
        total = len(info)
        matched = [x for x in info if x[1] is not None]
        unmatched = [x for x in info if x[1] is None]
        print(f"\nFolder: {s}  --  total tif: {total}, matched pattern: {len(matched)}, unmatched: {len(unmatched)}")
        if unmatched:
            print("  Example unmatched (may need renaming):")
            for f,_,_ in unmatched[:20]:
                print(f"    {f}")

        # build pid->modalities map for matched
        pid_map = defaultdict(list)
        for fname, pid, modality in matched:
            pid_map[pid].append((fname, modality))
        pair_count = 0
        for pid, lst in pid_map.items():
            mods = set(m for _, m in lst)
            if 'RGB' in mods and 'SAR' in mods:
                pair_count += 1
        overall_pairs[s] = (len(pid_map), pair_count)
        print(f"  unique pids: {len(pid_map)}, paired(pid with both RGB&SAR): {pair_count}")

    print('\nSummary per folder:')
    for s, (unique_pids, pair_count) in overall_pairs.items():
        print(f"  {s}: {unique_pids} unique pids, {pair_count} paired pids")

    print('\nNaming convention reminder:')
    print('  Filenames must match: <pid>_..._RGB.tif  OR  <pid>_..._SAR.tif')
    print('  where <pid> is digits (e.g., 0001, 12, 345).')

    print('\nIf your files use different modality suffixes (e.g., rgb.TIF, sar.tiff),')
    print('you can rename them with shell commands like:')
    print("  for f in $(find HOSS -iname '*rgb.tif'); do mv \"$f\" \"${f%.*}_RGB.tif\"; done")
    print('Or use the following python snippet to inspect and propose renames (non-destructive):')
    print("  python - <<'PY'\nimport os,re\npat=re.compile(r'.*(rgb|sar)\\.tif$', re.I)\nfor root,_,files in os.walk('HOSS'):\n  for f in files:\n    if pat.match(f):\n      print(os.path.join(root,f))\nPY')

    print('\nFinished. If you want, I can:')
    print('  - generate a non-destructive rename script to standardize filenames')
    print('  - produce a small visualizer to show paired examples after transforms')


if __name__ == '__main__':
    main()
