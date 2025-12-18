#!/usr/bin/env python3
"""
Validate AircraftShip dataset for potential cheating in evaluation.
Checks for overlaps between query and gallery sets.
"""

import os
import sys
sys.path.append('/home/ftt/CMFGR')

from datasets.aircraft_ship import AircraftShip

def main():
    # Load dataset
    dataset = AircraftShip(root='/home/ftt/CMFGR/data')

    # Extract paths
    query_paths = {os.path.basename(item[0]) for item in dataset.query}
    gallery_paths = {os.path.basename(item[0]) for item in dataset.gallery}

    # Check for overlaps
    overlap = query_paths & gallery_paths
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping images between query and gallery!")
        print("Sample overlaps:", list(overlap)[:5])
        return False
    else:
        print("No overlaps found between query and gallery sets.")
        return True

    # Check PIDs
    query_pids = {item[1] for item in dataset.query}
    gallery_pids = {item[1] for item in dataset.gallery}
    pid_overlap = query_pids & gallery_pids
    print(f"Query PIDs: {len(query_pids)}, Gallery PIDs: {len(gallery_pids)}")
    print(f"PID overlap: {len(pid_overlap)}")

    # Check camids
    query_camids = {item[2] for item in dataset.query}
    gallery_camids = {item[2] for item in dataset.gallery}
    print(f"Query camids: {query_camids}, Gallery camids: {gallery_camids}")

if __name__ == "__main__":
    main()