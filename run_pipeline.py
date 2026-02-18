#!/usr/bin/env python3
"""
Master Orchestrator — Runs all database pipelines and deduplication.

Systematic Review: "Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

Pipeline order:
1. PubMed/MEDLINE
2. Scopus
3. ACM Digital Library (via OpenAlex)
4. IEEE Xplore (via OpenAlex)
5. Cross-database deduplication

All pipelines use shared_screening.py for consistent criteria.
"""

import subprocess
import sys
import time
import os

BASE_DIR = '/home/user/Bias-Review-Paper'

PIPELINES = [
    ('PubMed/MEDLINE', 'pipeline_pubmed.py'),
    ('Scopus', 'pipeline_scopus.py'),
    ('ACM Digital Library', 'pipeline_acm.py'),
    ('IEEE Xplore', 'pipeline_ieee.py'),
]


def run_script(name, script):
    """Run a pipeline script and report result."""
    path = os.path.join(BASE_DIR, script)
    print(f"\n{'#'*70}")
    print(f"# RUNNING: {name} ({script})")
    print(f"{'#'*70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, path],
        cwd=BASE_DIR,
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  {name} completed successfully ({elapsed:.1f}s)")
    else:
        print(f"\n  WARNING: {name} exited with code {result.returncode} ({elapsed:.1f}s)")

    return result.returncode


def main():
    print("=" * 70)
    print("SYSTEMATIC REVIEW — FULL PIPELINE RUN")
    print("Approaches for Assessing and Mitigating Algorithmic Bias in Health AI")
    print("=" * 70)
    print(f"\nDatabases: PubMed, Scopus, ACM (OpenAlex), IEEE (OpenAlex)")
    print(f"Screening: Shared criteria via shared_screening.py")
    print(f"Full-text: PMC for PubMed (pass through if unavailable); none for others")
    print(f"Dedup: DOI + normalized title matching across all databases")

    overall_start = time.time()
    results = {}

    # Run each pipeline
    for name, script in PIPELINES:
        rc = run_script(name, script)
        results[name] = rc

    # Run deduplication
    print(f"\n{'#'*70}")
    print(f"# RUNNING: Cross-Database Deduplication")
    print(f"{'#'*70}\n")
    rc = run_script('Deduplication', 'deduplicate.py')
    results['Deduplication'] = rc

    # Final summary
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print("PIPELINE RUN COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"\nPipeline results:")
    for name, rc in results.items():
        status = 'OK' if rc == 0 else f'FAILED (code {rc})'
        print(f"  {name}: {status}")

    print(f"\nOutput files:")
    for fname in ['PubMed_Screening_Results.xlsx', 'Scopus_Screening_Results.xlsx',
                   'ACM_Screening_Results.xlsx', 'IEEE_Screening_Results.xlsx',
                   'Combined_Deduplicated_Results.xlsx']:
        path = os.path.join(BASE_DIR, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {'OK' if exists else 'MISSING'} {fname} ({size//1024}KB)" if exists else f"  MISSING {fname}")

    print(f"\nJSON data files:")
    for fname in ['pubmed_screening_data.json', 'scopus_screening_data.json',
                   'acm_screening_data.json', 'ieee_screening_data.json',
                   'deduplicated_results.json']:
        path = os.path.join(BASE_DIR, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {'OK' if exists else 'MISSING'} {fname} ({size//1024}KB)" if exists else f"  MISSING {fname}")


if __name__ == '__main__':
    main()
