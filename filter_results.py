#!/usr/bin/env python3
"""
Split deduplicated results into two files:
1. Full-text screened papers (no reviews)
2. Non-full-text screened / pass-through papers (no reviews)

Review papers are dropped entirely.
"""

import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = '/home/user/Bias-Review-Paper'

REVIEW_TYPES = {
    'Systematic Review', 'Scoping Review', 'Narrative Review',
    'Meta-Analysis', 'Review',
}


def is_review(paper):
    return paper.get('study_type', '') in REVIEW_TYPES


def is_passthrough(paper):
    ft = (paper.get('ft_status', '') or '').lower()
    return 'passed through' in ft


def build_excel(papers, title, out_path, sheet_name='Papers'):
    """Build a styled Excel workbook for a list of papers."""
    wb = Workbook()
    wb.remove(wb.active)

    h_font = Font(name='Calibri', bold=True, size=11, color='FFFFFF')
    h_fill = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid')
    h_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    c_align = Alignment(vertical='top', wrap_text=True)
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))
    alt_fill = PatternFill(start_color='EDF2F9', end_color='EDF2F9', fill_type='solid')

    # --- Summary ---
    ws_s = wb.create_sheet('Summary', index=0)
    by_db = {}
    for p in papers:
        db = p.get('source_db_label', 'Unknown')
        by_db[db] = by_db.get(db, 0) + 1
    by_type = {}
    for p in papers:
        st = p.get('study_type', 'Unknown')
        by_type[st] = by_type.get(st, 0) + 1

    summary_rows = [
        [title],
        [''],
        ['Total papers:', str(len(papers))],
        [''],
        ['By source:'],
    ]
    for db_label in ['PubMed/MEDLINE', 'Scopus', 'IEEE Xplore', 'ACM Digital Library']:
        c = by_db.get(db_label, 0)
        if c > 0:
            summary_rows.append([f'  {db_label}:', str(c)])
    summary_rows.append([''])
    summary_rows.append(['By study type:'])
    for st, c in sorted(by_type.items(), key=lambda x: -x[1]):
        summary_rows.append([f'  {st}:', str(c)])

    for ri, rd in enumerate(summary_rows, 1):
        for ci, v in enumerate(rd, 1):
            cell = ws_s.cell(row=ri, column=ci, value=v)
            if ri == 1:
                cell.font = Font(bold=True, size=14, color='1F4E79')
            elif ci == 1 and ':' in str(v):
                cell.font = Font(bold=True)
    ws_s.column_dimensions['A'].width = 35
    ws_s.column_dimensions['B'].width = 85

    # --- Papers sheet ---
    headers = [
        ('No.', 5), ('Source DB', 15), ('DOI', 30), ('Title', 65), ('Authors', 20),
        ('Year', 6), ('Journal/Venue', 35), ('URL', 35),
        ('Full-Text Status', 25),
        ('Study Type', 20), ('AI/ML Method', 30), ('Health Domain', 25),
        ('Bias Axes (Q1)', 30), ('Lifecycle Stage (Q2)', 25),
        ('Assessment vs Mitigation (Q2)', 22), ('Approach/Method', 35),
        ('Clinical Setting (Q3)', 25), ('Key Findings', 50),
    ]
    ws = wb.create_sheet(sheet_name)
    for ci, (h, w) in enumerate(headers, 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = h_font
        cell.fill = h_fill
        cell.alignment = h_align
        cell.border = border
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.freeze_panes = 'A2'

    for idx, p in enumerate(papers, 1):
        r = idx + 1
        vals = [
            idx, p.get('source_db_label', ''), p.get('doi', ''),
            p.get('title', ''), p.get('authors', ''),
            p.get('year', ''), p.get('journal', ''), p.get('url', ''),
            p.get('ft_status', ''),
            p.get('study_type', ''), p.get('ai_ml_method', ''),
            p.get('health_domain', ''),
            p.get('bias_axes', ''), p.get('lifecycle_stage', ''),
            p.get('assessment_or_mitigation', ''),
            p.get('approach_method', ''),
            p.get('clinical_setting', ''),
            p.get('key_findings', ''),
        ]
        for ci, v in enumerate(vals, 1):
            cell = ws.cell(row=r, column=ci, value=str(v) if v else '')
            cell.alignment = c_align
            cell.border = border
        if idx % 2 == 0:
            for ci in range(1, len(headers) + 1):
                ws.cell(row=r, column=ci).fill = alt_fill

    wb.save(out_path)
    print(f"  Saved: {out_path}  ({len(papers)} papers)")


def main():
    print("=" * 70)
    print("Splitting Results: Full-Text Screened vs Pass-Through (no reviews)")
    print("=" * 70)

    # Load and restore full dataset
    dedup_path = f'{BASE_DIR}/deduplicated_results.json'
    with open(dedup_path) as f:
        data = json.load(f)

    all_papers = data.get('unique', []) + data.get('removed', [])
    print(f"\nTotal papers (all): {len(all_papers)}")

    # Split into 3 buckets
    reviews = []
    fulltext_screened = []
    not_fulltext_screened = []

    for p in all_papers:
        if is_review(p):
            reviews.append(p)
        elif is_passthrough(p):
            not_fulltext_screened.append(p)
        else:
            fulltext_screened.append(p)

    print(f"\nReviews (dropped): {len(reviews)}")
    print(f"Full-text screened (no reviews): {len(fulltext_screened)}")
    print(f"Not full-text screened (no reviews): {len(not_fulltext_screened)}")

    # Review breakdown
    rev_types = {}
    for p in reviews:
        st = p.get('study_type', '')
        rev_types[st] = rev_types.get(st, 0) + 1
    print(f"\nDropped reviews by type:")
    for st, c in sorted(rev_types.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {st}")

    # File 1: Full-text screened, no reviews
    print(f"\n--- File 1: Full-Text Screened ---")
    build_excel(
        fulltext_screened,
        'Full-Text Screened Papers (Reviews Excluded)',
        f'{BASE_DIR}/FullText_Screened_Papers.xlsx',
        sheet_name='FullText_Screened',
    )

    # File 2: Not full-text screened, no reviews
    print(f"\n--- File 2: Not Full-Text Screened ---")
    build_excel(
        not_fulltext_screened,
        'Papers Without Full-Text Screening (Reviews Excluded)',
        f'{BASE_DIR}/NonFullText_Papers.xlsx',
        sheet_name='Non_FullText',
    )

    # Final summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Original: {len(all_papers)} unique papers")
    print(f"Reviews dropped: {len(reviews)}")
    print(f"FullText_Screened_Papers.xlsx: {len(fulltext_screened)} papers")
    print(f"NonFullText_Papers.xlsx: {len(not_fulltext_screened)} papers")
    print(f"Total kept: {len(fulltext_screened) + len(not_fulltext_screened)}")


if __name__ == '__main__':
    main()
