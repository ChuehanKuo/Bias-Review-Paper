#!/usr/bin/env python3
"""
ACM Digital Library Pipeline for Systematic Review:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

Uses OpenAlex API with ACM publisher filter.
1. Search OpenAlex (ACM publisher)
2. Deduplicate
3. Title+Abstract screening (shared criteria)
4. No full text available — all TA-included papers pass through
5. Extract structured columns
6. Build Excel + JSON backup
"""

import json
import urllib.request
import urllib.parse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_screening as ss

# ============================================================
# ACM PUBLISHER ID in OpenAlex
# ============================================================
ACM_PUBLISHER_ID = 'p4310319798'

# ============================================================
# QUERIES
# ============================================================
QUERIES = [
    {"id": "Q1", "label": "Core: algorithmic bias/fairness + health", "search": "algorithmic bias fairness health healthcare clinical medical"},
    {"id": "Q2", "label": "AI bias mitigation health", "search": "AI bias mitigation debiasing health healthcare clinical"},
    {"id": "Q3", "label": "Machine learning fairness health", "search": "machine learning fairness bias health clinical medical"},
    {"id": "Q4", "label": "Bias assessment AI healthcare", "search": "bias assessment evaluation audit artificial intelligence healthcare"},
    {"id": "Q5", "label": "Racial gender bias clinical AI", "search": "racial gender bias clinical prediction algorithm machine learning"},
    {"id": "Q6", "label": "Health disparities AI algorithm", "search": "health disparities algorithmic bias machine learning deep learning"},
    {"id": "Q7", "label": "Fairness-aware ML health", "search": "fairness-aware machine learning equalized odds demographic parity health"},
    {"id": "Q8", "label": "Bias EHR clinical decision support", "search": "bias electronic health record clinical decision support algorithm"},
    {"id": "Q9", "label": "Bias medical imaging AI", "search": "bias medical imaging radiology dermatology deep learning AI"},
    {"id": "Q10", "label": "Equitable AI health", "search": "equitable AI health equity fairness algorithm clinical"},
    {"id": "Q11", "label": "NLP bias clinical health", "search": "natural language processing bias clinical health medical NLP"},
    {"id": "Q12", "label": "Bias federated learning health", "search": "bias fairness federated learning health clinical"},
    {"id": "Q13", "label": "LLM bias health", "search": "large language model bias health clinical medical"},
    {"id": "Q14", "label": "Disparate impact health prediction", "search": "disparate impact health prediction model algorithm bias"},
    {"id": "Q15", "label": "Bias risk prediction clinical", "search": "bias risk prediction mortality readmission clinical algorithm"},
]

BASE_DIR = '/home/user/Bias-Review-Paper'


# ============================================================
# OpenAlex API
# ============================================================

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ''
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return ' '.join(w for _, w in words)


def search_openalex(query, publisher_id, per_page=200, max_results=2000):
    results = {}
    cursor = '*'
    fetched = 0

    while cursor and fetched < max_results:
        params = urllib.parse.urlencode({
            'search': query,
            'filter': f'primary_location.source.publisher_lineage:{publisher_id}',
            'per_page': per_page,
            'cursor': cursor,
            'mailto': 'review@example.com',
            'select': 'id,doi,title,publication_year,primary_location,authorships,'
                      'abstract_inverted_index,keywords,type,cited_by_count'
        })
        url = f'https://api.openalex.org/works?{params}'

        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            for work in data.get('results', []):
                doi = work.get('doi', '') or ''
                openalex_id = work.get('id', '')
                key = doi if doi else openalex_id
                if not key or key in results:
                    continue

                abstract = reconstruct_abstract(work.get('abstract_inverted_index'))
                authors = []
                for auth in work.get('authorships', []):
                    name = auth.get('author', {}).get('display_name', '')
                    if name:
                        authors.append(name)
                author_str = authors[0] + ' et al.' if len(authors) > 1 else (authors[0] if authors else 'N/A')
                source = work.get('primary_location', {}).get('source', {}) or {}
                venue = source.get('display_name', '')
                kw_list = work.get('keywords', []) or []
                keywords = '; '.join([k.get('display_name', '') for k in kw_list if k.get('display_name')])

                results[key] = {
                    'openalex_id': openalex_id,
                    'doi': doi.replace('https://doi.org/', '') if doi else '',
                    'title': work.get('title', '') or '',
                    'abstract': abstract,
                    'year': str(work.get('publication_year', '')) or '',
                    'journal': venue,
                    'authors': author_str,
                    'keywords': keywords,
                    'mesh_terms': '',
                    'pub_types': work.get('type', ''),
                    'url': doi if doi else openalex_id,
                    'cited_by_count': work.get('cited_by_count', 0),
                }
                fetched += 1

            cursor = data.get('meta', {}).get('next_cursor')
            if not data.get('results'):
                break

        except Exception as e:
            print(f"    ERROR: {e}")
            break

        time.sleep(0.2)

    return results, fetched


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ACM Digital Library Pipeline — Systematic Review (Fresh Run)")
    print("=" * 70)

    # Step 1: Search
    print(f"\nSTEP 1: Searching ACM via OpenAlex ({len(QUERIES)} queries)...")
    all_papers = {}
    query_stats = []

    for q in QUERIES:
        print(f"\n  {q['id']}: {q['label']}")
        papers, count = search_openalex(q['search'], ACM_PUBLISHER_ID)
        new_count = 0
        for key, paper in papers.items():
            if key not in all_papers:
                all_papers[key] = paper
                new_count += 1
        query_stats.append({
            'id': q['id'], 'label': q['label'], 'search': q['search'],
            'results_found': count, 'results': count,
            'new_unique': new_count, 'cumulative': len(all_papers)
        })
        print(f"    Found: {count} | New unique: {new_count} | Cumulative: {len(all_papers)}")
        time.sleep(0.3)

    print(f"\n  Total unique ACM papers: {len(all_papers)}")
    with_abstract = sum(1 for p in all_papers.values() if p.get('abstract'))
    print(f"  Papers with abstracts: {with_abstract}")

    # Step 2: Title+Abstract screening
    print(f"\nSTEP 2: Title+Abstract screening ({len(all_papers)} papers)...")
    ta_included = []
    ta_excluded = []

    for key, paper in all_papers.items():
        inc, reason = ss.screen_paper(paper)
        paper['include'] = inc
        paper['screen_reason'] = reason
        if inc:
            ta_included.append(paper)
        else:
            ta_excluded.append(paper)

    print(f"  INCLUDED: {len(ta_included)}")
    print(f"  EXCLUDED: {len(ta_excluded)}")

    # Step 3: No full text — pass through
    print(f"\nSTEP 3: Full text not available for ACM — passing through all {len(ta_included)} papers...")
    included = ta_included
    for p in included:
        p['ft_status'] = 'No full text available — passed through'
        p['ft_reason'] = 'ACM: full text paywalled'

    # Step 4: Extract columns
    print(f"\nSTEP 4: Extracting structured data for {len(included)} papers...")
    for p in included:
        ss.extract_all_columns(p)
    print(f"  Done.")

    included.sort(key=lambda x: x.get('year', '0'), reverse=True)

    all_excluded = []
    for p in ta_excluded:
        all_excluded.append({**p, 'exclusion_stage': 'Title+Abstract', 'exclusion_reason': p.get('screen_reason', '')})

    # Step 5: Build Excel
    print(f"\nSTEP 5: Building Excel...")
    meta = {
        'total_unique': len(all_papers),
        'ta_included': len(ta_included),
        'ft_screened': 0,
        'ft_unavailable': len(ta_included),
        'ft_excluded': 0,
    }
    config = {
        'db_name': 'ACM Digital Library (via OpenAlex)',
        'db_label': 'ACM',
        'header_color': '8B4513',
        'alt_row_color': 'FFF5EE',
        'id_field': 'doi',
        'id_label': 'DOI',
        'output_path': f'{BASE_DIR}/ACM_Screening_Results.xlsx',
        'date': '2026-02-17',
        'search_desc': 'OpenAlex API with ACM publisher filter (p4310319798)',
    }
    ss.build_excel(included, all_excluded, query_stats, meta, config)

    # Save JSON
    json_path = f'{BASE_DIR}/acm_screening_data.json'
    with open(json_path, 'w') as f:
        json.dump({
            'included': included,
            'ta_excluded': ta_excluded,
            'query_stats': query_stats,
            'meta': meta,
            'db': 'acm'
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Summary
    print(f"\n{'='*70}")
    print("ACM SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Total unique papers: {len(all_papers)}")
    print(f"  With abstracts: {with_abstract}")
    print(f"  Title+Abstract included: {len(ta_included)}")
    print(f"  FINAL INCLUDED: {len(included)}")
    print(f"  TOTAL EXCLUDED: {len(all_excluded)}")

    reasons = {}
    for p in ta_excluded:
        r = p.get('screen_reason', 'Unknown')
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\nExclusion breakdown:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {r}")

    ss.print_column_summary(included)


if __name__ == '__main__':
    main()
