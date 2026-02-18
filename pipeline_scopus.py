#!/usr/bin/env python3
"""
Scopus Pipeline for Systematic Review:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

1. Search Scopus via Elsevier API with focused queries
2. Fetch abstracts via OpenAlex
3. Title+Abstract screening (shared criteria)
4. Full-text screening via PMC (DOI lookup)
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
# QUERIES (Scopus Boolean syntax)
# ============================================================

QUERIES = [
    {
        "id": "Q1",
        "label": "Core: algorithmic bias/fairness + health",
        "query": (
            'TITLE-ABS-KEY("algorithmic bias" OR "algorithmic fairness" OR "AI bias" OR "AI fairness" '
            'OR "machine learning bias" OR "machine learning fairness" OR "bias mitigation" '
            'OR "debiasing" OR "fairness-aware" OR "bias detection" OR "bias audit" '
            'OR "disparate impact" OR "demographic parity" OR "equalized odds") '
            'AND TITLE-ABS-KEY("health" OR "healthcare" OR "clinical" OR "medical" OR "biomedical")'
        )
    },
    {
        "id": "Q2",
        "label": "Bias assessment/mitigation approaches in health AI",
        "query": (
            'TITLE-ABS-KEY("bias" OR "fairness") '
            'AND TITLE-ABS-KEY("assess" OR "mitigat" OR "detect" OR "evaluat" OR "framework" OR "approach") '
            'AND TITLE-ABS-KEY("artificial intelligence" OR "machine learning" OR "deep learning" OR "algorithm") '
            'AND TITLE-ABS-KEY("health" OR "healthcare" OR "clinical" OR "medical")'
        )
    },
    {
        "id": "Q3",
        "label": "Specific bias axes in clinical AI",
        "query": (
            'TITLE-ABS-KEY("racial bias" OR "gender bias" OR "age bias" OR "socioeconomic bias" OR "ethnic bias") '
            'AND TITLE-ABS-KEY("artificial intelligence" OR "machine learning" OR "deep learning" '
            'OR "clinical prediction" OR "clinical algorithm" OR "clinical decision support")'
        )
    }
]

BASE_DIR = '/home/user/Bias-Review-Paper'


# ============================================================
# API KEY
# ============================================================

def load_api_key():
    env_path = os.path.join(BASE_DIR, '.env')
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('SCOPUS_API_KEY='):
                    return line.split('=', 1)[1].strip()
    except Exception:
        pass
    return os.environ.get('SCOPUS_API_KEY', '')


# ============================================================
# SCOPUS SEARCH API
# ============================================================

def reconstruct_abstract(inverted_index):
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inverted_index:
        return ''
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return ' '.join(w for _, w in words)


def fetch_abstracts_via_openalex(papers):
    """
    Fetch abstracts for papers via OpenAlex (free API) using DOI lookup.
    Falls back to title search if no DOI.
    """
    fetched = 0
    total = len(papers)
    batch_dois = []
    batch_papers = []

    for p in papers:
        doi = p.get('doi', '')
        if doi:
            batch_dois.append(doi)
            batch_papers.append(p)

    # Batch query OpenAlex by DOI (up to 50 at a time using filter)
    for i in range(0, len(batch_dois), 50):
        chunk_dois = batch_dois[i:i+50]
        chunk_papers = batch_papers[i:i+50]
        doi_filter = '|'.join(f'https://doi.org/{d}' if not d.startswith('http') else d for d in chunk_dois)
        params = urllib.parse.urlencode({
            'filter': f'doi:{doi_filter}',
            'per_page': 50,
            'mailto': 'review@example.com',
            'select': 'doi,abstract_inverted_index,keywords'
        })
        url = f'https://api.openalex.org/works?{params}'
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            # Build DOI -> abstract map
            for work in data.get('results', []):
                w_doi = (work.get('doi', '') or '').replace('https://doi.org/', '').lower()
                abstract = reconstruct_abstract(work.get('abstract_inverted_index'))
                if not abstract:
                    continue
                # Match back to paper
                for p in chunk_papers:
                    p_doi = p.get('doi', '').lower()
                    if p_doi == w_doi and not p.get('abstract'):
                        p['abstract'] = abstract
                        kw_list = work.get('keywords', []) or []
                        if kw_list and not p.get('keywords'):
                            p['keywords'] = '; '.join(k.get('display_name', '') for k in kw_list if isinstance(k, dict) and k.get('display_name'))
                        fetched += 1
        except Exception as e:
            pass  # Continue with what we have

        if (i + 50) % 200 == 0:
            print(f"    Processed {min(i+50, len(batch_dois))}/{len(batch_dois)} DOIs ({fetched} abstracts found)")
        time.sleep(0.2)

    print(f"  Fetched {fetched} abstracts via OpenAlex (from {len(batch_dois)} DOIs)")
    return fetched


def search_scopus(query, api_key, max_results=5000):
    """
    Search Scopus using the Elsevier Search API.
    Returns dict of papers keyed by Scopus ID or DOI.
    """
    results = {}
    start = 0
    count = 25  # Scopus default page size

    while start < max_results:
        params = urllib.parse.urlencode({
            'query': query,
            'count': count,
            'start': start,
            'sort': 'relevancy',
            'field': 'dc:identifier,dc:title,dc:creator,prism:publicationName,prism:coverDate,prism:doi,dc:description,authkeywords,subtypeDescription,citedby-count',
        })
        url = f'https://api.elsevier.com/content/search/scopus?{params}'
        req = urllib.request.Request(url, headers={
            'X-ELS-APIKey': api_key,
            'Accept': 'application/json',
        })

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            search_results = data.get('search-results', {})
            total = int(search_results.get('opensearch:totalResults', '0'))
            entries = search_results.get('entry', [])

            if not entries:
                break

            # Check for error entries
            if len(entries) == 1 and entries[0].get('error'):
                print(f"    API error: {entries[0].get('error')}")
                break

            for entry in entries:
                scopus_id = entry.get('dc:identifier', '').replace('SCOPUS_ID:', '')
                doi = entry.get('prism:doi', '')
                title = entry.get('dc:title', '')
                key = doi if doi else scopus_id
                if not key or key in results:
                    continue

                # Parse date -> year
                cover_date = entry.get('prism:coverDate', '')
                year = cover_date[:4] if cover_date else ''

                results[key] = {
                    'scopus_id': scopus_id,
                    'doi': doi,
                    'title': title,
                    'abstract': entry.get('dc:description', '') or '',
                    'journal': entry.get('prism:publicationName', '') or '',
                    'year': year,
                    'authors': entry.get('dc:creator', '') or 'N/A',
                    'keywords': entry.get('authkeywords', '') or '',
                    'mesh_terms': '',
                    'pub_types': entry.get('subtypeDescription', '') or '',
                    'url': f'https://doi.org/{doi}' if doi else f'https://www.scopus.com/record/display.uri?eid={scopus_id}',
                    'cited_by_count': entry.get('citedby-count', ''),
                }

            start += count
            if start >= total:
                break

        except urllib.error.HTTPError as e:
            print(f"    HTTP Error {e.code}: {e.reason}")
            if e.code == 429:
                print("    Rate limited — waiting 10s...")
                time.sleep(10)
                continue
            elif e.code == 401:
                print("    Authentication failed — check API key")
                break
            else:
                break
        except Exception as e:
            print(f"    ERROR: {e}")
            break

        time.sleep(0.3)

    return results, len(results)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Scopus Pipeline — Systematic Review (New)")
    print("=" * 70)

    api_key = load_api_key()
    if not api_key:
        print("ERROR: No Scopus API key found in .env or environment")
        return

    print(f"  API key loaded: {api_key[:8]}...")

    # Step 1: Search
    print(f"\nSTEP 1: Searching Scopus ({len(QUERIES)} queries)...")
    all_papers = {}
    query_stats = []

    for q in QUERIES:
        print(f"\n  {q['id']}: {q['label']}")
        papers, count = search_scopus(q['query'], api_key)
        new_count = 0
        for key, paper in papers.items():
            if key not in all_papers:
                all_papers[key] = paper
                new_count += 1
        query_stats.append({
            'id': q['id'], 'label': q['label'], 'query': q['query'],
            'results_found': count, 'results': count,
            'new_unique': new_count, 'cumulative': len(all_papers)
        })
        print(f"    Found: {count} | New unique: {new_count} | Cumulative: {len(all_papers)}")
        time.sleep(0.5)

    print(f"\n  Total unique Scopus papers: {len(all_papers)}")
    with_abstract = sum(1 for p in all_papers.values() if p.get('abstract'))
    print(f"  Papers with abstracts: {with_abstract}")

    # Step 2: Fetch abstracts via OpenAlex (Scopus API doesn't return abstracts with this key)
    no_abstract = [p for p in all_papers.values() if not p.get('abstract')]
    if no_abstract:
        print(f"\nSTEP 2: Fetching abstracts for {len(no_abstract)} papers via OpenAlex (free)...")
        fetch_abstracts_via_openalex(no_abstract)
        with_abstract = sum(1 for p in all_papers.values() if p.get('abstract'))
        print(f"  Papers with abstracts now: {with_abstract} / {len(all_papers)}")

    # Step 3: Title+Abstract screening
    print(f"\nSTEP 3: Title+Abstract screening ({len(all_papers)} papers)...")
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

    # Step 4: Full-text screening via PMC (DOI lookup)
    print(f"\nSTEP 4: Full-text screening via PMC for {len(ta_included)} papers...")
    included, ft_excluded, ft_stats = ss.run_fulltext_screening(ta_included, id_field='doi', db_label='Scopus')

    # Step 5: Extract columns
    print(f"\nSTEP 5: Extracting structured data for {len(included)} papers...")
    for p in included:
        ss.extract_all_columns(p)
    print(f"  Done.")

    included.sort(key=lambda x: x.get('year', '0'), reverse=True)

    # Build excluded list
    all_excluded = []
    for p in ta_excluded:
        all_excluded.append({**p, 'exclusion_stage': 'Title+Abstract', 'exclusion_reason': p.get('screen_reason', '')})
    for p in ft_excluded:
        all_excluded.append({**p, 'exclusion_stage': 'Full-Text', 'exclusion_reason': p.get('ft_reason', '')})

    # Step 6: Build Excel
    print(f"\nSTEP 6: Building Excel...")
    meta = {
        'total_unique': len(all_papers),
        'ta_included': len(ta_included),
        'ft_screened': ft_stats['ft_screened'],
        'ft_unavailable': ft_stats['ft_unavailable'],
        'ft_excluded': ft_stats['ft_excluded'],
    }
    config = {
        'db_name': 'Scopus',
        'db_label': 'Scopus',
        'header_color': 'E87722',  # Elsevier orange
        'alt_row_color': 'FFF3E6',
        'id_field': 'scopus_id',
        'id_label': 'Scopus ID',
        'output_path': f'{BASE_DIR}/Scopus_Screening_Results.xlsx',
        'date': '2026-02-17',
        'search_desc': 'Elsevier Scopus Search API with focused Boolean queries',
    }
    ss.build_excel(included, all_excluded, query_stats, meta, config)

    # Save JSON
    json_path = f'{BASE_DIR}/scopus_screening_data.json'
    with open(json_path, 'w') as f:
        json.dump({
            'included': included,
            'ft_excluded': ft_excluded,
            'ta_excluded': ta_excluded,
            'query_stats': query_stats,
            'meta': meta,
            'db': 'scopus'
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SCOPUS SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Total unique papers: {len(all_papers)}")
    print(f"  With abstracts: {with_abstract}")
    print(f"  Title+Abstract included: {len(ta_included)}")
    print(f"  Full-text screened: {ft_stats['ft_screened']}")
    print(f"  Full-text excluded: {ft_stats['ft_excluded']}")
    print(f"  Passed through (no full text): {ft_stats['ft_unavailable']}")
    print(f"  FINAL INCLUDED: {len(included)}")
    print(f"  TOTAL EXCLUDED: {len(all_excluded)}")

    # TA exclusion breakdown
    reasons = {}
    for p in ta_excluded:
        r = p.get('screen_reason', 'Unknown')
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\nTitle+Abstract exclusion breakdown:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {r}")

    ss.print_column_summary(included)


if __name__ == '__main__':
    main()
