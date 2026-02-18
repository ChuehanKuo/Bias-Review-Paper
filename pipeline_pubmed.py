#!/usr/bin/env python3
"""
PubMed/MEDLINE Pipeline for Systematic Review:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

1. Search PubMed with focused Boolean queries (no cap)
2. Deduplicate PMIDs
3. Fetch title + abstract metadata
4. Title+Abstract screening (shared criteria)
5. Full-text screening via PMC (pass through if unavailable)
6. Extract structured columns
7. Build Excel + JSON backup
"""

import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared_screening as ss

# ============================================================
# QUERIES
# ============================================================

QUERIES = [
    {
        "id": "Q1",
        "label": "Core: algorithmic bias/fairness terms + health",
        "query": (
            '("algorithmic bias" OR "algorithmic fairness" OR "AI bias" OR "AI fairness" '
            'OR "machine learning bias" OR "machine learning fairness" OR "bias mitigation" '
            'OR "debiasing" OR "fairness-aware" OR "bias detection" OR "bias audit" '
            'OR "disparate impact" OR "demographic parity" OR "equalized odds") '
            'AND '
            '("health" OR "healthcare" OR "clinical" OR "medical" OR "biomedical")'
        )
    },
    {
        "id": "Q2",
        "label": "Bias assessment/mitigation approaches in health AI",
        "query": (
            '("bias" OR "fairness") '
            'AND ("assess" OR "mitigat" OR "detect" OR "evaluat" OR "framework" OR "approach" OR "method") '
            'AND ("artificial intelligence" OR "machine learning" OR "deep learning" OR "algorithm") '
            'AND ("health" OR "healthcare" OR "clinical" OR "medical")'
        )
    },
    {
        "id": "Q3",
        "label": "Specific bias axes in clinical AI",
        "query": (
            '("racial bias" OR "gender bias" OR "age bias" OR "socioeconomic bias" OR "ethnic bias") '
            'AND '
            '("artificial intelligence" OR "machine learning" OR "deep learning" '
            'OR "clinical prediction" OR "clinical algorithm" OR "clinical decision support")'
        )
    }
]

BASE_DIR = '/home/user/Bias-Review-Paper'


# ============================================================
# PUBMED API
# ============================================================

def search_pubmed(query, retmax=10000):
    encoded = urllib.parse.quote(query)
    count_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax=0&retmode=json"
    try:
        with urllib.request.urlopen(urllib.request.Request(count_url), timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        total = int(data.get('esearchresult', {}).get('count', '0'))
    except Exception as e:
        print(f"  ERROR: {e}")
        return [], 0

    all_ids = []
    for start in range(0, total, retmax):
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax={retmax}&retstart={start}&retmode=json"
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            all_ids.extend(data.get('esearchresult', {}).get('idlist', []))
        except Exception as e:
            print(f"  ERROR at offset {start}: {e}")
        time.sleep(0.4)

    return all_ids, total


def fetch_papers(pmids, batch_size=200):
    papers = {}
    total_batches = (len(pmids) - 1) // batch_size + 1
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        ids_str = ','.join(batch)
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids_str}&rettype=xml&retmode=xml"
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp:
                xml_data = resp.read().decode('utf-8')
            root = ET.fromstring(xml_data)
            for article in root.findall('.//PubmedArticle'):
                pmid_elem = article.find('.//PMID')
                if pmid_elem is None:
                    continue
                pmid = pmid_elem.text
                title_elem = article.find('.//ArticleTitle')
                title = ''.join(title_elem.itertext()) if title_elem is not None else ''
                abstract_parts = []
                for at in article.findall('.//AbstractText'):
                    label = at.get('Label', '')
                    text = ''.join(at.itertext())
                    abstract_parts.append(f"{label}: {text}" if label else text)
                abstract = ' '.join(abstract_parts)
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ''
                year = ''
                for yp in ['.//PubDate/Year', './/ArticleDate/Year', './/PubDate/MedlineDate']:
                    ye = article.find(yp)
                    if ye is not None and ye.text:
                        year = ye.text[:4]
                        break
                doi = ''
                for aid in article.findall('.//ArticleId'):
                    if aid.get('IdType') == 'doi':
                        doi = aid.text
                        break
                if not doi:
                    for eloc in article.findall('.//ELocationID'):
                        if eloc.get('EIdType') == 'doi':
                            doi = eloc.text
                            break
                authors = []
                for au in article.findall('.//Author'):
                    last = au.find('LastName')
                    fore = au.find('ForeName')
                    if last is not None and last.text:
                        name = last.text + (f" {fore.text}" if fore is not None and fore.text else '')
                        authors.append(name)
                author_str = authors[0] + ' et al.' if len(authors) > 1 else (authors[0] if authors else 'N/A')
                keywords = [kw.text for kw in article.findall('.//Keyword') if kw.text]
                mesh = [m.text for m in article.findall('.//MeshHeading/DescriptorName') if m.text]
                pub_types = [pt.text for pt in article.findall('.//PublicationType') if pt.text]

                papers[pmid] = {
                    'pmid': pmid, 'title': title, 'abstract': abstract,
                    'journal': journal, 'year': year, 'doi': doi,
                    'authors': author_str,
                    'keywords': '; '.join(keywords[:20]),
                    'mesh_terms': '; '.join(mesh[:20]),
                    'pub_types': '; '.join(pub_types),
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
                }
            print(f"  Batch {i//batch_size+1}/{total_batches} done ({len(papers)} total)")
        except Exception as e:
            print(f"  ERROR batch {i//batch_size+1}: {e}")
        time.sleep(0.4)
    return papers


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PubMed/MEDLINE Pipeline — Systematic Review (Fresh Run)")
    print("=" * 70)

    # Step 1: Search
    print(f"\nSTEP 1: Searching PubMed ({len(QUERIES)} queries, no cap)...")
    all_pmids = set()
    query_stats = []
    for q in QUERIES:
        print(f"\n  {q['id']}: {q['label']}")
        ids, total = search_pubmed(q['query'])
        new = set(ids) - all_pmids
        all_pmids.update(ids)
        query_stats.append({
            'id': q['id'], 'label': q['label'], 'query': q['query'],
            'total_in_pubmed': total, 'results': total,
            'new_unique': len(new), 'cumulative': len(all_pmids)
        })
        print(f"  Total: {total} | Retrieved: {len(ids)} | New: {len(new)} | Cumulative: {len(all_pmids)}")
        time.sleep(0.5)
    print(f"\n  Total unique PMIDs: {len(all_pmids)}")

    # Step 2: Fetch metadata
    print(f"\nSTEP 2: Fetching titles + abstracts...")
    papers = fetch_papers(sorted(all_pmids))
    print(f"  Fetched: {len(papers)}")

    # Step 3: Title+Abstract screening
    print(f"\nSTEP 3: Title+Abstract screening ({len(papers)} papers)...")
    ta_included = []
    ta_excluded = []
    for pmid, p in papers.items():
        inc, reason = ss.screen_paper(p)
        p['include'] = inc
        p['screen_reason'] = reason
        if inc:
            ta_included.append(p)
        else:
            ta_excluded.append(p)
    print(f"  INCLUDED: {len(ta_included)}")
    print(f"  EXCLUDED: {len(ta_excluded)}")

    # Step 4: Full-text screening via PMC
    print(f"\nSTEP 4: Full-text screening via PMC for {len(ta_included)} papers...")
    included, ft_excluded, ft_stats = ss.run_fulltext_screening(ta_included, id_field='pmid', db_label='PubMed')

    # Step 5: Extract columns
    print(f"\nSTEP 5: Extracting structured data for {len(included)} papers...")
    for p in included:
        ss.extract_all_columns(p)
    print(f"  Done.")

    included.sort(key=lambda x: x.get('year', '0'), reverse=True)

    # Combine excluded
    all_excluded = []
    for p in ta_excluded:
        all_excluded.append({**p, 'exclusion_stage': 'Title+Abstract', 'exclusion_reason': p.get('screen_reason', '')})
    for p in ft_excluded:
        all_excluded.append({**p, 'exclusion_stage': 'Full-Text', 'exclusion_reason': p.get('ft_reason', '')})

    # Step 6: Build Excel
    print(f"\nSTEP 6: Building Excel...")
    meta = {
        'total_unique': len(all_pmids),
        'total_fetched': len(papers),
        'ta_included': len(ta_included),
        'ft_screened': ft_stats['ft_screened'],
        'ft_unavailable': ft_stats['ft_unavailable'],
        'ft_excluded': ft_stats['ft_excluded'],
    }
    config = {
        'db_name': 'PubMed/MEDLINE',
        'db_label': 'PubMed',
        'header_color': '2F5496',
        'alt_row_color': 'F2F7FB',
        'id_field': 'pmid',
        'id_label': 'PMID',
        'output_path': f'{BASE_DIR}/PubMed_Screening_Results.xlsx',
        'date': '2026-02-17',
        'search_desc': 'NCBI E-utilities API with focused Boolean queries (no retrieval cap)',
    }
    ss.build_excel(included, all_excluded, query_stats, meta, config)

    # Save JSON
    json_path = f'{BASE_DIR}/pubmed_screening_data.json'
    with open(json_path, 'w') as f:
        json.dump({
            'included': included,
            'ft_excluded': ft_excluded,
            'ta_excluded': ta_excluded,
            'query_stats': query_stats,
            'meta': meta,
            'db': 'pubmed'
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Summary
    print(f"\n{'='*70}")
    print("PUBMED SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Total unique PMIDs: {len(all_pmids)}")
    print(f"  Fetched: {len(papers)}")
    print(f"  Title+Abstract included: {len(ta_included)}")
    print(f"  Full-text screened: {ft_stats['ft_screened']}")
    print(f"  Full-text excluded: {ft_stats['ft_excluded']}")
    print(f"  Passed through (no full text): {ft_stats['ft_unavailable']}")
    print(f"  FINAL INCLUDED: {len(included)}")
    print(f"  TOTAL EXCLUDED: {len(all_excluded)}")

    ss.print_column_summary(included)


if __name__ == '__main__':
    main()
