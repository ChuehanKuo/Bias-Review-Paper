#!/usr/bin/env python3
"""
PubMed Step 1: Search + Fetch + Remove Reviews

Systematic Review: "Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

- Runs focused queries via NCBI E-utilities
- Deduplicates PMIDs across queries
- Fetches metadata (title, abstract, authors, etc.)
- Removes review papers (systematic review, scoping review, narrative review, meta-analysis)
- Saves results as JSON for next step
"""

import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET

BASE_DIR = '/home/user/Bias-Review-Paper'

# ============================================================
# SEARCH QUERIES
# ============================================================

QUERIES = [
    {
        "id": "Q1",
        "label": "Core: algorithmic bias/fairness + health",
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
    },
    {
        "id": "Q4",
        "label": "Fairness metrics and frameworks in health",
        "query": (
            '("fairness metric" OR "fairness framework" OR "bias framework" OR "equity framework" '
            'OR "AI Fairness 360" OR "fairlearn" OR "aequitas") '
            'AND '
            '("health" OR "healthcare" OR "clinical" OR "medical")'
        )
    },
    {
        "id": "Q5",
        "label": "Bias in medical imaging / clinical NLP",
        "query": (
            '("bias" OR "fairness") '
            'AND ("deep learning" OR "neural network" OR "convolutional" OR "natural language processing") '
            'AND ("medical imaging" OR "radiology" OR "pathology" OR "dermatology" '
            'OR "clinical notes" OR "electronic health record")'
        )
    },
    {
        "id": "Q6",
        "label": "Health disparities + AI/ML",
        "query": (
            '"health disparities" '
            'AND ("machine learning" OR "artificial intelligence" OR "deep learning" OR "algorithm") '
            'AND ("bias" OR "fairness" OR "equity")'
        )
    },
    {
        "id": "Q7",
        "label": "Equitable AI in clinical settings",
        "query": (
            '("equitable" OR "equity") '
            'AND ("artificial intelligence" OR "machine learning") '
            'AND ("clinical" OR "patient" OR "hospital" OR "diagnosis" OR "treatment")'
        )
    },
]


# ============================================================
# PUBMED API
# ============================================================

def search_pubmed(query, retmax=10000):
    """Search PubMed and return all PMIDs + total count."""
    encoded = urllib.parse.quote(query)
    # Get count first
    count_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax=0&retmode=json"
    try:
        with urllib.request.urlopen(urllib.request.Request(count_url), timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        total = int(data.get('esearchresult', {}).get('count', '0'))
    except Exception as e:
        print(f"    ERROR getting count: {e}")
        return [], 0

    # Fetch all IDs
    all_ids = []
    for start in range(0, min(total, retmax), 500):
        url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
               f"db=pubmed&term={encoded}&retmax=500&retstart={start}&retmode=json")
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            ids = data.get('esearchresult', {}).get('idlist', [])
            all_ids.extend(ids)
        except Exception as e:
            print(f"    ERROR at offset {start}: {e}")
        time.sleep(0.35)

    return all_ids, total


def fetch_papers(pmids, batch_size=200):
    """Fetch full metadata for a list of PMIDs."""
    papers = {}
    total_batches = (len(pmids) - 1) // batch_size + 1

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        ids_str = ','.join(batch)
        url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
               f"db=pubmed&id={ids_str}&rettype=xml&retmode=xml")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SystematicReview/1.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                xml_data = resp.read().decode('utf-8')

            root = ET.fromstring(xml_data)
            for article in root.findall('.//PubmedArticle'):
                pmid_elem = article.find('.//PMID')
                if pmid_elem is None:
                    continue
                pmid = pmid_elem.text

                # Title
                title_elem = article.find('.//ArticleTitle')
                title = ''.join(title_elem.itertext()) if title_elem is not None else ''

                # Abstract
                abstract_parts = []
                for at in article.findall('.//AbstractText'):
                    label = at.get('Label', '')
                    text = ''.join(at.itertext())
                    abstract_parts.append(f"{label}: {text}" if label else text)
                abstract = ' '.join(abstract_parts)

                # Journal
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ''

                # Year
                year = ''
                for yp in ['.//PubDate/Year', './/ArticleDate/Year', './/PubDate/MedlineDate']:
                    ye = article.find(yp)
                    if ye is not None and ye.text:
                        year = ye.text[:4]
                        break

                # DOI
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

                # Authors
                authors = []
                for au in article.findall('.//Author'):
                    last = au.find('LastName')
                    fore = au.find('ForeName')
                    if last is not None and last.text:
                        name = last.text + (f" {fore.text}" if fore is not None and fore.text else '')
                        authors.append(name)
                author_str = authors[0] + ' et al.' if len(authors) > 1 else (authors[0] if authors else 'N/A')

                # Keywords, MeSH, Pub types
                keywords = [kw.text for kw in article.findall('.//Keyword') if kw.text]
                mesh = [m.text for m in article.findall('.//MeshHeading/DescriptorName') if m.text]
                pub_types = [pt.text for pt in article.findall('.//PublicationType') if pt.text]

                papers[pmid] = {
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'journal': journal,
                    'year': year,
                    'doi': doi,
                    'authors': author_str,
                    'keywords': '; '.join(keywords[:20]),
                    'mesh_terms': '; '.join(mesh[:20]),
                    'pub_types': '; '.join(pub_types),
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    'source_db': 'pubmed',
                    'source_db_label': 'PubMed/MEDLINE',
                }

            batch_num = i // batch_size + 1
            print(f"    Batch {batch_num}/{total_batches} done ({len(papers)} total)")
        except Exception as e:
            print(f"    ERROR batch {i//batch_size+1}: {e}")
        time.sleep(0.4)

    return papers


# ============================================================
# REVIEW DETECTION
# ============================================================

REVIEW_TYPES = {
    'Systematic Review', 'Scoping Review', 'Narrative Review',
    'Meta-Analysis', 'Review',
}

def classify_study_type(paper):
    """Classify study type for review detection."""
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    pub_types = (paper.get('pub_types', '') or '').lower()
    combined = title + ' ' + abstract + ' ' + pub_types

    if any(t in combined for t in ['systematic review', 'prisma', 'systematic literature review']):
        return 'Systematic Review'
    if any(t in combined for t in ['scoping review', 'scoping study']):
        return 'Scoping Review'
    if any(t in combined for t in ['narrative review', 'literature review', 'review article']):
        return 'Narrative Review'
    if 'meta-analysis' in combined:
        return 'Meta-Analysis'
    # Check PubMed publication type field specifically
    if 'review' in pub_types:
        return 'Review'
    return None  # Not a review


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STEP 1: PubMed Search + Fetch + Remove Reviews")
    print("=" * 70)

    # --- Search ---
    print(f"\nSearching PubMed ({len(QUERIES)} queries)...\n")
    all_pmids = set()
    query_stats = []

    for q in QUERIES:
        print(f"  {q['id']}: {q['label']}")
        ids, total = search_pubmed(q['query'])
        new = set(ids) - all_pmids
        all_pmids.update(ids)
        query_stats.append({
            'id': q['id'],
            'label': q['label'],
            'query': q['query'],
            'total_in_pubmed': total,
            'retrieved': len(ids),
            'new_unique': len(new),
            'cumulative': len(all_pmids),
        })
        print(f"    Total: {total} | Retrieved: {len(ids)} | New unique: {len(new)} | Cumulative: {len(all_pmids)}")
        time.sleep(0.5)

    print(f"\n  Total unique PMIDs: {len(all_pmids)}")

    # --- Fetch ---
    print(f"\nFetching metadata for {len(all_pmids)} papers...\n")
    papers = fetch_papers(sorted(all_pmids))
    print(f"\n  Successfully fetched: {len(papers)}")

    # --- Remove reviews ---
    print(f"\nRemoving review papers...")
    non_reviews = []
    reviews = []

    for pmid, p in papers.items():
        study_type = classify_study_type(p)
        if study_type and study_type in REVIEW_TYPES:
            p['study_type'] = study_type
            p['exclusion_reason'] = f'Review paper: {study_type}'
            reviews.append(p)
        else:
            non_reviews.append(p)

    # Review breakdown
    review_breakdown = {}
    for p in reviews:
        st = p.get('study_type', 'Unknown')
        review_breakdown[st] = review_breakdown.get(st, 0) + 1

    print(f"\n  Reviews removed: {len(reviews)}")
    for st, c in sorted(review_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {c:>4}  {st}")
    print(f"  Non-review papers remaining: {len(non_reviews)}")

    # --- Save ---
    output = {
        'query_stats': query_stats,
        'total_unique_pmids': len(all_pmids),
        'total_fetched': len(papers),
        'reviews_removed': len(reviews),
        'papers_remaining': len(non_reviews),
        'review_breakdown': review_breakdown,
        'non_reviews': non_reviews,
        'reviews': reviews,
    }

    out_path = f'{BASE_DIR}/pubmed_step1_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"STEP 1 SUMMARY")
    print(f"{'='*70}")
    print(f"  Queries run:              {len(QUERIES)}")
    print(f"  Total unique PMIDs:       {len(all_pmids)}")
    print(f"  Successfully fetched:     {len(papers)}")
    print(f"  Reviews removed:          {len(reviews)}")
    print(f"  Papers for screening:     {len(non_reviews)}")
    print(f"\n  Query breakdown:")
    for qs in query_stats:
        print(f"    {qs['id']}: {qs['total_in_pubmed']:>5} total | {qs['new_unique']:>4} new | {qs['cumulative']:>5} cumul  [{qs['label']}]")

    return output


if __name__ == '__main__':
    main()
