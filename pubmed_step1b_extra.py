#!/usr/bin/env python3
"""
PubMed Step 1b: Fetch extra PMIDs from additional queries and merge into Step 1 results.
"""

import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET

BASE_DIR = '/home/user/Bias-Review-Paper'

EXTRA_QUERIES = [
    {"id": "E1", "label": "MeSH: AI + Healthcare Disparities",
     "query": '"Artificial Intelligence"[MeSH] AND "Healthcare Disparities"[MeSH]'},
    {"id": "E2", "label": "MeSH: ML + Bias",
     "query": '"Machine Learning"[MeSH] AND "Bias"[MeSH]'},
    {"id": "E3", "label": "MeSH: AI + Prejudice/Discrimination",
     "query": '"Artificial Intelligence"[MeSH] AND ("Prejudice"[MeSH] OR "Social Discrimination"[MeSH])'},
    {"id": "E4", "label": "Underdiagnosis + algorithm/AI",
     "query": '("underdiagnosis" OR "underdiagnosed") AND ("algorithm" OR "machine learning" OR "artificial intelligence") AND ("bias" OR "disparity")'},
    {"id": "E5", "label": "Risk score + race/disparity",
     "query": '("risk score" OR "risk prediction" OR "prediction model") AND ("race" OR "racial" OR "disparity" OR "disparities") AND ("bias" OR "fairness" OR "equity")'},
    {"id": "E6", "label": "Specific tools: AIF360, Fairlearn, etc.",
     "query": '("AI Fairness 360" OR "AIF360" OR "Fairlearn" OR "aequitas" OR "What-If Tool") AND ("health" OR "clinical" OR "medical")'},
    {"id": "E7", "label": "Social determinants + ML + bias",
     "query": '"social determinants" AND ("machine learning" OR "artificial intelligence" OR "algorithm") AND ("bias" OR "fairness" OR "equity")'},
    {"id": "E8", "label": "EHR/clinical data + algorithmic bias",
     "query": '("electronic health record" OR "EHR" OR "clinical data") AND ("algorithmic bias" OR "model bias" OR "prediction bias" OR "fairness")'},
    {"id": "E9", "label": "Skin tone / dermatology AI bias",
     "query": '("skin tone" OR "skin color" OR "Fitzpatrick") AND ("algorithm" OR "deep learning" OR "AI" OR "machine learning") AND ("bias" OR "fairness" OR "performance")'},
    {"id": "E10", "label": "Pulse oximetry / wearable bias",
     "query": '("pulse oximetry" OR "oximeter" OR "SpO2" OR "wearable") AND ("bias" OR "accuracy" OR "disparity") AND ("race" OR "skin" OR "pigment")'},
]


def search_pubmed(query, retmax=10000):
    encoded = urllib.parse.quote(query)
    count_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax=0&retmode=json"
    try:
        with urllib.request.urlopen(urllib.request.Request(count_url), timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        total = int(data.get('esearchresult', {}).get('count', '0'))
    except Exception as e:
        print(f"    ERROR: {e}")
        return [], 0

    all_ids = []
    for start in range(0, min(total, retmax), 500):
        url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
               f"db=pubmed&term={encoded}&retmax=500&retstart={start}&retmode=json")
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            all_ids.extend(data.get('esearchresult', {}).get('idlist', []))
        except Exception as e:
            print(f"    ERROR at offset {start}: {e}")
        time.sleep(0.35)
    return all_ids, total


def fetch_papers(pmids, batch_size=200):
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
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    'source_db': 'pubmed',
                    'source_db_label': 'PubMed/MEDLINE',
                }
            print(f"    Batch {i//batch_size+1}/{total_batches} done ({len(papers)} total)")
        except Exception as e:
            print(f"    ERROR batch {i//batch_size+1}: {e}")
        time.sleep(0.4)
    return papers


REVIEW_TYPES = {'Systematic Review', 'Scoping Review', 'Narrative Review', 'Meta-Analysis', 'Review'}

def classify_study_type(paper):
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
    if 'review' in pub_types:
        return 'Review'
    return None


def main():
    print("=" * 70)
    print("STEP 1b: Fetch extra PMIDs and merge")
    print("=" * 70)

    # Load existing
    with open(f'{BASE_DIR}/pubmed_step1_results.json') as f:
        data = json.load(f)

    existing_pmids = set()
    for p in data['non_reviews'] + data['reviews']:
        existing_pmids.add(p['pmid'])
    print(f"\nExisting PMIDs: {len(existing_pmids)}")

    # Search extra queries
    new_pmids = set()
    extra_stats = []
    for q in EXTRA_QUERIES:
        print(f"\n  {q['id']}: {q['label']}")
        ids, total = search_pubmed(q['query'])
        new = set(ids) - existing_pmids - new_pmids
        new_pmids.update(new)
        extra_stats.append({
            'id': q['id'], 'label': q['label'], 'query': q['query'],
            'total_in_pubmed': total, 'retrieved': len(ids),
            'new_unique': len(new), 'cumulative_new': len(new_pmids),
        })
        print(f"    Total: {total} | Retrieved: {len(ids)} | New: {len(new)} | Cumulative new: {len(new_pmids)}")
        time.sleep(0.5)

    print(f"\n  New unique PMIDs to fetch: {len(new_pmids)}")

    if not new_pmids:
        print("  No new PMIDs found. Done.")
        return

    # Fetch new papers
    print(f"\nFetching metadata for {len(new_pmids)} new papers...")
    new_papers = fetch_papers(sorted(new_pmids))
    print(f"  Fetched: {len(new_papers)}")

    # Remove reviews from new papers
    new_non_reviews = []
    new_reviews = []
    for pmid, p in new_papers.items():
        st = classify_study_type(p)
        if st and st in REVIEW_TYPES:
            p['study_type'] = st
            p['exclusion_reason'] = f'Review paper: {st}'
            new_reviews.append(p)
        else:
            new_non_reviews.append(p)

    print(f"\n  New reviews removed: {len(new_reviews)}")
    print(f"  New non-reviews: {len(new_non_reviews)}")

    # Merge into existing data
    data['non_reviews'].extend(new_non_reviews)
    data['reviews'].extend(new_reviews)
    data['query_stats'].extend(extra_stats)
    data['total_unique_pmids'] = len(existing_pmids) + len(new_pmids)
    data['total_fetched'] = data['total_fetched'] + len(new_papers)
    data['reviews_removed'] = len(data['reviews'])
    data['papers_remaining'] = len(data['non_reviews'])

    # Update review breakdown
    rb = {}
    for p in data['reviews']:
        st = p.get('study_type', 'Unknown')
        rb[st] = rb.get(st, 0) + 1
    data['review_breakdown'] = rb

    # Save
    out_path = f'{BASE_DIR}/pubmed_step1_results.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"UPDATED STEP 1 SUMMARY")
    print(f"{'='*70}")
    print(f"  Total queries:            {len(data['query_stats'])} (7 original + 10 extra)")
    print(f"  Total unique PMIDs:       {data['total_unique_pmids']}")
    print(f"  Total fetched:            {data['total_fetched']}")
    print(f"  Total reviews removed:    {data['reviews_removed']}")
    for st, c in sorted(rb.items(), key=lambda x: -x[1]):
        print(f"    {c:>5}  {st}")
    print(f"  Papers for screening:     {data['papers_remaining']}")
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
