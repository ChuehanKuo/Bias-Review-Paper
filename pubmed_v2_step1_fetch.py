#!/usr/bin/env python3
"""
PubMed v2 Step 1: Extract PMIDs from PubMed2 sheet and fetch full metadata from NCBI.

Input:  [0211] IEEE John_Screening 1 & 2.xlsx 的副本.xlsx → PubMed2 sheet (1899 PMIDs)
Output: pubmed_v2_fetched.json  (all papers with abstracts, MeSH, pub types, etc.)
"""

import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
import openpyxl

BASE_DIR = '/home/user/Bias-Review-Paper'
INPUT_FILE = f'{BASE_DIR}/[0211] IEEE John_Screening 1 & 2.xlsx 的副本.xlsx'
OUTPUT_FILE = f'{BASE_DIR}/pubmed_v2_fetched.json'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
BATCH_SIZE = 200

def extract_pmids():
    """Extract all PMIDs from PubMed2 sheet."""
    wb = openpyxl.load_workbook(INPUT_FILE, read_only=True)
    ws = wb['PubMed2']
    pmids = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            pmids.append(str(int(row[0])))
    wb.close()
    print(f"Extracted {len(pmids)} PMIDs from PubMed2 sheet")
    return pmids


def fetch_batch(pmids, retries=4):
    """Fetch metadata for a batch of PMIDs from NCBI efetch."""
    params = urllib.parse.urlencode({
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml',
    })
    url = f'{EFETCH_URL}?{params}'

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'BiasReviewBot/1.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode('utf-8')
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"    Fetch error (attempt {attempt+1}/{retries}): {e}, retrying in {wait}s...")
            time.sleep(wait)
    print(f"    FAILED to fetch batch after {retries} attempts")
    return None


def parse_xml(xml_str):
    """Parse efetch XML into structured paper dicts."""
    papers = []
    root = ET.fromstring(xml_str)

    for article in root.findall('.//PubmedArticle'):
        try:
            medline = article.find('MedlineCitation')
            pmid_el = medline.find('PMID')
            pmid = pmid_el.text if pmid_el is not None else ''

            art = medline.find('Article')

            # Title
            title_el = art.find('ArticleTitle')
            title = ''.join(title_el.itertext()) if title_el is not None else ''

            # Abstract
            abstract_el = art.find('Abstract')
            if abstract_el is not None:
                parts = []
                for at in abstract_el.findall('AbstractText'):
                    label = at.get('Label', '')
                    text = ''.join(at.itertext()) or ''
                    if label:
                        parts.append(f"{label}: {text}")
                    else:
                        parts.append(text)
                abstract = ' '.join(parts)
            else:
                abstract = ''

            # Journal
            journal_el = art.find('.//Title')
            journal = journal_el.text if journal_el is not None else ''

            # Year
            year = ''
            for ypath in ['.//PubDate/Year', './/PubDate/MedlineDate']:
                y_el = art.find(ypath)
                if y_el is not None and y_el.text:
                    year = y_el.text[:4]
                    break

            # Authors
            authors = []
            for author in art.findall('.//Author'):
                ln = author.find('LastName')
                fn = author.find('ForeName')
                if ln is not None:
                    name = ln.text
                    if fn is not None:
                        name += f" {fn.text}"
                    authors.append(name)

            # DOI
            doi = ''
            for eid in art.findall('.//ELocationID'):
                if eid.get('EIdType') == 'doi':
                    doi = eid.text or ''
                    break
            if not doi:
                for aid in article.findall('.//ArticleId'):
                    if aid.get('IdType') == 'doi':
                        doi = aid.text or ''
                        break

            # MeSH terms
            mesh_terms = []
            for mh in medline.findall('.//MeshHeading/DescriptorName'):
                if mh.text:
                    mesh_terms.append(mh.text)

            # Keywords
            keywords = []
            for kw in medline.findall('.//Keyword'):
                if kw.text:
                    keywords.append(kw.text)

            # Publication types
            pub_types = []
            for pt in art.findall('.//PublicationType'):
                if pt.text:
                    pub_types.append(pt.text)

            # PMCID
            pmcid = ''
            for aid in article.findall('.//ArticleId'):
                if aid.get('IdType') == 'pmc':
                    pmcid = aid.text or ''
                    break

            papers.append({
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': '; '.join(authors),
                'year': year,
                'journal': journal,
                'doi': doi,
                'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                'mesh_terms': '; '.join(mesh_terms),
                'keywords': '; '.join(keywords),
                'pub_types': '; '.join(pub_types),
                'pmcid': pmcid,
            })
        except Exception as e:
            print(f"    Parse error for article: {e}")

    return papers


def main():
    print("=" * 70)
    print("PubMed v2 Step 1: Extract PMIDs & Fetch Metadata")
    print("=" * 70)

    pmids = extract_pmids()
    unique_pmids = list(dict.fromkeys(pmids))  # preserve order, dedupe
    print(f"Unique PMIDs: {len(unique_pmids)}")

    all_papers = []
    total_batches = (len(unique_pmids) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(unique_pmids), BATCH_SIZE):
        batch = unique_pmids[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"\nBatch {batch_num}/{total_batches}: fetching {len(batch)} PMIDs...")

        xml_str = fetch_batch(batch)
        if xml_str:
            papers = parse_xml(xml_str)
            all_papers.extend(papers)
            print(f"  Parsed {len(papers)} papers (total: {len(all_papers)})")
        else:
            print(f"  SKIPPED batch {batch_num}")

        time.sleep(0.5)

    # Remove reviews
    non_reviews = []
    reviews_removed = 0
    for p in all_papers:
        pt_lower = p['pub_types'].lower()
        title_lower = p['title'].lower()
        is_review = any(t in pt_lower for t in ['review', 'systematic review', 'meta-analysis', 'scoping review'])
        is_review = is_review or any(t in title_lower for t in [
            'systematic review', 'scoping review', 'meta-analysis', 'narrative review',
            'literature review', 'integrative review', 'rapid review', 'umbrella review',
            'a review of', 'review of the', 'state of the art review'
        ])
        if is_review:
            reviews_removed += 1
        else:
            non_reviews.append(p)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  PMIDs from sheet:   {len(unique_pmids)}")
    print(f"  Successfully fetched: {len(all_papers)}")
    print(f"  Reviews removed:    {reviews_removed}")
    print(f"  Papers for screening: {len(non_reviews)}")

    # Save
    output = {
        'total_pmids': len(unique_pmids),
        'total_fetched': len(all_papers),
        'reviews_removed': reviews_removed,
        'papers': non_reviews,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
