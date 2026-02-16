#!/usr/bin/env python3
"""
Exhaustive PubMed search for systematic review on
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"
Runs all 50 queries and deduplicates results.
"""

import urllib.request
import urllib.parse
import json
import time
import sys
import os

QUERIES = [
    '"algorithmic bias" healthcare',
    '"AI fairness" clinical medicine',
    '"machine learning bias" clinical',
    '"machine learning fairness" health',
    '"bias mitigation" "artificial intelligence" health',
    '"health disparities" "machine learning" bias',
    '"clinical prediction model" bias fairness algorithm',
    '"equitable AI" health',
    '"racial bias" "machine learning" clinical',
    '"gender bias" AI clinical healthcare',
    '"algorithmic discrimination" healthcare',
    '"fair machine learning" health clinical',
    '"bias detection" "artificial intelligence" medical',
    '"disparate impact" clinical algorithm',
    '"underrepresentation" "training data" health AI',
    '"skin color bias" OR "skin tone bias" AI dermatology',
    '"bias" "deep learning" "medical imaging"',
    '"fairness" "natural language processing" "clinical notes"',
    '"EHR bias" machine learning',
    '"socioeconomic bias" "prediction model" health',
    '"age bias" "machine learning" clinical',
    '"language bias" AI healthcare',
    '"disability bias" artificial intelligence health',
    '"insurance bias" "algorithm" health',
    '"geographic bias" "machine learning" health',
    '"bias audit" clinical algorithm AI',
    '"model fairness" "electronic health record"',
    '"debiasing" healthcare AI machine learning',
    '"calibration bias" clinical prediction model',
    '"subgroup performance" AI clinical fairness',
    '"health equity" "artificial intelligence" bias algorithm',
    '"risk prediction" bias "machine learning" race',
    '"sepsis prediction" bias fairness',
    '"mortality prediction" bias AI fairness',
    '"readmission prediction" bias algorithm',
    '"chest X-ray" AI bias fairness',
    '"dermatology" AI bias "skin color"',
    '"radiology" AI bias fairness',
    '"pathology" AI bias fairness',
    '"ophthalmology" AI bias fairness',
    '"mental health" AI bias algorithm',
    '"cardiology" AI bias algorithm prediction',
    '"oncology" AI bias algorithm prediction',
    '"emergency department" AI bias algorithm',
    '"FDA" AI bias algorithm regulation',
    '"clinical decision support" bias fairness machine learning',
    '"precision medicine" bias fairness AI',
    '"wearable" AI bias health',
    '"federated learning" fairness health',
    '"transfer learning" bias clinical',
]

BASE_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

all_pmids = set()
query_results = {}

def fetch_url(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PubMedSearch/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  ERROR fetching URL after {retries} attempts: {e}", file=sys.stderr)
                return None

print("=" * 80)
print("EXHAUSTIVE PUBMED SEARCH - 50 QUERIES")
print("=" * 80)

for i, query in enumerate(QUERIES, 1):
    print(f"\n--- Query {i}/50: {query} ---")

    params = urllib.parse.urlencode({
        'db': 'pubmed',
        'term': query,
        'retmax': 250,
        'retmode': 'json',
        'sort': 'relevance'
    })

    search_url = f"{BASE_SEARCH}?{params}"
    result = fetch_url(search_url)

    if not result:
        print(f"  FAILED to search")
        time.sleep(1)
        continue

    try:
        data = json.loads(result)
        ids = data.get('esearchresult', {}).get('idlist', [])
        total = data.get('esearchresult', {}).get('count', '0')
        new_ids = [pid for pid in ids if pid not in all_pmids]

        print(f"  Total in PubMed: {total} | Fetched: {len(ids)} | New unique: {len(new_ids)}")

        all_pmids.update(ids)
        query_results[i] = {
            'query': query,
            'total': total,
            'fetched': len(ids),
            'new': len(new_ids),
            'ids': ids
        }
    except json.JSONDecodeError:
        print(f"  FAILED to parse search results")

    # Respect NCBI rate limit (3 requests/second without API key)
    time.sleep(0.4)

print(f"\n\n{'='*80}")
print(f"TOTAL UNIQUE PMIDs COLLECTED: {len(all_pmids)}")
print(f"{'='*80}")

# Now fetch details for all PMIDs in batches of 200
print(f"\nFetching article details for {len(all_pmids)} unique articles...")

all_articles = {}
pmid_list = list(all_pmids)

for batch_start in range(0, len(pmid_list), 200):
    batch = pmid_list[batch_start:batch_start+200]
    batch_num = batch_start // 200 + 1
    total_batches = (len(pmid_list) + 199) // 200
    print(f"  Fetching batch {batch_num}/{total_batches} ({len(batch)} articles)...")

    params = urllib.parse.urlencode({
        'db': 'pubmed',
        'id': ','.join(batch),
        'retmode': 'json'
    })

    summary_url = f"{BASE_SUMMARY}?{params}"
    result = fetch_url(summary_url)

    if not result:
        print(f"  FAILED to fetch batch {batch_num}")
        time.sleep(1)
        continue

    try:
        data = json.loads(result)
        result_data = data.get('result', {})
        for pmid in batch:
            if pmid in result_data:
                article = result_data[pmid]
                title = article.get('title', 'N/A')
                # Get journal
                source = article.get('source', 'N/A')
                fulljournalname = article.get('fulljournalname', source)
                # Get year
                pubdate = article.get('pubdate', 'N/A')
                year = pubdate.split()[0] if pubdate else 'N/A'
                # Get DOI from articleids
                doi = ''
                for aid in article.get('articleids', []):
                    if aid.get('idtype') == 'doi':
                        doi = aid.get('value', '')
                        break

                all_articles[pmid] = {
                    'pmid': pmid,
                    'title': title,
                    'journal': fulljournalname,
                    'year': year,
                    'doi': doi,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ERROR parsing batch {batch_num}: {e}")

    time.sleep(0.4)

# Sort by year (newest first), then by title
sorted_articles = sorted(all_articles.values(), key=lambda x: (x.get('year', '0'), x.get('title', '')), reverse=True)

# Write results to file
output_path = "/home/user/Bias-Review-Paper/pubmed_search_results.txt"
with open(output_path, 'w') as f:
    f.write("EXHAUSTIVE PUBMED SEARCH RESULTS\n")
    f.write("Systematic Review: Approaches for Assessing and Mitigating Algorithmic Bias in Health AI\n")
    f.write(f"Date: 2026-02-16\n")
    f.write(f"Total queries: 50\n")
    f.write(f"Total unique articles: {len(sorted_articles)}\n")
    f.write("=" * 120 + "\n\n")

    # Query summary
    f.write("QUERY SUMMARY\n")
    f.write("-" * 80 + "\n")
    for qnum in sorted(query_results.keys()):
        qr = query_results[qnum]
        f.write(f"Q{qnum:02d}: {qr['query']}\n")
        f.write(f"     Total: {qr['total']} | Fetched: {qr['fetched']} | New unique: {qr['new']}\n")
    f.write("\n" + "=" * 120 + "\n\n")

    # All articles
    f.write("ALL ARTICLES (sorted by year, newest first)\n")
    f.write("-" * 120 + "\n\n")
    for idx, article in enumerate(sorted_articles, 1):
        f.write(f"{idx}. {article['title']}\n")
        f.write(f"   PMID: {article['pmid']} | Year: {article['year']} | Journal: {article['journal']}\n")
        f.write(f"   URL: {article['url']}\n")
        if article['doi']:
            f.write(f"   DOI: https://doi.org/{article['doi']}\n")
        f.write("\n")

print(f"\nResults written to: {output_path}")
print(f"Total unique articles found: {len(sorted_articles)}")

# Also write a CSV for easy import
csv_path = "/home/user/Bias-Review-Paper/pubmed_search_results.csv"
with open(csv_path, 'w') as f:
    f.write("PMID,Title,Year,Journal,URL,DOI\n")
    for article in sorted_articles:
        title = article['title'].replace('"', '""')
        journal = article['journal'].replace('"', '""')
        doi_url = f"https://doi.org/{article['doi']}" if article['doi'] else ''
        f.write(f'"{article["pmid"]}","{title}","{article["year"]}","{journal}","{article["url"]}","{doi_url}"\n')

print(f"CSV written to: {csv_path}")
