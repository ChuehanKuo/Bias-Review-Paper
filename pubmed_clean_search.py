#!/usr/bin/env python3
"""
PubMed Clean Search Strategy for Systematic Review:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

Clean approach:
- 3 well-structured Boolean queries (no overlap, comprehensive)
- No retrieval cap — get ALL results
- Deduplicate by PMID at the end
- Screen by title+abstract together
"""

import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET

# ============================================================
# SEARCH STRATEGY: 3 focused queries
# ============================================================

QUERIES = [
    # Query 1: Core — algorithmic/AI bias + fairness in health
    # Papers that are specifically ABOUT bias or fairness in AI/ML for health
    {
        "id": "Q1",
        "label": "Core: Algorithmic bias/fairness + health AI",
        "query": (
            '("algorithmic bias" OR "algorithmic fairness" OR "AI bias" OR "AI fairness" '
            'OR "machine learning bias" OR "machine learning fairness" OR "bias mitigation" '
            'OR "debiasing" OR "fair machine learning" OR "fairness-aware" OR "bias detection" '
            'OR "disparate impact" OR "demographic parity" OR "equalized odds" OR "bias audit") '
            'AND '
            '("health" OR "healthcare" OR "clinical" OR "medical" OR "biomedical" OR "patient")'
        )
    },
    # Query 2: Health disparities/equity + AI/ML
    # Papers about disparities driven by AI/ML algorithms
    {
        "id": "Q2",
        "label": "Health disparities/equity driven by AI/ML",
        "query": (
            '("health disparities" OR "health equity" OR "health inequity" OR "healthcare disparities" '
            'OR "racial bias" OR "gender bias" OR "socioeconomic bias" OR "ethnic bias") '
            'AND '
            '("artificial intelligence" OR "machine learning" OR "deep learning" OR "algorithm" '
            'OR "predictive model" OR "clinical prediction" OR "decision support")'
        )
    },
    # Query 3: Fairness in specific clinical AI domains
    # Papers about fairness in medical imaging, NLP, EHR-based prediction, etc.
    {
        "id": "Q3",
        "label": "Fairness in clinical AI domains (imaging, EHR, NLP)",
        "query": (
            '("fairness" OR "equitable" OR "bias" OR "disparity") '
            'AND '
            '("clinical algorithm" OR "clinical AI" OR "medical AI" OR "health AI" '
            'OR "clinical decision support" OR "risk prediction" OR "medical imaging") '
            'AND '
            '("machine learning" OR "deep learning" OR "artificial intelligence" OR "neural network")'
        )
    }
]


def search_pubmed(query, retmax=10000):
    """Search PubMed and return ALL PMIDs (no cap)."""
    encoded = urllib.parse.quote(query)

    # First, get the count
    count_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax=0&retmode=json"
    try:
        with urllib.request.urlopen(urllib.request.Request(count_url), timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        total_count = int(data.get('esearchresult', {}).get('count', '0'))
    except Exception as e:
        print(f"  ERROR getting count: {e}")
        return [], 0

    print(f"  Total results in PubMed: {total_count}")

    # Fetch ALL PMIDs (in batches of 10000 if needed)
    all_ids = []
    for start in range(0, total_count, retmax):
        batch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={encoded}&retmax={retmax}&retstart={start}&retmode=json"
        )
        try:
            with urllib.request.urlopen(urllib.request.Request(batch_url), timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            ids = data.get('esearchresult', {}).get('idlist', [])
            all_ids.extend(ids)
            print(f"  Retrieved {len(all_ids)}/{total_count} PMIDs...")
        except Exception as e:
            print(f"  ERROR fetching batch at {start}: {e}")
        time.sleep(0.4)

    return all_ids, total_count


def fetch_papers(pmids, batch_size=200):
    """Fetch title + abstract + metadata for all PMIDs."""
    papers = {}
    total_batches = (len(pmids) - 1) // batch_size + 1

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        ids_str = ','.join(batch)
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={ids_str}&rettype=xml&retmode=xml"
        )

        try:
            req = urllib.request.Request(url)
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
                for abs_text in article.findall('.//AbstractText'):
                    label = abs_text.get('Label', '')
                    text = ''.join(abs_text.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = ' '.join(abstract_parts)

                # Journal
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ''

                # Year
                year = ''
                for ypath in ['.//PubDate/Year', './/ArticleDate/Year', './/PubDate/MedlineDate']:
                    ye = article.find(ypath)
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
                for author in article.findall('.//Author'):
                    last = author.find('LastName')
                    fore = author.find('ForeName')
                    if last is not None and last.text:
                        name = last.text
                        if fore is not None and fore.text:
                            name += f" {fore.text}"
                        authors.append(name)
                if len(authors) > 1:
                    author_str = authors[0] + ' et al.'
                elif authors:
                    author_str = authors[0]
                else:
                    author_str = 'N/A'

                # Keywords
                keywords = []
                for kw in article.findall('.//Keyword'):
                    if kw.text:
                        keywords.append(kw.text)

                # MeSH terms
                mesh_terms = []
                for mesh in article.findall('.//MeshHeading/DescriptorName'):
                    if mesh.text:
                        mesh_terms.append(mesh.text)

                # Publication type
                pub_types = []
                for pt in article.findall('.//PublicationType'):
                    if pt.text:
                        pub_types.append(pt.text)

                papers[pmid] = {
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'journal': journal,
                    'year': year,
                    'doi': doi,
                    'authors': author_str,
                    'keywords': '; '.join(keywords[:20]),
                    'mesh_terms': '; '.join(mesh_terms[:20]),
                    'pub_types': '; '.join(pub_types),
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
                }

            batch_num = i // batch_size + 1
            print(f"  Fetched batch {batch_num}/{total_batches} ({len(papers)} papers so far)")
        except Exception as e:
            print(f"  ERROR fetching batch {i//batch_size + 1}: {e}")

        time.sleep(0.4)

    return papers


def screen_paper(paper):
    """
    Screen paper by title+abstract together.

    INCLUDE: Papers specifically ABOUT bias/fairness in AI/ML applied to health.
             The paper's main focus should be on algorithmic bias — not just mentioning it.

    EXCLUDE:
    - Human cognitive/implicit biases (not about AI)
    - Statistical bias in epidemiology (without AI/ML)
    - Papers that only mention bias in passing (e.g., limitations section)
    - Not health-related
    """
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    combined = title + ' ' + abstract
    keywords = (paper.get('keywords', '') or '').lower()
    mesh = (paper.get('mesh_terms', '') or '').lower()
    all_text = combined + ' ' + keywords + ' ' + mesh

    # ---- MUST have AI/ML component ----
    ai_terms = [
        'machine learning', 'deep learning', 'artificial intelligence',
        'neural network', 'algorithm', 'predictive model', 'prediction model',
        'classifier', 'classification', 'natural language processing',
        'nlp', 'computer vision', 'automated', 'computational model',
        'random forest', 'logistic regression', 'xgboost', 'gradient boosting',
        'convolutional', 'transformer', 'large language model', 'llm',
        'decision support', 'risk score', 'risk prediction',
        'data-driven', 'supervised learning', 'unsupervised learning',
        'federated learning', 'reinforcement learning', 'foundation model',
        'generative ai', 'chatgpt', 'gpt-4'
    ]
    has_ai = any(term in all_text for term in ai_terms)

    # ---- MUST have health/medical component ----
    health_terms = [
        'health', 'clinical', 'medical', 'patient', 'hospital', 'disease',
        'diagnosis', 'treatment', 'care', 'physician', 'surgery', 'therapeutic',
        'pathology', 'radiology', 'dermatology', 'cardiology', 'oncology',
        'ophthalmology', 'psychiatry', 'mental health', 'ehr', 'electronic health',
        'emr', 'biomedical', 'pharmaceutical', 'mortality', 'readmission',
        'sepsis', 'icu', 'emergency', 'chest x-ray', 'mammograph', 'mri',
        'ecg', 'genomic', 'cancer', 'diabetes', 'cardiovascular',
        'epidemiol', 'public health', 'healthcare', 'medicine', 'nursing'
    ]
    has_health = any(term in all_text for term in health_terms)

    # ---- MUST have bias/fairness as a CENTRAL topic ----
    # Strong indicators: bias/fairness is in the TITLE or is a major topic in abstract
    strong_bias_title = any(term in title for term in [
        'bias', 'fairness', 'fair ', 'unfair', 'equitable', 'equity',
        'disparity', 'disparities', 'discrimination', 'debiasing', 'debias',
        'underdiagnos', 'underrepresent', 'inequit'
    ])

    # Abstract-level: needs substantial discussion of bias/fairness
    bias_keywords_in_abstract = [
        'algorithmic bias', 'algorithmic fairness', 'ai bias', 'ai fairness',
        'machine learning bias', 'machine learning fairness', 'model bias',
        'prediction bias', 'bias mitigation', 'bias detection', 'bias assessment',
        'fairness-aware', 'fair machine learning', 'debiasing', 'debias',
        'disparate impact', 'demographic parity', 'equalized odds',
        'equal opportunity', 'calibration across', 'subgroup fairness',
        'racial bias', 'gender bias', 'age bias', 'socioeconomic bias',
        'ethnic bias', 'bias in ai', 'bias in machine learning',
        'bias in algorithm', 'bias in prediction', 'bias in model',
        'fairness metric', 'fairness constraint', 'fairness evaluation',
        'health disparit', 'health equit', 'health inequit',
        'underdiagnosis bias', 'representation bias', 'sampling bias',
        'label bias', 'measurement bias', 'data bias',
        'bias in clinical', 'bias in health', 'biased algorithm',
        'biased model', 'biased prediction', 'unfair',
        'mitigating bias', 'addressing bias', 'reducing bias',
        'assessing bias', 'evaluating bias', 'detecting bias',
        'sources of bias', 'types of bias'
    ]
    bias_count_abstract = sum(1 for term in bias_keywords_in_abstract if term in abstract)

    # Paper must have bias/fairness as a substantial topic
    has_substantial_bias = strong_bias_title or bias_count_abstract >= 2

    if not has_ai:
        return False, 'EXCLUDE: No AI/ML/algorithm component'
    if not has_health:
        return False, 'EXCLUDE: Not health/medical related'
    if not has_substantial_bias:
        return False, 'EXCLUDE: Bias/fairness not a central topic (only mentioned in passing)'

    # ---- Exclude human cognitive biases (not about AI) ----
    human_bias_phrases = [
        'implicit bias training', 'implicit bias among clinician',
        'implicit bias among physician', 'implicit bias among provider',
        'clinician bias', 'physician bias', 'provider bias',
        'cognitive bias in decision making', 'unconscious bias training',
        'bias in clinical judgment', 'implicit association test',
        'weight bias', 'obesity bias among', 'anti-fat bias'
    ]
    ai_bias_phrases = [
        'algorithmic bias', 'ai bias', 'model bias', 'prediction bias',
        'bias mitigation', 'bias in ai', 'bias in machine learning',
        'fair machine learning', 'algorithmic fairness', 'model fairness',
        'debiasing algorithm', 'fairness-aware', 'bias detection',
        'disparate impact', 'demographic parity'
    ]
    is_human_bias = any(term in all_text for term in human_bias_phrases)
    has_ai_bias = any(term in all_text for term in ai_bias_phrases)

    if is_human_bias and not has_ai_bias:
        return False, 'EXCLUDE: About human/clinician cognitive biases, not AI algorithmic bias'

    # ---- Exclude pure statistical/epidemiological bias ----
    stat_bias_phrases = [
        'selection bias in cohort', 'recall bias in survey',
        'information bias in epidemiol', 'publication bias in meta',
        'attrition bias in trial', 'reporting bias in review',
        'selection bias in observational study'
    ]
    is_stat_only = any(term in all_text for term in stat_bias_phrases)
    if is_stat_only and not has_ai_bias:
        return False, 'EXCLUDE: Statistical/epidemiological bias without AI/ML context'

    return True, 'INCLUDE: Paper about bias/fairness in AI/ML applied to health'


def main():
    print("=" * 70)
    print("PubMed Clean Search — Systematic Review")
    print("Algorithmic Bias in Health AI")
    print("=" * 70)

    # Step 1: Search
    all_pmids = set()
    query_stats = []

    for q in QUERIES:
        print(f"\n{q['id']}: {q['label']}")
        print(f"  Query: {q['query'][:100]}...")
        ids, total = search_pubmed(q['query'], retmax=10000)
        new = set(ids) - all_pmids
        all_pmids.update(ids)
        query_stats.append({
            'id': q['id'],
            'label': q['label'],
            'query': q['query'],
            'total_in_pubmed': total,
            'retrieved': len(ids),
            'new_unique': len(new),
            'cumulative': len(all_pmids)
        })
        print(f"  Retrieved: {len(ids)} | New unique: {len(new)} | Cumulative: {len(all_pmids)}")
        time.sleep(0.5)

    print(f"\n{'='*70}")
    print(f"Total unique PMIDs (deduplicated): {len(all_pmids)}")
    print(f"{'='*70}")

    # Step 2: Fetch
    print(f"\nFetching titles + abstracts for {len(all_pmids)} papers...")
    pmid_list = sorted(all_pmids)
    papers = fetch_papers(pmid_list, batch_size=200)

    with_abs = sum(1 for p in papers.values() if p['abstract'])
    print(f"\nFetched: {len(papers)} papers ({with_abs} with abstracts)")

    # Step 3: Screen
    print(f"\nScreening {len(papers)} papers by title+abstract...")
    included = []
    excluded = []
    for pmid, paper in papers.items():
        inc, reason = screen_paper(paper)
        paper['include'] = inc
        paper['screen_reason'] = reason
        if inc:
            included.append(paper)
        else:
            excluded.append(paper)

    print(f"\n{'='*70}")
    print(f"INCLUDED: {len(included)}")
    print(f"EXCLUDED: {len(excluded)}")
    print(f"{'='*70}")

    # Exclusion breakdown
    reasons = {}
    for p in excluded:
        r = p['screen_reason']
        reasons[r] = reasons.get(r, 0) + 1
    print("\nExclusion breakdown:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {r}")

    # Year distribution of included
    years = {}
    for p in included:
        y = p.get('year', 'Unknown')
        years[y] = years.get(y, 0) + 1
    print("\nIncluded papers by year:")
    for y in sorted(years.keys(), reverse=True):
        print(f"  {y}: {years[y]}")

    # Save
    all_data = {
        'included': included,
        'excluded': excluded,
        'query_stats': query_stats,
        'total_unique_pmids': len(all_pmids),
        'total_fetched': len(papers)
    }
    with open('/home/user/Bias-Review-Paper/pubmed_clean_screening.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved to pubmed_clean_screening.json")


if __name__ == '__main__':
    main()
