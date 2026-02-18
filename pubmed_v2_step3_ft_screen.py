#!/usr/bin/env python3
"""
PubMed v2 Step 3: Full-Text Screening (Stage 2)

1. Look up PMCIDs for the 351 T/A-included papers
2. Fetch full text from PMC for those with PMCIDs
3. Apply full-text screening criteria
4. Split into: ft_included, ft_excluded, no_fulltext

Output: pubmed_v2_step3_results.json
"""

import json, time, re, urllib.request, urllib.parse, xml.etree.ElementTree as ET

BASE_DIR = '/home/user/Bias-Review-Paper'
INPUT_FILE = f'{BASE_DIR}/pubmed_v2_step2_results.json'
OUTPUT_FILE = f'{BASE_DIR}/pubmed_v2_step3_results.json'

ID_CONVERT_URL = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'
PMC_OA_URL = 'https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi'
EFETCH_PMC_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
BATCH_SIZE = 100

# ============================================================
# FULL-TEXT SCREENING CRITERIA
# ============================================================

AI_TERMS = [
    'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
    'nlp', 'natural language processing', 'random forest', 'logistic regression',
    'support vector', 'gradient boosting', 'xgboost', 'decision tree',
    'transformer', 'bert', 'gpt', 'large language model', 'llm',
    'convolutional', 'recurrent', 'supervised learning', 'unsupervised',
    'reinforcement learning', 'federated learning', 'algorithm',
    'predictive model', 'prediction model', 'classification',
    'computer vision', 'image recognition', 'clinical decision support',
]

HEALTH_TERMS = [
    'health', 'clinical', 'medical', 'patient', 'hospital', 'disease',
    'diagnosis', 'treatment', 'prognosis', 'mortality', 'imaging',
    'radiology', 'oncology', 'cardiology', 'dermatology', 'ehr',
    'electronic health record', 'biomedical', 'healthcare', 'care',
    'surgery', 'emergency', 'icu', 'sepsis', 'cancer', 'diabetes',
]

BIAS_TITLE_TERMS = [
    'bias', 'fairness', 'fair ', 'unfair', 'equity', 'inequity',
    'disparity', 'disparities', 'discrimination', 'equitable',
]

APPROACH_INDICATORS = [
    # Fairness metrics
    'demographic parity', 'equalized odds', 'equal opportunity',
    'disparate impact', 'predictive parity', 'calibration across',
    'fairness metric', 'fairness measure',
    # Bias detection / assessment
    'bias assessment', 'bias detection', 'bias evaluation', 'bias audit',
    'bias analysis', 'bias measurement', 'subgroup analysis',
    'stratified analysis', 'disaggregated', 'performance gap',
    'performance disparity', 'model audit', 'algorithmic audit',
    # Mitigation / debiasing
    'bias mitigation', 'debiasing', 'debias', 'reweighting', 'reweighing',
    'resampling', 'oversampling', 'undersampling', 'adversarial debiasing',
    'fairness constraint', 'fair representation', 'threshold adjustment',
    'calibration', 'data augmentation',
    # Fairness frameworks
    'aif360', 'fairlearn', 'aequitas', 'responsible ai',
    'trustworthy ai', 'ethical ai', 'fair machine learning',
    # Assessment verbs
    'evaluate fairness', 'assess bias', 'measure bias', 'quantify bias',
    'detect bias', 'identify bias', 'examine bias', 'analyze bias',
    'mitigate bias', 'reduce bias', 'address bias', 'correct bias',
    # Axes
    'racial bias', 'gender bias', 'age bias', 'socioeconomic',
    'health equity', 'health disparity', 'health disparities',
    'underrepresented', 'minority', 'vulnerable population',
]


def lookup_pmcids(pmids, retries=4):
    """Convert PMIDs to PMCIDs using NCBI ID converter."""
    mapping = {}
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i+BATCH_SIZE]
        params = urllib.parse.urlencode({
            'ids': ','.join(batch),
            'format': 'json',
            'tool': 'BiasReviewBot',
            'email': 'review@example.com',
        })
        url = f'{ID_CONVERT_URL}?{params}'

        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'BiasReviewBot/1.0'})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    for rec in data.get('records', []):
                        pmid = rec.get('pmid', '')
                        pmcid = rec.get('pmcid', '')
                        if pmid and pmcid:
                            mapping[pmid] = pmcid
                    break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"    ID convert error (attempt {attempt+1}): {e}, retrying in {wait}s...")
                time.sleep(wait)
        time.sleep(0.4)

    return mapping


def fetch_pmc_fulltext(pmcid, retries=3):
    """Fetch full text from PMC efetch."""
    params = urllib.parse.urlencode({
        'db': 'pmc',
        'id': pmcid,
        'rettype': 'xml',
    })
    url = f'{EFETCH_PMC_URL}?{params}'

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'BiasReviewBot/1.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                xml_bytes = resp.read()
                root = ET.fromstring(xml_bytes)
                texts = []
                for elem in root.iter():
                    if elem.text:
                        texts.append(elem.text)
                    if elem.tail:
                        texts.append(elem.tail)
                return ' '.join(texts)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    return None


def screen_fulltext(paper, fulltext):
    """Apply full-text screening criteria."""
    ft_lower = fulltext.lower()

    # Must have AI and health terms
    has_ai = any(t in ft_lower for t in AI_TERMS)
    has_health = any(t in ft_lower for t in HEALTH_TERMS)
    if not has_ai or not has_health:
        return False, 'No AI/ML + health in full text'

    # Count approach indicators
    approach_count = sum(1 for t in APPROACH_INDICATORS if t in ft_lower)

    # Check bias in title
    title_lower = (paper.get('title', '') or '').lower()
    has_bias_title = any(t in title_lower for t in BIAS_TITLE_TERMS)

    # Inclusion logic:
    # Option A: approach_count >= 2 AND bias in title
    # Option B: approach_count >= 3
    # Option C: bias in title AND approach_count >= 1
    if approach_count >= 3:
        return True, f'Included (approach_count={approach_count})'
    if has_bias_title and approach_count >= 2:
        return True, f'Included (bias_title + approach_count={approach_count})'
    if has_bias_title and approach_count >= 1:
        return True, f'Included (bias_title + approach_count={approach_count})'

    if approach_count < 1:
        return False, 'No approach indicators in full text'
    return False, f'Insufficient evidence (approach_count={approach_count}, bias_title={has_bias_title})'


def main():
    print("=" * 70)
    print("PubMed v2 Step 3: Full-Text Screening")
    print("=" * 70)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    papers = data['included']
    print(f"T/A included papers: {len(papers)}")

    # Phase 1: Lookup PMCIDs
    print(f"\nPhase 1: Looking up PMCIDs...")
    pmids = [p['pmid'] for p in papers]
    pmcid_map = lookup_pmcids(pmids)

    # Also use PMCIDs already in the data
    for p in papers:
        if p.get('pmcid') and p['pmid'] not in pmcid_map:
            pmcid_map[p['pmid']] = p['pmcid']

    has_pmcid = [p for p in papers if p['pmid'] in pmcid_map]
    no_pmcid = [p for p in papers if p['pmid'] not in pmcid_map]

    print(f"  PMCIDs found: {len(has_pmcid)}")
    print(f"  No PMCID:     {len(no_pmcid)}")

    # Phase 2: Fetch & screen full texts
    print(f"\nPhase 2: Fetching & screening full texts...")
    ft_included = []
    ft_excluded = []
    ft_failed = []
    exclusion_reasons = {}

    for i, p in enumerate(has_pmcid):
        pmcid = pmcid_map[p['pmid']]
        p['pmcid'] = pmcid

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(has_pmcid)}] Fetching {pmcid}...")

        fulltext = fetch_pmc_fulltext(pmcid)
        if fulltext is None:
            ft_failed.append(p)
            p['ft_status'] = 'Fetch failed'
            time.sleep(0.3)
            continue

        passed, reason = screen_fulltext(p, fulltext)
        if passed:
            p['ft_status'] = 'Included (2-stage)'
            ft_included.append(p)
        else:
            p['ft_status'] = f'Excluded: {reason}'
            ft_excluded.append(p)
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

        time.sleep(0.35)

    # Mark no-fulltext papers
    for p in no_pmcid:
        p['ft_status'] = 'No PMC full text'

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  T/A included:          {len(papers)}")
    print(f"  PMCIDs found:          {len(has_pmcid)}")
    print(f"  No PMC full text:      {len(no_pmcid)}")
    print(f"  Full texts screened:   {len(ft_included) + len(ft_excluded)}")
    print(f"  FT included (2-stage): {len(ft_included)}")
    print(f"  FT excluded:           {len(ft_excluded)}")
    print(f"  FT fetch failed:       {len(ft_failed)}")
    print(f"\nExclusion reasons:")
    for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
        print(f"    {count:>4}  {reason}")

    output = {
        # Pipeline metadata
        'source_file': data['source_file'],
        'source_sheet': data['source_sheet'],
        'total_pmids_in_sheet': data['total_pmids_in_sheet'],
        'total_fetched': data['total_fetched'],
        'reviews_removed': data['reviews_removed'],
        'ta_screened': data['papers_screened'],
        'ta_included_count': data['ta_included_count'],
        'ta_excluded_count': data['ta_excluded_count'],
        'ta_exclusion_reasons': data['ta_exclusion_reasons'],
        'screening_criteria': data['screening_criteria'],
        # Full-text screening
        'phase1_pmcid_lookup': {
            'pmcids_found': len(has_pmcid),
            'no_pmcid': len(no_pmcid),
        },
        'phase2_fulltext_screening': {
            'ft_screened': len(ft_included) + len(ft_excluded),
            'ft_included': len(ft_included),
            'ft_excluded': len(ft_excluded),
            'ft_fetch_failed': len(ft_failed),
            'exclusion_reasons': exclusion_reasons,
        },
        'ft_screening_criteria': {
            'approach_indicators': APPROACH_INDICATORS,
            'bias_title_terms': BIAS_TITLE_TERMS,
        },
        # Paper lists
        'ft_included': ft_included,
        'ft_excluded': ft_excluded,
        'no_fulltext': no_pmcid,
        'ft_failed': ft_failed,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
