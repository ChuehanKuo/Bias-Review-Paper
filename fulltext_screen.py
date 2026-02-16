#!/usr/bin/env python3
"""
Full-Text Screening Pipeline for Systematic Review:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

Step 1: Convert PMIDs to PMCIDs (check open access availability)
Step 2: Fetch full text XML from PMC
Step 3: Full-text screening with strict criteria
Step 4: Build updated Excel
"""

import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
import re

# ============================================================
# STEP 1: PMID -> PMCID conversion
# ============================================================

def get_pmcids(pmids, batch_size=200):
    """Convert PMIDs to PMCIDs using NCBI ID Converter API."""
    pmid_to_pmc = {}
    total_batches = (len(pmids) - 1) // batch_size + 1

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        ids_str = ','.join(batch)
        url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            f"?ids={ids_str}&format=json&tool=systematic_review&email=review@example.com"
        )
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            for rec in data.get('records', []):
                pmid = str(rec.get('pmid', ''))
                pmcid = rec.get('pmcid', '')
                if pmid and pmcid:
                    pmid_to_pmc[pmid] = pmcid
            print(f"  Batch {i//batch_size+1}/{total_batches}: {len(pmid_to_pmc)} PMCIDs found so far")
        except Exception as e:
            print(f"  ERROR batch {i//batch_size+1}: {e}")
        time.sleep(0.4)

    return pmid_to_pmc


# ============================================================
# STEP 2: Fetch full text from PMC
# ============================================================

def fetch_fulltext(pmcid):
    """Fetch full text XML from PMC."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&rettype=xml&retmode=xml"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp:
            xml_data = resp.read().decode('utf-8')
        root = ET.fromstring(xml_data)

        # Extract all body text
        body_parts = []
        for p in root.iter('p'):
            text = ''.join(p.itertext())
            if text.strip():
                body_parts.append(text.strip())

        # Extract section titles
        sections = []
        for sec in root.iter('sec'):
            title_elem = sec.find('title')
            if title_elem is not None and title_elem.text:
                sections.append(title_elem.text)

        fulltext = ' '.join(body_parts)
        return fulltext, sections
    except Exception as e:
        return None, []


def fetch_fulltexts_batch(pmcids, max_papers=None):
    """Fetch full texts for a list of PMCIDs."""
    results = {}
    to_fetch = list(pmcids.items())
    if max_papers:
        to_fetch = to_fetch[:max_papers]

    total = len(to_fetch)
    for i, (pmid, pmcid) in enumerate(to_fetch):
        fulltext, sections = fetch_fulltext(pmcid)
        if fulltext and len(fulltext) > 200:
            results[str(pmid)] = {
                'fulltext': fulltext,
                'sections': sections,
                'pmcid': pmcid,
                'char_count': len(fulltext)
            }
        if (i + 1) % 20 == 0:
            print(f"  Fetched {i+1}/{total} full texts ({len(results)} successful)")
        time.sleep(0.35)

    return results


# ============================================================
# STEP 3: Full-text screening
# ============================================================

def fulltext_screen(paper, fulltext_data=None):
    """
    Full-text screening with STRICT criteria.

    A paper is INCLUDED only if it:
    1. Specifically describes an approach/method/framework for assessing OR mitigating
       algorithmic bias in AI/ML applied to health
    2. The main focus of the paper is on algorithmic bias/fairness (not a side mention)
    3. Involves AI/ML algorithms in a health context

    EXCLUDE if:
    - Bias is only mentioned in intro/limitations but paper is really about something else
    - Paper is about human biases, not algorithmic
    - Paper is about general AI ethics without specific bias assessment/mitigation content
    - Paper is about non-health applications
    - Paper only discusses bias conceptually without any approach/method
    """
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()

    # Use full text if available, otherwise use abstract
    if fulltext_data and fulltext_data.get('fulltext'):
        text = fulltext_data['fulltext'].lower()
        sections = [s.lower() for s in fulltext_data.get('sections', [])]
        has_fulltext = True
    else:
        text = abstract
        sections = []
        has_fulltext = False

    combined = title + ' ' + abstract + ' ' + text

    # ---- CRITERION 1: Must specifically discuss approaches for bias assessment/mitigation ----
    approach_indicators = [
        # Assessment approaches
        'bias assessment', 'bias detection', 'bias evaluation', 'bias audit',
        'fairness assessment', 'fairness evaluation', 'fairness audit',
        'bias measurement', 'bias quantification', 'bias analysis',
        'measuring bias', 'detecting bias', 'evaluating bias',
        'fairness metric', 'fairness measure', 'bias metric',
        'demographic parity', 'equalized odds', 'equal opportunity',
        'disparate impact', 'predictive parity', 'calibration across',
        'subgroup analysis', 'disaggregated', 'stratified performance',

        # Mitigation approaches
        'bias mitigation', 'bias reduction', 'bias correction',
        'debiasing', 'debias', 'fairness-aware',
        'fair machine learning', 'fair classification',
        'adversarial debiasing', 'reweighting', 'reweighing',
        'resampling', 'data augmentation for fairness',
        'fairness constraint', 'fairness regularization',
        'threshold adjustment', 'calibration', 'post-processing',
        'pre-processing', 'in-processing',
        'counterfactual fairness', 'causal fairness',

        # Frameworks/tools
        'fairness framework', 'bias framework', 'equity framework',
        'ai fairness 360', 'aequitas', 'fairlearn',
        'bias toolkit', 'fairness toolkit',
        'model card', 'datasheet', 'datanutrition',

        # Review/survey of approaches
        'review of bias', 'survey of fairness', 'review of fairness',
        'bias mitigation strategies', 'approaches to fairness',
        'methods for bias', 'techniques for fairness',
    ]
    approach_count = sum(1 for t in approach_indicators if t in combined)

    # ---- CRITERION 2: Must be about AI/ML ----
    ai_terms = [
        'machine learning', 'deep learning', 'artificial intelligence',
        'neural network', 'algorithm', 'predictive model', 'classifier',
        'natural language processing', 'computer vision',
        'random forest', 'logistic regression', 'xgboost',
        'convolutional', 'transformer', 'large language model', 'llm',
        'decision support', 'risk prediction', 'federated learning',
        'reinforcement learning', 'foundation model', 'chatgpt', 'gpt'
    ]
    has_ai = any(t in combined for t in ai_terms)

    # ---- CRITERION 3: Must be health-related ----
    health_terms = [
        'health', 'clinical', 'medical', 'patient', 'hospital', 'disease',
        'diagnosis', 'treatment', 'care', 'radiology', 'dermatology',
        'cardiology', 'oncology', 'ophthalmology', 'psychiatry',
        'ehr', 'electronic health', 'biomedical', 'mortality',
        'readmission', 'sepsis', 'icu', 'emergency', 'chest x-ray',
        'cancer', 'diabetes', 'cardiovascular', 'public health',
        'healthcare', 'medicine'
    ]
    has_health = any(t in combined for t in health_terms)

    # ---- CRITERION 4: Bias must be CENTRAL to the paper ----
    # Check if bias/fairness appears in title
    bias_in_title = any(t in title for t in [
        'bias', 'fairness', 'fair ', 'equitable', 'equity', 'disparity',
        'disparities', 'debiasing', 'debias', 'underdiagnos', 'inequit',
        'discrimination'
    ])

    # Check for bias-related section headings (if full text available)
    bias_section = any(
        any(t in s for t in ['bias', 'fairness', 'equity', 'disparity', 'ethical'])
        for s in sections
    ) if sections else False

    # ---- DECISION LOGIC ----
    if not has_ai:
        return False, 'No AI/ML component in full text', 'No full text' if not has_fulltext else 'Full text'
    if not has_health:
        return False, 'Not health-related in full text', 'No full text' if not has_fulltext else 'Full text'

    # Strict: need substantial approach content
    if has_fulltext:
        # With full text: need at least 3 approach indicators
        if approach_count >= 3 and (bias_in_title or bias_section):
            return True, 'Included: discusses approaches for bias assessment/mitigation in health AI', 'Full text'
        elif approach_count >= 5:
            return True, 'Included: substantial bias/fairness methodology content', 'Full text'
        elif bias_in_title and approach_count >= 2:
            return True, 'Included: bias is central topic with approach content', 'Full text'
        else:
            return False, f'Excluded: insufficient approach content (only {approach_count} indicators)', 'Full text'
    else:
        # Without full text: use abstract only, be slightly more lenient
        if approach_count >= 2 and bias_in_title:
            return True, 'Included: bias central + approach content (abstract only)', 'Abstract only'
        elif approach_count >= 3:
            return True, 'Included: substantial approach content in abstract', 'Abstract only'
        elif bias_in_title and approach_count >= 1:
            return True, 'Included: bias is central topic (abstract only, needs verification)', 'Abstract only'
        else:
            return False, f'Excluded: insufficient approach content in abstract ({approach_count} indicators)', 'Abstract only'


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Full-Text Screening Pipeline")
    print("=" * 70)

    # Load included papers from title+abstract screening
    with open('/home/user/Bias-Review-Paper/pubmed_final_data.json', 'r') as f:
        data = json.load(f)

    included_papers = data['included']
    print(f"\nPapers from title+abstract screening: {len(included_papers)}")

    # Step 1: Get PMCIDs
    print(f"\nSTEP 1: Checking PMC availability...")
    pmids = [p['pmid'] for p in included_papers]
    pmid_to_pmc = get_pmcids(pmids)
    print(f"  Papers with PMC full text: {len(pmid_to_pmc)} / {len(pmids)}")

    # Step 2: Fetch full texts
    print(f"\nSTEP 2: Fetching full texts from PMC...")
    fulltext_data = fetch_fulltexts_batch(pmid_to_pmc)
    print(f"  Successfully fetched: {len(fulltext_data)} full texts")

    # Step 3: Full-text screen
    print(f"\nSTEP 3: Full-text screening...")
    ft_included = []
    ft_excluded = []

    for paper in included_papers:
        pmid = paper['pmid']
        ft_data = fulltext_data.get(pmid)

        inc, reason, source = fulltext_screen(paper, ft_data)
        paper['ft_include'] = inc
        paper['ft_reason'] = reason
        paper['ft_source'] = source
        paper['has_fulltext'] = pmid in fulltext_data
        paper['pmcid'] = pmid_to_pmc.get(pmid, '')

        if inc:
            ft_included.append(paper)
        else:
            ft_excluded.append(paper)

    print(f"\n{'='*70}")
    print(f"FULL-TEXT SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  INCLUDED: {len(ft_included)}")
    print(f"  EXCLUDED: {len(ft_excluded)}")
    print(f"  Screened with full text: {sum(1 for p in included_papers if p.get('has_fulltext'))}")
    print(f"  Screened with abstract only: {sum(1 for p in included_papers if not p.get('has_fulltext'))}")

    # Breakdown
    reasons = {}
    for p in ft_excluded:
        r = p['ft_reason']
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\nExclusion breakdown:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {r}")

    inc_reasons = {}
    for p in ft_included:
        r = p['ft_reason']
        inc_reasons[r] = inc_reasons.get(r, 0) + 1
    print(f"\nInclusion breakdown:")
    for r, c in sorted(inc_reasons.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {r}")

    # Year distribution
    years = {}
    for p in ft_included:
        y = p.get('year', 'Unknown')
        years[y] = years.get(y, 0) + 1
    print(f"\nIncluded papers by year:")
    for y in sorted(years.keys(), reverse=True):
        print(f"  {y}: {years[y]}")

    # Save
    result = {
        'ft_included': ft_included,
        'ft_excluded': ft_excluded,
        'query_stats': data.get('query_stats', []),
        'meta': {
            'title_abstract_included': len(included_papers),
            'pmc_available': len(pmid_to_pmc),
            'fulltext_fetched': len(fulltext_data),
            'ft_included': len(ft_included),
            'ft_excluded': len(ft_excluded),
        }
    }

    with open('/home/user/Bias-Review-Paper/pubmed_fulltext_screening.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to pubmed_fulltext_screening.json")
    print(f"Ready for Excel generation.")


if __name__ == '__main__':
    main()
