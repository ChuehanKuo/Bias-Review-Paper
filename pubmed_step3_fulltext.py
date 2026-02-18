#!/usr/bin/env python3
"""
PubMed Step 3: Full-Text Screening (Stage 2)

Takes 1,146 T/A-included papers and:
1. Looks up PMCIDs via NCBI ID converter (PMID -> PMCID)
2. Splits into has-full-text vs no-full-text
3. Fetches actual full text from PMC for those with PMCIDs
4. Applies full-text screening criteria (same as shared_screening.py)
5. Outputs:
   - ft_included: passed both Stage 1 + Stage 2
   - ft_excluded: passed Stage 1 but failed Stage 2
   - no_fulltext: passed Stage 1, no full text available (Stage 1 only)
"""

import json
import time
import urllib.request
import xml.etree.ElementTree as ET

BASE_DIR = '/home/user/Bias-Review-Paper'

# ============================================================
# FULL-TEXT SCREENING CRITERIA (same as shared_screening.py)
# ============================================================

APPROACH_INDICATORS = [
    'bias assessment', 'bias detection', 'bias evaluation', 'bias audit',
    'fairness assessment', 'fairness evaluation', 'fairness audit',
    'bias measurement', 'bias quantification', 'bias analysis',
    'measuring bias', 'detecting bias', 'evaluating bias',
    'fairness metric', 'fairness measure', 'bias metric',
    'demographic parity', 'equalized odds', 'equal opportunity',
    'disparate impact', 'predictive parity', 'calibration across',
    'subgroup analysis', 'disaggregated', 'stratified performance',
    'bias mitigation', 'bias reduction', 'bias correction',
    'debiasing', 'debias', 'fairness-aware',
    'fair machine learning', 'fair classification',
    'adversarial debiasing', 'reweighting', 'reweighing',
    'resampling', 'data augmentation for fairness',
    'fairness constraint', 'fairness regularization',
    'threshold adjustment', 'calibration', 'post-processing',
    'pre-processing', 'in-processing',
    'counterfactual fairness', 'causal fairness',
    'fairness framework', 'bias framework', 'equity framework',
    'ai fairness 360', 'aequitas', 'fairlearn',
    'bias toolkit', 'fairness toolkit',
    'model card', 'datasheet',
    'review of bias', 'survey of fairness', 'review of fairness',
    'bias mitigation strategies', 'approaches to fairness',
    'methods for bias', 'techniques for fairness',
]

AI_TERMS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'neural network', 'algorithm', 'predictive model', 'classifier',
    'natural language processing', 'computer vision',
    'random forest', 'logistic regression', 'xgboost',
    'convolutional', 'transformer', 'large language model', 'llm',
    'decision support', 'risk prediction', 'federated learning',
    'reinforcement learning', 'foundation model', 'chatgpt', 'gpt',
]

HEALTH_TERMS = [
    'health', 'clinical', 'medical', 'patient', 'hospital', 'disease',
    'diagnosis', 'treatment', 'care', 'radiology', 'dermatology',
    'cardiology', 'oncology', 'ophthalmology', 'psychiatry',
    'ehr', 'electronic health', 'biomedical', 'mortality',
    'readmission', 'sepsis', 'icu', 'emergency', 'chest x-ray',
    'cancer', 'diabetes', 'cardiovascular', 'public health',
    'healthcare', 'medicine',
]

BIAS_TITLE_TERMS = [
    'bias', 'fairness', 'fair ', 'equitable', 'equity', 'disparity',
    'disparities', 'debiasing', 'debias', 'underdiagnos', 'inequit',
    'discrimination',
]


def fulltext_screen(paper, full_text):
    """
    Screen using actual full-text content.
    Returns (included: bool, reason: str).
    """
    title = (paper.get('title', '') or '').lower()
    text = full_text.lower()
    combined = title + ' ' + text

    approach_count = sum(1 for t in APPROACH_INDICATORS if t in combined)
    has_ai = any(t in combined for t in AI_TERMS)
    has_health = any(t in combined for t in HEALTH_TERMS)
    bias_in_title = any(t in title for t in BIAS_TITLE_TERMS)

    if not has_ai:
        return False, 'No AI/ML component in full text'
    if not has_health:
        return False, 'Not health-related in full text'

    if approach_count >= 2 and bias_in_title:
        return True, f'Included: bias central + approach content ({approach_count} indicators)'
    elif approach_count >= 3:
        return True, f'Included: substantial approach content ({approach_count} indicators)'
    elif bias_in_title and approach_count >= 1:
        return True, f'Included: bias is central topic ({approach_count} indicators)'
    else:
        return False, f'Excluded: insufficient approach content ({approach_count} indicators)'


# ============================================================
# PMC FUNCTIONS
# ============================================================

def batch_get_pmcids(pmids, batch_size=200):
    """Convert PMIDs to PMCIDs using NCBI ID converter."""
    pmid_to_pmcid = {}
    total_batches = (len(pmids) - 1) // batch_size + 1

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        ids_str = ','.join(batch)
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={ids_str}&format=json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SystematicReview/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            for rec in data.get('records', []):
                pmcid = rec.get('pmcid', '')
                pmid = str(rec.get('pmid', ''))
                if pmcid and pmid:
                    pmid_to_pmcid[pmid] = pmcid
        except Exception as e:
            print(f"    PMCID batch error: {e}")
        batch_num = i // batch_size + 1
        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"    PMCID lookup batch {batch_num}/{total_batches} ({len(pmid_to_pmcid)} found)")
        time.sleep(0.4)

    return pmid_to_pmcid


def fetch_pmc_fulltext(pmcid, retries=2):
    """Fetch full text from PMC as plain text."""
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
           f"db=pmc&id={pmcid}&rettype=xml&retmode=xml")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SystematicReview/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read().decode('utf-8')
            root = ET.fromstring(xml_data)

            # Try body first (actual article text)
            body = root.find('.//body')
            if body is not None:
                text = ' '.join(body.itertext())
                if len(text) > 200:
                    return text

            # Fall back to abstract
            abstract = root.find('.//abstract')
            if abstract is not None:
                text = ' '.join(abstract.itertext())
                if len(text) > 100:
                    return text

            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STEP 3: Full-Text Screening")
    print("=" * 70)

    # Load Step 2 results
    with open(f'{BASE_DIR}/pubmed_step2_results.json') as f:
        data = json.load(f)

    papers = data['ta_included']
    print(f"\nT/A included papers: {len(papers)}")

    # --- Phase 1: Look up PMCIDs ---
    print(f"\nPhase 1: Looking up PMCIDs for {len(papers)} papers...")
    pmids = [p['pmid'] for p in papers]
    pmcid_map = batch_get_pmcids(pmids)
    print(f"\n  PMCIDs found: {len(pmcid_map)} / {len(pmids)}")

    # Split
    has_fulltext_papers = []
    no_fulltext_papers = []

    for p in papers:
        pmcid = pmcid_map.get(p['pmid'])
        if pmcid:
            p['pmcid'] = pmcid
            has_fulltext_papers.append(p)
        else:
            p['pmcid'] = None
            p['ft_status'] = 'No PMC full text available'
            no_fulltext_papers.append(p)

    print(f"\n  Has PMC full text: {len(has_fulltext_papers)}")
    print(f"  No PMC full text:  {len(no_fulltext_papers)}")

    # --- Phase 2: Fetch full text and screen ---
    print(f"\nPhase 2: Fetching and screening {len(has_fulltext_papers)} papers with PMC full text...")

    ft_included = []
    ft_excluded = []
    ft_fetch_failed = []

    for i, p in enumerate(has_fulltext_papers):
        pmcid = p['pmcid']
        full_text = fetch_pmc_fulltext(pmcid)

        if full_text and len(full_text) > 200:
            inc, reason = fulltext_screen(p, full_text)
            p['ft_text_length'] = len(full_text)

            if inc:
                p['ft_status'] = f'Full text screened — INCLUDED ({pmcid})'
                p['ft_reason'] = reason
                ft_included.append(p)
            else:
                p['ft_status'] = f'Full text screened — EXCLUDED ({pmcid})'
                p['ft_reason'] = reason
                ft_excluded.append(p)
        else:
            # PMC fetch failed — treat as no full text
            p['ft_status'] = f'PMC fetch failed ({pmcid})'
            p['ft_reason'] = 'Full text not retrievable'
            ft_fetch_failed.append(p)

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(has_fulltext_papers):
            print(f"    Processed {i+1}/{len(has_fulltext_papers)} "
                  f"(included: {len(ft_included)}, excluded: {len(ft_excluded)}, "
                  f"fetch failed: {len(ft_fetch_failed)})")
        time.sleep(0.15)

    # Add fetch-failed to no_fulltext
    no_fulltext_papers.extend(ft_fetch_failed)

    # --- Exclusion breakdown ---
    ft_exc_reasons = {}
    for p in ft_excluded:
        r = p.get('ft_reason', 'Unknown')
        ft_exc_reasons[r] = ft_exc_reasons.get(r, 0) + 1

    # --- Save ---
    output = {
        'ft_screening_criteria': {
            'approach_indicators': APPROACH_INDICATORS,
            'ai_terms': AI_TERMS,
            'health_terms': HEALTH_TERMS,
            'bias_title_terms': BIAS_TITLE_TERMS,
            'logic': (
                'Include if full text has: (1) AI/ML terms, AND (2) health terms, '
                'AND meets one of: '
                '(a) approach_count >= 2 AND bias in title, '
                '(b) approach_count >= 3, '
                '(c) bias in title AND approach_count >= 1. '
                'Approach indicators count occurrences of 50+ bias assessment/mitigation terms.'
            ),
        },
        'phase1_pmcid_lookup': {
            'total_papers': len(papers),
            'pmcids_found': len(pmcid_map),
            'no_pmcid': len(papers) - len(pmcid_map),
        },
        'phase2_fulltext_screening': {
            'papers_with_fulltext': len(has_fulltext_papers),
            'ft_included': len(ft_included),
            'ft_excluded': len(ft_excluded),
            'ft_fetch_failed': len(ft_fetch_failed),
            'exclusion_reasons': ft_exc_reasons,
        },
        'final_counts': {
            'two_stage_screened': len(ft_included),
            'one_stage_only_no_fulltext': len(no_fulltext_papers),
            'ft_excluded': len(ft_excluded),
        },
        'ft_included': ft_included,
        'ft_excluded': ft_excluded,
        'no_fulltext': no_fulltext_papers,
        # Carry forward
        'screening_criteria': data['screening_criteria'],
        'query_stats': data['query_stats'],
        'reviews_removed': data['reviews_removed'],
        'review_breakdown': data['review_breakdown'],
        'total_unique_pmids': data['total_unique_pmids'],
        'total_fetched': data['total_fetched'],
        'ta_included_count': data['ta_included_count'],
        'ta_excluded_count': data['ta_excluded_count'],
        'ta_exclusion_reasons': data['exclusion_reasons'],
    }

    out_path = f'{BASE_DIR}/pubmed_step3_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"STEP 3 SUMMARY")
    print(f"{'='*70}")
    print(f"  T/A included papers:              {len(papers)}")
    print(f"  PMCIDs found:                     {len(pmcid_map)}")
    print(f"  No PMC full text:                 {len(papers) - len(pmcid_map)}")
    print(f"  PMC fetch failed:                 {len(ft_fetch_failed)}")
    print(f"")
    print(f"  Full-text screened:               {len(ft_included) + len(ft_excluded)}")
    print(f"    INCLUDED (2-stage screened):     {len(ft_included)}")
    print(f"    EXCLUDED by full text:           {len(ft_excluded)}")
    print(f"  No full text (1-stage only):       {len(no_fulltext_papers)}")
    print(f"")
    print(f"  Full-text exclusion reasons:")
    for r, c in sorted(ft_exc_reasons.items(), key=lambda x: -x[1]):
        print(f"    {c:>4}  {r}")
    print(f"\n  Saved: {out_path}")
    print(f"\n  FINAL FILES TO GENERATE (Step 4):")
    print(f"    File 1: {len(ft_included)} papers (2-stage screened, full-text)")
    print(f"    File 2: {len(no_fulltext_papers)} papers (1-stage only, no full text)")


if __name__ == '__main__':
    main()
