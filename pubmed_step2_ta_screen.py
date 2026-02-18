#!/usr/bin/env python3
"""
PubMed Step 2: Title + Abstract Screening (Stage 1)

Screens 10,118 non-review papers using keyword-based criteria:
1. Must have AI/ML component
2. Must be health-related
3. Bias/fairness must be a central topic (not just mentioned in passing)
4. Exclude papers about human cognitive biases only

Saves results for Step 3.
"""

import json

BASE_DIR = '/home/user/Bias-Review-Paper'

# ============================================================
# SCREENING CRITERIA
# ============================================================

AI_TERMS = [
    'machine learning', 'deep learning', 'artificial intelligence', 'neural network',
    'algorithm', 'predictive model', 'prediction model', 'classifier', 'classification',
    'natural language processing', 'nlp', 'computer vision', 'random forest',
    'logistic regression', 'xgboost', 'convolutional', 'transformer',
    'large language model', 'llm', 'decision support', 'risk prediction',
    'federated learning', 'reinforcement learning', 'foundation model',
    'chatgpt', 'gpt-4', 'supervised learning',
]

HEALTH_TERMS = [
    'health', 'clinical', 'medical', 'patient', 'hospital', 'disease',
    'diagnosis', 'treatment', 'care', 'pathology', 'radiology', 'dermatology',
    'cardiology', 'oncology', 'ophthalmology', 'psychiatry', 'mental health',
    'ehr', 'electronic health', 'biomedical', 'mortality', 'readmission',
    'sepsis', 'icu', 'emergency', 'chest x-ray', 'mammograph', 'cancer',
    'diabetes', 'cardiovascular', 'public health', 'healthcare', 'medicine',
]

# Strong bias/fairness terms that indicate centrality when in title
STRONG_TITLE_TERMS = [
    'bias', 'fairness', 'fair ', 'unfair', 'equitable', 'equity',
    'disparity', 'disparities', 'discrimination', 'debiasing', 'debias',
    'underdiagnos', 'underrepresent', 'inequit',
]

# AI-specific bias terms (in abstract)
AI_BIAS_TERMS = [
    'algorithmic bias', 'algorithmic fairness', 'ai bias', 'ai fairness',
    'machine learning bias', 'machine learning fairness', 'model bias',
    'bias mitigation', 'bias detection', 'bias assessment', 'fairness-aware',
    'fair machine learning', 'debiasing', 'disparate impact', 'demographic parity',
    'equalized odds', 'equal opportunity', 'fairness metric', 'fairness evaluation',
    'bias in ai', 'bias in machine learning', 'bias in algorithm', 'biased model',
    'biased prediction', 'mitigating bias', 'addressing bias', 'reducing bias',
    'assessing bias', 'evaluating bias', 'detecting bias', 'sources of bias',
    'racial bias', 'gender bias', 'age bias', 'socioeconomic bias', 'ethnic bias',
    'underdiagnosis bias', 'representation bias', 'data bias', 'label bias',
    'health disparit', 'health equit', 'health inequit', 'unfair',
]

# Human-only bias terms (to exclude papers about clinician bias, not algorithmic)
HUMAN_ONLY_TERMS = [
    'implicit bias training', 'implicit bias among', 'clinician bias',
    'physician bias', 'provider bias', 'unconscious bias training',
    'implicit association test', 'weight bias', 'anti-fat bias',
]

# AI-specific terms that override the human-only exclusion
AI_SPECIFIC_TERMS = [
    'algorithmic bias', 'ai bias', 'model bias', 'bias mitigation',
    'bias in ai', 'bias in machine learning', 'algorithmic fairness',
    'fairness-aware', 'debiasing algorithm',
]


def screen_paper(paper):
    """
    Title+Abstract screening.
    Returns (included: bool, reason: str).
    """
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    combined = title + ' ' + abstract
    kw = (paper.get('keywords', '') or '').lower()
    mesh = (paper.get('mesh_terms', '') or '').lower()
    all_text = combined + ' ' + kw + ' ' + mesh

    # Criterion 1: Must have AI/ML component
    has_ai = any(t in all_text for t in AI_TERMS)
    if not has_ai:
        return False, 'No AI/ML component'

    # Criterion 2: Must be health-related
    has_health = any(t in all_text for t in HEALTH_TERMS)
    if not has_health:
        return False, 'Not health-related'

    # Criterion 3: Bias/fairness must be central
    strong_in_title = any(t in title for t in STRONG_TITLE_TERMS)
    bias_count = sum(1 for t in AI_BIAS_TERMS if t in abstract)
    has_substantial_bias = strong_in_title or bias_count >= 2

    if not has_substantial_bias:
        return False, 'Bias/fairness not central topic'

    # Criterion 4: Exclude human cognitive biases only
    if any(t in all_text for t in HUMAN_ONLY_TERMS) and not any(t in all_text for t in AI_SPECIFIC_TERMS):
        return False, 'Human cognitive biases only'

    return True, 'Included'


def main():
    print("=" * 70)
    print("STEP 2: Title + Abstract Screening")
    print("=" * 70)

    # Load Step 1 results
    with open(f'{BASE_DIR}/pubmed_step1_results.json') as f:
        data = json.load(f)

    papers = data['non_reviews']
    print(f"\nPapers to screen: {len(papers)}")

    # Screen
    included = []
    excluded = []
    exclusion_reasons = {}

    for p in papers:
        inc, reason = screen_paper(p)
        p['ta_included'] = inc
        p['ta_reason'] = reason

        if inc:
            included.append(p)
        else:
            excluded.append(p)
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    # Results
    print(f"\n  INCLUDED: {len(included)}")
    print(f"  EXCLUDED: {len(excluded)}")
    print(f"\n  Exclusion breakdown:")
    for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
        print(f"    {count:>5}  {reason}")

    # Quick stats on included papers
    with_abstract = sum(1 for p in included if len(p.get('abstract', '') or '') > 50)
    no_abstract = len(included) - with_abstract
    print(f"\n  Included papers with abstract: {with_abstract}")
    print(f"  Included papers without/short abstract: {no_abstract}")

    # Year distribution of included
    years = {}
    for p in included:
        y = p.get('year', 'Unknown')
        years[y] = years.get(y, 0) + 1
    print(f"\n  Year distribution (included, top 10):")
    for y, c in sorted(years.items(), key=lambda x: -x[1])[:10]:
        print(f"    {y}: {c}")

    # Save
    output = {
        'screening_criteria': {
            'ai_terms': AI_TERMS,
            'health_terms': HEALTH_TERMS,
            'strong_title_terms': STRONG_TITLE_TERMS,
            'ai_bias_terms': AI_BIAS_TERMS,
            'human_only_terms': HUMAN_ONLY_TERMS,
            'ai_specific_terms': AI_SPECIFIC_TERMS,
            'logic': (
                'Include if: (1) has AI/ML terms, AND (2) has health terms, '
                'AND (3) bias/fairness is central (strong term in title OR >= 2 AI bias terms in abstract), '
                'AND (4) not about human cognitive biases only.'
            ),
        },
        'total_screened': len(papers),
        'ta_included_count': len(included),
        'ta_excluded_count': len(excluded),
        'exclusion_reasons': exclusion_reasons,
        'ta_included': included,
        'ta_excluded': excluded,
        # Carry forward from Step 1
        'query_stats': data['query_stats'],
        'reviews_removed': data['reviews_removed'],
        'review_breakdown': data['review_breakdown'],
        'total_unique_pmids': data['total_unique_pmids'],
        'total_fetched': data['total_fetched'],
    }

    out_path = f'{BASE_DIR}/pubmed_step2_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"STEP 2 SUMMARY")
    print(f"{'='*70}")
    print(f"  Papers screened:     {len(papers)}")
    print(f"  T/A INCLUDED:        {len(included)}")
    print(f"  T/A EXCLUDED:        {len(excluded)}")
    print(f"\n  Saved: {out_path}")
    print(f"\n  Ready for Step 3: split into has-full-text vs no-full-text")


if __name__ == '__main__':
    main()
