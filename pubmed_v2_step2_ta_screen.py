#!/usr/bin/env python3
"""
PubMed v2 Step 2: Title + Abstract Screening (Stage 1)

Applies the same 4-criterion T/A screen to the 1222 non-review papers.
Output: pubmed_v2_step2_results.json
"""

import json, re

BASE_DIR = '/home/user/Bias-Review-Paper'
INPUT_FILE = f'{BASE_DIR}/pubmed_v2_fetched.json'
OUTPUT_FILE = f'{BASE_DIR}/pubmed_v2_step2_results.json'

# ============================================================
# SCREENING CRITERIA  (same as original pipeline)
# ============================================================

AI_TERMS = [
    'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
    'natural language processing', 'nlp', 'computer vision', 'random forest',
    'decision tree', 'support vector', 'svm', 'logistic regression',
    'gradient boosting', 'xgboost', 'ensemble', 'supervised learning',
    'unsupervised learning', 'reinforcement learning', 'transfer learning',
    'federated learning', 'convolutional neural', 'recurrent neural',
    'transformer', 'bert', 'gpt', 'large language model', 'llm',
    'generative ai', 'foundation model', 'predictive model',
    'prediction model', 'clinical prediction', 'risk prediction',
    'classification model', 'regression model', 'clustering',
    'automated', 'algorithm', 'computational', 'data-driven',
    'clinical decision support', 'computer-aided', 'image recognition',
    'feature selection', 'dimensionality reduction',
]

HEALTH_TERMS = [
    'health', 'clinical', 'medical', 'patient', 'hospital', 'disease',
    'diagnosis', 'treatment', 'therapy', 'prognosis', 'mortality',
    'morbidity', 'surgical', 'radiology', 'pathology', 'oncology',
    'cardiology', 'dermatology', 'ophthalmology', 'psychiatry',
    'mental health', 'emergency', 'icu', 'intensive care', 'ehr',
    'electronic health record', 'biomedical', 'pharmaceutical',
    'drug', 'genomic', 'imaging', 'screening', 'vaccination',
    'epidemiology', 'public health', 'healthcare', 'care',
    'nursing', 'pharmacy', 'dental', 'rehabilitation',
    'chronic', 'acute', 'outpatient', 'inpatient', 'primary care',
    'sepsis', 'pneumonia', 'diabetes', 'cancer', 'tumor', 'stroke',
    'heart', 'lung', 'kidney', 'liver', 'brain',
]

STRONG_TITLE_TERMS = [
    'bias', 'fairness', 'fair ', 'unfair', 'equity', 'inequity',
    'disparity', 'disparities', 'discrimination', 'algorithmic bias',
    'racial bias', 'gender bias', 'health equity', 'health disparities',
    'equitable', 'inequitable',
]

AI_BIAS_TERMS = [
    'algorithmic bias', 'model bias', 'prediction bias', 'data bias',
    'selection bias', 'label bias', 'measurement bias', 'sampling bias',
    'representation bias', 'training bias', 'dataset bias',
    'fairness', 'unfairness', 'equalized odds', 'demographic parity',
    'equal opportunity', 'disparate impact', 'calibration bias',
    'subgroup', 'performance gap', 'performance disparity',
    'underrepresent', 'overrepresent', 'health equity',
    'health disparity', 'health disparities', 'racial disparity',
    'racial disparities', 'gender disparity', 'sex-based',
    'race-based', 'ethnic', 'socioeconomic', 'underserved',
    'marginalized', 'vulnerable population', 'minority',
    'bias mitigation', 'bias detection', 'bias assessment',
    'bias evaluation', 'bias audit', 'debiasing', 'debias',
    'fair machine learning', 'fair ai', 'responsible ai',
    'trustworthy ai', 'ethical ai',
]

HUMAN_ONLY_BIAS_TERMS = [
    'cognitive bias', 'confirmation bias', 'anchoring bias',
    'availability bias', 'recall bias', 'observer bias',
    'interviewer bias', 'response bias', 'reporting bias',
    'publication bias', 'attrition bias', 'detection bias',
    'information bias', 'lead-time bias', 'length bias',
    'surveillance bias', 'referral bias', 'volunteer bias',
    'healthy worker', 'berkson', 'neyman',
]

AI_SPECIFIC_OVERRIDE = [
    'algorithmic', 'algorithm', 'machine learning', 'deep learning',
    'artificial intelligence', 'model bias', 'prediction bias',
    'training bias', 'data bias', 'fairness', 'fair ',
    'equalized odds', 'demographic parity', 'disparate impact',
]


def screen_paper(paper):
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    combined = title + ' ' + abstract

    # Criterion 1: AI/ML
    has_ai = any(t in combined for t in AI_TERMS)
    if not has_ai:
        return False, 'No AI/ML terms'

    # Criterion 2: Health
    has_health = any(t in combined for t in HEALTH_TERMS)
    if not has_health:
        return False, 'No health terms'

    # Criterion 3: Bias/fairness centrality
    has_strong_title = any(t in title for t in STRONG_TITLE_TERMS)
    ai_bias_count = sum(1 for t in AI_BIAS_TERMS if t in combined)
    if not has_strong_title and ai_bias_count < 2:
        return False, 'Bias/fairness not central'

    # Criterion 4: Exclude human-only biases
    has_human_only = any(t in combined for t in HUMAN_ONLY_BIAS_TERMS)
    if has_human_only:
        has_ai_specific = any(t in combined for t in AI_SPECIFIC_OVERRIDE)
        if not has_ai_specific:
            return False, 'Human cognitive bias only'

    return True, 'Included'


def main():
    print("=" * 70)
    print("PubMed v2 Step 2: Title + Abstract Screening")
    print("=" * 70)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    papers = data['papers']
    print(f"Papers to screen: {len(papers)}")

    included = []
    excluded = []
    exclusion_reasons = {}

    for p in papers:
        passed, reason = screen_paper(p)
        if passed:
            included.append(p)
        else:
            excluded.append(p)
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    print(f"\nRESULTS:")
    print(f"  Included: {len(included)}")
    print(f"  Excluded: {len(excluded)}")
    print(f"\nExclusion reasons:")
    for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:>5}  {reason}")

    output = {
        'source_file': '[0211] IEEE John_Screening 1 & 2.xlsx 的副本.xlsx',
        'source_sheet': 'PubMed2',
        'total_pmids_in_sheet': data['total_pmids'],
        'total_fetched': data['total_fetched'],
        'reviews_removed': data['reviews_removed'],
        'papers_screened': len(papers),
        'ta_included_count': len(included),
        'ta_excluded_count': len(excluded),
        'ta_exclusion_reasons': exclusion_reasons,
        'screening_criteria': {
            'ai_terms': AI_TERMS,
            'health_terms': HEALTH_TERMS,
            'strong_title_terms': STRONG_TITLE_TERMS,
            'ai_bias_terms': AI_BIAS_TERMS,
            'human_only_terms': HUMAN_ONLY_BIAS_TERMS,
            'ai_specific_terms': AI_SPECIFIC_OVERRIDE,
        },
        'included': included,
        'excluded': excluded,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
