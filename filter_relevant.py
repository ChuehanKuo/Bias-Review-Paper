#!/usr/bin/env python3
"""
Filter PubMed results for relevance to:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

Uses title-based keyword filtering to identify papers that are specifically
about bias/fairness in health AI (not just papers that mention bias incidentally).
"""

import csv

# Keywords that strongly suggest the paper is about algorithmic bias in health AI
STRONG_KEYWORDS = [
    'algorithmic bias', 'algorithmic fairness', 'ai bias', 'ai fairness',
    'machine learning bias', 'machine learning fairness', 'model bias',
    'model fairness', 'bias mitigation', 'bias detection', 'debiasing',
    'de-biasing', 'fair machine learning', 'fair ai', 'equitable ai',
    'equitable artificial intelligence', 'health equity',
    'algorithmic discrimination', 'disparate impact', 'bias audit',
    'fairness metric', 'fairness framework', 'fairness-aware',
    'bias in artificial intelligence', 'bias in ai', 'biased algorithm',
    'racial bias', 'gender bias', 'ethnic bias', 'race bias',
    'skin color bias', 'skin tone bias', 'underrepresentation',
    'underserved', 'health disparit', 'healthcare disparit',
    'racial disparit', 'ethnic disparit', 'socioeconomic bias',
    'age bias', 'language bias', 'disability bias', 'insurance bias',
    'geographic bias', 'calibration bias', 'subgroup performance',
    'subgroup analysis', 'disparate performance', 'performance gap',
    'performance disparit', 'equit', 'inequit',
    'clinical prediction bias', 'prediction bias', 'predictive bias',
    'dataset bias', 'data bias', 'training data bias', 'selection bias',
    'measurement bias', 'label bias', 'sampling bias',
    'responsible ai', 'trustworthy ai', 'ethical ai', 'ai ethics',
    'bias in clinical', 'bias in health', 'bias in medical',
    'fairness in clinical', 'fairness in health', 'fairness in medical',
    'fair clinical', 'fair health', 'fair predict',
    'bias in deep learning', 'bias in machine learning',
    'algorithmic harm', 'algorithmic accountability',
    'discrimination in', 'discriminatory',
]

# Domain-specific patterns (bias + health domain)
BIAS_WORDS = ['bias', 'fairness', 'fair ', 'equit', 'inequit', 'disparit', 'discriminat']
HEALTH_AI_WORDS = ['ai', 'artificial intelligence', 'machine learning', 'deep learning',
                    'algorithm', 'predict', 'model', 'neural network', 'nlp',
                    'clinical decision', 'decision support', 'automated']
HEALTH_WORDS = ['health', 'clinical', 'medical', 'patient', 'hospital', 'ehr',
                'electronic health record', 'diagnosis', 'diagnostic', 'prognos',
                'radiology', 'dermatology', 'ophthalmology', 'pathology', 'cardiol',
                'oncology', 'sepsis', 'mortality', 'readmission', 'chest x-ray',
                'imaging', 'drug', 'pharma', 'treatment', 'surgical', 'emergency',
                'icu', 'intensive care', 'mental health', 'psychiatr']

def is_relevant(title):
    """Check if title is relevant to algorithmic bias in health AI."""
    title_lower = title.lower()

    # Check for strong keywords (direct match)
    for kw in STRONG_KEYWORDS:
        if kw in title_lower:
            return True, 'strong_keyword'

    # Check for combination: bias word + AI word + health word
    has_bias = any(bw in title_lower for bw in BIAS_WORDS)
    has_ai = any(aw in title_lower for aw in HEALTH_AI_WORDS)
    has_health = any(hw in title_lower for hw in HEALTH_WORDS)

    if has_bias and has_ai:
        return True, 'bias+ai'
    if has_bias and has_health and has_ai:
        return True, 'bias+health+ai'

    return False, None

# Read CSV
input_path = '/home/user/Bias-Review-Paper/pubmed_search_results.csv'
output_path = '/home/user/Bias-Review-Paper/pubmed_filtered_results.csv'
output_txt = '/home/user/Bias-Review-Paper/pubmed_filtered_results.txt'

relevant = []
all_papers = []

with open(input_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_papers.append(row)
        is_rel, reason = is_relevant(row['Title'])
        if is_rel:
            row['relevance_reason'] = reason
            relevant.append(row)

# Sort by year descending
relevant.sort(key=lambda x: x.get('Year', '0'), reverse=True)

print(f"Total papers: {len(all_papers)}")
print(f"Relevant papers (title-filtered): {len(relevant)}")

# Write filtered CSV
with open(output_path, 'w') as f:
    f.write("PMID,Title,Year,Journal,URL,DOI,RelevanceReason\n")
    for r in relevant:
        title = r['Title'].replace('"', '""')
        journal = r['Journal'].replace('"', '""')
        f.write(f'"{r["PMID"]}","{title}","{r["Year"]}","{journal}","{r["URL"]}","{r["DOI"]}","{r.get("relevance_reason","")}"\n')

# Write filtered TXT
with open(output_txt, 'w') as f:
    f.write("FILTERED PUBMED SEARCH RESULTS - RELEVANT TO ALGORITHMIC BIAS IN HEALTH AI\n")
    f.write("=" * 120 + "\n")
    f.write(f"Total papers from search: {len(all_papers)}\n")
    f.write(f"Relevant papers (title-filtered): {len(relevant)}\n")
    f.write(f"NOTE: This is title-based filtering only. Abstract screening still needed.\n")
    f.write("=" * 120 + "\n\n")

    for idx, r in enumerate(relevant, 1):
        f.write(f"{idx}. {r['Title']}\n")
        f.write(f"   Year: {r['Year']} | Journal: {r['Journal']}\n")
        f.write(f"   URL: {r['URL']}\n")
        if r.get('DOI'):
            f.write(f"   DOI: {r['DOI']}\n")
        f.write(f"   [Matched: {r.get('relevance_reason', 'N/A')}]\n\n")

print(f"\nFiltered results written to:")
print(f"  CSV: {output_path}")
print(f"  TXT: {output_txt}")

# Print year distribution
from collections import Counter
year_counts = Counter(r['Year'] for r in relevant)
print(f"\nYear distribution of relevant papers:")
for year in sorted(year_counts.keys(), reverse=True):
    print(f"  {year}: {year_counts[year]} papers")
