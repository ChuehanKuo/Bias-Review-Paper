#!/usr/bin/env python3
"""
PubMed Full Database Screening for Systematic Review:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

Step 1: Search PubMed with comprehensive queries
Step 2: Fetch title + abstract for all unique PMIDs
Step 3: Screen using title+abstract together
Step 4: Export to Excel
"""

import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
import re
import csv

# ============================================================
# STEP 1: SEARCH PUBMED
# ============================================================

QUERIES = [
    # Core algorithmic bias + health queries
    '("algorithmic bias" OR "algorithmic fairness") AND ("health" OR "healthcare" OR "clinical" OR "medical")',
    '("AI bias" OR "AI fairness") AND ("health" OR "healthcare" OR "clinical" OR "medical") AND ("algorithm" OR "machine learning" OR "deep learning")',
    '("machine learning bias" OR "machine learning fairness") AND ("health" OR "healthcare" OR "clinical" OR "medical")',
    '"bias mitigation" AND ("artificial intelligence" OR "machine learning") AND ("health" OR "healthcare" OR "clinical")',
    '"health disparities" AND ("machine learning" OR "artificial intelligence" OR "deep learning") AND ("bias" OR "fairness")',
    '"clinical prediction" AND ("bias" OR "fairness") AND ("algorithm" OR "machine learning" OR "AI")',
    '"equitable AI" AND ("health" OR "healthcare" OR "clinical")',
    # Specific bias axes
    '"racial bias" AND ("machine learning" OR "artificial intelligence" OR "algorithm") AND ("health" OR "clinical" OR "medical")',
    '"gender bias" AND ("machine learning" OR "artificial intelligence" OR "algorithm") AND ("health" OR "clinical" OR "medical")',
    '"age bias" AND ("machine learning" OR "artificial intelligence") AND ("health" OR "clinical")',
    '"socioeconomic" AND "bias" AND ("machine learning" OR "AI") AND ("health" OR "clinical")',
    # Specific methods/settings
    '"bias" AND "deep learning" AND ("medical imaging" OR "radiology" OR "pathology")',
    '"fairness" AND "natural language processing" AND ("clinical" OR "health" OR "medical")',
    '"bias" AND "electronic health record" AND ("machine learning" OR "AI" OR "algorithm")',
    '"bias" AND "clinical decision support" AND ("machine learning" OR "AI" OR "algorithm")',
    # Specific clinical domains
    '"bias" AND ("machine learning" OR "AI" OR "deep learning") AND "dermatology"',
    '"bias" AND ("machine learning" OR "AI" OR "deep learning") AND "radiology"',
    '"bias" AND ("machine learning" OR "AI" OR "deep learning") AND "ophthalmology"',
    '"bias" AND ("machine learning" OR "AI" OR "deep learning") AND "cardiology"',
    '"bias" AND ("machine learning" OR "AI" OR "deep learning") AND "mental health"',
    '"bias" AND ("machine learning" OR "AI" OR "deep learning") AND "oncology"',
    # Fairness frameworks & assessment
    '"fairness" AND ("health" OR "healthcare" OR "clinical") AND ("machine learning" OR "deep learning" OR "AI")',
    '"bias audit" AND ("health" OR "clinical" OR "medical") AND ("AI" OR "algorithm")',
    '"disparate impact" AND ("health" OR "clinical") AND ("algorithm" OR "machine learning" OR "prediction")',
    # Additional
    '"debiasing" AND ("health" OR "healthcare" OR "clinical") AND ("AI" OR "machine learning")',
    '"health equity" AND ("artificial intelligence" OR "machine learning") AND ("bias" OR "fairness" OR "algorithm")',
    '"skin color" AND "bias" AND ("AI" OR "deep learning" OR "algorithm")',
    '"underdiagnosis" AND "bias" AND ("AI" OR "machine learning" OR "algorithm")',
    '"federated learning" AND "fairness" AND ("health" OR "clinical" OR "medical")',
    '"large language model" AND "bias" AND ("health" OR "clinical" OR "medical")',
    # Broad catch-all
    '"algorithmic bias" AND "healthcare"',
    '"AI fairness" AND "healthcare"',
    '"fair machine learning" AND "health"',
    '"bias detection" AND "clinical" AND ("AI" OR "machine learning")',
    '"prediction model" AND "racial bias" AND ("health" OR "clinical")',
    '"mortality prediction" AND ("bias" OR "fairness") AND ("AI" OR "machine learning")',
    '"readmission" AND ("bias" OR "fairness") AND ("AI" OR "machine learning" OR "algorithm")',
    '"sepsis" AND ("bias" OR "fairness") AND ("AI" OR "machine learning" OR "prediction")',
    '"chest X-ray" AND ("bias" OR "fairness") AND ("AI" OR "deep learning")',
    '"wearable" AND "bias" AND ("AI" OR "machine learning") AND "health"',
]


def search_pubmed(query, retmax=500):
    """Search PubMed and return list of PMIDs."""
    encoded = urllib.parse.quote(query)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax={retmax}&retmode=json"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        ids = data.get('esearchresult', {}).get('idlist', [])
        count = data.get('esearchresult', {}).get('count', '0')
        return ids, int(count)
    except Exception as e:
        print(f"  ERROR searching: {e}")
        return [], 0


def fetch_abstracts(pmids, batch_size=100):
    """Fetch title + abstract + metadata for a batch of PMIDs."""
    papers = {}
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        ids_str = ','.join(batch)
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids_str}&rettype=xml&retmode=xml"

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
                            name += f" {fore.text[0]}"
                        authors.append(name)
                author_str = authors[0] + ' et al.' if len(authors) > 1 else (authors[0] if authors else 'N/A')

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

            print(f"  Fetched batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1} ({len(batch)} papers)")
        except Exception as e:
            print(f"  ERROR fetching batch {i//batch_size + 1}: {e}")

        time.sleep(0.4)

    return papers


# ============================================================
# STEP 3: TITLE+ABSTRACT SCREENING
# ============================================================

def screen_paper(paper):
    """
    Screen a paper using title+abstract together.
    Returns (include: bool, reason: str)

    INCLUDE: Papers about bias/fairness in ML/AI algorithms applied to health
    EXCLUDE: Human cognitive biases, statistical bias without AI/ML, non-health
    """
    title = (paper.get('title', '') or '').lower()
    abstract = (paper.get('abstract', '') or '').lower()
    combined = title + ' ' + abstract
    keywords = (paper.get('keywords', '') or '').lower()
    mesh = (paper.get('mesh_terms', '') or '').lower()
    all_text = combined + ' ' + keywords + ' ' + mesh

    # ----- EXCLUSION CHECKS (applied first) -----

    # Must be about AI/ML/algorithms
    ai_terms = [
        'machine learning', 'deep learning', 'artificial intelligence',
        'neural network', 'algorithm', 'predictive model', 'prediction model',
        'classifier', 'classification model', 'natural language processing',
        'nlp', 'computer vision', 'automated', 'computational',
        'random forest', 'logistic regression', 'xgboost', 'gradient boosting',
        'convolutional', 'transformer', 'large language model', 'llm',
        'decision support', 'clinical decision', 'risk score', 'risk prediction',
        'ai ', ' ai,', ' ai.', 'a.i.', 'ai-', 'ml ', ' ml,', ' ml.',
        'data-driven', 'data driven', 'supervised learning', 'unsupervised',
        'federated learning', 'reinforcement learning', 'foundation model'
    ]
    has_ai = any(term in all_text for term in ai_terms)

    # Must be about health/healthcare/medicine
    health_terms = [
        'health', 'clinical', 'medical', 'patient', 'hospital', 'disease',
        'diagnosis', 'treatment', 'care', 'physician', 'doctor', 'nurse',
        'surgery', 'therapeutic', 'pathology', 'radiology', 'dermatology',
        'cardiology', 'oncology', 'ophthalmology', 'psychiatry', 'mental health',
        'ehr', 'electronic health record', 'emr', 'biomedical', 'pharmaceutical',
        'drug', 'mortality', 'readmission', 'sepsis', 'icu', 'emergency',
        'chest x-ray', 'mammography', 'ct scan', 'mri', 'ecg', 'ekg',
        'genomic', 'genetic', 'cancer', 'diabetes', 'cardiovascular',
        'epidemiol', 'public health', 'healthcare', 'medicine'
    ]
    has_health = any(term in all_text for term in health_terms)

    # Must be about bias/fairness (in AI context)
    bias_terms = [
        'bias', 'fairness', 'fair ', 'unfair', 'disparity', 'disparities',
        'discrimination', 'inequity', 'inequitable', 'equitable', 'equity',
        'underrepresent', 'underserved', 'marginalized', 'disadvantaged',
        'racial', 'gender', 'ethnic', 'socioeconomic', 'demographic',
        'debiasing', 'debias', 'mitigat', 'algorithmic fairness',
        'disparate impact', 'equal opportunity', 'demographic parity',
        'calibration', 'subgroup', 'underdiagnos'
    ]
    has_bias = any(term in all_text for term in bias_terms)

    if not has_ai:
        return False, 'EXCLUDE: No AI/ML/algorithm component identified'
    if not has_health:
        return False, 'EXCLUDE: Not related to health/healthcare/medicine'
    if not has_bias:
        return False, 'EXCLUDE: No bias/fairness component identified'

    # Exclude papers primarily about human cognitive biases (not AI)
    human_bias_indicators = [
        'implicit bias training', 'implicit bias among', 'clinician bias',
        'physician bias', 'provider bias', 'cognitive bias in decision',
        'unconscious bias training', 'bias in clinical judgment',
        'confirmation bias in diagnosis', 'anchoring bias in',
        'implicit association test'
    ]
    # But only exclude if there's no AI angle
    is_human_bias = any(term in all_text for term in human_bias_indicators)
    ai_bias_indicators = [
        'algorithmic bias', 'ai bias', 'model bias', 'prediction bias',
        'training data bias', 'machine learning bias', 'fairness-aware',
        'bias mitigation', 'bias in ai', 'bias in machine learning',
        'fair machine learning', 'algorithmic fairness', 'model fairness',
        'debiasing algorithm', 'bias in prediction', 'bias in algorithm'
    ]
    has_ai_bias = any(term in all_text for term in ai_bias_indicators)

    if is_human_bias and not has_ai_bias:
        return False, 'EXCLUDE: About human/clinician cognitive biases, not AI/algorithmic bias'

    # Exclude pure statistical/epidemiological bias without AI
    stat_bias_only = [
        'selection bias in cohort', 'recall bias in survey',
        'information bias in epidemiol', 'publication bias meta-analysis',
        'selection bias in observational', 'attrition bias',
        'reporting bias in systematic review'
    ]
    is_stat_bias = any(term in all_text for term in stat_bias_only)
    if is_stat_bias and not has_ai_bias:
        return False, 'EXCLUDE: About statistical/epidemiological bias, not AI/algorithmic bias'

    # ----- INCLUSION -----
    return True, 'INCLUDE: Paper about bias/fairness in AI/ML applied to health'


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PubMed Full Database Screening")
    print("Systematic Review: Algorithmic Bias in Health AI")
    print("=" * 70)

    # Step 1: Search
    print(f"\nSTEP 1: Searching PubMed with {len(QUERIES)} queries...")
    all_pmids = set()
    query_stats = []

    for i, query in enumerate(QUERIES):
        ids, total_count = search_pubmed(query, retmax=500)
        new_ids = set(ids) - all_pmids
        all_pmids.update(ids)
        query_stats.append({
            'query': query[:80],
            'total_in_pubmed': total_count,
            'retrieved': len(ids),
            'new_unique': len(new_ids)
        })
        print(f"  Q{i+1:02d}: {total_count:>5} total | {len(ids):>4} retrieved | {len(new_ids):>4} new | cumulative: {len(all_pmids)}")
        time.sleep(0.35)

    print(f"\nTotal unique PMIDs: {len(all_pmids)}")

    # Step 2: Fetch abstracts
    print(f"\nSTEP 2: Fetching titles + abstracts for {len(all_pmids)} papers...")
    pmid_list = sorted(all_pmids)
    papers = fetch_abstracts(pmid_list, batch_size=100)

    with_abstract = sum(1 for p in papers.values() if p['abstract'])
    print(f"  Papers with abstracts: {with_abstract}")
    print(f"  Papers without abstracts: {len(papers) - with_abstract}")

    # Step 3: Screen
    print(f"\nSTEP 3: Screening {len(papers)} papers by title+abstract...")
    included = []
    excluded = []

    for pmid, paper in papers.items():
        include, reason = screen_paper(paper)
        paper['include'] = include
        paper['screen_reason'] = reason
        if include:
            included.append(paper)
        else:
            excluded.append(paper)

    print(f"  INCLUDED: {len(included)}")
    print(f"  EXCLUDED: {len(excluded)}")

    # Save all data
    all_data = {
        'included': included,
        'excluded': excluded,
        'query_stats': query_stats,
        'total_pmids': len(all_pmids),
        'total_fetched': len(papers)
    }

    with open('/home/user/Bias-Review-Paper/pubmed_screening_data.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\nData saved to pubmed_screening_data.json")
    print(f"Ready for Excel generation.")


if __name__ == '__main__':
    main()
