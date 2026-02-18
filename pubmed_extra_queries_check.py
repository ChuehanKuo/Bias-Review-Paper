#!/usr/bin/env python3
"""Quick check: how many additional PMIDs would extra queries find?"""

import json
import urllib.request
import urllib.parse
import time

# Load existing PMIDs
with open('/home/user/Bias-Review-Paper/pubmed_step1_results.json') as f:
    data = json.load(f)
existing = set()
for p in data['non_reviews'] + data['reviews']:
    existing.add(p['pmid'])
print(f"Existing PMIDs: {len(existing)}")

EXTRA_QUERIES = [
    {
        "id": "E1",
        "label": "MeSH: AI + Healthcare Disparities",
        "query": '"Artificial Intelligence"[MeSH] AND "Healthcare Disparities"[MeSH]'
    },
    {
        "id": "E2",
        "label": "MeSH: ML + Bias",
        "query": '"Machine Learning"[MeSH] AND "Bias"[MeSH]'
    },
    {
        "id": "E3",
        "label": "MeSH: AI + Prejudice/Discrimination",
        "query": '"Artificial Intelligence"[MeSH] AND ("Prejudice"[MeSH] OR "Social Discrimination"[MeSH])'
    },
    {
        "id": "E4",
        "label": "Underdiagnosis + algorithm/AI",
        "query": '("underdiagnosis" OR "underdiagnosed") AND ("algorithm" OR "machine learning" OR "artificial intelligence") AND ("bias" OR "disparity")'
    },
    {
        "id": "E5",
        "label": "Risk score + race/disparity",
        "query": '("risk score" OR "risk prediction" OR "prediction model") AND ("race" OR "racial" OR "disparity" OR "disparities") AND ("bias" OR "fairness" OR "equity")'
    },
    {
        "id": "E6",
        "label": "Specific tools: AIF360, Fairlearn, etc.",
        "query": '("AI Fairness 360" OR "AIF360" OR "Fairlearn" OR "aequitas" OR "What-If Tool") AND ("health" OR "clinical" OR "medical")'
    },
    {
        "id": "E7",
        "label": "Social determinants + ML + bias",
        "query": '"social determinants" AND ("machine learning" OR "artificial intelligence" OR "algorithm") AND ("bias" OR "fairness" OR "equity")'
    },
    {
        "id": "E8",
        "label": "EHR/clinical data + algorithmic bias",
        "query": '("electronic health record" OR "EHR" OR "clinical data") AND ("algorithmic bias" OR "model bias" OR "prediction bias" OR "fairness")'
    },
    {
        "id": "E9",
        "label": "Skin tone / dermatology AI bias",
        "query": '("skin tone" OR "skin color" OR "Fitzpatrick") AND ("algorithm" OR "deep learning" OR "AI" OR "machine learning") AND ("bias" OR "fairness" OR "performance")'
    },
    {
        "id": "E10",
        "label": "Pulse oximetry / wearable bias",
        "query": '("pulse oximetry" OR "oximeter" OR "SpO2" OR "wearable") AND ("bias" OR "accuracy" OR "disparity") AND ("race" OR "skin" OR "pigment")'
    },
]

all_new = set()
for q in EXTRA_QUERIES:
    encoded = urllib.parse.quote(q['query'])
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded}&retmax=10000&retmode=json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        ids = set(result.get('esearchresult', {}).get('idlist', []))
        total = int(result.get('esearchresult', {}).get('count', '0'))
        new = ids - existing - all_new
        all_new.update(new)
        print(f"  {q['id']}: {total:>5} total | {len(ids):>5} retrieved | {len(new):>4} NEW  [{q['label']}]")
    except Exception as e:
        print(f"  {q['id']}: ERROR - {e}")
    time.sleep(0.35)

print(f"\nTotal new PMIDs from extra queries: {len(all_new)}")
print(f"Combined total would be: {len(existing) + len(all_new)}")
