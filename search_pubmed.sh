#!/bin/bash
# PubMed search script using E-utilities
# Usage: search_pubmed.sh "query" output_file

QUERY="$1"
OUTPUT="$2"
ENCODED_QUERY=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$QUERY'))")

# Search PubMed - get up to 200 results per query
SEARCH_URL="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=${ENCODED_QUERY}&retmax=200&retmode=json"
SEARCH_RESULT=$(curl -s "$SEARCH_URL")

# Extract PMIDs
PMIDS=$(echo "$SEARCH_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ids = data.get('esearchresult', {}).get('idlist', [])
print(','.join(ids))
")

COUNT=$(echo "$SEARCH_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('esearchresult', {}).get('count', '0'))
")

echo "Query: $QUERY | Total results: $COUNT | Fetched IDs: $(echo $PMIDS | tr ',' '\n' | wc -w)" >> "$OUTPUT.log"

if [ -z "$PMIDS" ] || [ "$PMIDS" = "" ]; then
    echo "No results for: $QUERY" >> "$OUTPUT.log"
    exit 0
fi

# Fetch details for these PMIDs
sleep 0.4
FETCH_URL="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=${PMIDS}&retmode=json"
curl -s "$FETCH_URL" > "$OUTPUT"

echo "Saved results to $OUTPUT" >> "$OUTPUT.log"
