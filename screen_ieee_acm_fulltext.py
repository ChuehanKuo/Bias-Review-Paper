#!/usr/bin/env python3
"""
Full-text screening for IEEE and ACM papers.

Tries multiple sources to get actual full text:
1. OpenAlex open_access.oa_url
2. Semantic Scholar full text
3. Direct HTML fetch from publisher (for open access papers)
4. Unpaywall API via DOI

Then applies the same fulltext_screen() criteria used for PubMed papers.
"""

import json
import time
import re
import urllib.request
import urllib.parse
import ssl
import html

from shared_screening import (
    fulltext_screen, APPROACH_INDICATORS, AI_TERMS, HEALTH_TERMS,
    extract_all_columns
)

BASE_DIR = '/home/user/Bias-Review-Paper'

# Lenient SSL for fetching
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'SystematicReview/1.0 (mailto:review@example.com)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def fetch_url(url, timeout=30):
    """Fetch URL content as string."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = resp.read()
            # Try utf-8 first, then latin-1
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                return data.decode('latin-1')
    except Exception as e:
        return None


def html_to_text(html_content):
    """Extract readable text from HTML."""
    if not html_content:
        return ''
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def try_openalex(doi):
    """Try to get full text URL from OpenAlex."""
    print(f"    [OpenAlex] Looking up {doi}...")
    encoded = urllib.parse.quote(f'https://doi.org/{doi}', safe='')
    url = f'https://api.openalex.org/works/{encoded}?mailto=review@example.com'
    content = fetch_url(url)
    if not content:
        print(f"    [OpenAlex] No response")
        return None, None

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"    [OpenAlex] Invalid JSON")
        return None, None

    oa = data.get('open_access', {})
    oa_url = oa.get('oa_url', '')
    is_oa = oa.get('is_oa', False)
    print(f"    [OpenAlex] is_oa={is_oa}, oa_url={oa_url or '(none)'}")

    # Also check for fulltext_origin / best_oa_location
    best_loc = data.get('best_oa_location', {}) or {}
    pdf_url = best_loc.get('pdf_url', '')
    landing_url = best_loc.get('landing_page_url', '')

    if pdf_url:
        print(f"    [OpenAlex] PDF URL: {pdf_url}")
    if landing_url and landing_url != oa_url:
        print(f"    [OpenAlex] Landing: {landing_url}")

    # Try to fetch full text from oa_url (HTML)
    if oa_url:
        text = fetch_and_extract(oa_url)
        if text and len(text) > 500:
            return text, f'OpenAlex OA ({oa_url})'

    # Try landing page
    if landing_url and landing_url != oa_url:
        text = fetch_and_extract(landing_url)
        if text and len(text) > 500:
            return text, f'OpenAlex landing ({landing_url})'

    return None, None


def try_semantic_scholar(doi):
    """Try Semantic Scholar API for full text / extended abstract."""
    print(f"    [S2] Looking up {doi}...")
    url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract,tldr,openAccessPdf'
    content = fetch_url(url)
    if not content:
        print(f"    [S2] No response")
        return None, None

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"    [S2] Invalid JSON")
        return None, None

    oa_pdf = data.get('openAccessPdf', {}) or {}
    pdf_url = oa_pdf.get('url', '')
    abstract_s2 = data.get('abstract', '')

    if pdf_url:
        print(f"    [S2] OA PDF: {pdf_url}")
        # Try HTML version of the page (we can't easily parse PDFs)
        text = fetch_and_extract(pdf_url)
        if text and len(text) > 500:
            return text, f'Semantic Scholar OA ({pdf_url})'

    return None, None


def try_unpaywall(doi):
    """Try Unpaywall API for open access full text."""
    print(f"    [Unpaywall] Looking up {doi}...")
    url = f'https://api.unpaywall.org/v2/{doi}?email=review@example.com'
    content = fetch_url(url)
    if not content:
        print(f"    [Unpaywall] No response")
        return None, None

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"    [Unpaywall] Invalid JSON")
        return None, None

    is_oa = data.get('is_oa', False)
    best = data.get('best_oa_location', {}) or {}
    oa_url = best.get('url_for_pdf', '') or best.get('url_for_landing_page', '') or best.get('url', '')

    print(f"    [Unpaywall] is_oa={is_oa}, url={oa_url or '(none)'}")

    if oa_url:
        text = fetch_and_extract(oa_url)
        if text and len(text) > 500:
            return text, f'Unpaywall ({oa_url})'

    return None, None


def try_direct_doi(doi):
    """Try fetching from DOI redirect (publisher page)."""
    print(f"    [Direct] Trying https://doi.org/{doi}...")
    url = f'https://doi.org/{doi}'
    text = fetch_and_extract(url)
    if text and len(text) > 500:
        return text, f'DOI direct ({url})'
    return None, None


def fetch_and_extract(url):
    """Fetch a URL and extract text content."""
    content = fetch_url(url, timeout=30)
    if not content:
        return None

    # If it looks like HTML, extract text
    if '<html' in content[:1000].lower() or '<body' in content[:2000].lower():
        text = html_to_text(content)
        return text

    # If it's plain text / XML-ish
    if len(content) > 200:
        text = html_to_text(content)
        return text

    return None


def get_full_text(paper):
    """
    Try multiple sources to get full text for a paper.
    Returns (full_text, source_description) or (None, None).
    """
    doi = paper.get('doi', '')
    if not doi:
        return None, None

    # 1. OpenAlex
    text, src = try_openalex(doi)
    if text:
        return text, src
    time.sleep(0.3)

    # 2. Semantic Scholar
    text, src = try_semantic_scholar(doi)
    if text:
        return text, src
    time.sleep(0.3)

    # 3. Unpaywall
    text, src = try_unpaywall(doi)
    if text:
        return text, src
    time.sleep(0.3)

    # 4. Direct DOI fetch
    text, src = try_direct_doi(doi)
    if text:
        return text, src

    return None, None


def main():
    print("=" * 70)
    print("Full-Text Screening for IEEE & ACM Papers")
    print("=" * 70)

    # Load data
    with open(f'{BASE_DIR}/deduplicated_results.json') as f:
        data = json.load(f)

    all_papers = data.get('unique', []) + data.get('removed', [])

    # Find IEEE and ACM papers that were "passed through"
    target_papers = [
        p for p in all_papers
        if p.get('source_db_label') in ('IEEE Xplore', 'ACM Digital Library')
        and 'passed through' in (p.get('ft_status', '') or '').lower()
    ]

    print(f"\nFound {len(target_papers)} IEEE/ACM pass-through papers to screen")

    results = {
        'screened': [],       # Full text found and screened -> included
        'excluded': [],       # Full text found and screened -> excluded
        'unavailable': [],    # No full text available anywhere
    }

    for i, p in enumerate(target_papers):
        db = p.get('source_db_label', '')
        doi = p.get('doi', '')
        title = (p.get('title', '') or '')[:70]
        print(f"\n--- [{i+1}/{len(target_papers)}] {db}: {title}...")
        print(f"    DOI: {doi}")

        full_text, source = get_full_text(p)

        if full_text and len(full_text) > 500:
            print(f"    FULL TEXT FOUND ({len(full_text)} chars) via {source}")

            # Apply the same screening criteria as PubMed
            included, reason = fulltext_screen(p, full_text)

            if included:
                p['ft_status'] = f'Full text screened ({source})'
                p['ft_reason'] = reason
                results['screened'].append(p)
                print(f"    RESULT: INCLUDED - {reason}")
            else:
                p['ft_status'] = f'Full text excluded ({source})'
                p['ft_reason'] = reason
                p['exclusion_stage'] = 'Full-Text'
                p['exclusion_reason'] = reason
                results['excluded'].append(p)
                print(f"    RESULT: EXCLUDED - {reason}")
        else:
            print(f"    NO FULL TEXT AVAILABLE")
            p['ft_status'] = f'No full text available — passed through'
            p['ft_reason'] = f'{db}: full text not accessible from any source'
            results['unavailable'].append(p)
            print(f"    RESULT: PASSED THROUGH (no full text)")

        time.sleep(0.5)  # Rate limiting

    # Summary
    print(f"\n{'='*70}")
    print(f"SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"Total papers:    {len(target_papers)}")
    print(f"Full text found: {len(results['screened']) + len(results['excluded'])}")
    print(f"  Included:      {len(results['screened'])}")
    print(f"  Excluded:      {len(results['excluded'])}")
    print(f"No full text:    {len(results['unavailable'])}")

    if results['excluded']:
        print(f"\nExcluded papers:")
        for p in results['excluded']:
            print(f"  - {p.get('title','')[:70]}...")
            print(f"    Reason: {p.get('ft_reason','')}")

    if results['unavailable']:
        print(f"\nUnavailable (passed through):")
        for p in results['unavailable']:
            print(f"  - {p.get('title','')[:70]}...")

    # Save updated results back to JSON
    # Rebuild unique/removed lists with updated papers
    updated_doi_map = {p.get('doi'): p for p in target_papers if p.get('doi')}

    for lst_key in ['unique', 'removed']:
        for i, p in enumerate(data[lst_key]):
            doi = p.get('doi', '')
            if doi and doi in updated_doi_map:
                data[lst_key][i] = updated_doi_map[doi]

    out_path = f'{BASE_DIR}/deduplicated_results.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated: {out_path}")

    return results


if __name__ == '__main__':
    results = main()
