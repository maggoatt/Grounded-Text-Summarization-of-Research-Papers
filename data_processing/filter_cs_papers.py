import os
import requests
import json
import gzip
import urllib.request
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("S2_API_Key")

headers = {"x-api-key": api_key}

CS_KEYWORDS = [
    "computer science", "machine learning", "deep learning", "neural network",
    "artificial intelligence", "nlp", "natural language processing",
    "computer vision", "reinforcement learning",
    "distributed systems", "database", "databases", "operating system",
    "compiler", "algorithms", "data structures", "software engineering",
    "security", "cryptography", "networking", "systems", "hci"
]

def extract_title_and_text(paper):
    """
    Extract title and text from paper, handling both schema formats.

    Returns: (title, text, has_content)
    """
    # Schema 1: Direct fields (newer format)
    if 'title' in paper and 'body' in paper:
        title = paper.get('title', '')
        body = paper.get('body', {})
        if isinstance(body, dict):
            text = body.get('text', '')
        else:
            text = ''

        has_content = bool(title) and bool(text)
        return title, text, has_content

    # Schema 2: Nested in content.annotations (older format)
    elif 'content' in paper:
        content = paper.get('content', {})

        # Check if content has the text field
        text = content.get('text', '')

        # Title is in annotations
        annotations = content.get('annotations', {})
        title_annotations = annotations.get('title')

        # Title is usually an array of spans, extract if available
        if title_annotations and isinstance(title_annotations, list) and len(title_annotations) > 0:
            # Title spans usually have 'text' field
            title = ' '.join(span.get('text', '') for span in title_annotations if isinstance(span, dict))
        else:
            title = ''

        has_content = bool(title) and bool(text)
        return title, text, has_content

    # Unknown schema
    return '', '', False


def is_cs_paper(title, text, keywords):
    """Check if paper is CS-related based on title and text."""
    title_lower = title.lower()

    # High precision: check title first
    if any(k.lower() in title_lower for k in keywords):
        return True

    # Fallback: check first 5000 chars of text
    text_sample = text[:5000].lower()
    return any(k.lower() in text_sample for k in keywords)


def clean_paper(paper):
    """Remove unnecessary fields from paper to save space."""
    # Fields to keep in annotations
    useful_annotations = [
        'title',
        'abstract',
        'author',
        'authoraffiliation',
        'authorfirstname',
        'authorlastname',
        'sectionheader',
        'paragraph',
        'venue',
        'publisher'
    ]

    cleaned = {
        'corpusid': paper.get('corpusid'),
        'externalids': paper.get('externalids'),
        'content': {
            'text': paper.get('content', {}).get('text'),
            'annotations': {}
        }
    }

    # Only keep useful annotations
    original_annotations = paper.get('content', {}).get('annotations', {})
    for key in useful_annotations:
        if key in original_annotations and original_annotations[key] is not None:
            cleaned['content']['annotations'][key] = original_annotations[key]

    # Also copy openaccessinfo if it exists (from newer schema)
    if 'openaccessinfo' in paper:
        cleaned['openaccessinfo'] = paper['openaccessinfo']

    # For newer schema, also copy direct fields
    if 'title' in paper:
        cleaned['title'] = paper['title']
    if 'authors' in paper:
        cleaned['authors'] = paper['authors']
    if 'body' in paper:
        cleaned['body'] = paper['body']

    return cleaned


def write_papers_to_jsonl(papers, output_file):
    """Append cleaned papers to JSONL file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for paper in papers:
            cleaned_paper = clean_paper(paper)
            f.write(json.dumps(cleaned_paper, ensure_ascii=False) + '\n')


def main():
    # Fetch the latest release
    response_latest_release = requests.get(
        'https://api.semanticscholar.org/datasets/v1/release/latest',
        headers=headers
    )

    if response_latest_release.status_code != 200:
        print(f"Failed to fetch latest release. Status: {response_latest_release.status_code}")
        return

    latest_release_id = response_latest_release.json()['release_id']
    print(f"Latest Release ID: {latest_release_id}")

    # Fetch dataset files
    response_dataset = requests.get(
        f'https://api.semanticscholar.org/datasets/v1/release/{latest_release_id}/dataset/s2orc',
        headers=headers
    )

    if response_dataset.status_code != 200:
        print(f"Failed to fetch dataset. Status: {response_dataset.status_code}")
        return

    dataset_info = response_dataset.json()
    urls = dataset_info["files"]

    print(f"Total files available: {len(urls)}")
    print("Starting from the LAST file (better chance of full-text content)...\n")

    TARGET_PAPERS = 10  # Number of CS papers to find
    document_count = 0
    papers_checked = 0
    papers_with_content = 0
    papers_without_content = 0

    # Start from the END of the file list (better schemas)
    for url_index in range(len(urls) - 1, -1, -1):
        if document_count >= TARGET_PAPERS:
            break

        url = urls[url_index]
        print(f"\n{'='*80}")
        print(f"Processing file {len(urls) - url_index}/{len(urls)} (index {url_index})...")
        print(f"{'='*80}")

        written_this_url = 0

        try:
            with urllib.request.urlopen(url) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    for line_num, raw in enumerate(gz, 1):
                        if document_count >= TARGET_PAPERS:
                            break

                        # Progress indicator
                        if line_num % 1000 == 0:
                            print(f"  Checked {line_num} papers in this file...", end='\r')

                        line = raw.decode("utf-8").strip()
                        if not line:
                            continue

                        try:
                            paper = json.loads(line)
                            papers_checked += 1

                            # Extract title and text (handles both schemas)
                            title, text, has_content = extract_title_and_text(paper)

                            if not has_content:
                                papers_without_content += 1
                                continue

                            papers_with_content += 1

                            # Check if it's a CS paper
                            if is_cs_paper(title, text, CS_KEYWORDS):
                                print(f"\n✓ MATCH #{document_count + 1}: {title[:100]}")

                                write_papers_to_jsonl([paper], "cs_papers.jsonl")
                                document_count += 1
                                written_this_url += 1

                        except json.JSONDecodeError as e:
                            print(f"\n  JSON decode error on line {line_num}: {e}")
                            continue

            print(f"\n  File {url_index}: Wrote {written_this_url} CS papers")
            print(f"  Total CS papers found: {document_count}/{TARGET_PAPERS}")

        except Exception as e:
            print(f"\n  Failed to read URL {url_index}: {e}")
            continue

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total papers checked: {papers_checked}")
    print(f"Papers WITH content: {papers_with_content}")
    print(f"Papers WITHOUT content (null): {papers_without_content}")
    print(f"CS papers found and saved: {document_count}")
    print(f"Output file: cs_papers.jsonl")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
