import json

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


def clean_jsonl_file(input_file, output_file):
    """Clean all papers in a JSONL file."""

    total = 0

    # Calculate sizes
    original_size = 0
    cleaned_size = 0

    with open(input_file, 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                original_size += len(line)
                paper = json.loads(line)
                cleaned_paper = clean_paper(paper)
                cleaned_line = json.dumps(cleaned_paper, ensure_ascii=False) + '\n'
                cleaned_size += len(cleaned_line)
                fout.write(cleaned_line)
                total += 1

    print(f"Cleaned {total} papers")
    print(f"Original size: {original_size:,} bytes ({original_size / 1024 / 1024:.2f} MB)")
    print(f"Cleaned size: {cleaned_size:,} bytes ({cleaned_size / 1024 / 1024:.2f} MB)")
    print(f"Saved: {original_size - cleaned_size:,} bytes ({(1 - cleaned_size/original_size)*100:.1f}% reduction)")
    print(f"\nCleaned file saved to: {output_file}")

    # Show what was removed
    print("\n" + "="*80)
    print("REMOVED FIELDS:")
    print("="*80)
    print("- bibentry (bibliography entries)")
    print("- bibref (citation references)")
    print("- bibtitle (bibliography titles)")
    print("- bibvenue (bibliography venues)")
    print("- bibauthor, bibauthorfirstname, bibauthorlastname (bibliography authors)")
    print("- figure (figures)")
    print("- figurecaption (figure captions)")
    print("- figureref (figure references)")
    print("- table (tables)")
    print("- tableref (table references)")
    print("- formula (mathematical formulas)")
    print("- externalids (external database IDs)")

    print("\n" + "="*80)
    print("KEPT FIELDS:")
    print("="*80)
    print("- corpusid")
    print("- content.text (full paper text)")
    print("- title, abstract")
    print("- author, authoraffiliation, authorfirstname, authorlastname")
    print("- sectionheader (section titles)")
    print("- paragraph (body paragraphs)")
    print("- venue, publisher")


if __name__ == "__main__":
    clean_jsonl_file("papers.jsonl", "papers_cleaned.jsonl")
