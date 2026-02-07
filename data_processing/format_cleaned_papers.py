"""
- Reads papers_cleaned.jsonl, converts papers from raw S2ORC into JSON structure
- Reads the char-offset indices from S2ORC format into full text
- Turns offset and paragraph into section_title and text key:value pairs
- Save each paper as its own JSON file in data/ dir.
"""

import json
import os


def parse_annotation_list(data):
    if data is None:
        return []
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return data
    return []


def get_text_and_annotations(paper):
    """
    Extract full text and annotations from a paper. handles
    content.* and body.* S2ORC schemas
    """
    full_text = None
    section_headers = []
    paragraphs = []

    # body.text + body.annotations
    body = paper.get('body') or {}
    if body.get('text'):
        full_text = body['text']
        annotations = body.get('annotations', {})
        section_headers = parse_annotation_list(annotations.get('section_header'))
        paragraphs = parse_annotation_list(annotations.get('paragraph'))

    # content.text + content.annotations
    if full_text is None:
        content = paper.get('content') or {}
        if content.get('text'):
            full_text = content['text']
            annotations = content.get('annotations', {})
            section_headers = parse_annotation_list(annotations.get('sectionheader'))
            paragraphs = parse_annotation_list(annotations.get('paragraph'))

    return full_text, section_headers, paragraphs


def extract_sections(full_text, section_headers, paragraphs):
    """
    Grab readable sections from character-offset annotations.

    - paragraphs assigned to most recent preceding section header
    - paragraphs that appear before the first section header are under "Untitled" section.
    """
    if not full_text:
        return []

    # Sort by start position
    headers = sorted(section_headers, key=lambda x: int(x['start']))
    paras = sorted(paragraphs, key=lambda x: int(x['start']))

    # (title, position) pairs from header offsets
    header_info = []
    for h in headers:
        start, end = int(h['start']), int(h['end'])
        title = full_text[start:end].strip()
        # some journals include leading chars, so clean any '|'
        title = title.lstrip('|').strip()
        if title:
            header_info.append({'title': title, 'start': start})

    # walk through paragraphs, switch sections when a header is passed
    sections = []
    current_title = "Untitled"
    current_paragraphs = []
    header_idx = 0

    for p in paras:
        p_start = int(p['start'])
        p_end = int(p['end'])
        p_text = full_text[p_start:p_end].strip()

        if not p_text:
            continue

        #  passsection headers that precede curr paragraph
        while header_idx < len(header_info) and header_info[header_idx]['start'] <= p_start:
            # save the accumulated section then switching
            if current_paragraphs:
                sections.append({
                    'section_title': current_title,
                    'text': '\n\n'.join(current_paragraphs)
                })
                current_paragraphs = []
            current_title = header_info[header_idx]['title']
            header_idx += 1

        current_paragraphs.append(p_text)

    # save the final section
    if current_paragraphs:
        sections.append({
            'section_title': current_title,
            'text': '\n\n'.join(current_paragraphs)
        })

    return sections


def format_paper(paper):
    """
    schema:
    {
        "corpusid": int,
        "title": str,
        "authors": [str],
        "url": str | None,
        "license": str | None,
        "sections": [
            {"section_title": str, "text": str},
            ...
        ]
    }
    """
    full_text, section_headers, paragraphs = get_text_and_annotations(paper)
    sections = extract_sections(full_text, section_headers, paragraphs)

    formatted = {
        'corpusid': paper.get('corpusid'),
        'title': paper.get('title', 'Unknown Title'),
        'authors': paper.get('authors', []),
    }

    # include metadata
    oai = paper.get('openaccessinfo') or {}
    formatted['url'] = oai.get('url')
    formatted['license'] = oai.get('license')

    formatted['sections'] = sections

    return formatted


def main(input_file, output_dir):
    """
    Processing section headers and corresponding paragraphs to align with summarization model pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            paper = json.loads(line)
            formatted = format_paper(paper)

            corpusid = formatted['corpusid']

            num_sections = len(formatted['sections'])
            title_preview = formatted['title'][:60]
            # remove those with few section numbers:
            if num_sections < 4:
                print(f"[WARNING] Only {num_sections} sections found, skipping '{title_preview}...'\n")
                continue

            output_path = os.path.join(output_dir, f"{corpusid}.json")

            with open(output_path, 'w', encoding='utf-8') as out:
                json.dump(formatted, out, indent=2, ensure_ascii=False)

            print(f"[{count + 1}] {title_preview}... => data/{corpusid}.json ({num_sections} sections)\n")
            count += 1

    print(f"\nSuccessfully formatted {count} papers into {output_dir}/")


if __name__ == "__main__":
    # identifying root path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_dir = os.path.join(project_root, "data")
    input_file = os.path.join(data_dir, "papers_cleaned.jsonl")
    output_dir = data_dir

    main(input_file, output_dir)
