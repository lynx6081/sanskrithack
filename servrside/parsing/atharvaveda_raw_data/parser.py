import re, json
from bs4 import BeautifulSoup
from collections import defaultdict

def parse_atharvaveda(html_file, output_json="../atharvaveda_merged.json"):
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Regex: (AVŚ_book,hymn.verseLine) text
    # Example: (AVŚ_1,1.1a) yé triṣaptā́ḥ ...
    pattern = re.compile(r"^\(AVŚ_(\d+),(\d+)\.(\d+)([a-z])\)\s+(.*)$")

    merged = defaultdict(list)

    for text in soup.find_all(text=True):
        line = text.strip()
        if not line.startswith("(AVŚ_"):
            continue

        m = pattern.match(line)
        if m:
            book, hymn, verse, line_letter, text_sa = m.groups()
            key = (book, hymn, verse)
            merged[key].append((line_letter, text_sa))

    result = []
    for (book, hymn, verse), lines in merged.items():
        # sort by line letter (a, c, e...)
        lines = sorted(lines, key=lambda x: x[0])
        full_text = " ".join([t for _, t in lines])

        result.append({
            "book": book,
            "hymn": hymn,
            "verse": verse,
            "text_sa": full_text.strip()
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted and merged {len(result)} verses → {output_json}")
    return result

# Example usage:
parse_atharvaveda("avs_acu.htm")
