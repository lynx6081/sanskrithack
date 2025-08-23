import re
import json
from bs4 import BeautifulSoup
from collections import defaultdict

def parse_and_merge_gretil(html_file, output_json="../rigveda_merged.json"):
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # regex to capture structure
    verse_pattern = re.compile(r"^RV_(\d{2})\.(\d{3})\.(\d{2})\.(\d)\{(\d{2})\}\s+(.*)$")

    merged = defaultdict(list)

    # iterate through all raw text nodes
    for text in soup.find_all(text=True):
        line = text.strip()
        if not line.startswith("RV_"):
            continue

        match = verse_pattern.match(line)
        if match:
            mandala, sukta, verse, line_no, varga, text_sa = match.groups()
            key = (mandala, sukta, verse)
            merged[key].append((int(line_no), text_sa))

    result = []
    for (mandala, sukta, verse), lines in merged.items():
        # sort lines numerically
        lines = sorted(lines, key=lambda x: x[0])
        full_text = " ".join([t for _, t in lines])

        result.append({
            "mandala": mandala,
            "sukta": sukta,
            "verse": verse,
            "text_sa": full_text.strip()
        })

    # save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted and merged {len(result)} verses → {output_json}")
    return result

# Rigveda Json implementation
parse_and_merge_gretil("rv_01_u.htm")
parse_and_merge_gretil("./rv_02_u.htm")
parse_and_merge_gretil("./rv_03_u.htm")
parse_and_merge_gretil("./rv_04_u.htm")
parse_and_merge_gretil("./rv_05_u.htm")
parse_and_merge_gretil("./rv_06_u.htm")
parse_and_merge_gretil("./rv_07_u.htm")
parse_and_merge_gretil("./rv_08_u.htm")
parse_and_merge_gretil("./rv_09_u.htm")
parse_and_merge_gretil("./rv_10_u.htm")



