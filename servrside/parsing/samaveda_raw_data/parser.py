import re, json
from bs4 import BeautifulSoup
from collections import defaultdict

def parse_samaveda(html_file, output_json="../samaveda_merged.json"):
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # pattern: arcika prapathaka ardha dasati verse+line text
    pattern = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)\s+(\d{4})([a-z])\s+(.*)$")

    merged = defaultdict(list)

    for text in soup.find_all(text=True):
        line = text.strip()
        if not line or not line[0].isdigit():
            continue

        m = pattern.match(line)
        if m:
            arcika, prapathaka, ardha, dasati, line_letter, text_sa = m.groups()
            verse_num = dasati  # treat as verse id
            key = (arcika, prapathaka, ardha, verse_num)
            merged[key].append((line_letter, text_sa))

    result = []
    for (arcika, prapathaka, ardha, verse), lines in merged.items():
        # sort by line letter (a, c, e…)
        lines = sorted(lines, key=lambda x: x[0])
        full_text = " ".join([t for _, t in lines])

        result.append({
            "arcika": arcika,
            "prapathaka": prapathaka,
            "ardha": ardha,
            "verse": verse,
            "text_sa": full_text.strip()
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted and merged {len(result)} verses → {output_json}")
    return result

# parsing implementation:
parse_samaveda("samavedu.htm")
