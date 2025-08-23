import re
import json
from bs4 import BeautifulSoup

# html_file = "./rv_01_u.htm"

def parse_gretil_html(html_file, output_json="./veda.json"):
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    verses = []
    verse_pattern = re.compile(r"^RV_(\d{2})\.(\d{3})\.(\d{2})\.(\d)\{(\d{2})\}\s+(.*)$")

    for br in soup.find_all(text=True):
        line = br.strip()
        if not line.startswith("RV_"):
            continue

        match = verse_pattern.match(line)
        if match:
            mandala, sukta, verse, line_no, varga, text_sa = match.groups()
            verses.append({
                "mandala": mandala,
                "sukta": sukta,
                "verse": verse,
                "line": line_no,
                "varga": varga,
                "text_sa": text_sa
            })

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted {len(verses)} verse lines → {output_json}")
    return verses

# Example usage:

parse_gretil_html("./rv_01_u.htm")
parse_gretil_html("./rv_02_u.htm")
parse_gretil_html("./rv_03_u.htm")
parse_gretil_html("./rv_04_u.htm")
parse_gretil_html("./rv_05_u.htm")
parse_gretil_html("./rv_06_u.htm")
parse_gretil_html("./rv_07_u.htm")
parse_gretil_html("./rv_08_u.htm")
parse_gretil_html("./rv_09_u.htm")
parse_gretil_html("./rv_10_u.htm")

