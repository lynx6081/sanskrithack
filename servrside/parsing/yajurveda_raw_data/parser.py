from bs4 import BeautifulSoup
import re
import json

# --- Load the HTML file ---
with open("maitrs_au.htm", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# --- Extract all text content ---
raw_text = soup.get_text(separator="\n")

# Regex to capture references of form //MS_1,1.1//
ref_pattern = re.compile(r"(.*?)//(MS_[0-9,\.]+)//")

# Keep track of current page marker
current_page = None
data = []

# Split by lines to process sequentially
for line in raw_text.split("\n"):
    line = line.strip()
    if not line:
        continue

    # Check for page markers like [Page I,1]
    page_match = re.search(r"\[Page (.*?)\]", line)
    if page_match:
        current_page = page_match.group(1)
        continue

    # Match reference and text
    ref_match = ref_pattern.search(line)
    if ref_match:
        text = ref_match.group(1).strip()
        ref = ref_match.group(2)

        entry = {
            "reference": ref,
            "page": current_page,
            "text": text
        }
        data.append(entry)

# --- Save to JSON ---
with open("../yajurveda_merged.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(data)} chunks into yajurveda_merged.json")
