import re
import csv
import pdfplumber

PDF_PATH = "Compendium of the Catechism of the Catholic Church.pdf"
CSV_PATH = "catechism_compendium.csv"

def extract_catechism():
    data = []
    current_section = ""
    current_chapter = ""
    current_question = None
    current_answer = []
    current_refs = ""

    # regex patterns
    q_pattern = re.compile(r"^(\d+)\.\s+(.*)")
    ref_pattern = re.compile(r"(\d+(-\d+)?(,\s*\d+(-\d+)?)*)$")

    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()

                # detect section/chapter headings
                if line.startswith("Part "):
                    current_section = line
                elif line.startswith("CHAPTER"):
                    current_chapter = line

                # detect a question
                match = q_pattern.match(line)
                if match:
                    # save previous Q&A
                    if current_question:
                        data.append([
                            q_num, current_question,
                            " ".join(current_answer).strip(),
                            current_refs, current_section, current_chapter
                        ])
                    # start new Q
                    q_num = match.group(1)
                    current_question = match.group(2).strip()
                    current_answer = []
                    current_refs = ""
                else:
                    # check if line looks like paragraph refs
                    if ref_pattern.match(line):
                        current_refs = line
                    else:
                        current_answer.append(line)

    # save the last Q
    if current_question:
        data.append([
            q_num, current_question,
            " ".join(current_answer).strip(),
            current_refs, current_section, current_chapter
        ])

    # write to CSV
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["QuestionNumber","Question","Answer","ParagraphRefs","Section","Chapter"])
        writer.writerows(data)

    print(f"✅ Extracted {len(data)} Q&A into {CSV_PATH}")

if __name__ == "__main__":
    extract_catechism()
