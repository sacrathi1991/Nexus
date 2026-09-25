"""
run_replace.py

Manual trigger for document replacement (same role run_ingestion.py plays
for ingestion) — reads ingestion/replace_manifest.csv and calls
services.replace_service.replace_document() once per row. No manual
prompts: an admin just fills in the CSV/Excel and this handles the rest —
same pattern as ingestion/manifest.csv, on purpose.

Run from the project root:
    python run_replace.py

Later, an admin UI's "Replace this document" button will call the exact
same replace_document() function through an API route — this file gets
thrown away at that point; replace_document() won't need to change. The
new files are picked up from ingestion/replacement_pdfs/ — a SEPARATE
folder from ingestion/sample_pdfs/, which holds the originally-ingested
files. Keeping them apart avoids ever confusing "the file currently live
in the knowledge base" with "the file waiting to replace it". Swapping
this folder for S3 later means changing where new_file_path points,
nothing else.
"""

import csv
import os

from services.replace_service import replace_document

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "ingestion", "replace_manifest.csv")
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "ingestion", "replacement_pdfs")

with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    print("ingestion/replace_manifest.csv has no rows yet — nothing to replace.")
    print("Columns needed: old_filename,new_filename,company,year,doc_type")

for row in rows:
    new_file_path = os.path.join(PDF_FOLDER, row["new_filename"])

    if not os.path.exists(new_file_path):
        print(f"Skipping — new file not found: {new_file_path}")
        continue

    print(f"Replacing '{row['old_filename']}' with '{row['new_filename']}' ...")
    try:
        result = replace_document(
            old_filename=row["old_filename"],
            new_file_path=new_file_path,
            new_filename=row["new_filename"],
            company=row["company"],
            year=int(row["year"]),
            doc_type=row["doc_type"],
        )
        print(f"  Done: {result}\n")
    except ValueError as e:
        # replace_document() raises this if old_filename doesn't match any
        # existing document — printed and skipped, rather than crashing
        # the whole batch over one bad row.
        print(f"  Skipped: {e}\n")

print("All replace_manifest.csv rows processed.")
