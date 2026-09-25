"""
ingestion/run_ingestion.py

Manual trigger for the ingestion pipeline (Stage 1 — see Day 6 discussion).
Reads ingestion/manifest.csv, and for each row, calls
services.ingestion_service.ingest_pdf() with the matching PDF from
ingestion/sample_pdfs/.

Run from the project root:
    python -m ingestion.run_ingestion

Later, api/ingestion_router.py and an S3-event trigger will call the exact
same ingest_pdf() function — only how it gets triggered changes. THIS file
is the part that will eventually be thrown away / replaced; ingest_pdf()
itself won't need to change.
"""

import csv
import os

from services.ingestion_service import ingest_pdf

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.csv")
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "sample_pdfs")
# os.path.dirname(__file__) = the ingestion/ folder itself, regardless of
# which directory you happen to run the command from — same fragile-path
# fix already applied back in config.py, reused here for the same reason.

with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    # DictReader turns each CSV row into a dict using the header row as
    # keys, e.g. {"filename": "tcs_annual_report_2022-23.pdf", "company": "TCS", ...}
    # instead of a plain list of values — makes row["company"] below
    # possible instead of having to remember column positions by index.
    rows = list(reader)
    # Read all rows into a plain list now, while the file is still open —
    # DictReader is normally a one-pass iterator that empties itself as you
    # read it, and would otherwise be tied to the file staying open.

print(f"Found {len(rows)} entries in manifest.csv\n")

for row in rows:
    filename = row["filename"]
    file_path = os.path.join(PDF_FOLDER, filename)
    # Combines the folder path with this row's filename to get the full
    # path to the actual PDF file on disk, e.g.
    # ".../ingestion/sample_pdfs/tcs_annual_report_2022-23.pdf"

    if not os.path.exists(file_path):
        # Guards against a manifest.csv row whose PDF was never downloaded,
        # or was renamed/deleted — skip it with a message instead of
        # crashing the whole loop over every other file.
        print(f"Skipping {filename} — file not found in {PDF_FOLDER}")
        continue

    print(f"Processing {filename} ...")
    ingest_pdf(
        # This is the ONE call that does everything — hash check, load,
        # chunk, embed, save — all of that logic lives in
        # services/ingestion_service.py, not here. This file's only job is
        # to read the manifest and call ingest_pdf() once per row.
        file_path=file_path,
        filename=filename,
        company=row["company"],
        year=int(row["year"]),   # CSV values are always strings — must convert to int explicitly
        doc_type=row["doc_type"],
    )
    print()

print("All manifest entries processed.")
