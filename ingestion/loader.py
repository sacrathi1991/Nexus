"""
ingestion/loader.py

Extracts usable text from a PDF, handling three different kinds of content
that plain text extraction alone would mishandle:

1. Normal text          -> pdfplumber's extract_text() (accurate, fast)
2. Tables                -> pdfplumber's extract_tables(), converted into a
                             plain-text grid so numbers/columns survive into
                             the chunk instead of being read as a jumbled
                             sentence (this matters a lot for financial
                             reports, which are full of tables)
3. Scanned/image-only     -> if a page has no extractable text at all (a
   pages                    photographed/scanned page, common in older or
                             signed documents), OCR is attempted as a
                             fallback IF pytesseract + Tesseract are
                             installed. If not, that page is skipped with a
                             warning instead of crashing the whole pipeline —
                             OCR is optional, not a hard requirement, since
                             most corporate PDFs (like the sample set here)
                             are digitally generated and never hit this path.

Called from: services/ingestion_service.py, at the very start of ingest_pdf()
             — this is the FIRST thing that happens to a file, before any
             chunking or embedding.

Returns one combined text string per PDF — page texts and table text are
joined together in reading order, then handed to ingestion/chunker.py.
"""

import pdfplumber
# pdfplumber is a PDF-parsing library. Unlike a basic PDF reader, it keeps
# track of the position/layout of text on the page, which is what lets it
# detect tables (extract_tables()) separately from paragraph text
# (extract_text()), instead of reading everything as one flat stream.


def _table_to_text(table: list[list]) -> str:
    """Turn a pdfplumber table (list of rows) into a readable pipe-separated block.

    Called from: extract_pdf_text() below, once per table found on a page.

    `table` looks like: [["Year", "Revenue", "Profit"], ["2023", "1000", "200"], ...]
    i.e. a list of rows, each row a list of cell values (one per column).
    This function turns that into plain text like:
        [TABLE]
        Year | Revenue | Profit
        2023 | 1000 | 200
        [/TABLE]
    The [TABLE]/[/TABLE] markers aren't required by anything downstream —
    they're just a visual cue so that if you ever print/inspect a chunk,
    you can immediately tell "this chunk contains a table" vs. prose.
    """
    lines = []
    for row in table:
        # Table cells can legitimately be None (an empty cell in the PDF).
        # str(cell) would turn that into the literal text "None", which we
        # don't want appearing in the data — so empty cells become "" instead.
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        lines.append(" | ".join(cells))
    return "[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]"


def _ocr_page(pdf_path: str, page_number: int) -> str:
    """Fallback for a page with no extractable text: render it as an image and OCR it.
    Only runs if pytesseract + PyMuPDF are installed — otherwise skips gracefully.

    Called from: extract_pdf_text() below, ONLY when a page has neither
    normal text nor a table on it — i.e. it's very likely a scanned image
    of a page rather than a digitally-generated one.

    page_number is 0-indexed (page 1 of the PDF = page_number 0), matching
    how pdfplumber's `pdf.pages` list is indexed in extract_pdf_text().
    """
    try:
        # These three imports only happen INSIDE this function, not at the
        # top of the file. That's deliberate: if they're not installed,
        # only OCR breaks (and only when actually needed) — everything
        # else in this file keeps working normally.
        import fitz  # PyMuPDF: renders a PDF page to an image
        import pytesseract  # wraps the Tesseract OCR engine
        from PIL import Image  # Python's standard image-handling library
        import io  # lets us treat raw image bytes as an in-memory file
    except ImportError:
        print(f"  Page {page_number + 1} has no extractable text and OCR packages "
              f"aren't installed — skipping this page (pip install PyMuPDF pytesseract Pillow "
              f"and install the Tesseract engine to enable OCR).")
        return ""

    try:
        doc = fitz.open(pdf_path)          # open the PDF with PyMuPDF (separate from pdfplumber's own handle)
        page = doc[page_number]            # jump to the specific page that needs OCR
        pix = page.get_pixmap(dpi=200)     # render that page as a raster image (200 DPI = readable but not huge)
        image = Image.open(io.BytesIO(pix.tobytes("png")))  # turn the raw image bytes into a PIL Image object
        return pytesseract.image_to_string(image)            # run OCR: image -> plain text
    except Exception as e:
        # If OCR itself fails for any reason (corrupt page, Tesseract not
        # on PATH, etc.), don't crash the whole ingestion run over one page
        # — just log it and move on with an empty string for this page.
        print(f"  OCR fallback failed on page {page_number + 1}: {e}")
        return ""


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text (including tables, with an OCR fallback for image-only pages) from a PDF.

    This is the ONLY function other files should call from this module —
    _table_to_text() and _ocr_page() are internal helpers (the leading
    underscore is a Python convention meaning "not meant to be imported
    elsewhere").

    Called from: services/ingestion_service.py's ingest_pdf() function,
    as: text = extract_pdf_text(file_path)
    """
    page_texts = []  # will hold one string per page that has any content

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"    {total_pages} pages to process (table detection makes this the slowest step)...")
        # pdf.pages is a list of Page objects, one per page in the PDF, in order.
        # enumerate() gives us both the page and its index (i), because
        # _ocr_page() needs to know WHICH page number to render if OCR is needed.
        for i, page in enumerate(pdf.pages):
            if (i + 1) % 20 == 0 or (i + 1) == total_pages:
                # Printing on EVERY page would flood the console for a
                # 300+ page PDF, so this only prints every 20 pages (and
                # always on the last one) — just enough to prove it's
                # actively moving, not frozen.
                print(f"    ...page {i + 1}/{total_pages}")

            text = page.extract_text() or ""
            # extract_text() returns None (not "") if the page has zero
            # extractable text — the `or ""` guards against that so the
            # rest of this loop can safely treat `text` as a string.

            tables = page.extract_tables()
            # Returns a list of tables found on this page (usually 0 or 1,
            # but could be more if a page has multiple tables stacked).
            table_blocks = [_table_to_text(t) for t in tables if t]
            # `if t` skips any empty table pdfplumber occasionally detects
            # (e.g. a table-like grid of blank cells).

            if not text.strip() and not table_blocks:
                # No normal text AND no tables were found on this page at
                # all -> this page is almost certainly a scanned image, so
                # try OCR as a last resort.
                text = _ocr_page(pdf_path, i)

            # Combine this page's normal text with its table blocks (if any)
            # into one string, then trim leading/trailing whitespace.
            page_content = "\n\n".join([text] + table_blocks).strip()
            if page_content:
                # Only keep pages that actually produced something — a
                # completely blank page (e.g. a deliberate spacer page in
                # the PDF) contributes nothing and is skipped.
                page_texts.append(page_content)

    # Join every page's content together, separated by a blank line, into
    # one big string representing the ENTIRE document. This is what gets
    # handed to ingestion/chunker.py next.
    return "\n\n".join(page_texts)
