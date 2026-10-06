"""Turn the cleaned OCBC product table into one document per product.

Reads:  data/products_clean/products.json  (cleaned public API data, pulled 2026-10-05)
Writes: data/docs/<doc_id>.md               (human-readable copies, for inspection only)

Retrieval works on the in-memory dicts from build_documents(), not the .md files.
No customer data is involved here; every product fact comes from OCBC's public API.

Run from the project folder:  python -m src.data_gen
"""
import json
import os
import re

PRODUCTS_PATH = "data/products_clean/products.json"
DOCS_DIR = "data/docs"

# Text fields that become document sections, in reading order (most general first).
# key_risks is listed even though it's empty in this pull, so it appears automatically if a later pull fills it.
SECTION_TITLES = {
    "description": "Description",
    "overview": "Overview",
    "how_it_works": "How it works",
    "eligibility": "Eligibility",
    "initial_deposit": "Initial deposit",
    "minimum_balance": "Minimum balance",
    "fees_and_charges": "Fees and charges",
    "interest_rates": "Interest rates",
    "loan_amount": "Loan amount",
    "benefits": "Benefits",
    "key_risks": "Key risks",
    "remarks": "Remarks",
    "how_to_apply": "How to apply",
}

# Fields kept as metadata on the document, not as searchable sections.
# The disclaimer is the same boilerplate on every product, so as a section it would only add retrieval noise.
# It is still kept, and the freshness note (step 5) is built from these fields.
META_FIELDS = ["product_name", "category", "sub_category", "source_file",
               "pulled_on", "website_link", "disclaimer"]


def load_products(path=PRODUCTS_PATH):
    """Load the cleaned product list (one dict per product)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_doc_id(product_name):
    """Make a stable, filename-safe id from a product name, e.g. 'Bonus+ Savings Account' -> 'bonus-plus-savings-account'."""
    name = product_name.lower().replace("+", " plus ")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_document(product):
    """Build one document dict from a product: metadata plus the non-empty sections in a fixed order."""
    sections = {
        title: product[field].strip()
        for field, title in SECTION_TITLES.items()
        if (product.get(field) or "").strip()
    }
    doc = {"doc_id": make_doc_id(product["product_name"]), "sections": sections}
    doc.update({k: product.get(k, "") for k in META_FIELDS})
    return doc


def build_documents(products=None):
    """Build documents for every product. Raises on duplicate doc_ids so citations stay unambiguous."""
    products = load_products() if products is None else products
    docs = [build_document(p) for p in products]
    ids = [d["doc_id"] for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate doc_ids: {sorted(dupes)}")
    return docs


def render_markdown(doc):
    """Render a document as Markdown: a metadata header, then one '##' heading per section."""
    lines = [
        f"# {doc['product_name']}",
        "",
        f"- **Category:** {doc['category']}" + (f" / {doc['sub_category']}" if doc["sub_category"] and doc["sub_category"] != doc["category"] else ""),
        f"- **Source file:** {doc['source_file']}",
        f"- **Pulled on:** {doc['pulled_on']} (indicative; may be outdated)",
        f"- **Website:** {doc['website_link']}",
        f"- **Disclaimer:** {doc['disclaimer']}",
    ]
    for title, text in doc["sections"].items():
        lines += ["", f"## {title}", "", text]
    return "\n".join(lines) + "\n"


def write_documents(docs, out_dir=DOCS_DIR):
    """Write each document to <out_dir>/<doc_id>.md and return the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for doc in docs:
        path = os.path.join(out_dir, f"{doc['doc_id']}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_markdown(doc))
        paths.append(path)
    return paths


if __name__ == "__main__":
    docs = build_documents()
    paths = write_documents(docs)
    print(f"Built {len(docs)} documents, {sum(len(d['sections']) for d in docs)} sections -> {DOCS_DIR}/\n")
    for d in docs:
        print(f"  {d['doc_id']:<30} {len(d['sections']):>2} sections  {', '.join(d['sections'])}")
    print("\n--- sample: " + paths[1] + " ---\n")
    print(render_markdown(docs[1]))
