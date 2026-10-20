"""Two chunking strategies, evaluated side by side (docs/chunking_comparison.md).

A. recursive  — LangChain RecursiveCharacterTextSplitter (1000 / 200) over the page's clean text. The common baseline.
B. structured — one chunk per heading section (FAQ Q+A kept whole), one atomic chunk per price record, videos grouped,
                and every chunk prefixed with a context header (page, treatment/clinic, location, section path).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from placidway_bot.ingest.models import PageDoc, PriceRecord, Section

PAGE_TYPE_LABEL = {
    "clinic_profile": "Clinic profile",
    "package": "Treatment package",
    "clinic_videos": "Clinic videos",
    "price_list": "Clinic price list",
    "price_compare": "Price comparison (multiple clinics and countries)",
}
SECTION_MAX = 1500        # structured: split a section beyond this many characters
SECTION_MERGE = 700       # structured: merge consecutive tiny sibling sections up to this size
VIDEOS_PER_CHUNK = 4


@dataclass
class Chunk:
    chunk_id: str
    text: str                 # what is embedded / keyword-indexed / shown to the LLM
    slug: str
    source_url: str
    page_title: str
    page_type: str
    section: str
    kind: str                 # text | section | faq | price | video
    strategy: str
    treatment: str = ""
    clinic: str = ""
    location: str = ""
    content_hash: str = ""
    fetched_at: str = ""

    def metadata(self) -> dict:
        meta = asdict(self)
        meta.pop("text")
        return meta


def _base(doc: PageDoc, strategy: str) -> dict:
    return dict(slug=doc.slug, source_url=doc.url, page_title=doc.title, page_type=doc.page_type, strategy=strategy,
                treatment=doc.treatment or "", clinic=doc.clinic or "", location=doc.location or "",
                content_hash=doc.content_hash, fetched_at=doc.fetched_at)


def page_plain_text(doc: PageDoc) -> str:
    """The page as clean text in reading order — the input a basic pipeline would chunk."""
    parts = [doc.h1 or doc.title]
    parts += [f"{s.heading}\n{s.text}" for s in doc.sections]
    if doc.videos:
        parts.append("Videos:\n" + "\n".join(f"- {v.title} ({v.category}): {v.description}" for v in doc.videos))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- A. recursive
def chunk_recursive(doc: PageDoc, size: int = 1000, overlap: int = 200) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [Chunk(chunk_id=f"{doc.slug}::recursive::{i:03d}", text=piece, section="", kind="text",
                  **_base(doc, "recursive"))
            for i, piece in enumerate(splitter.split_text(page_plain_text(doc)))]


def chunk_plain_text(slug: str, url: str, title: str, page_type: str, text: str,
                     strategy: str = "recursive", size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Recursive chunking of unstructured text (used for the Chrome visible-text captures)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [Chunk(chunk_id=f"{slug}::{strategy}::{i:03d}", text=piece, slug=slug, source_url=url, page_title=title,
                  page_type=page_type, section="", kind="text", strategy=strategy)
            for i, piece in enumerate(splitter.split_text(text))]


# --------------------------------------------------------------------------- B. structured
def context_header(doc: PageDoc) -> str:
    subject = doc.treatment or doc.clinic or doc.h1 or doc.title
    where = f" | {doc.location}" if doc.location else ""
    clinic = f" | {doc.clinic}" if doc.clinic and doc.clinic != subject else ""
    return f"[{PAGE_TYPE_LABEL.get(doc.page_type, doc.page_type)}: {subject}{clinic}{where}]\nPage: {doc.title}"


def _section_path(sec: Section) -> str:
    return " > ".join([*sec.path, sec.heading])


def _is_faq(sec: Section) -> bool:
    return sec.heading.rstrip().endswith("?") and any("faq" in p.lower() or "frequently" in p.lower() for p in sec.path)


def _merge_small(sections: list[Section]) -> list[tuple[str, str, str]]:
    """(section path, text, kind) with tiny consecutive siblings (same parent path) merged together."""
    out: list[tuple[str, str, str]] = []
    for sec in sections:
        kind = "faq" if _is_faq(sec) else "section"
        # the "Section: <path>" line already names the heading, so the body is just the text
        text = f"Q: {sec.heading}\nA: {sec.text}" if kind == "faq" else sec.text
        if (kind == "section" and out and out[-1][2] == "section" and len(out[-1][1]) + len(text) < SECTION_MERGE
                and out[-1][0].rsplit(" > ", 1)[0] == _section_path(sec).rsplit(" > ", 1)[0]):
            # a merged sibling keeps its own heading so the boundary stays visible
            out[-1] = (out[-1][0], f"{out[-1][1]}\n\n{sec.heading}\n{text}", "section")
        else:
            out.append((_section_path(sec), text, kind))
    return out


def price_sentence(p: PriceRecord) -> str:
    price = p.price_text if p.currency in p.price_text else f"{p.price_text} {p.currency}"
    parts = [f"Price: {p.item}: {price} ({p.qualifier})."]
    if p.clinic:
        parts.append(f"Clinic: {p.clinic}.")
    if p.location:
        parts.append(f"Location: {p.location}.")
    if p.details:
        parts.append(f"Details: {p.details.rstrip('.')}.")
    parts.append(f"Page section: {p.section}.")
    return " ".join(parts)


def chunk_structured(doc: PageDoc) -> list[Chunk]:
    header = context_header(doc)
    base = _base(doc, "structured")
    chunks: list[Chunk] = []

    def add(text: str, section: str, kind: str) -> None:
        chunks.append(Chunk(chunk_id=f"{doc.slug}::structured::{len(chunks):03d}", text=f"{header}\n{text}",
                            section=section, kind=kind, **base))

    splitter = RecursiveCharacterTextSplitter(chunk_size=SECTION_MAX - 300, chunk_overlap=150)
    for path, text, kind in _merge_small(doc.sections):
        pieces = splitter.split_text(text) if len(text) > SECTION_MAX else [text]
        for piece in pieces:
            add(f"Section: {path}\n{piece}", path, kind)
    for p in doc.prices:
        add(price_sentence(p), p.section, "price")
    for i in range(0, len(doc.videos), VIDEOS_PER_CHUNK):
        group = doc.videos[i:i + VIDEOS_PER_CHUNK]
        lines = [f"- {v.title} ({v.category or 'video'}): {v.description} {v.youtube_url or ''}".rstrip() for v in group]
        add("Section: Patient and treatment videos\n" + "\n".join(lines), "Videos", "video")
    return chunks


STRATEGIES = {"recursive": chunk_recursive, "structured": chunk_structured}
