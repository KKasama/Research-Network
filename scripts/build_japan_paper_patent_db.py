#!/usr/bin/env python3
"""
build_japan_paper_patent_db.py
==============================

Step 1 of the Japan-only paper⇄patent network pre-build pipeline.

What this does:
  1. Calls the Lens.org Scholarly API for papers affiliated with the top 8
     Japanese research universities (旧帝大 + 東工大), filtered to only
     those cited by at least one patent.
  2. Sorts by patent_citations_count DESC and pulls the top N papers
     (default: 10,000).
  3. Resolves DOIs to OpenAlex Work IDs (via OpenAlex API) where the Lens
     response doesn't already include them.
  4. Persists everything to a local DuckDB file with two tables:
       - papers (one row per paper)
       - paper_patent_links (many-to-many: paper ⇄ citing patent)

Usage:
    export LENS_API_KEY="your_lens_token"
    python scripts/build_japan_paper_patent_db.py \
        --output data/japan_papers_patents.duckdb \
        --top-papers 10000 \
        --mailto your-email@example.org

Quick smoke test (100 papers, no OpenAlex enrichment):
    python scripts/build_japan_paper_patent_db.py \
        --output data/test.duckdb --top-papers 100 --no-openalex

Verify ROR IDs without doing a full build:
    python scripts/build_japan_paper_patent_db.py --verify-rors

References:
  - Lens Scholar API:  https://docs.api.lens.org/request-scholar.html
  - ROR registry:      https://ror.org
  - OpenAlex API:      https://docs.openalex.org/how-to-use-the-api/api-overview
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import requests

try:
    import duckdb
except ImportError:
    sys.exit(
        "duckdb not installed. Install with:\n"
        "    pip install duckdb requests\n"
    )

LOG = logging.getLogger("build_japan_db")

LENS_SCHOLAR_URL = "https://api.lens.org/scholarly/search"
OPENALEX_BASE = "https://api.openalex.org/works"

# ──────────────────────────────────────────────────────────────────────────
# Top 8 Japanese research universities (旧帝大 + 東工大).
# ROR IDs from https://ror.org. The ROR ID is the canonical, stable
# identifier; institution names may vary across affiliations.
# ──────────────────────────────────────────────────────────────────────────
TOP_INSTITUTIONS = [
    {"ror": "057zh3y96", "name_en": "University of Tokyo",           "name_ja": "東京大学"},
    {"ror": "02kpeqv85", "name_en": "Kyoto University",              "name_ja": "京都大学"},
    {"ror": "035t8zc32", "name_en": "Osaka University",              "name_ja": "大阪大学"},
    {"ror": "01dq60k83", "name_en": "Tohoku University",             "name_ja": "東北大学"},
    {"ror": "00p4k0j84", "name_en": "Kyushu University",             "name_ja": "九州大学"},
    {"ror": "04chrp450", "name_en": "Nagoya University",             "name_ja": "名古屋大学"},
    {"ror": "02e16g702", "name_en": "Hokkaido University",           "name_ja": "北海道大学"},
    {"ror": "0112mx960", "name_en": "Tokyo Institute of Technology", "name_ja": "東京工業大学"},
]


# ──────────────────────────────────────────────────────────────────────────
# Lens API: scroll-based pagination
# ──────────────────────────────────────────────────────────────────────────
def lens_scroll_search(
    api_key: str,
    query: dict,
    sort: list,
    include: list,
    size: int = 500,
) -> Iterator[dict]:
    """Iterate over all matching scholar records using cursor-based pagination."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "query": query,
        "sort": sort,
        "include": include,
        "size": size,
        "scroll": "1m",
    }
    scroll_id: Optional[str] = None
    page = 0

    while True:
        body = {"scroll_id": scroll_id, "scroll": "1m"} if scroll_id else payload
        r = requests.post(LENS_SCHOLAR_URL, headers=headers, json=body, timeout=120)
        page += 1

        if r.status_code == 204:
            LOG.info("Lens API returned 204 (no more results)")
            return
        if r.status_code == 429:
            LOG.warning("Rate limit hit, sleeping 60s")
            time.sleep(60)
            continue
        if r.status_code == 401:
            raise RuntimeError("Lens API key is invalid (401). Check LENS_API_KEY.")
        if r.status_code != 200:
            raise RuntimeError(
                f"Lens API error {r.status_code}: {r.text[:500]}"
            )

        data = r.json()
        items = data.get("data") or []
        total = data.get("total")
        if page == 1 and total is not None:
            LOG.info(f"Lens reports {total:,} total matching scholarly works")

        if not items:
            return

        for item in items:
            yield item

        scroll_id = data.get("scroll_id")
        if not scroll_id:
            return


# ──────────────────────────────────────────────────────────────────────────
# Top-papers fetch: Japan top-8 institutions × cited by patent
# ──────────────────────────────────────────────────────────────────────────
def _build_scope_query(
    ror_ids: list[str] | None,
    country_code: str | None,
) -> dict:
    """
    Build the Lens Scholar `query` clause.

    Pass either:
      - ror_ids: filter to authors affiliated with any of these ROR IDs
      - country_code: filter to authors with at least one institutional
        affiliation in the given country (e.g., "JP" for all-Japan mode)
    """
    must: list[dict] = [{"match": {"has_patent_citations": True}}]
    if ror_ids:
        must.append({"terms": {"author.affiliation.ror_id": ror_ids}})
    elif country_code:
        must.append({"term": {"author.affiliation.address.country_code": country_code}})
    else:
        raise ValueError("Either ror_ids or country_code must be provided.")
    return {"bool": {"must": must}}


def fetch_top_papers(
    api_key: str,
    ror_ids: list[str] | None = None,
    country_code: str | None = None,
    max_papers: int = 10000,
    page_size: int = 500,
) -> Iterator[dict]:
    """
    Yield top N papers (by patent citation count, descending) matching the
    given scope (either an explicit ROR list or a country_code).

    The function is a generator — caller must consume each item quickly to
    keep the Lens scroll context alive (1-minute TTL). With Phase-1-only
    work (normalise + DuckDB insert, no per-item HTTP), throughput is well
    under the TTL limit even on slow machines.
    """
    query = _build_scope_query(ror_ids, country_code)
    # NOTE: Lens has inconsistent naming between search and response fields.
    # Sort/search uses `referenced_by_patent_count` (the new recommended name;
    # the deprecated alias `patent_citation_count` (singular) also works), but
    # the response payload returns `patent_citations_count` (plural).
    sort = [{"referenced_by_patent_count": "desc"}]
    # Projection: response field names. `referenced_by_count` is NOT valid
    # for include — use scholarly_citations_count (matches Lens UI label).
    include = [
        "lens_id",
        "title",
        "year_published",
        "publication_type",
        "external_ids",
        "source",
        "authors",
        "patent_citations",
        "patent_citations_count",
        "scholarly_citations_count",
    ]

    seen = 0
    for item in lens_scroll_search(api_key, query, sort, include, size=page_size):
        if seen >= max_papers:
            return
        yield item
        seen += 1
        if seen % 5000 == 0:
            LOG.info(f"  Phase 1 Lens fetch: {seen:,} papers received...")


# ──────────────────────────────────────────────────────────────────────────
# Normalisation: Lens response → DB row
# ──────────────────────────────────────────────────────────────────────────
def normalise_paper(item: dict) -> dict:
    """Flatten a Lens scholar item into a row suitable for our `papers` table."""
    # External IDs (DOI / OpenAlex / PMID / etc.)
    doi = openalex_id = pmid = ""
    for ext in item.get("external_ids") or []:
        t = (ext.get("type") or "").lower()
        v = ext.get("value") or ""
        if t == "doi" and not doi:
            doi = v
        elif t == "openalex" and not openalex_id:
            openalex_id = v
        elif t == "pmid" and not pmid:
            pmid = v

    # Authors and affiliations
    authors = item.get("authors") or []
    first_author = ""
    if authors:
        first = authors[0]
        first_author = " ".join(
            x for x in [first.get("first_name"), first.get("last_name")] if x
        ).strip()

    institutions: set[str] = set()
    matched_rors: set[str] = set()
    for au in authors:
        for aff in au.get("affiliations") or []:
            name = aff.get("name")
            if name:
                institutions.add(name)
            # ROR is nested in `affiliations[].ids[]` as {type: "ror", value: "..."}
            # NOT under a top-level "ror_id" key.
            for id_ent in aff.get("ids") or []:
                if (id_ent.get("type") or "").lower() == "ror":
                    val = id_ent.get("value") or ""
                    if val:
                        matched_rors.add(val)

    # Citing patents
    cite_patents = item.get("patent_citations") or []
    patent_ids = [p.get("lens_id", "") for p in cite_patents if p.get("lens_id")]

    source = item.get("source") or {}

    return {
        "lens_id": item.get("lens_id") or "",
        "title": item.get("title") or "",
        "year": item.get("year_published"),
        "publication_type": item.get("publication_type") or "",
        "doi": doi,
        "openalex_id": openalex_id,
        "pmid": pmid,
        "source_title": source.get("title") or "",
        "first_author": first_author,
        "n_authors": len(authors),
        "institutions": "; ".join(sorted(institutions))[:2000],
        "ror_ids": "; ".join(sorted(matched_rors)),
        "patent_citation_count": item.get("patent_citations_count") or 0,
        "scholarly_citation_count": item.get("scholarly_citations_count") or 0,
        "_patent_lens_ids": patent_ids,
    }


# ──────────────────────────────────────────────────────────────────────────
# OpenAlex DOI → Work ID resolver (with simple in-memory cache)
# ──────────────────────────────────────────────────────────────────────────
class OpenAlexResolver:
    def __init__(self, mailto: str = ""):
        self.mailto = mailto
        self._cache: dict[str, str] = {}
        self._session = requests.Session()

    def resolve(self, doi: str) -> Optional[str]:
        if not doi:
            return None
        if doi in self._cache:
            return self._cache[doi]
        try:
            params = {"select": "id"}
            if self.mailto:
                params["mailto"] = self.mailto
            r = self._session.get(
                f"{OPENALEX_BASE}/doi:{doi}",
                timeout=30,
                params=params,
            )
            if r.status_code == 200:
                wid = (r.json().get("id") or "").replace("https://openalex.org/", "")
                self._cache[doi] = wid
                return wid
        except Exception as e:
            LOG.debug(f"OpenAlex resolve failed for {doi}: {e}")
        self._cache[doi] = ""
        return None

    def resolve_batch(self, dois: list[str]) -> dict[str, str]:
        """
        Bulk resolve up to 50 DOIs per request. Returns a mapping
        {original_doi -> openalex_work_id (or "")}.

        OpenAlex's Works endpoint supports a pipe-separated list of up to
        50 DOIs in a single filter. This is dramatically faster than the
        per-DOI form: ~7,000 batch requests cover ~352K DOIs in ~3 minutes
        on the polite-pool tier.
        """
        if not dois:
            return {}
        # Normalise DOIs: lowercase + strip URL prefix for matching
        norm_map: dict[str, str] = {}  # normalised -> original
        unique_norm: list[str] = []
        for d in dois:
            if not d:
                continue
            n = d.lower().replace("https://doi.org/", "").strip()
            if n and n not in norm_map:
                norm_map[n] = d
                unique_norm.append(n)

        out: dict[str, str] = {d: "" for d in dois}

        BATCH = 50
        for i in range(0, len(unique_norm), BATCH):
            chunk = unique_norm[i : i + BATCH]
            cached_only = [n for n in chunk if n in self._cache]
            for n in cached_only:
                out[norm_map[n]] = self._cache[n]
            uncached = [n for n in chunk if n not in self._cache]
            if not uncached:
                continue
            params = {
                "filter": "doi:" + "|".join(uncached),
                "per-page": 50,
                "select": "id,doi",
            }
            if self.mailto:
                params["mailto"] = self.mailto
            try:
                r = self._session.get(OPENALEX_BASE, params=params, timeout=60)
                if r.status_code != 200:
                    LOG.debug(
                        f"OpenAlex batch returned {r.status_code}: {r.text[:200]}"
                    )
                    # Mark all as resolved-but-empty so we don't retry
                    for n in uncached:
                        self._cache[n] = ""
                    continue
                results = r.json().get("results") or []
                returned: dict[str, str] = {}
                for w in results:
                    w_doi = (w.get("doi") or "").lower().replace(
                        "https://doi.org/", ""
                    ).strip()
                    w_id = (w.get("id") or "").replace(
                        "https://openalex.org/", ""
                    )
                    if w_doi and w_id:
                        returned[w_doi] = w_id
                for n in uncached:
                    wid = returned.get(n, "")
                    self._cache[n] = wid
                    out[norm_map[n]] = wid
            except Exception as e:
                LOG.debug(f"OpenAlex batch failed for {len(uncached)} DOIs: {e}")
                for n in uncached:
                    self._cache[n] = ""

        return out


# ──────────────────────────────────────────────────────────────────────────
# DuckDB schema and writer
# ──────────────────────────────────────────────────────────────────────────
SCHEMA_PAPERS = """
CREATE TABLE papers (
    lens_id                  VARCHAR PRIMARY KEY,
    title                    VARCHAR,
    year                     INTEGER,
    publication_type         VARCHAR,
    doi                      VARCHAR,
    openalex_id              VARCHAR,
    pmid                     VARCHAR,
    source_title             VARCHAR,
    first_author             VARCHAR,
    n_authors                INTEGER,
    institutions             VARCHAR,
    ror_ids                  VARCHAR,
    patent_citation_count    INTEGER,
    scholarly_citation_count INTEGER
)
"""

SCHEMA_LINKS = """
CREATE TABLE paper_patent_links (
    paper_lens_id  VARCHAR,
    patent_lens_id VARCHAR,
    PRIMARY KEY (paper_lens_id, patent_lens_id)
)
"""

SCHEMA_META = """
CREATE TABLE build_meta (
    built_at      TIMESTAMP,
    top_papers    INTEGER,
    institutions  VARCHAR,
    notes         VARCHAR
)
"""


def build_db(
    output_path: Path,
    paper_iter: Iterator[dict],
    institutions_label: str,
    top_papers: int,
    enrich_openalex: bool = True,
    mailto: str = "",
) -> tuple[int, int]:
    """
    Create a DuckDB at `output_path` and populate it in three phases:

      Phase 1  Stream Lens scholar items into `papers` and
               `paper_patent_links` (no per-item OpenAlex lookups, so the
               Lens scroll TTL is respected even for very large fetches).
      Phase 2  Bulk-resolve DOIs to OpenAlex Work IDs in batches of 50,
               UPDATE papers in place. ~5-10x faster than per-DOI calls
               and constant memory.
      Phase 3  Build indexes, write build_meta.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        LOG.info(f"Removing existing DB: {output_path}")
        output_path.unlink()

    con = duckdb.connect(str(output_path))
    # Lower memory overhead for the bulk-insert workload. DuckDB holds row
    # ordering metadata for each row in an active transaction; for our
    # workload (millions of small rows) this can balloon to many GB and
    # OOM the process. We don't care about insertion order, so disable it.
    try:
        con.execute("SET preserve_insertion_order = false")
    except Exception:
        pass
    con.execute(SCHEMA_PAPERS)
    con.execute(SCHEMA_LINKS)
    con.execute(SCHEMA_META)

    n_papers = 0
    n_links = 0
    n_oa_resolved = 0

    try:
        # ────────────── Phase 1: stream Lens → DB ──────────────
        LOG.info("Phase 1/3: streaming Lens results into DuckDB...")
        # IMPORTANT: use INSERT OR IGNORE rather than try/except. DuckDB's
        # MVCC transactions enter an aborted state on a constraint violation
        # within an active transaction; subsequent statements then fail with
        # 'TransactionContext Error: Current transaction is aborted'. INSERT
        # OR IGNORE silently skips the duplicate row without raising, which
        # keeps the transaction healthy.
        #
        # ALSO: commit periodically. A single open transaction across millions
        # of inserts retains undo information in memory and can OOM
        # (we observed ~12.7 GiB RAM usage on a 350K-paper run before this
        # fix). Periodic COMMIT releases that memory.
        COMMIT_EVERY = 5000  # papers per transaction window
        con.execute("BEGIN")
        n_seen = 0
        last_log = 0
        for raw in paper_iter:
            paper = normalise_paper(raw)
            if not paper["lens_id"]:
                continue

            con.execute(
                "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    paper["lens_id"], paper["title"], paper["year"],
                    paper["publication_type"], paper["doi"], paper["openalex_id"],
                    paper["pmid"], paper["source_title"], paper["first_author"],
                    paper["n_authors"], paper["institutions"], paper["ror_ids"],
                    paper["patent_citation_count"], paper["scholarly_citation_count"],
                ],
            )
            n_seen += 1

            # Dedupe patent IDs within this paper before insert (otherwise
            # the same (paper_lens_id, patent_lens_id) pair can be sent
            # multiple times).
            seen_pids: set[str] = set()
            for pid in paper["_patent_lens_ids"]:
                if not pid or pid in seen_pids:
                    continue
                seen_pids.add(pid)
                con.execute(
                    "INSERT OR IGNORE INTO paper_patent_links VALUES (?,?)",
                    [paper["lens_id"], pid],
                )

            # Periodic commit to bound memory.
            if n_seen % COMMIT_EVERY == 0:
                con.execute("COMMIT")
                con.execute("BEGIN")

            if n_seen - last_log >= 5000:
                LOG.info(f"  Phase 1: {n_seen:,} papers seen...")
                last_log = n_seen
        con.execute("COMMIT")

        # Authoritative counts from the DB (after dedupe by PK)
        n_papers = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        n_links = con.execute("SELECT COUNT(*) FROM paper_patent_links").fetchone()[0]
        LOG.info(
            f"Phase 1 complete: {n_papers:,} unique papers /"
            f" {n_links:,} unique paper-patent links"
            f" (seen {n_seen:,} Lens items total)"
        )

        # ────────────── Phase 2: batched OpenAlex ──────────────
        if enrich_openalex and n_papers > 0:
            todo_rows = con.execute(
                "SELECT lens_id, doi FROM papers "
                "WHERE doi <> '' AND (openalex_id IS NULL OR openalex_id = '')"
            ).fetchall()
            n_todo = len(todo_rows)
            LOG.info(
                f"Phase 2/3: batched OpenAlex DOI->WorkID resolution"
                f" for {n_todo:,} papers..."
            )

            resolver = OpenAlexResolver(mailto=mailto)
            BATCH = 50              # OpenAlex DOIs per HTTP request
            COMMIT_EVERY_BATCHES = 100  # commit every 100 batches (~5K papers)
            done = 0
            con.execute("BEGIN")
            for i in range(0, n_todo, BATCH):
                batch_rows = todo_rows[i : i + BATCH]
                dois = [r[1] for r in batch_rows]
                results = resolver.resolve_batch(dois)
                for lens_id, doi in batch_rows:
                    wid = results.get(doi, "")
                    if wid:
                        con.execute(
                            "UPDATE papers SET openalex_id = ? WHERE lens_id = ?",
                            [wid, lens_id],
                        )
                        n_oa_resolved += 1
                done += len(batch_rows)
                # Periodic commit to keep transaction memory bounded.
                if ((i // BATCH) + 1) % COMMIT_EVERY_BATCHES == 0:
                    con.execute("COMMIT")
                    con.execute("BEGIN")
                if (i // BATCH) % 20 == 0:
                    LOG.info(
                        f"  Phase 2: {done:,}/{n_todo:,} processed"
                        f" / {n_oa_resolved:,} resolved"
                    )
            con.execute("COMMIT")
            LOG.info(
                f"Phase 2 complete: {n_oa_resolved:,} OpenAlex Work IDs resolved"
            )
        else:
            LOG.info("Phase 2/3: skipped (--no-openalex or empty DB)")

        # ────────────── Phase 3: indexes + meta ────────────────
        LOG.info("Phase 3/3: building indexes and writing build metadata...")
        con.execute("CREATE INDEX idx_papers_doi          ON papers(doi)")
        con.execute("CREATE INDEX idx_papers_openalex     ON papers(openalex_id)")
        con.execute("CREATE INDEX idx_papers_year         ON papers(year)")
        con.execute("CREATE INDEX idx_papers_pcc          ON papers(patent_citation_count)")
        con.execute("CREATE INDEX idx_links_patent_lensid ON paper_patent_links(patent_lens_id)")

        con.execute(
            "INSERT INTO build_meta VALUES (now(), ?, ?, ?)",
            [
                top_papers,
                institutions_label,
                json.dumps(
                    {
                        "openalex_enrichment": enrich_openalex,
                        "openalex_resolved": n_oa_resolved,
                    },
                    ensure_ascii=False,
                ),
            ],
        )
    finally:
        con.close()

    LOG.info(
        f"DB built: {n_papers:,} papers, {n_links:,} paper-patent links"
        f" → {output_path}"
    )
    if enrich_openalex:
        LOG.info(f"  OpenAlex enrichment: {n_oa_resolved:,} resolved via DOI")
    return n_papers, n_links


# ──────────────────────────────────────────────────────────────────────────
# Standalone helpers
# ──────────────────────────────────────────────────────────────────────────
def verify_rors(ror_ids: list[str]) -> None:
    """Quick check: print each ROR ID with its registered name from ror.org.

    ROR API v2 (current) uses a different response schema than v1:
        v1: {"name": "...", "country": {"country_code": "..."}}
        v2: {"names": [{"value": "...", "types": ["ror_display"]}, ...],
             "locations": [{"geonames_details": {"country_code": "..."}}]}

    We try v2 first, fall back to v1 for older endpoints.
    """
    def parse_v2(d: dict) -> tuple[str, str]:
        # Find the display name
        display = ""
        for entry in d.get("names") or []:
            if "ror_display" in (entry.get("types") or []):
                display = entry.get("value", "")
                break
        if not display:  # fallback: first name
            names = d.get("names") or []
            if names:
                display = names[0].get("value", "")
        # Find country code
        country = ""
        for loc in d.get("locations") or []:
            cc = (loc.get("geonames_details") or {}).get("country_code")
            if cc:
                country = cc
                break
        return display or "?", country or "?"

    def parse_v1(d: dict) -> tuple[str, str]:
        name = d.get("name", "?")
        country = (d.get("country") or {}).get("country_code", "?")
        return name, country

    print("Verifying ROR IDs against the ROR registry...\n")
    for ror_id in ror_ids:
        url_v2 = f"https://api.ror.org/v2/organizations/{ror_id}"
        url_v1 = f"https://api.ror.org/organizations/{ror_id}"
        try:
            r = requests.get(url_v2, timeout=10)
            if r.status_code == 200:
                name, country = parse_v2(r.json())
                print(f"  OK  {ror_id}  -->  {name} ({country})")
                continue
            if r.status_code == 404:
                print(f"  FAIL {ror_id}  -->  HTTP 404 (ROR ID does not exist)")
                continue
            # Try v1 fallback for non-404 errors
            r = requests.get(url_v1, timeout=10)
            if r.status_code == 200:
                name, country = parse_v1(r.json())
                print(f"  OK  {ror_id}  -->  {name} ({country})  [v1]")
            else:
                print(f"  FAIL {ror_id}  -->  HTTP {r.status_code}")
        except Exception as e:
            print(f"  FAIL {ror_id}  -->  {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output",
        default="data/japan_papers_patents.duckdb",
        help="Path to output DuckDB file. Default: data/japan_papers_patents.duckdb",
    )
    p.add_argument(
        "--top-papers",
        type=int,
        default=10000,
        help="Maximum number of papers to fetch (default: 10,000)",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Lens API scroll page size (max 1000; default 500)",
    )
    p.add_argument(
        "--no-openalex",
        action="store_true",
        help="Skip OpenAlex DOI->WorkID resolution (faster but loses Work IDs)",
    )
    p.add_argument(
        "--mailto",
        default=os.environ.get("OPENALEX_MAILTO", ""),
        help="Email for OpenAlex polite pool (recommended).",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("LENS_API_KEY", ""),
        help="Lens API token. Defaults to LENS_API_KEY env var.",
    )
    p.add_argument(
        "--verify-rors",
        action="store_true",
        help="Just verify the configured ROR IDs against ror.org and exit.",
    )
    p.add_argument(
        "--all-japan",
        action="store_true",
        help=(
            "Switch from the default ROR-list scope (top 8 universities) "
            "to a country-wide scope using "
            "author.affiliation.address.country_code: 'JP'. "
            "This covers ALL Japanese-affiliated papers cited by patents "
            "(~352K total per Lens UI). Combine with a larger --top-papers."
        ),
    )
    p.add_argument(
        "--country",
        default="JP",
        help=(
            "Country code used by --all-japan (alpha-2; default JP). "
            "Useful if you want to repurpose this script for another country."
        ),
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ror_ids = [inst["ror"] for inst in TOP_INSTITUTIONS]

    if args.verify_rors:
        verify_rors(ror_ids)
        return 0

    if not args.api_key:
        sys.stderr.write(
            "ERROR: Lens API key required.\n"
            "  Set the env var (recommended):\n"
            "      export LENS_API_KEY=\"your_lens_token\"\n"
            "  Or pass it explicitly:\n"
            "      python build_japan_paper_patent_db.py --api-key your_lens_token ...\n"
            "  Get a key at https://www.lens.org/lens/user/subscriptions\n"
        )
        return 2

    if args.all_japan:
        institutions_label = (
            f"All institutions in country={args.country.upper()} "
            f"(via author.affiliation.address.country_code)"
        )
        LOG.info(
            f"Scope: ALL institutions in country '{args.country.upper()}' "
            f"(country-wide mode)"
        )
        scope_kwargs = {"country_code": args.country.upper(), "ror_ids": None}
    else:
        institutions_label = "; ".join(
            f"{i['name_en']} ({i['ror']})" for i in TOP_INSTITUTIONS
        )
        LOG.info("Scope: Top 8 Japanese research universities")
        for inst in TOP_INSTITUTIONS:
            LOG.info(
                f"  - {inst['name_ja']}  / {inst['name_en']}  (ROR: {inst['ror']})"
            )
        scope_kwargs = {"ror_ids": ror_ids, "country_code": None}

    LOG.info(f"Fetching top {args.top_papers:,} papers by patent citation count")
    LOG.info(f"OpenAlex enrichment: {'OFF' if args.no_openalex else 'ON (batched)'}")

    output = Path(args.output).expanduser().resolve()
    LOG.info(f"Output DuckDB: {output}")

    paper_iter = fetch_top_papers(
        args.api_key,
        max_papers=args.top_papers,
        page_size=args.page_size,
        **scope_kwargs,
    )

    n_papers, n_links = build_db(
        output,
        paper_iter,
        institutions_label,
        args.top_papers,
        enrich_openalex=not args.no_openalex,
        mailto=args.mailto,
    )

    print()
    print("=" * 60)
    print(f"DuckDB:        {output}")
    print(f"Papers:        {n_papers:,}")
    print(f"Patent links:  {n_links:,}")
    print("=" * 60)
    print()
    print("Quick query example (Python):")
    print(f"    import duckdb")
    print(f"    con = duckdb.connect('{output}', read_only=True)")
    print(f"    con.sql('SELECT title, patent_citation_count'")
    print(f"            ' FROM papers ORDER BY patent_citation_count DESC LIMIT 10').show()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
