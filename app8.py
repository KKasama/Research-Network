"""
Research Network Portal v2
──────────────────────────
ステップ1: データ収集 → ローカルJSON保存
ステップ2: 分析手法選択（KeyBERT / BERTopic / K-means / 共著 / 引用）
ステップ3: 可視化（アプリ内 / VOSviewer / Retina）
"""

import streamlit as st
import json, requests, os, re, logging, warnings
from collections import defaultdict
from pathlib import Path
import datetime
import xml.etree.ElementTree as ET

# ── モデルロード時の冗長ログを抑制 ──
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY",  "error")
os.environ.setdefault("HF_HUB_VERBOSITY",        "error")

for _lg in ("sentence_transformers", "sentence_transformers.models.Transformer",
            "transformers", "huggingface_hub", "huggingface_hub.utils",
            "huggingface_hub.file_download", "filelock"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
except Exception:
    pass

try:
    from huggingface_hub import utils as _hf_utils
    _hf_utils.logging.set_verbosity_error()
except Exception:
    pass

st.set_page_config(page_title="Research Network v2", page_icon="🔬", layout="wide")

# ── データ保存フォルダ ──
DATA_DIR = Path.home() / "research_data"
DATA_DIR.mkdir(exist_ok=True)

# ── 言語 ──
if "lang" not in st.session_state:
    st.session_state.lang = "ja"
lang = st.session_state.lang

def tl(ja, en):
    return ja if lang == "ja" else en

def _oa_email():
    """OpenAlex polite pool 用メールアドレスを session_state から取得する"""
    return st.session_state.get("openalex_email", "").strip() or "research@example.com"

# ── KeyBERTモデル定義 ──
KEYBERT_MODELS = {
    "SciBERT": (
        "allenai/scibert_scivocab_uncased",
        tl("📚 学術論文全般（推奨）", "📚 General academic papers (recommended)"),
    ),
    "BioBERT": (
        "dmis-lab/biobert-base-cased-v1.2",
        tl("🧬 生命科学・医療・心理・福祉系", "🧬 Life science / medicine / psychology"),
    ),
    "MiniLM (English)": (
        "all-MiniLM-L6-v2",
        tl("🌐 英語・一般（汎用・高速）", "🌐 English, general (fast)"),
    ),
    "Multilingual MiniLM": (
        "paraphrase-multilingual-MiniLM-L12-v2",
        tl("🌍 多言語・日本語対応", "🌍 Multilingual, Japanese supported"),
    ),
}

# BERTopic / K-means は汎用モデルのみ使用（ドメイン特化モデルは不適）
BERTOPIC_MODELS = {
    "MiniLM (English)": (
        "all-MiniLM-L6-v2",
        tl("🌐 英語・汎用（高速）", "🌐 English, general (fast)"),
    ),
    "Multilingual MiniLM": (
        "paraphrase-multilingual-MiniLM-L12-v2",
        tl("🌍 多言語対応", "🌍 Multilingual"),
    ),
    "MPNet (high accuracy)": (
        "all-mpnet-base-v2",
        tl("🎯 英語・高精度（低速）", "🎯 English, high accuracy (slow)"),
    ),
}

# ────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_works(filters, per_page=500, mailto=""):
    base = "https://api.openalex.org/works"
    fields = "id,title,publication_year,doi,authorships,cited_by_count,abstract_inverted_index,topics,concepts,referenced_works"
    works, cursor = [], "*"
    per_req = min(per_page, 200)
    _mailto = mailto or "research@example.com"
    while len(works) < per_page:
        params = {
            "filter": ",".join(filters),
            "per_page": per_req,
            "cursor": cursor,
            "select": fields,
            "mailto": _mailto
        }
        try:
            r = requests.get(base, params=params, timeout=20)
            data = r.json()
            batch = data.get("results", [])
            if not batch: break
            works.extend(batch)
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor: break
        except Exception as e:
            st.error("API error: " + str(e))
            break
    return works[:per_page]


@st.cache_data(ttl=3600)
def fetch_works_pubmed(query, max_papers=500):
    """PubMed E-utilities API から論文を取得し、OpenAlex 互換フォーマットに変換する"""
    # Step 1: esearch で PMID リストを取得
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    try:
        r = requests.get(search_url, params={
            "db": "pubmed",
            "term": query,
            "retmax": max_papers,
            "retmode": "json",
        }, timeout=20)
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        st.error("PubMed search error: " + str(e))
        return []

    if not ids:
        return []

    # Step 2: efetch で XML 詳細を取得（100件ずつ）
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    works = []
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        try:
            r2 = requests.get(fetch_url, params={
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "rettype": "abstract",
                "retmode": "xml",
            }, timeout=30)
            root = ET.fromstring(r2.text)
        except Exception as e:
            st.error("PubMed fetch error: " + str(e))
            continue

        for article in root.findall(".//PubmedArticle"):
            try:
                # PMID
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text.strip() if pmid_el is not None else ""

                # Title
                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else ""

                # Year
                year = None
                pub_date = article.find(".//PubDate")
                if pub_date is not None:
                    y_el = pub_date.find("Year")
                    if y_el is not None:
                        try:
                            year = int(y_el.text.strip())
                        except Exception:
                            year = None

                # Abstract
                abstract_parts = []
                for ab_text in article.findall(".//AbstractText"):
                    part = "".join(ab_text.itertext()).strip()
                    if part:
                        label = ab_text.get("Label", "")
                        if label:
                            abstract_parts.append(label + ": " + part)
                        else:
                            abstract_parts.append(part)
                abstract = " ".join(abstract_parts)

                # Authors（所属施設も取得）
                all_authors = article.findall(".//Author")
                authorships = []
                for pos_idx, author_el in enumerate(all_authors):
                    last = author_el.findtext("LastName", "")
                    fore = author_el.findtext("ForeName", "")
                    name = (last + " " + fore).strip() if last else fore
                    if not name:
                        collective = author_el.findtext("CollectiveName", "")
                        name = collective
                    if name:
                        position = "first" if pos_idx == 0 else (
                            "last" if pos_idx == len(all_authors) - 1 else "middle"
                        )
                        # 所属施設を取得
                        affil_list = []
                        for affil_info in author_el.findall(".//AffiliationInfo"):
                            affil_text = affil_info.findtext("Affiliation", "").strip()
                            if affil_text:
                                affil_list.append({"display_name": affil_text, "ror": None})
                        authorships.append({
                            "author": {"id": "", "display_name": name},
                            "author_position": position,
                            "institutions": affil_list,
                        })

                # DOI
                doi = ""
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = "https://doi.org/" + article_id.text.strip()
                        break

                # Journal
                journal = article.findtext(".//Journal/Title", "") or article.findtext(".//ISOAbbreviation", "")

                works.append({
                    "id": "pmid:" + pmid,
                    "title": title,
                    "publication_year": year,
                    "doi": doi,
                    "authorships": authorships,
                    "cited_by_count": 0,
                    "abstract_inverted_index": {},
                    "_abstract": abstract,
                    "referenced_works": [],
                    "topics": [],
                    "_source": "pubmed",
                    "_pmid": pmid,
                    "_journal": journal,
                })
            except Exception:
                continue

    return works


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_patents_lens(keyword, api_key, max_patents=50, inventor_filter="",
                       applicant_filter="", year_from=None, year_to=None,
                       ipc_code="", jurisdictions=None, doc_types=None, npl_only=True):
    """Lens.orgで特許検索 → NPL引用を含む特許リストを返す"""
    url = "https://api.lens.org/patent/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # bool queryでキーワード検索
    must_clauses = [
        {"query_string": {"query": keyword, "fields": ["title", "abstract", "claim"]}},
    ]
    if npl_only:
        must_clauses.append({"term": {"cites_npl": True}})
    if inventor_filter.strip():
        must_clauses.append({"match": {"inventor.name": inventor_filter.strip()}})
    if applicant_filter.strip():
        must_clauses.append({"match": {"applicant.name": applicant_filter.strip()}})
    if ipc_code.strip():
        must_clauses.append({
            "query_string": {
                "query": ipc_code.strip().rstrip("*") + "*",
                "fields": ["biblio.classifications_ipcr.symbol"],
            }
        })

    # 年範囲フィルタ
    if year_from or year_to:
        range_clause = {}
        if year_from:
            range_clause["gte"] = int(year_from)
        if year_to:
            range_clause["lte"] = int(year_to)
        must_clauses.append({"range": {"year_published": range_clause}})

    # 管轄（出願国）フィルタ
    if jurisdictions:
        must_clauses.append({"terms": {"jurisdiction": [j.upper() for j in jurisdictions]}})

    # 特許タイプフィルタ
    if doc_types:
        must_clauses.append({"terms": {"doc_type": doc_types}})

    payload = {
        "query": {"bool": {"must": must_clauses}},
        "size": min(max_patents, 100),
        "sort": [{"year_published": "desc"}],
        "include": [
            "lens_id",
            "biblio.invention_title",
            "biblio.parties",
            "biblio.references_cited",
            "biblio.classifications_ipcr",
            "abstract",
            "date_published",
            "year_published",
            "jurisdiction",
            "doc_type",
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 401:
            st.error(tl(
                "Lens.org APIキーが無効です。正しいキーを入力してください。",
                "Lens.org API key is invalid. Please enter a valid key."
            ))
            return []
        if r.status_code == 429:
            st.error(tl(
                "Lens.org APIのレート制限に達しました。しばらく待ってから再試行してください。",
                "Lens.org API rate limit reached. Please wait and try again."
            ))
            return []
        if r.status_code != 200:
            st.error(tl(
                f"Lens.org APIエラー (HTTP {r.status_code}): {r.text[:200]}",
                f"Lens.org API error (HTTP {r.status_code}): {r.text[:200]}"
            ))
            return []
        data = r.json()
    except Exception as e:
        st.error(tl(f"Lens.org接続エラー: {e}", f"Lens.org connection error: {e}"))
        return []

    results = data.get("data", [])
    patents = []
    for item in results:
        biblio = item.get("biblio", {})

        # Title (prefer English)
        title = ""
        for t in biblio.get("invention_title", []):
            if t.get("lang") == "en":
                title = t.get("text", "")
                break
        if not title:
            titles = biblio.get("invention_title", [])
            if titles:
                title = titles[0].get("text", "")

        # Abstract (prefer English)
        abstract = ""
        for ab in item.get("abstract", []):
            if ab.get("lang") == "en":
                abstract = ab.get("text", "")
                break
        if not abstract:
            abs_list = item.get("abstract", [])
            if abs_list:
                abstract = abs_list[0].get("text", "")

        # Inventors
        parties = biblio.get("parties", {})
        inventors = [
            inv.get("extracted_name", {}).get("value", "")
            for inv in parties.get("inventors", [])
            if inv.get("extracted_name", {}).get("value")
        ]

        # Applicants
        applicants = [
            app.get("extracted_name", {}).get("value", "")
            for app in parties.get("applicants", [])
            if app.get("extracted_name", {}).get("value")
        ]

        # IPC codes
        ipc_codes = [
            cl.get("symbol", "")
            for cl in biblio.get("classifications_ipcr", {}).get("classifications", [])
            if cl.get("symbol")
        ]

        # NPL citations（DOIはexternal_ids[type=doi]から取得 → なければテキストからregex）
        npl_citations = []
        doi_pattern = re.compile(r'10\.\d{4,9}/[^\s,;"\']+(?=[,\s"\']|$)')
        for cit in biblio.get("references_cited", {}).get("citations", []):
            nplcit = cit.get("nplcit")
            if nplcit:
                text = nplcit.get("text", "")
                lens_scholar_id = nplcit.get("lens_id", "")
                # external_idsからDOI取得（正式フィールド）
                doi = ""
                for ext in nplcit.get("external_ids", []):
                    if ext.get("type", "").lower() == "doi":
                        doi = ext.get("value", "")
                        break
                # fallback: テキストからDOI抽出
                if not doi:
                    m = doi_pattern.search(text)
                    if m:
                        doi = m.group(0).rstrip(".,;)")
                npl_citations.append({
                    "text": text,
                    "doi": doi,
                    "lens_scholar_id": lens_scholar_id,
                })

        patents.append({
            "lens_id": item.get("lens_id", ""),
            "title": title,
            "date_published": item.get("date_published", ""),
            "year_published": item.get("year_published", None),
            "jurisdiction": item.get("jurisdiction", ""),
            "doc_type": item.get("doc_type", ""),
            "inventors": inventors,
            "applicants": applicants,
            "ipc_codes": ipc_codes,
            "abstract": abstract,
            "npl_citations": npl_citations,
        })

    return patents


def resolve_npl_to_works(npl_list):
    """NPLリストのDOIをOpenAlexで照合し、論文メタデータを返す
    優先順位: 1) nplcit.external_ids[doi]  2) テキストからregex  3) lens_scholar_id経由
    """
    doi_pattern = re.compile(r'10\.\d{4,9}/[^\s,;"\']+(?=[,\s"\']|$)')
    seen_dois = set()
    works = []

    for npl in npl_list:
        doi = npl.get("doi", "")
        text = npl.get("text", "")
        # fallback: テキストからDOI抽出
        if not doi:
            m = doi_pattern.search(text)
            if m:
                doi = m.group(0).rstrip(".,;)")
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip().rstrip(".")
        if doi_clean and doi_clean not in seen_dois:
            seen_dois.add(doi_clean)
            try:
                r = requests.get(
                    f"https://api.openalex.org/works/https://doi.org/{doi_clean}",
                    params={"mailto": _oa_email()}, timeout=10
                )
                if r.status_code == 200:
                    w = r.json()
                    if w.get("id"):
                        works.append(w)
                        continue
            except Exception:
                pass
        # fallback: lens_scholar_id でOpenAlex external_id検索
        lens_sid = npl.get("lens_scholar_id", "")
        if lens_sid and lens_sid not in seen_dois:
            seen_dois.add(lens_sid)
            try:
                r = requests.get(
                    "https://api.openalex.org/works",
                    params={"filter": f"ids.lens:{lens_sid}", "mailto": _oa_email()},
                    timeout=10
                )
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    if results:
                        works.append(results[0])
            except Exception:
                pass

    return works


def build_patent_paper_network(patents, resolved_works):
    """特許→論文の引用ネットワークをVOSviewer JSON形式で構築"""
    # Build DOI lookup for resolved works
    doi_to_work = {}
    for w in resolved_works:
        doi = (w.get("doi") or "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip("/")
        if doi:
            doi_to_work[doi] = w

    doi_pattern = re.compile(r'10\.\d{4,9}/[^\s,;"\']+(?=[,\s"\']|$)')

    items = []
    links = []
    node_id_counter = [1]

    # Patent nodes (cluster=1)
    patent_node_map = {}  # lens_id -> node_id
    for pat in patents:
        nid = node_id_counter[0]
        node_id_counter[0] += 1
        patent_node_map[pat["lens_id"]] = nid
        npl_count = len(pat.get("npl_citations", []))
        year_str = (pat.get("date_published") or "")[:4]
        label = (pat.get("title") or pat["lens_id"])[:60]
        if year_str:
            label = f"{label} ({year_str})"
        item = {
            "id": nid,
            "label": label,
            "cluster": 1,
            "weights": {"NPL Citations": npl_count},
            "description": "Patent: " + pat["lens_id"],
        }
        items.append(item)

    # Paper nodes (cluster=2)
    paper_node_map = {}  # doi -> node_id
    for w in resolved_works:
        doi = (w.get("doi") or "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip("/")
        if not doi or doi in paper_node_map:
            continue
        nid = node_id_counter[0]
        node_id_counter[0] += 1
        paper_node_map[doi] = nid
        title = (w.get("title") or doi)[:60]
        year = w.get("publication_year")
        if year:
            title = f"{title} ({year})"
        item = {
            "id": nid,
            "label": title,
            "cluster": 2,
            "weights": {"Citations": w.get("cited_by_count", 0) or 0},
            "description": "Paper: " + (w.get("doi") or ""),
        }
        if year:
            item["scores"] = {"Year": int(year)}
        if w.get("doi"):
            item["url"] = w["doi"]
        items.append(item)

    # Patent→Paper edges
    seen_links = set()
    patent_to_dois = {}  # lens_id -> set of dois cited
    for pat in patents:
        cited_dois = set()
        for npl in pat.get("npl_citations", []):
            doi = npl.get("doi", "")
            if not doi:
                m = doi_pattern.search(npl.get("text", ""))
                if m:
                    doi = m.group(0).rstrip(".,;)")
            doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip("/")
            if doi_clean and doi_clean in paper_node_map:
                cited_dois.add(doi_clean)
                src = patent_node_map[pat["lens_id"]]
                tgt = paper_node_map[doi_clean]
                edge_key = (src, tgt)
                if edge_key not in seen_links:
                    seen_links.add(edge_key)
                    links.append({"source_id": src, "target_id": tgt, "strength": 1})
        patent_to_dois[pat["lens_id"]] = cited_dois

    # Paper→Paper edges via bibliographic coupling (shared patent citations)
    paper_dois = list(paper_node_map.keys())
    doi_to_patents = defaultdict(set)
    for lens_id, dois in patent_to_dois.items():
        for doi in dois:
            doi_to_patents[doi].add(lens_id)

    paper_cooccur = defaultdict(int)
    for doi1 in paper_dois:
        for doi2 in paper_dois:
            if doi1 >= doi2:
                continue
            shared = len(doi_to_patents[doi1] & doi_to_patents[doi2])
            if shared >= 1:
                paper_cooccur[(doi1, doi2)] = shared

    for (doi1, doi2), strength in paper_cooccur.items():
        src = paper_node_map[doi1]
        tgt = paper_node_map[doi2]
        edge_key = (min(src, tgt), max(src, tgt))
        if edge_key not in seen_links:
            seen_links.add(edge_key)
            links.append({"source_id": src, "target_id": tgt, "strength": strength})

    return {"items": items, "links": links}


def reconstruct_abstract(inv_index, work=None):
    """OpenAlex inverted index から abstract を再構成する。
    PubMed論文の場合は _abstract フィールドを直接返す。"""
    if work is not None and work.get("_abstract"):
        return work["_abstract"]
    if not inv_index: return ""
    pos_word = {}
    for word, positions in inv_index.items():
        for p in positions:
            pos_word[p] = word
    return " ".join(pos_word[p] for p in sorted(pos_word.keys()))

def save_dataset(name, works, meta):
    """論文データをローカルJSONに保存"""
    path = DATA_DIR / (name + ".json")
    payload = {
        "name": name,
        "saved_at": datetime.datetime.now().isoformat(),
        "meta": meta,
        "works": works
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def load_dataset(name):
    path = DATA_DIR / (name + ".json")
    if not path.exists(): return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def list_datasets():
    return sorted([p.stem for p in DATA_DIR.glob("*.json")])

# ────────────────────────────────────────────
# 引用トレンド取得（OpenAlex API）
# ────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_citing_works_full(work_id: str, max_papers: int = 200, mailto: str = ""):
    """
    指定論文を引用している論文を、書誌結合に必要な referenced_works を含む
    完全データで取得する（被引用数順・最大 max_papers 件）。
    """
    short_id = work_id.rstrip("/").split("/")[-1]
    fields = ("id,title,publication_year,doi,authorships,"
              "cited_by_count,referenced_works,topics")
    results, cursor = [], "*"
    per_req = min(max_papers, 200)
    _mailto = mailto or "research@example.com"
    while len(results) < max_papers:
        params = {
            "filter":   f"cites:{short_id}",
            "sort":     "cited_by_count:desc",
            "per_page": per_req,
            "cursor":   cursor,
            "select":   fields,
            "mailto":   _mailto,
        }
        try:
            r = requests.get("https://api.openalex.org/works",
                             params=params, timeout=20)
            data  = r.json()
            batch = data.get("results", [])
            if not batch:
                break
            results.extend(batch)
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        except Exception:
            break
    return results[:max_papers]

# ────────────────────────────────────────────
# 中心性計算
# ────────────────────────────────────────────
def compute_centrality(vos_data):
    try:
        import networkx as nx
    except ImportError:
        return {}
    items = vos_data.get("items", [])
    links = vos_data.get("links", [])
    if not items or not links:
        return {}
    id_to_label = {item["id"]: item["label"] for item in items}
    G = nx.Graph()
    for item in items:
        G.add_node(item["id"])
    for link in links:
        src = link.get("source_id") or link.get("source")
        tgt = link.get("target_id") or link.get("target")
        w   = float(link.get("strength", 1))
        if src and tgt and src in id_to_label and tgt in id_to_label:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += w
            else:
                G.add_edge(src, tgt, weight=w)
    if len(G.nodes) == 0:
        return {}
    degree      = nx.degree_centrality(G)
    try:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes}
    try:
        pagerank    = nx.pagerank(G, weight="weight", max_iter=200)
    except Exception:
        pagerank    = {n: 0.0 for n in G.nodes}
    result = {}
    for nid in G.nodes:
        label = id_to_label.get(nid, nid)
        result[label] = {
            "degree":      round(degree.get(nid, 0), 4),
            "betweenness": round(betweenness.get(nid, 0), 4),
            "pagerank":    round(pagerank.get(nid, 0), 4),
        }
    return result

# ────────────────────────────────────────────
# ネットワーク構築
# ────────────────────────────────────────────
def build_coauth(works, min_links=2):
    author_info, work_authors = {}, []
    for w in works:
        auths = []
        for a in w.get("authorships", []):
            aid  = a.get("author", {}).get("id", "") or ""
            name = a.get("author", {}).get("display_name", "") or ""
            # PubMedはIDが空 → 著者名をキーとして使用
            key  = aid if aid else name
            if key and name:
                author_info[key] = name
                auths.append(key)
        work_authors.append(auths)
    links = defaultdict(int)
    doc_count = defaultdict(int)
    for auths in work_authors:
        for a in auths: doc_count[a] += 1
        for i in range(len(auths)):
            for j in range(i+1, len(auths)):
                links[(min(auths[i], auths[j]), max(auths[i], auths[j]))] += 1
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a,b),s in links.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    items = [{"id": aid, "label": author_info[aid], "weights": {"Documents": doc_count[aid]}} for aid in connected]
    return {"items": items, "links": link_list}

def build_keyword_cooccurrence(works, work_keywords, min_links=2):
    kw_count = defaultdict(int)
    work_kws = []
    for w in works:
        kws = work_keywords.get(w.get("id",""), [])
        work_kws.append(kws)
        for k in kws: kw_count[k] += 1
    links = defaultdict(int)
    for kws in work_kws:
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                pair = (min(kws[i], kws[j]), max(kws[i], kws[j]))
                links[pair] += 1
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a,b),s in links.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    items = [{"id": k, "label": k, "weights": {"Occurrences": kw_count[k]}} for k in connected]
    return {"items": items, "links": link_list}

def build_citation_network(works, citation_type="bibliographic_coupling", min_links=1):
    """書誌結合 or 直接引用ネットワークを構築（ノードIDはDOI優先）"""
    from collections import defaultdict
    work_info = {w.get("id",""): w for w in works}
    work_ids  = set(work_info.keys())

    def to_node_id(wid):
        """OpenAlex IDをDOI IDに変換（なければOpenAlex IDを使用）"""
        w = work_info.get(wid, {})
        doi = (w.get("doi", "") or "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip("/")
        return doi if doi else wid

    if citation_type == "bibliographic_coupling":
        refs = {w.get("id",""): set(w.get("referenced_works",[])) for w in works}
        wlist = list(work_ids)
        links = defaultdict(int)
        for i in range(len(wlist)):
            for j in range(i+1, len(wlist)):
                shared = len(refs[wlist[i]] & refs[wlist[j]])
                if shared >= min_links:
                    links[(wlist[i], wlist[j])] = shared
        link_list = [{"source_id": to_node_id(a), "target_id": to_node_id(b), "strength": s}
                     for (a,b), s in links.items()]
    else:  # direct_citation
        seen, link_list = set(), []
        for w in works:
            wid = w.get("id","")
            for ref in w.get("referenced_works",[]):
                if ref in work_ids and ref != wid:
                    src, tgt = to_node_id(wid), to_node_id(ref)
                    if (src, tgt) not in seen:
                        seen.add((src, tgt))
                        link_list.append({"source_id": src, "target_id": tgt, "strength": 1})

    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    doi_to_work = {to_node_id(w.get("id","")): w for w in works}

    # VOSviewerはid=整数必須のため、DOI文字列→整数にマッピング
    doi_to_int = {nid: idx for idx, nid in enumerate(sorted(connected), 1)}

    items = []
    for nid, int_id in doi_to_int.items():
        w = doi_to_work.get(nid, {})
        year    = w.get("publication_year") or ""
        title   = (w.get("title","") or nid)[:50]
        doi_url = w.get("doi", "") or ""
        item = {
            "id":      int_id,
            "label":   f"{title} ({year})" if year else title,
            "weights": {"Citations": w.get("cited_by_count", 0)},
        }
        if year:
            item["scores"] = {"Year": int(year)}
        if doi_url:
            item["url"] = doi_url
        items.append(item)

    link_list_int = [
        {"source_id": doi_to_int[l["source_id"]],
         "target_id": doi_to_int[l["target_id"]],
         "strength":  l["strength"]}
        for l in link_list
        if l["source_id"] in doi_to_int and l["target_id"] in doi_to_int
    ]
    return {"items": items, "links": link_list_int}


# ────────────────────────────────────────────
# 引用系譜（Citation Genealogy）
# ────────────────────────────────────────────
def fetch_referenced_works_batch(work_ids, mailto=""):
    """OpenAlex ID リストから論文データをバッチ取得（最大50件/リクエスト）"""
    if not work_ids:
        return {}
    results = {}
    id_list = [wid for wid in work_ids if wid]
    chunk_size = 50
    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i:i + chunk_size]
        short_ids = []
        for wid in chunk:
            m = re.search(r'(W\d+)', str(wid))
            if m:
                short_ids.append(m.group(1))
        if not short_ids:
            continue
        params = {
            "filter": f"openalex:{'|'.join(short_ids)}",
            "per-page": chunk_size,
            "select": "id,title,publication_year,doi,cited_by_count,authorships,referenced_works",
        }
        if mailto:
            params["mailto"] = mailto
        try:
            resp = requests.get("https://api.openalex.org/works", params=params, timeout=25)
            if resp.ok:
                for w in resp.json().get("results", []):
                    wid = w.get("id", "")
                    if wid:
                        results[wid] = w
        except Exception:
            pass
    return results


def build_citation_genealogy(seed_works, generations=3, max_per_gen=20, mailto=""):
    """
    seed_works を第0世代として referenced_works を辿り最大 N 世代の系譜を構築。
    Returns: (nodes, edges)
      nodes : {id -> {label, title, gen, year}}
      edges : [(source_id, target_id)]
    """
    all_nodes  = {}   # id -> dict
    all_edges  = []   # [(src, tgt)]
    work_cache = {}   # id -> work data

    # ─ Gen 0 ─
    for w in seed_works:
        wid = w.get("id", "") or ""
        if not wid:
            continue
        title = (w.get("title", "") or wid)
        year  = str(w.get("publication_year") or "")
        label = (f"{title[:45]}… ({year})" if len(title) > 45 else f"{title} ({year})") if year else title[:50]
        all_nodes[wid]  = {"label": label, "title": title, "gen": 0, "year": year}
        work_cache[wid] = w

    current_layer = set(all_nodes.keys())

    for gen in range(1, generations + 1):
        if not current_layer:
            break

        # Collect referenced IDs from current layer (limit per paper)
        next_ref_ids = set()
        for wid in current_layer:
            w = work_cache.get(wid, {})
            refs = w.get("referenced_works", [])[:max_per_gen]
            for ref in refs:
                if ref:
                    all_edges.append((wid, ref))
                    if ref not in all_nodes:
                        next_ref_ids.add(ref)

        if not next_ref_ids:
            break

        # Fetch next generation details
        fetched = fetch_referenced_works_batch(next_ref_ids, mailto)
        work_cache.update(fetched)

        next_layer = set()
        for ref_id in next_ref_ids:
            w = fetched.get(ref_id, {})
            title = (w.get("title", "") or ref_id) if w else ref_id
            year  = str(w.get("publication_year") or "") if w else ""
            if w:
                label = (f"{title[:45]}… ({year})" if len(title) > 45 else f"{title} ({year})") if year else title[:50]
                next_layer.add(ref_id)
            else:
                m = re.search(r'W(\d+)', str(ref_id))
                label = f"W{m.group(1)}" if m else ref_id
            all_nodes[ref_id] = {"label": label, "title": title, "gen": gen, "year": year}

        current_layer = next_layer

    return all_nodes, all_edges


def render_genealogy_pyvis(all_nodes, all_edges):
    """引用系譜をPyVisで有向グラフとして描画。世代ごとに色分け。"""
    try:
        from pyvis.network import Network
        import tempfile

        GEN_COLORS = {0: "#4477CC", 1: "#44AA66", 2: "#CC9922", 3: "#CC4444"}

        net = Network(
            height="700px", width="100%",
            bgcolor="#f9f9f9", font_color="#333333",
            directed=True
        )
        net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=150)

        # Add nodes
        node_ids_in_graph = set(all_nodes.keys())
        for nid, info in all_nodes.items():
            gen   = info.get("gen", 0)
            color = GEN_COLORS.get(gen, "#aaaaaa")
            size  = max(8, 28 - gen * 5)
            label = info.get("label", nid)[:30]
            title = (
                f"<b>Gen {gen}</b><br>"
                f"{info.get('title', '')[:80]}<br>"
                f"Year: {info.get('year', '—')}"
            )
            net.add_node(str(nid), label=label, size=size, color=color, title=title)

        # Add edges (only between nodes that exist in the graph)
        seen_edges = set()
        for src, tgt in all_edges:
            if src in node_ids_in_graph and tgt in node_ids_in_graph:
                key = (str(src), str(tgt))
                if key not in seen_edges:
                    seen_edges.add(key)
                    net.add_edge(str(src), str(tgt), arrows="to", color="#aaaaaa", width=1)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            html_content = open(f.name).read()

        st.components.v1.html(html_content, height=720, scrolling=False)

    except ImportError:
        st.warning(tl(
            "pyvis 未インストール。`pip install pyvis` を実行してください。",
            "pyvis not installed. Run `pip install pyvis`."
        ))


def fetch_citing_works(work_id, max_results=20, mailto=""):
    """work_id を引用している論文を被引用数順で取得（OpenAlex cites フィルタ）"""
    m = re.search(r'(W\d+)', str(work_id))
    if not m:
        return []
    params = {
        "filter":   f"cites:{m.group(1)}",
        "sort":     "cited_by_count:desc",
        "per-page": min(max_results, 50),
        "select":   "id,title,publication_year,doi,cited_by_count,authorships,referenced_works",
    }
    if mailto:
        params["mailto"] = mailto
    try:
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=25)
        if resp.ok:
            return resp.json().get("results", [])
    except Exception:
        pass
    return []


def build_forward_citation_genealogy(seed_works, generations=3, max_per_gen=20, mailto=""):
    """
    seed_works を第0世代として、被引用論文（自分を引用している論文）を世代ごとに辿る。
    Returns: (nodes, edges)
      nodes : {id -> {label, title, gen, year}}
      edges : [(citing_id, cited_id)]  ← 引用している論文 → 引用されている論文
    """
    all_nodes  = {}
    all_edges  = []

    # ─ Gen 0 ─
    for w in seed_works:
        wid = w.get("id", "") or ""
        if not wid:
            continue
        title = (w.get("title", "") or wid)
        year  = str(w.get("publication_year") or "")
        label = (f"{title[:45]}… ({year})" if len(title) > 45 else f"{title} ({year})") if year else title[:50]
        all_nodes[wid] = {"label": label, "title": title, "gen": 0, "year": year}

    current_layer = set(all_nodes.keys())

    for gen in range(1, generations + 1):
        if not current_layer:
            break

        # 現世代の各論文を引用している論文を取得
        # 層が大きくなるほど per_paper_limit を絞って爆発防止
        per_paper = max(3, max_per_gen // max(1, len(current_layer)))

        next_layer = set()
        for wid in current_layer:
            citing_list = fetch_citing_works(wid, per_paper, mailto)
            for w in citing_list:
                cid = w.get("id", "")
                if not cid:
                    continue
                # エッジ: 引用している論文 → 引用されている論文
                all_edges.append((cid, wid))
                if cid not in all_nodes:
                    title = (w.get("title", "") or cid)
                    year  = str(w.get("publication_year") or "")
                    label = (f"{title[:45]}… ({year})" if len(title) > 45 else f"{title} ({year})") if year else title[:50]
                    all_nodes[cid] = {"label": label, "title": title, "gen": gen, "year": year}
                    next_layer.add(cid)

        current_layer = next_layer

    return all_nodes, all_edges


def build_bertopic_network(works, model_key, min_links=2, min_topic_size=10):
    """BERTopicでトピッククラスタリング → ネットワーク化"""
    import html as _html
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    # テキスト準備：HTMLエンティティ除去・空文字除外
    texts, wids = [], []
    for w in works:
        title    = w.get("title", "") or ""
        abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}), work=w)
        text     = _html.unescape((title + " " + abstract).strip())
        if len(text) > 20:
            texts.append(text[:512])
            wids.append(w.get("id", ""))

    if len(texts) < 10:
        return {"items": [], "links": []}, {}, {}

    emb_model  = SentenceTransformer(model_key)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    topic_model = BERTopic(
        embedding_model=emb_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
    )
    topics, _ = topic_model.fit_transform(texts)

    # トピックラベル・件数取得
    topic_info   = topic_model.get_topic_info()
    topic_labels = {row["Topic"]: row["Name"]  for _, row in topic_info.iterrows()}
    topic_counts = {row["Topic"]: row["Count"] for _, row in topic_info.iterrows()}

    # ワーク→トピックのマッピング（外れ値 -1 は除外）
    work_topic = {wid: t for wid, t in zip(wids, topics) if t != -1}

    # ── エッジ：著者ごとに担当トピックを集約してから共起カウント（O(authors)）──
    author_topics = defaultdict(set)
    topic_count   = defaultdict(int)
    for w in works:
        wid = w.get("id", "")
        t   = work_topic.get(wid, -1)
        if t == -1:
            continue
        topic_count[t] += 1
        for a in w.get("authorships", []):
            aid = a.get("author", {}).get("id", "")
            if aid:
                author_topics[aid].add(t)

    topic_cooccur = defaultdict(int)
    for aid, tset in author_topics.items():
        tlist = sorted(tset)
        for i, t1 in enumerate(tlist):
            for t2 in tlist[i + 1:]:
                topic_cooccur[(t1, t2)] += 1

    valid_topics = {t for t in topic_count if t != -1}
    link_list = [
        {"source_id": str(a), "target_id": str(b), "strength": s}
        for (a, b), s in topic_cooccur.items()
        if s >= min_links and a in valid_topics and b in valid_topics
    ]

    # ── ノード：エッジがなくても全有効トピックを表示 ──
    items = [
        {
            "id":          str(t),
            "label":       topic_labels.get(t, f"Topic {t}")[:60],
            "weights":     {"Papers": topic_count.get(t, 0)},
            "description": f"Topic {t}",
        }
        for t in sorted(valid_topics)
    ]

    return {"items": items, "links": link_list}, work_topic, topic_labels

def build_kmeans_network(works, model_key, n_clusters=10, min_links=2):
    """K-meansクラスタリング → ネットワーク化"""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    from sentence_transformers import SentenceTransformer

    texts, wids = [], []
    for w in works:
        title = w.get("title","") or ""
        abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}), work=w)
        text = (title + " " + abstract).strip()
        if text:
            texts.append(text[:512])
            wids.append(w.get("id",""))

    emb_model = SentenceTransformer(model_key)
    embeddings = emb_model.encode(texts, show_progress_bar=False)
    embeddings = normalize(embeddings)

    kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    work_cluster = {wid: int(lbl) for wid, lbl in zip(wids, labels)}
    cluster_count = defaultdict(int)
    for lbl in labels: cluster_count[int(lbl)] += 1

    # クラスター代表キーワード（各クラスターの論文タイトルから抽出）
    cluster_texts = defaultdict(list)
    for text, lbl in zip(texts, labels):
        cluster_texts[int(lbl)].append(text)
    cluster_labels = {}
    for c, ctexts in cluster_texts.items():
        words = " ".join(ctexts[:5]).lower().split()
        freq = defaultdict(int)
        stopwords = {"the","a","an","of","in","and","to","for","with","on","at","by","from","is","are","was","were","be","been","that","this","which","as","it","its"}
        for w in words:
            w = re.sub(r'[^a-z]','',w)
            if len(w) > 3 and w not in stopwords:
                freq[w] += 1
        top = sorted(freq.items(), key=lambda x: -x[1])[:3]
        cluster_labels[c] = "C"+str(c)+": " + " / ".join(k for k,_ in top) if top else "Cluster "+str(c)

    # クラスター共著ネットワーク
    cluster_cooccur = defaultdict(int)
    for w in works:
        wid = w.get("id","")
        c1 = work_cluster.get(wid, -1)
        if c1 == -1: continue
        for a in w.get("authorships",[]):
            aid = a.get("author",{}).get("id","")
            for w2 in works:
                wid2 = w2.get("id","")
                c2 = work_cluster.get(wid2, -1)
                if c2 == -1 or c2 == c1: continue
                for a2 in w2.get("authorships",[]):
                    if a2.get("author",{}).get("id","") == aid:
                        pair = (min(c1,c2), max(c1,c2))
                        cluster_cooccur[pair] += 1

    link_list = [{"source_id": str(a), "target_id": str(b), "strength": s}
                 for (a,b),s in cluster_cooccur.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    items = [{"id": str(c), "label": cluster_labels.get(c, "Cluster "+str(c)),
              "weights": {"Papers": cluster_count[c]}}
             for c in set(int(x) for x in connected)]

    return {"items": items, "links": link_list}, work_cluster, cluster_labels

# ────────────────────────────────────────────
# 最大連結成分抽出
# ────────────────────────────────────────────
def largest_connected_component(vos_data):
    from collections import defaultdict, deque
    items = vos_data.get("items", [])
    links = vos_data.get("links", [])
    if not items:
        return vos_data

    adj = defaultdict(set)
    for l in links:
        adj[l["source_id"]].add(l["target_id"])
        adj[l["target_id"]].add(l["source_id"])

    visited = set()
    components = []
    for item in items:
        nid = str(item["id"])
        if nid not in visited:
            comp = []
            q = deque([nid])
            while q:
                n = q.popleft()
                if n in visited:
                    continue
                visited.add(n)
                comp.append(n)
                q.extend(adj[n])
            components.append(comp)

    if not components:
        return vos_data

    largest = set(max(components, key=len))
    filtered_items = [i for i in items if str(i["id"]) in largest]
    filtered_links = [l for l in links
                      if str(l["source_id"]) in largest and str(l["target_id"]) in largest]
    return {"items": filtered_items, "links": filtered_links}

# ────────────────────────────────────────────
# GEXF変換
# ────────────────────────────────────────────
def to_gexf(vos_data):
    import xml.etree.ElementTree as ET
    items = vos_data.get("items", [])
    links = vos_data.get("links", [])
    gexf = ET.Element("gexf", {"xmlns":"http://gexf.net/1.3","xmlns:viz":"http://gexf.net/1.3/viz","version":"1.3"})
    meta = ET.SubElement(gexf, "meta")
    ET.SubElement(meta, "description").text = "Research Network Portal"
    graph = ET.SubElement(gexf, "graph", {"mode":"static","defaultedgetype":"undirected"})
    attrs_node = ET.SubElement(graph, "attributes", {"class":"node","mode":"static"})
    weight_keys = sorted(set(wk for item in items for wk in item.get("weights",{}).keys()))
    for i, wk in enumerate(weight_keys):
        ET.SubElement(attrs_node, "attribute", {"id":str(i),"title":wk,"type":"float"})
    if any(item.get("description") for item in items):
        ET.SubElement(attrs_node, "attribute", {"id":str(len(weight_keys)),"title":"description","type":"string"})
    attrs_edge = ET.SubElement(graph, "attributes", {"class":"edge","mode":"static"})
    ET.SubElement(attrs_edge, "attribute", {"id":"0","title":"strength","type":"float"})
    nodes_el = ET.SubElement(graph, "nodes")
    for item in items:
        node = ET.SubElement(nodes_el, "node", {"id":str(item.get("id",item.get("label",""))),"label":str(item.get("label",""))})
        weights = item.get("weights",{})
        attvals = ET.SubElement(node, "attvalues")
        for i, wk in enumerate(weight_keys):
            if wk in weights:
                ET.SubElement(attvals, "attvalue", {"for":str(i),"value":str(weights[wk])})
        desc = item.get("description","")
        if desc:
            ET.SubElement(attvals, "attvalue", {"for":str(len(weight_keys)),"value":desc})
        main_weight = list(weights.values())[0] if weights else 1
        ET.SubElement(node, "viz:size", {"value":str(max(1.0, float(main_weight)))})
    edges_el = ET.SubElement(graph, "edges")
    for i, link in enumerate(links):
        edge = ET.SubElement(edges_el, "edge", {"id":str(i),"source":str(link.get("source_id","")),"target":str(link.get("target_id","")),"weight":str(link.get("strength",1))})
        attvals = ET.SubElement(edge, "attvalues")
        ET.SubElement(attvals, "attvalue", {"for":"0","value":str(link.get("strength",1))})
    ET.indent(gexf, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(gexf, encoding="unicode")

# ────────────────────────────────────────────
# PyVis インタラクティブ表示
# ────────────────────────────────────────────
def render_pyvis(vos_data):
    try:
        from pyvis.network import Network
        import tempfile, base64
        items = vos_data.get("items", [])
        links = vos_data.get("links", [])
        if not items: return

        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

        key = list(items[0]["weights"].keys())[0] if items else "weight"
        max_w = max((item["weights"].get(key,1) for item in items), default=1)

        # クラスター色
        colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                  "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4"]
        for idx, item in enumerate(items):
            w = item["weights"].get(key, 1)
            size = 10 + 40 * (w / max_w)
            color = colors[idx % len(colors)]
            net.add_node(str(item.get("id", item["label"])),
                        label=item["label"][:20],
                        size=size, color=color,
                        title=item["label"] + "\n" + key + ": " + str(w))

        for link in links:
            net.add_edge(str(link["source_id"]), str(link["target_id"]),
                        value=link.get("strength",1))

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            html_content = open(f.name).read()

        st.components.v1.html(html_content, height=620, scrolling=False)

    except ImportError:
        st.warning(tl("pyvis未インストール。`pip install pyvis` を実行してください。",
                      "pyvis not installed. Run `pip install pyvis`."))

# ════════════════════════════════════════════
# UI
# ════════════════════════════════════════════

# サイドバー：言語切替
with st.sidebar:
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("EN" if lang=="ja" else "日本語"):
            st.session_state.lang = "en" if lang=="ja" else "ja"
            st.rerun()

    st.title("🔬 Research Network v2")
    st.markdown("---")

    # ── API 設定（OpenAlex / Lens.org）──
    with st.expander(tl("⚙️ API 設定", "⚙️ API Settings"), expanded=False):

        # ── OpenAlex ──
        st.markdown(f"**🔓 OpenAlex**")
        st.caption(tl(
            "メールアドレスを登録すると Polite Pool が適用され、レート制限が緩和されます。",
            "Registering your email enables OpenAlex Polite Pool for better rate limits."
        ))
        _email_input = st.text_input(
            tl("メールアドレス（任意）", "Email Address (optional)"),
            value=st.session_state.get("openalex_email", ""),
            placeholder="your@email.com",
            key="openalex_email_input"
        )
        if _email_input != st.session_state.get("openalex_email", ""):
            st.session_state["openalex_email"] = _email_input
            st.rerun()
        if st.session_state.get("openalex_email", ""):
            st.success(tl(
                f"✅ Polite Pool 有効: {st.session_state['openalex_email']}",
                f"✅ Polite Pool active: {st.session_state['openalex_email']}"
            ))
        else:
            st.info(tl("⚠️ 未設定（デフォルトPoolを使用中）", "⚠️ Not set (using default pool)"))

        st.markdown("---")

        # ── Lens.org ──
        st.markdown(f"**🔑 Lens.org**")
        st.caption(tl(
            "特許検索（NPL引用ネットワーク）に使用します。Lens.orgで発行したAPIトークンを入力してください。",
            "Used for patent search and NPL citation network. Enter your Lens.org API token."
        ))
        _lens_key_input = st.text_input(
            tl("Lens.org APIキー", "Lens.org API Key"),
            value=st.session_state.get("lens_api_key", ""),
            type="password",
            placeholder=tl("Lens.orgで発行したトークン", "Your Lens.org token"),
            key="lens_api_key_sidebar"
        )
        if _lens_key_input != st.session_state.get("lens_api_key", ""):
            st.session_state["lens_api_key"] = _lens_key_input
            st.rerun()
        if st.session_state.get("lens_api_key", ""):
            st.success(tl("✅ Lens.org APIキー設定済み", "✅ Lens.org API Key configured"))
        else:
            st.warning(tl("⚠️ 未設定（特許検索には必須）", "⚠️ Not set (required for patent search)"))

    st.markdown("---")

    # ステップ表示
    step = st.radio(tl("ステップ","Step"), [
        tl("① データ収集・保存","① Collect & Save"),
        tl("② 分析・可視化","② Analyze & Visualize"),
        tl("③ KAKEN助成金分析","③ KAKEN Grant Analysis"),
    ])

# ────────────────────────────────────────────
# OpenAlex 検索ヘルパー
# ────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_topics_data():
    topics_path = Path.home() / "Downloads" / "topics.json"
    alt_paths = [Path("topics.json"), Path(__file__).parent / "topics.json"]
    for p in [topics_path] + alt_paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return []

def search_openalex(query, target, max_results=50):
    _mailto = _oa_email()
    try:
        if target in ("Title + Abstract", "タイトル＋抄録"):
            filt = "title.search:" + query
            count_url = "https://api.openalex.org/works?filter=" + filt + "&per_page=1&select=id&mailto=" + _mailto
            count = requests.get(count_url, timeout=15).json().get("meta",{}).get("count",0)
            url = "https://api.openalex.org/works?filter=" + filt + "&per_page=" + str(max_results) + "&select=id,doi,title,publication_year,authorships&mailto=" + _mailto
            results = requests.get(url, timeout=20).json().get("results",[])
            return results, count, "works"
        elif target in ("Author Name", "著者名"):
            url = "https://api.openalex.org/authors?search=" + query + "&per_page=" + str(max_results) + "&select=id,display_name,works_count,last_known_institutions&mailto=" + _mailto
            results = requests.get(url, timeout=15).json().get("results",[])
            return results, len(results), "authors"
        elif target in ("Affiliation Name", "機関名"):
            url = "https://api.openalex.org/institutions?search=" + query + "&per_page=" + str(max_results) + "&select=id,display_name,ror,country_code,type,works_count&mailto=" + _mailto
            results = requests.get(url, timeout=15).json().get("results",[])
            return results, len(results), "institutions"
        elif target in ("Concept", "コンセプト"):
            c_url = "https://api.openalex.org/concepts?search=" + query + "&per_page=5&select=id,display_name,works_count&mailto=" + _mailto
            concepts = requests.get(c_url, timeout=15).json().get("results",[])
            if not concepts: return [], 0, "works"
            cid = concepts[0]["id"]
            filt = "concepts.id:" + cid
            count_url = "https://api.openalex.org/works?filter=" + filt + "&per_page=1&select=id&mailto=" + _mailto
            count = requests.get(count_url, timeout=15).json().get("meta",{}).get("count",0)
            url = "https://api.openalex.org/works?filter=" + filt + "&per_page=" + str(max_results) + "&select=id,doi,title,publication_year,authorships&mailto=" + _mailto
            results = requests.get(url, timeout=20).json().get("results",[])
            for r in results:
                r["_concept_name"] = concepts[0].get("display_name", query)
                r["_concept_id"] = cid
            return results, count, "works_concept"
        else:
            return [], 0, "works"
    except:
        return [], 0, "works"

# セッション初期化
for _k, _v in [("s1_selected_topics",[]), ("s1_ego_author_ids",[]),
                ("s1_ego_author_names",[]), ("s1_org_ror_id",None),
                ("s1_org_name",""), ("s1_search_results",[]),
                ("s1_search_result_type","works"), ("s1_search_count",0),
                ("s1_affiliation_kw","")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════
# ステップ1: データ収集・保存
# ════════════════════════════════════════════
if tl("① データ収集・保存","① Collect & Save") in step:
    st.header(tl("📥 ステップ1: データ収集・保存","📥 Step 1: Collect & Save Data"))

    with st.expander(tl("📂 保存済みデータセット","📂 Saved Datasets"), expanded=False):
        datasets = list_datasets()
        if datasets:
            cols_ds = st.columns(3)
            for i, ds in enumerate(datasets):
                d = load_dataset(ds)
                if d:
                    n = len(d.get("works",[]))
                    saved = d.get("saved_at","")[:10]
                    cols_ds[i % 3].markdown(f"**{ds}**  \n{n}件 / {saved}")
        else:
            st.info(tl("保存済みデータなし","No saved datasets"))

    # ── データソース選択 ──
    st.subheader(tl("🗄️ データソース選択","🗄️ Data Source"))
    data_source = st.radio(
        tl("データソース","Data source"),
        ["OpenAlex", "PubMed", tl("🔬 特許 (Lens.org)", "🔬 Patents (Lens.org)")],
        horizontal=True,
        key="s1_data_source",
    )

    # ══════════════════════════════════════════
    # PubMed モード
    # ══════════════════════════════════════════
    if data_source == "PubMed":
        st.info(tl(
            "PubMed (NCBI) から論文を取得します。被引用数・参考文献データは取得できません。",
            "Fetching papers from PubMed (NCBI). Citation counts and references are not available."
        ))

        # ── キーワード（必須） + 検索フィールド ──
        PM_FIELD_OPTIONS = {
            tl("全フィールド（PubMed標準）","All Fields (PubMed default)"): "",
            tl("タイトル・抄録 [tiab]","Title / Abstract [tiab]"): "[Title/Abstract]",
            tl("タイトルのみ [ti]","Title only [ti]"): "[Title]",
            tl("MeSH用語 [mh]","MeSH Terms [mh]"): "[MeSH Terms]",
        }
        kw_col, field_col = st.columns([3, 2])
        pm_kw = kw_col.text_input(
            tl("🔍 キーワード","🔍 Keyword"),
            key="s1_pm_kw",
            placeholder=tl("例: lung cancer","e.g. lung cancer")
        )
        pm_field_label = field_col.selectbox(
            tl("検索フィールド","Search field"),
            list(PM_FIELD_OPTIONS.keys()),
            key="s1_pm_field"
        )
        pm_field_tag = PM_FIELD_OPTIONS[pm_field_label]

        PM_LANGUAGES = {
            tl("指定なし","Any"): "",
            "English": "english",
            "Japanese": "japanese",
            "Chinese": "chinese",
            "French": "french",
            "German": "german",
            "Spanish": "spanish",
            "Korean": "korean",
        }

        PM_COUNTRIES = [
            ("Japan",          "🇯🇵 " + tl("日本","Japan")),
            ("United States",  "🇺🇸 " + tl("米国","United States")),
            ("China",          "🇨🇳 " + tl("中国","China")),
            ("United Kingdom", "🇬🇧 " + tl("英国","United Kingdom")),
            ("Germany",        "🇩🇪 " + tl("ドイツ","Germany")),
            ("France",         "🇫🇷 " + tl("フランス","France")),
            ("South Korea",    "🇰🇷 " + tl("韓国","South Korea")),
            ("Australia",      "🇦🇺 " + tl("オーストラリア","Australia")),
            ("Canada",         "🇨🇦 " + tl("カナダ","Canada")),
            ("Italy",          "🇮🇹 " + tl("イタリア","Italy")),
            ("India",          "🇮🇳 " + tl("インド","India")),
            ("Brazil",         "🇧🇷 " + tl("ブラジル","Brazil")),
            ("Spain",          "🇪🇸 " + tl("スペイン","Spain")),
            ("Netherlands",    "🇳🇱 " + tl("オランダ","Netherlands")),
            ("Sweden",         "🇸🇪 " + tl("スウェーデン","Sweden")),
            ("Switzerland",    "🇨🇭 " + tl("スイス","Switzerland")),
            ("Taiwan",         "🇹🇼 " + tl("台湾","Taiwan")),
            ("Singapore",      "🇸🇬 " + tl("シンガポール","Singapore")),
        ]
        _pm_country_labels = [label for _, label in PM_COUNTRIES]
        _pm_country_map    = {label: eng for eng, label in PM_COUNTRIES}

        # ── 詳細フィルタ ──
        with st.expander(tl("🔧 詳細フィルタ（著者・機関・国・年・出版タイプ・言語）",
                            "🔧 Advanced Filters (Author / Affiliation / Country / Year / Type / Language)"),
                         expanded=False):

            fc1, fc2 = st.columns(2)
            pm_author = fc1.text_input(
                tl("👤 著者名 [au]","👤 Author [au]"),
                key="s1_pm_author",
                placeholder=tl("例: Yamamoto K","e.g. Yamamoto K")
            )
            pm_affil = fc2.text_input(
                tl("🏷️ 所属施設 [ad]","🏷️ Affiliation [ad]"),
                key="s1_pm_affil",
                placeholder=tl("例: Tohoku University","e.g. Tohoku University")
            )

            pm_year_enabled = st.toggle(
                tl("📅 年フィルタを有効にする","📅 Enable year filter"),
                value=False,
                key="s1_pm_year_enabled"
            )
            if pm_year_enabled:
                yc1, yc2 = st.columns(2)
                pm_year_from = yc1.number_input(
                    tl("開始年","Year from"),
                    min_value=1900, max_value=2026, value=2015, step=1,
                    key="s1_pm_year_from"
                )
                pm_year_to = yc2.number_input(
                    tl("終了年","Year to"),
                    min_value=1900, max_value=2026, value=2026, step=1,
                    key="s1_pm_year_to"
                )
            else:
                pm_year_from = st.session_state.get("s1_pm_year_from", 2015)
                pm_year_to   = st.session_state.get("s1_pm_year_to",   2026)

            PM_PUB_TYPES = [
                "Journal Article", "Review", "Systematic Review",
                "Meta-Analysis", "Clinical Trial",
                "Randomized Controlled Trial", "Case Reports",
                "Comparative Study", "Editorial", "Letter",
            ]
            pm_pub_types = st.multiselect(
                tl("📄 出版タイプ [pt]（複数選択可）","📄 Publication Type [pt] (multi-select)"),
                PM_PUB_TYPES,
                key="s1_pm_pub_types"
            )

            pm_lang_label = st.selectbox(
                tl("🌐 言語 [la]","🌐 Language [la]"),
                list(PM_LANGUAGES.keys()),
                key="s1_pm_lang"
            )
            pm_lang = PM_LANGUAGES[pm_lang_label]

            pm_country_labels = st.multiselect(
                tl("🌏 国フィルタ [ad]（複数選択可・OR検索）",
                   "🌏 Country filter [ad] (multi-select, OR search)"),
                _pm_country_labels,
                key="s1_pm_countries"
            )
            if pm_country_labels:
                st.caption(tl(
                    "※ 著者所属テキスト（[ad]）に国名が含まれる論文を絞り込みます。",
                    "※ Filters papers where the author affiliation text [ad] contains the country name."
                ))

        # ── クエリ組み立て ──
        def build_pubmed_query():
            parts = []
            kw = st.session_state.get("s1_pm_kw", "").strip()
            field_tag = PM_FIELD_OPTIONS.get(
                st.session_state.get("s1_pm_field",
                    tl("全フィールド（PubMed標準）","All Fields (PubMed default)")), ""
            )
            if kw:
                # フレーズ検索：スペースあり → クォート付き
                kw_q = f'"{kw}"' if " " in kw else kw
                parts.append(f"{kw_q}{field_tag}" if field_tag else kw_q)
            author = st.session_state.get("s1_pm_author", "").strip()
            if author:
                parts.append(f'"{author}"[Author]')
            affil = st.session_state.get("s1_pm_affil", "").strip()
            if affil:
                parts.append(f'"{affil}"[Affiliation]')
            # 年フィルタ：トグルONの時のみ付加
            if st.session_state.get("s1_pm_year_enabled", False):
                yf = st.session_state.get("s1_pm_year_from", 2015)
                yt = st.session_state.get("s1_pm_year_to",   2026)
                parts.append(f"{yf}:{yt}[dp]")
            for pt in st.session_state.get("s1_pm_pub_types", []):
                parts.append(f'"{pt}"[pt]')
            lang = PM_LANGUAGES.get(
                st.session_state.get("s1_pm_lang", tl("指定なし","Any")), ""
            )
            if lang:
                parts.append(f"{lang}[la]")
            # 国フィルタ：複数選択時はOR結合
            sel_country_labels = st.session_state.get("s1_pm_countries", [])
            if sel_country_labels:
                country_terms = [f'"{_pm_country_map[lb]}"[Affiliation]'
                                 for lb in sel_country_labels if lb in _pm_country_map]
                if len(country_terms) == 1:
                    parts.append(country_terms[0])
                elif country_terms:
                    parts.append("(" + " OR ".join(country_terms) + ")")
            return " AND ".join(parts)

        pm_query = build_pubmed_query()

        # ── クエリプレビュー ──
        if pm_query:
            with st.expander(tl("🔍 送信クエリ（確認用）","🔍 Query preview (debug)"), expanded=False):
                st.code(pm_query, language="text")

        # ── 件数確認 ──
        pm_max = st.slider(tl("最大取得件数","Max papers"), 50, 2000, 500, 50, key="s1_pm_max")

        col_cnt_pm, col_run_pm = st.columns([1, 2])
        with col_cnt_pm:
            if st.button(tl("🔢 件数を確認","🔢 Check count"),
                         disabled=not pm_query, use_container_width=True, key="s1_pm_cnt_btn"):
                with st.spinner(tl("件数を取得中...","Counting...")):
                    try:
                        r_cnt = requests.get(
                            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                            params={"db": "pubmed", "term": pm_query,
                                    "rettype": "count", "retmode": "json"},
                            timeout=15
                        )
                        cnt = int(r_cnt.json().get("esearchresult", {}).get("count", 0))
                        st.session_state["s1_pm_count"] = cnt
                    except Exception:
                        st.session_state["s1_pm_count"] = -1

        if "s1_pm_count" in st.session_state:
            cnt = st.session_state["s1_pm_count"]
            if cnt < 0:
                st.warning(tl("件数取得失敗","Failed to get count"))
            elif cnt == 0:
                st.warning(tl("⚠️ 該当論文なし","⚠️ No papers found"))
            elif cnt < 200:
                st.warning(tl(f"📄 {cnt:,}件 — 少なめ", f"📄 {cnt:,} papers — few"))
            elif cnt <= 3000:
                st.success(tl(f"📄 {cnt:,}件 — 良好", f"📄 {cnt:,} papers — good"))
            else:
                st.info(tl(f"📄 {cnt:,}件 — 多め（取得上限: {pm_max}件）",
                           f"📄 {cnt:,} papers — many (fetch limit: {pm_max})"))

        def _auto_dataset_name_pm():
            base = st.session_state.get("s1_pm_kw", "") or "pubmed_dataset"
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            return re.sub(r"[^\w]", "_", base)[:25] + "_" + ts

        with col_run_pm:
            if st.button(
                tl("▶ PubMedを検索してデータを保存","▶ Search PubMed & Save"),
                type="primary", use_container_width=True, key="s1_pm_run_btn",
                disabled=not pm_query
            ):
                with st.spinner(tl("PubMedを検索中...","Searching PubMed...")):
                    pm_works = fetch_works_pubmed(pm_query, pm_max)
                if pm_works:
                    pm_name = _auto_dataset_name_pm()
                    pm_meta = {
                        "keyword": pm_kw,
                        "query": pm_query,
                        "source": "pubmed",
                        "max_papers": pm_max,
                    }
                    pm_path = save_dataset(pm_name, pm_works, pm_meta)
                    st.success(tl(f"✅ {len(pm_works):,}件保存 → {pm_path}",
                                  f"✅ {len(pm_works):,} papers saved → {pm_path}"))
                    st.session_state["loaded_dataset"] = pm_name
                    st.session_state["loaded_works"] = pm_works
                    st.subheader(tl("プレビュー（上位5件）","Preview (top 5)"))
                    for w in pm_works[:5]:
                        title   = w.get("title","") or ""
                        year    = str(w.get("publication_year","") or "")
                        doi     = w.get("doi","") or ""
                        journal = w.get("_journal","") or ""
                        authors = [a.get("author",{}).get("display_name","")
                                   for a in w.get("authorships",[])[:2]]
                        affils  = []
                        for a in w.get("authorships",[])[:1]:
                            for inst in a.get("institutions",[])[:1]:
                                affils.append(inst.get("display_name",""))
                        st.markdown(f"**{title[:80]}**")
                        st.caption(
                            "  |  ".join(filter(None, [
                                year,
                                ", ".join(filter(None, authors)),
                                affils[0][:50] if affils else "",
                                journal,
                                f"[DOI]({doi})" if doi else "",
                            ]))
                        )
                        st.markdown("---")
                else:
                    st.warning(tl("該当なし","No results found"))

    if data_source == "PubMed":
        # PubMed モードでは以下のOpenAlex専用UIは不要
        st.stop()

    # ══════════════════════════════════════════
    # Lens.org 特許検索モード
    # ══════════════════════════════════════════
    if tl("特許", "Patents") in data_source:
        st.info(tl(
            "Lens.org APIを使って特許を検索し、NPL（非特許文献）引用からOpenAlexで論文を照合します。",
            "Search patents via Lens.org API and resolve NPL (non-patent literature) citations to papers via OpenAlex."
        ))
        # APIキーはサイドバーの「⚙️ API 設定」から取得
        lens_api_key = st.session_state.get("lens_api_key", "")
        if not lens_api_key:
            st.warning(tl(
                "⚠️ Lens.org APIキーが未設定です。左サイドバーの「⚙️ API 設定」から入力してください。",
                "⚠️ Lens.org API Key not set. Please enter it in '⚙️ API Settings' in the sidebar."
            ))
        else:
            st.success(tl("✅ Lens.org APIキー設定済み（サイドバーで変更可能）",
                          "✅ Lens.org API Key configured (changeable in sidebar)"))
        lens_keyword = st.text_input(
            tl("🔍 検索キーワード（タイトル・抄録・クレーム）", "🔍 Keyword (Title / Abstract / Claim)"),
            key="lens_keyword",
            placeholder=tl("例: CRISPR gene editing", "e.g. CRISPR gene editing"),
        )

        # ── 詳細フィルタ ──
        with st.expander(tl("🔧 詳細フィルタ（発明者・出願人・年・IPC・国・タイプ）",
                            "🔧 Advanced Filters (Inventor / Applicant / Year / IPC / Country / Type)"),
                         expanded=False):

            lf_c1, lf_c2 = st.columns(2)
            lens_inventor = lf_c1.text_input(
                tl("👤 発明者名 (inventor.name)", "👤 Inventor name"),
                key="lens_inventor",
                placeholder=tl("例: Zhang Feng", "e.g. Zhang Feng"),
            )
            lens_applicant = lf_c2.text_input(
                tl("🏢 出願人/機関 (applicant.name)", "🏢 Applicant / Assignee"),
                key="lens_applicant",
                placeholder=tl("例: Broad Institute", "e.g. Broad Institute"),
            )

            ly_c1, ly_c2 = st.columns(2)
            lens_year_from = ly_c1.number_input(
                tl("📅 開始年", "📅 Year from"),
                min_value=1900, max_value=2026, value=2010, step=1,
                key="lens_year_from",
            )
            lens_year_to = ly_c2.number_input(
                tl("📅 終了年", "📅 Year to"),
                min_value=1900, max_value=2026, value=2026, step=1,
                key="lens_year_to",
            )

            lens_ipc = st.text_input(
                tl("🏷️ IPC分類コード（前方一致）", "🏷️ IPC Classification Code (prefix match)"),
                key="lens_ipc",
                placeholder=tl("例: A61K（医薬品）、C12N（微生物・遺伝子）、H04L（通信）",
                               "e.g. A61K (pharmaceuticals), C12N (genetics), H04L (communications)"),
            )

            JURISDICTIONS = {
                "JP 🇯🇵": "JP", "US 🇺🇸": "US", "EP 🇪🇺": "EP",
                "WO (PCT) 🌐": "WO", "CN 🇨🇳": "CN", "KR 🇰🇷": "KR",
                "DE 🇩🇪": "DE", "GB 🇬🇧": "GB", "FR 🇫🇷": "FR",
                "AU 🇦🇺": "AU", "CA 🇨🇦": "CA",
            }
            lens_juris_labels = st.multiselect(
                tl("🌏 出願国/管轄（複数選択可）", "🌏 Jurisdiction (multi-select)"),
                list(JURISDICTIONS.keys()),
                key="lens_jurisdictions",
            )
            lens_juris_codes = [JURISDICTIONS[lb] for lb in lens_juris_labels]

            DOC_TYPES = {
                tl("指定なし", "Any"): [],
                tl("登録済み特許のみ", "Granted patents only"): ["granted_patent"],
                tl("出願公開のみ", "Applications only"): ["patent_application"],
                tl("両方", "Both"): ["granted_patent", "patent_application"],
            }
            lens_doc_type_label = st.selectbox(
                tl("📄 特許タイプ", "📄 Patent type"),
                list(DOC_TYPES.keys()),
                key="lens_doc_type",
            )
            lens_doc_types = DOC_TYPES[lens_doc_type_label]

            lens_npl_only = st.toggle(
                tl("NPL引用あり特許のみ（論文との紐付けに必要）",
                   "NPL-citing patents only (required for paper linkage)"),
                value=True,
                key="lens_npl_only",
            )

        lens_max = st.slider(
            tl("取得特許数", "Max patents to fetch"),
            min_value=10, max_value=200, value=50, step=10,
            key="lens_max",
        )

        _lens_search_disabled = not (lens_api_key.strip() and lens_keyword.strip())
        if st.button(
            tl("▶ Lens.orgを検索してデータを保存", "▶ Search Lens.org & Save"),
            type="primary", use_container_width=True,
            disabled=_lens_search_disabled,
            key="lens_search_btn",
        ):
            with st.spinner(tl("Lens.orgで特許を検索中...", "Searching patents on Lens.org...")):
                _patents = fetch_patents_lens(
                    lens_keyword.strip(),
                    lens_api_key.strip(),
                    lens_max,
                    inventor_filter=lens_inventor.strip(),
                    applicant_filter=lens_applicant.strip(),
                    year_from=lens_year_from,
                    year_to=lens_year_to,
                    ipc_code=lens_ipc.strip(),
                    jurisdictions=lens_juris_codes if lens_juris_codes else None,
                    doc_types=lens_doc_types if lens_doc_types else None,
                    npl_only=lens_npl_only,
                )

            if not _patents:
                st.warning(tl("特許が見つかりませんでした。キーワードやAPIキーを確認してください。",
                               "No patents found. Check your keyword and API key."))
            else:
                # Collect all NPL citations
                _all_npls = []
                for pat in _patents:
                    _all_npls.extend(pat.get("npl_citations", []))

                _unique_npls = []
                _seen_texts = set()
                for npl in _all_npls:
                    doi = npl.get("doi", "")
                    text = npl.get("text", "")
                    key_str = doi if doi else text[:80]
                    if key_str and key_str not in _seen_texts:
                        _seen_texts.add(key_str)
                        _unique_npls.append(npl)

                st.info(tl(
                    f"特許 {len(_patents)}件 / NPL引用（ユニーク） {len(_unique_npls)}件 を検出",
                    f"Found {len(_patents)} patents / {len(_unique_npls)} unique NPL citations"
                ))

                if lens_npl_only and _unique_npls:
                    with st.spinner(tl("NPL引用をOpenAlexで照合中...", "Resolving NPL citations via OpenAlex...")):
                        _resolved_works = resolve_npl_to_works(_unique_npls)
                    st.success(tl(
                        f"✅ {len(_resolved_works)}件の論文を照合しました",
                        f"✅ Resolved {len(_resolved_works)} papers"
                    ))
                else:
                    _resolved_works = []

                # Save dataset
                _lens_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                _lens_name = re.sub(r"[^\w]", "_", lens_keyword.strip())[:20] + "_lens_" + _lens_ts
                _lens_meta = {
                    "source": "lens_patent",
                    "query": lens_keyword.strip(),
                    "inventor": lens_inventor.strip(),
                    "applicant": lens_applicant.strip(),
                    "year_from": lens_year_from,
                    "year_to": lens_year_to,
                    "ipc_code": lens_ipc.strip(),
                    "jurisdictions": lens_juris_codes,
                    "doc_types": lens_doc_types,
                    "npl_only": lens_npl_only,
                    "max_patents": lens_max,
                    "n_patents": len(_patents),
                    "n_resolved_works": len(_resolved_works),
                }
                # Save both works (resolved papers) and patents
                _lens_path = DATA_DIR / (_lens_name + ".json")
                _lens_payload = {
                    "name": _lens_name,
                    "saved_at": datetime.datetime.now().isoformat(),
                    "meta": _lens_meta,
                    "works": _resolved_works,
                    "patents": _patents,
                }
                with open(_lens_path, "w", encoding="utf-8") as f:
                    json.dump(_lens_payload, f, ensure_ascii=False, indent=2)

                st.success(tl(f"✅ 保存しました → {_lens_path}",
                               f"✅ Saved → {_lens_path}"))
                st.session_state["loaded_dataset"] = _lens_name
                st.session_state["loaded_works"] = _resolved_works

                # Preview patents
                st.subheader(tl("特許プレビュー（上位5件）", "Patent Preview (top 5)"))
                for pat in _patents[:5]:
                    st.markdown(f"**{pat.get('title', 'No title')[:80]}**")
                    _inv_str  = ", ".join(pat.get("inventors", [])[:3])
                    _app_str  = ", ".join(pat.get("applicants", [])[:2])
                    _ipc_str  = ", ".join(pat.get("ipc_codes", [])[:3])
                    _npl_n    = len(pat.get("npl_citations", []))
                    _juris    = pat.get("jurisdiction", "")
                    _dtype    = pat.get("doc_type", "")
                    st.caption("  |  ".join(filter(None, [
                        pat.get("date_published","")[:10],
                        f"👤 {_inv_str}" if _inv_str else "",
                        f"🏢 {_app_str}" if _app_str else "",
                        f"🏷️ {_ipc_str}" if _ipc_str else "",
                        f"🌏 {_juris}" if _juris else "",
                        f"📄 {_dtype}" if _dtype else "",
                        f"NPL引用: {_npl_n}件",
                        f"ID: {pat.get('lens_id','')}",
                    ])))
                    st.markdown("---")

        st.stop()

    # ── 以下は OpenAlex モード専用 ──
    topics_all = load_topics_data()
    topic_map  = {t_["display_name"]: t_["id"].replace("https://openalex.org/T","") for t_ in topics_all}
    all_label  = tl("-- すべて --","-- All --")

    # ══════════════════════════════════════════
    # STEP A: トピック階層ブラウズ
    # ══════════════════════════════════════════
    st.subheader(tl("① トピックを選ぶ","① Browse & Select Topics"))

    if not topics_all:
        st.info(tl(
            "topics.json が見つかりません。`~/Downloads/topics.json` または同フォルダに置いてください。",
            "topics.json not found. Place it in ~/Downloads/ or the same folder as app8.py."
        ))
    else:
        col_browse, col_selected = st.columns([3, 1])

        with col_browse:
            # 階層フィルタ
            b_col1, b_col2, b_col3 = st.columns(3)
            domains = [all_label] + sorted(set(t_["domain"]["display_name"] for t_ in topics_all if t_.get("domain")))
            sel_domain = b_col1.selectbox("Domain", domains, key="s1_domain")
            filtered = topics_all
            if sel_domain != all_label:
                filtered = [t_ for t_ in filtered if t_.get("domain",{}).get("display_name") == sel_domain]

            fields = [all_label] + sorted(set(t_["field"]["display_name"] for t_ in filtered if t_.get("field")))
            sel_field = b_col2.selectbox("Field", fields, key="s1_field")
            if sel_field != all_label:
                filtered = [t_ for t_ in filtered if t_.get("field",{}).get("display_name") == sel_field]

            subfields = [all_label] + sorted(set(t_["subfield"]["display_name"] for t_ in filtered if t_.get("subfield")))
            sel_sub = b_col3.selectbox("Subfield", subfields, key="s1_sub")
            if sel_sub != all_label:
                filtered = [t_ for t_ in filtered if t_.get("subfield",{}).get("display_name") == sel_sub]

            topic_filter_kw = st.text_input(
                tl("トピック名で絞り込み","Filter topics by name"),
                key="s1_tf", placeholder=tl("例: battery, diabetes","e.g. battery, diabetes")
            )
            if topic_filter_kw:
                filtered = [t_ for t_ in filtered if topic_filter_kw.lower() in t_["display_name"].lower()]

            st.caption(f"**{len(filtered)}** " + tl("件のトピック（チェックで選択）","topics — check to select"))

            # チェックボックス一覧
            for t_ in filtered[:100]:
                tid   = t_["id"].replace("https://openalex.org/T","")
                label = t_["display_name"]
                checked = label in st.session_state.s1_selected_topics
                if st.checkbox(label, value=checked, key="s1_cb_"+tid):
                    if label not in st.session_state.s1_selected_topics:
                        st.session_state.s1_selected_topics.append(label)
                else:
                    if label in st.session_state.s1_selected_topics:
                        st.session_state.s1_selected_topics.remove(label)

        with col_selected:
            st.markdown(tl("**✅ 選択済みトピック**","**✅ Selected Topics**"))
            if st.session_state.s1_selected_topics:
                for tp in st.session_state.s1_selected_topics:
                    st.markdown(f"- {tp}")
                if st.button(tl("🗑 全クリア","🗑 Clear all"), key="s1_clear_topics"):
                    st.session_state.s1_selected_topics = []
                    st.rerun()
            else:
                st.info(tl("まだ選択なし","None selected"))

    # ══════════════════════════════════════════
    # STEP B: キーワード・著者・機関で絞り込み（任意）
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader(tl("② キーワード・著者・機関で絞り込む（任意）",
                    "② Narrow by Keyword / Author / Institution (optional)"))

    tab_kw, tab_author, tab_inst, tab_affil = st.tabs([
        tl("📝 キーワード","📝 Keyword"),
        tl("👤 著者名","👤 Author"),
        tl("🏢 機関名","🏢 Institution"),
        tl("🏷️ 所属施設","🏷️ Affiliation"),
    ])

    with tab_kw:
        free_kw = st.text_input(
            tl("タイトル絞り込みキーワード（トピックとAND条件）",
               "Title keyword filter (AND with topics)"),
            key="s1_freekw",
            placeholder=tl("例: solid electrolyte","e.g. solid electrolyte")
        )

    with tab_author:
        s_author_q = st.text_input(tl("著者名","Author name"), key="s1_aq",
                                   placeholder=tl("例: Komaba Shinichi","e.g. Komaba Shinichi"))
        if st.button(tl("著者を検索","Search author"), key="s1_abtn") and s_author_q:
            with st.spinner(tl("著者を検索中...","Searching authors...")):
                res, _, _ = search_openalex(s_author_q, "Author Name")
                st.session_state.s1_search_results = res
                st.session_state.s1_search_result_type = "authors"
        if st.session_state.s1_search_result_type == "authors" and st.session_state.s1_search_results:
            st.markdown(tl("**候補から選択してください（複数選択可）**","**Select from results (multiple allowed)**"))
            for a in st.session_state.s1_search_results:
                aid  = a.get("id","").split("/")[-1]  # short form e.g. A12345
                name = a.get("display_name","")
                wc   = a.get("works_count",0)
                insts = a.get("last_known_institutions",[])
                inst  = insts[0].get("display_name","") if insts else ""
                lbl   = f"{name}  ({wc}件 / {inst[:25]})" if inst else f"{name}  ({wc}件)"
                already = aid in st.session_state.s1_ego_author_ids
                btn_lbl = ("✅ " if already else "") + lbl
                if st.button(btn_lbl, key="s1_a_"+aid):
                    if already:
                        idx = st.session_state.s1_ego_author_ids.index(aid)
                        st.session_state.s1_ego_author_ids.pop(idx)
                        st.session_state.s1_ego_author_names.pop(idx)
                    else:
                        st.session_state.s1_ego_author_ids.append(aid)
                        st.session_state.s1_ego_author_names.append(name)
                    st.rerun()
        if st.session_state.s1_ego_author_ids:
            for i, (aid, aname) in enumerate(zip(st.session_state.s1_ego_author_ids,
                                                  st.session_state.s1_ego_author_names)):
                col1, col2 = st.columns([4, 1])
                col1.success("✅ " + aname)
                if col2.button(tl("削除","Remove"), key=f"s1_ca_{i}"):
                    st.session_state.s1_ego_author_ids.pop(i)
                    st.session_state.s1_ego_author_names.pop(i)
                    st.rerun()
            if st.button(tl("全クリア","Clear all"), key="s1_ca_all"):
                st.session_state.s1_ego_author_ids  = []
                st.session_state.s1_ego_author_names = []
                st.rerun()

    with tab_inst:
        # ── A: ROR IDで直接指定（最も確実） ──
        st.markdown(tl("**🔑 ROR IDで直接指定（推奨・最も確実）**",
                       "**🔑 Enter ROR ID directly (recommended)**"))
        st.caption(tl(
            "ROR URLまたはROR番号を入力してください。"
            "　例: `https://ror.org/006yn3y28`　または　`006yn3y28`",
            "Enter ROR URL or ROR number.  "
            "e.g. `https://ror.org/006yn3y28` or `006yn3y28`"
        ))
        _ror_input_col, _ror_btn_col = st.columns([4, 1])
        _ror_raw = _ror_input_col.text_input(
            tl("ROR URL / ROR番号","ROR URL / ROR number"),
            key="s1_ror_direct",
            placeholder="https://ror.org/006yn3y28  または  006yn3y28",
            label_visibility="collapsed"
        )
        _ror_confirm = _ror_btn_col.button(
            tl("確定","Confirm"), key="s1_ror_confirm_btn", use_container_width=True
        )

        if _ror_confirm and _ror_raw.strip():
            # URL・番号どちらでも正規化
            _ror_clean = _ror_raw.strip().rstrip("/")
            if "ror.org/" in _ror_clean:
                _ror_id = _ror_clean.split("ror.org/")[-1]
            else:
                _ror_id = _ror_clean
            _ror_full = f"https://ror.org/{_ror_id}"

            with st.spinner(tl("RORで機関情報を取得中...","Fetching institution by ROR...")):
                try:
                    _r = requests.get(
                        f"https://api.openalex.org/institutions/{_ror_full}",
                        params={"mailto": _oa_email()}, timeout=10
                    )
                    _inst_data = _r.json()
                    _inst_name = _inst_data.get("display_name", "")
                    if _inst_name:
                        st.session_state.s1_org_ror_id = _ror_full
                        st.session_state.s1_org_name   = _inst_name
                        st.session_state.s1_search_results = []
                        st.rerun()
                    else:
                        st.error(tl(
                            f"ROR ID `{_ror_id}` が見つかりませんでした。番号を確認してください。",
                            f"ROR ID `{_ror_id}` not found. Please check the number."
                        ))
                except Exception as e:
                    st.error(f"API error: {e}")

        st.markdown(tl("---  または 機関名で検索  ---","---  or search by name  ---"))

        # ── B: 機関名テキスト検索 ──
        s_inst_q = st.text_input(tl("機関名","Institution name"), key="s1_iq",
                                 placeholder=tl("例: National Institute for Materials Science",
                                               "e.g. National Institute for Materials Science"))
        if st.button(tl("機関を検索","Search institution"), key="s1_ibtn") and s_inst_q:
            with st.spinner(tl("機関を検索中...","Searching institutions...")):
                res, _, _ = search_openalex(s_inst_q, "Affiliation Name")
                st.session_state.s1_search_results = res
                st.session_state.s1_search_result_type = "institutions"
        if st.session_state.s1_search_result_type == "institutions" and st.session_state.s1_search_results:
            st.markdown(tl("**候補から選択（ROR IDを確認してください）**",
                           "**Select from results (check ROR ID)**"))
            for inst in st.session_state.s1_search_results:
                iid     = inst.get("id","")
                name    = inst.get("display_name","")
                ror     = inst.get("ror","") or ""
                country = inst.get("country_code","")
                wc      = inst.get("works_count",0)
                ror_s   = ror.replace("https://ror.org/","") if ror else iid.split("/")[-1]
                lbl     = f"{name}  ({country} / {wc}件)  [ROR: {ror_s}]"
                if st.button(lbl, key="s1_i_"+ror_s):
                    st.session_state.s1_org_ror_id = ror or iid
                    st.session_state.s1_org_name   = name
                    st.session_state.s1_search_results = []
                    st.rerun()

        # ── 選択中の機関 ──
        if st.session_state.s1_org_ror_id:
            _ror_disp = st.session_state.s1_org_ror_id.replace("https://ror.org/","")
            st.success(f"✅ {st.session_state.s1_org_name}　[ROR: {_ror_disp}]")
            if st.button(tl("クリア","Clear"), key="s1_ci"):
                st.session_state.s1_org_ror_id = None
                st.session_state.s1_org_name   = ""
                st.rerun()

    with tab_affil:
        st.caption(tl(
            "著者の所属施設名をテキストで直接入力して絞り込みます。"
            "ROR IDは不要です。部分一致で検索されます。",
            "Filter papers by author affiliation name (text search, partial match). No ROR ID needed."
        ))
        affil_kw = st.text_input(
            tl("所属施設名","Affiliation name"),
            key="s1_affiliation_kw",
            placeholder=tl("例: 東京大学, Tohoku University","e.g. Tokyo, Tohoku University")
        )
        if affil_kw:
            st.success(f"✅ {tl('所属施設フィルタ：','Affiliation filter:')} {affil_kw}")
            if st.button(tl("クリア","Clear"), key="s1_affil_clear"):
                st.session_state.s1_affiliation_kw = ""
                st.rerun()

    # ══════════════════════════════════════════
    # STEP C: フィルタ・保存設定
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader(tl("③ フィルタ・保存設定","③ Filters & Save Settings"))

    fc1, fc2, fc3 = st.columns(3)
    year_from = fc1.number_input("From", 1990, 2026, 2020, key="s1_yf")
    year_to   = fc2.number_input("To",   1990, 2026, 2026, key="s1_yt")
    per_page  = fc3.slider(tl("最大取得件数","Max papers"), 100, 3000, 500, 100, key="s1_pp")
    oa_only   = st.toggle(tl("OAのみ","Open Access only"), key="s1_oa")

    # 国フィルタ
    COUNTRIES = [
        ("JP","🇯🇵 日本 / Japan"),
        ("US","🇺🇸 米国 / USA"),
        ("CN","🇨🇳 中国 / China"),
        ("GB","🇬🇧 英国 / UK"),
        ("DE","🇩🇪 ドイツ / Germany"),
        ("FR","🇫🇷 フランス / France"),
        ("KR","🇰🇷 韓国 / Korea"),
        ("AU","🇦🇺 オーストラリア / Australia"),
        ("CA","🇨🇦 カナダ / Canada"),
        ("IT","🇮🇹 イタリア / Italy"),
        ("IN","🇮🇳 インド / India"),
        ("BR","🇧🇷 ブラジル / Brazil"),
        ("ES","🇪🇸 スペイン / Spain"),
        ("NL","🇳🇱 オランダ / Netherlands"),
        ("SE","🇸🇪 スウェーデン / Sweden"),
        ("CH","🇨🇭 スイス / Switzerland"),
        ("TW","🇹🇼 台湾 / Taiwan"),
        ("SG","🇸🇬 シンガポール / Singapore"),
        ("RU","🇷🇺 ロシア / Russia"),
        ("IL","🇮🇱 イスラエル / Israel"),
    ]
    country_options = [label for _, label in COUNTRIES]
    sel_country_labels = st.multiselect(
        tl("🌏 国・地域フィルタ（複数選択可）","🌏 Country / Region filter (multi-select)"),
        country_options, key="s1_countries"
    )
    sel_country_codes = [code for code, label in COUNTRIES if label in sel_country_labels]

    # 国コード直接入力フィルタ（任意）
    s1_country_code_input = st.text_input(
        tl(
            "🌐 国コード直接入力フィルタ（任意・例: JP, US）",
            "🌐 Country code filter (optional, e.g. JP, US)"
        ),
        key="s1_country_code_input",
        placeholder=tl("空欄=フィルタなし　例: JP", "Leave empty for no filter. e.g. JP"),
        help=tl(
            "ISO 3166-1 alpha-2 の国コードを入力すると OpenAlex の "
            "`institutions.country_code` フィルタに追加されます。"
            "上の複数選択と組み合わせることもできます。",
            "Enter an ISO 3166-1 alpha-2 country code to add an "
            "`institutions.country_code` filter to OpenAlex. "
            "Can be combined with the multi-select above."
        ),
    )
    s1_country_code_clean = s1_country_code_input.strip().upper() if s1_country_code_input else ""
    if s1_country_code_clean and s1_country_code_clean not in sel_country_codes:
        sel_country_codes.append(s1_country_code_clean)

    def _auto_dataset_name():
        base = (
            (st.session_state.s1_ego_author_names[0] if st.session_state.s1_ego_author_names else "") or
            st.session_state.s1_org_name or
            (st.session_state.s1_selected_topics[0] if st.session_state.s1_selected_topics else "") or
            st.session_state.get("s1_freekw", "") or
            "dataset"
        )
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        return re.sub(r"[^\w]", "_", base)[:25] + "_" + ts

    # 現在の検索条件サマリ
    summary = []
    if st.session_state.s1_selected_topics:
        summary.append(tl("🗂 トピック: ","🗂 Topics: ") + ", ".join(st.session_state.s1_selected_topics[:3]))
    if st.session_state.s1_ego_author_ids:
        summary.append(tl("👤 著者: ","👤 Author: ") + ", ".join(st.session_state.s1_ego_author_names))
    if st.session_state.s1_org_ror_id:
        summary.append(tl("🏢 機関: ","🏢 Institution: ") + st.session_state.s1_org_name)
    if st.session_state.get("s1_affiliation_kw",""):
        summary.append(tl("🏷️ 所属施設: ","🏷️ Affiliation: ") + st.session_state.s1_affiliation_kw)
    if st.session_state.get("s1_freekw",""):
        summary.append(tl("📝 キーワード: ","📝 Keyword: ") + st.session_state.s1_freekw)
    if sel_country_codes:
        summary.append(tl("🌏 国: ","🌏 Countries: ") + ", ".join(sel_country_codes))
    if summary:
        st.info("  |  ".join(summary))

    st.markdown("---")

    # ── 件数プレビューボタン ──
    def build_filters_from_state():
        """現在の選択条件からfiltersリストを構築"""
        f = []
        if st.session_state.s1_selected_topics:
            tids = ["T" + topic_map[tp] for tp in st.session_state.s1_selected_topics if tp in topic_map]
            if len(tids) == 1: f.append("topics.id:" + tids[0])
            elif tids: f.append("topics.id:" + "|".join(tids))
        if st.session_state.s1_ego_author_ids:
            aids = st.session_state.s1_ego_author_ids
            f.append("authorships.author.id:" + (aids[0] if len(aids) == 1 else "|".join(aids)))
        if st.session_state.s1_org_ror_id:
            ror = st.session_state.s1_org_ror_id
            if not ror.startswith("https://ror.org/"): ror = "https://ror.org/" + ror
            f.append("authorships.institutions.ror:" + ror)
        if st.session_state.get("s1_affiliation_kw",""):
            f.append("authorships.institutions.display_name.search:" + st.session_state.s1_affiliation_kw)
        if st.session_state.get("s1_freekw","") and not st.session_state.s1_selected_topics:
            f.append("title.search:" + st.session_state.s1_freekw)
        f.append("publication_year:" + str(year_from) + "-" + str(year_to))
        if oa_only: f.append("is_oa:true")
        if sel_country_codes:
            f.append("authorships.institutions.country_code:" +
                     (sel_country_codes[0] if len(sel_country_codes)==1 else "|".join(sel_country_codes)))
        return f

    can_run = bool(
        st.session_state.s1_selected_topics or
        st.session_state.s1_ego_author_ids or
        st.session_state.s1_org_ror_id or
        st.session_state.get("s1_freekw","") or
        st.session_state.get("s1_affiliation_kw","")
    )

    # 件数確認ボタン
    col_cnt, col_run = st.columns([1, 2])
    with col_cnt:
        if st.button(tl("🔢 件数を確認","🔢 Check count"),
                     disabled=not can_run, use_container_width=True):
            with st.spinner(tl("件数を取得中...","Counting...")):
                try:
                    f = build_filters_from_state()
                    r = requests.get(
                        "https://api.openalex.org/works",
                        params={"filter": ",".join(f), "per_page": 1,
                                "select": "id", "mailto": _oa_email()},
                        timeout=15)
                    count = r.json().get("meta",{}).get("count", 0)
                    st.session_state["s1_count_result"] = count
                    st.session_state["s1_count_filter"] = ",".join(f)
                except:
                    st.session_state["s1_count_result"] = -1

    # 件数表示
    if "s1_count_result" in st.session_state:
        cnt = st.session_state["s1_count_result"]
        if cnt < 0:
            st.warning(tl("件数取得失敗","Failed to get count"))
        elif cnt == 0:
            st.warning(tl("⚠️ 該当論文なし","⚠️ No papers found"))
        elif cnt < 200:
            st.warning(tl(f"📄 {cnt:,}件 — 少なめ", f"📄 {cnt:,} papers — few"))
        elif cnt <= 2000:
            st.success(tl(f"📄 {cnt:,}件 — 良好", f"📄 {cnt:,} papers — good"))
        else:
            st.info(tl(f"📄 {cnt:,}件 — 多め（取得上限: {per_page}件）",
                       f"📄 {cnt:,} papers — many (fetch limit: {per_page})"))
        if "s1_count_filter" in st.session_state:
            with st.expander(tl("🔍 送信フィルタ（確認用）","🔍 Applied filter (debug)")):
                st.code(st.session_state["s1_count_filter"])

    with col_run:
        if st.button(tl("▶ 検索してデータを保存","▶ Search & Save"),
                     type="primary", use_container_width=True, disabled=not can_run):
            filters = build_filters_from_state()
            meta_info = {}
            if st.session_state.s1_selected_topics:
                meta_info["topics"] = st.session_state.s1_selected_topics
            elif st.session_state.s1_ego_author_ids:
                meta_info["author"] = ", ".join(st.session_state.s1_ego_author_names)
            elif st.session_state.s1_org_ror_id:
                meta_info["institution"] = st.session_state.s1_org_name
            elif st.session_state.get("s1_freekw",""):
                meta_info["keyword"] = st.session_state.s1_freekw
            if sel_country_codes:
                meta_info["countries"] = sel_country_codes
            meta_info.update({"year_from": year_from, "year_to": year_to, "filters": filters})

            dataset_name = _auto_dataset_name()
            with st.spinner(tl("OpenAlexを検索中...","Searching OpenAlex...")):
                works = fetch_works(filters, per_page, mailto=_oa_email())

            if works:
                path = save_dataset(dataset_name, works, meta_info)
                st.session_state["s1_count_result"] = len(works)
                st.success(tl(f"✅ {len(works):,}件保存 → {path}",
                              f"✅ {len(works):,} papers saved → {path}"))
                st.session_state["loaded_dataset"] = dataset_name
                st.session_state["loaded_works"] = works

                st.subheader(tl("プレビュー（上位5件）","Preview (top 5)"))
                for w in works[:5]:
                    title = w.get("title","") or ""
                    year  = str(w.get("publication_year",""))
                    doi   = w.get("doi","") or ""
                    authors = [a.get("author",{}).get("display_name","") for a in w.get("authorships",[])[:2]]
                    st.markdown(f"**{title[:80]}**")
                    st.caption(year + "  |  " + ", ".join(filter(None,authors)) +
                               (f"  |  [DOI]({doi})" if doi else ""))
                    st.markdown("---")
            else:
                st.warning(tl("該当なし","No results found"))

# ════════════════════════════════════════════
# ステップ2: 分析・可視化
# ════════════════════════════════════════════
else:
    st.header(tl("🔬 ステップ2: 分析・可視化","🔬 Step 2: Analyze & Visualize"))

    # データセット選択
    datasets = list_datasets()
    if not datasets:
        st.warning(tl("まずステップ1でデータを保存してください。",
                      "Please save data in Step 1 first."))
        st.stop()

    selected_ds = st.selectbox(
        tl("📂 使用するデータセット","📂 Dataset to use"), datasets,
        index=datasets.index(st.session_state.get("loaded_dataset", datasets[0]))
               if st.session_state.get("loaded_dataset") in datasets else 0
    )

    data = load_dataset(selected_ds)
    if not data:
        st.error(tl("データ読み込み失敗","Failed to load dataset"))
        st.stop()

    works = data["works"]
    meta  = data.get("meta", {})

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric(tl("論文数","Papers"), len(works))
    col_info2.metric(tl("保存日","Saved"), data.get("saved_at","")[:10])
    col_info3.metric(tl("クエリ","Query"), meta.get("query","")[:20])

    st.markdown("---")

    # ── 研究シナリオガイド ──
    with st.expander(tl("🗺️ 研究シナリオガイド","🗺️ Research Scenario Guide"), expanded=False):
        st.markdown(tl(
            """このアプリで実行できる代表的な **6つの研究シナリオ** と、使うべき機能の対応表です。

| # | シナリオ | 推奨手法 | ステップ |
|---|----------|----------|----------|
| 1 | **研究コミュニティの把握** 誰が中心的な研究者か、どのグループが存在するか | 共著ネットワーク → 中心性ランキング | Step2 |
| 2 | **研究トレンドの発見** どのトピックが急成長しているか | BERTopic / K-means クラスタリング → ホワイトスペース可視化 | Step2 |
| 3 | **ホワイトスペース（空白領域）の特定** 論文数が少なく引用が高いニッチ分野を発見 | BERTopic / K-means → ホワイトスペース散布図 | Step2 |
| 4 | **国際動向の比較** どの国が主導しているか | 国別論文数グラフ（Step2） + 国フィルタ（Step1） | Step1+2 |
| 5 | **重要論文・ハブの特定** 最も影響力のある論文・研究者は誰か | KeyBERT共起 → PageRankランキング + 重要論文ランキング | Step2 |
| 6 | **研究費対効果の評価** 助成金と論文アウトカムの関係 | KAKEN分析 + 重要論文の被引用数（費用対効果の推定） | Step2+3 |

> 💡 **ヒント**: シナリオ3の「ホワイトスペース」は BERTopic または K-means 実行後に自動表示されます。
""",
            """Here are **6 research scenarios** you can explore with this app and which features to use:

| # | Scenario | Recommended Features | Step |
|---|----------|----------------------|------|
| 1 | **Map the research community** Who are the key researchers? What groups exist? | Co-authorship network → Centrality ranking | Step 2 |
| 2 | **Discover research trends** Which topics are growing fast? | BERTopic / K-means clustering → White Space viz | Step 2 |
| 3 | **Identify white spaces** Find niches with few papers but high citation impact | BERTopic / K-means → White Space scatter plot | Step 2 |
| 4 | **International comparison** Which countries lead the field? | Papers-by-country chart (Step 2) + country filter (Step 1) | Step 1+2 |
| 5 | **Find key papers & hubs** Most influential papers and researchers | KeyBERT co-occurrence → PageRank + Key Papers ranking | Step 2 |
| 6 | **Evaluate research funding ROI** Relationship between grants and publication outcomes | KAKEN analysis + citation counts (funding cost per citation) | Step 2+3 |

> 💡 **Tip**: The "White Space" scatter plot for Scenario 3 appears automatically after running BERTopic or K-means analysis.
"""
        ))

    st.markdown("---")

    # ── 分析設定（サイドバー） ──
    with st.sidebar:
        st.markdown("---")
        st.subheader(tl("⚙️ 分析設定","⚙️ Analysis Settings"))

        analysis_type = st.radio(tl("分析手法","Analysis method"), [
            tl("共著ネットワーク","Co-authorship Network"),
            tl("KeyBERT キーワード共起","KeyBERT Keyword Co-occurrence"),
            tl("BERTopic クラスタリング","BERTopic Clustering"),
            tl("K-means クラスタリング","K-means Clustering"),
            tl("引用分析","Citation Analysis"),
            tl("🌳 引用系譜（3世代）","🌳 Citation Genealogy (3 Generations)"),
        ])

        # 手法説明
        _method_info = {
            tl("共著ネットワーク","Co-authorship Network"): tl(
                """論文の著者情報から「誰と誰が一緒に論文を書いたか」を辺（リンク）として可視化するネットワークです。
ノードが著者、辺の太さが共著回数を表します。研究コミュニティの構造や中心的な研究者の把握に有効です。

**主要文献**
- Barabási et al. (2002). Evolution of the social network of scientific collaborations. *Physica A*, 311(3–4), 590–614. [DOI](https://doi.org/10.1016/S0378-4371(02)00736-7)
- Newman, M. E. J. (2001). The structure of scientific collaboration networks. *PNAS*, 98(2), 404–409. [DOI](https://doi.org/10.1073/pnas.021544898)""",
                """Visualizes co-authorship relationships as a network: nodes are authors, edges represent co-authored papers, and edge weight indicates frequency. Useful for identifying research communities and key researchers.

**Key References**
- Barabási et al. (2002). Evolution of the social network of scientific collaborations. *Physica A*, 311(3–4), 590–614. [DOI](https://doi.org/10.1016/S0378-4371(02)00736-7)
- Newman, M. E. J. (2001). The structure of scientific collaboration networks. *PNAS*, 98(2), 404–409. [DOI](https://doi.org/10.1073/pnas.021544898)"""),
            tl("KeyBERT キーワード共起","KeyBERT Keyword Co-occurrence"): tl(
                """各論文のタイトル・アブストラクトから **KeyBERT** を用いてキーワードを抽出し、同一論文内に同時出現したキーワードペアを辺として可視化します。
BERTの文埋め込みを利用するため、TF-IDFより意味的に重要な語句を抽出できます。

**主要文献**
- Grootendorst, M. (2020). KeyBERT: Minimal keyword extraction with BERT. *Zenodo*. [DOI](https://doi.org/10.5281/zenodo.4461265)
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT 2019*, 4171–4186. [DOI](https://doi.org/10.18653/v1/N19-1423)""",
                """Extracts keywords from titles and abstracts using **KeyBERT** (BERT-based semantic similarity), then builds a co-occurrence network of keywords appearing together in the same paper.

**Key References**
- Grootendorst, M. (2020). KeyBERT: Minimal keyword extraction with BERT. *Zenodo*. [DOI](https://doi.org/10.5281/zenodo.4461265)
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT 2019*, 4171–4186. [DOI](https://doi.org/10.18653/v1/N19-1423)"""),
            tl("BERTopic クラスタリング","BERTopic Clustering"): tl(
                """論文テキストを Sentence-BERT で埋め込み → UMAP で次元削減 → HDBSCAN で密度ベースクラスタリング → c-TF-IDF でトピックラベルを自動生成する手法です。
トピック数を事前に指定する必要がなく、意味的に近い論文群を自動でまとめます。

**主要文献**
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv*:2203.05794. [arXiv](https://arxiv.org/abs/2203.05794)
- McInnes, L. et al. (2017). HDBSCAN: Hierarchical density based clustering. *JOSS*, 2(11), 205. [DOI](https://doi.org/10.21105/joss.00205)
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*, 3982–3992. [DOI](https://doi.org/10.18653/v1/D19-1410)""",
                """Embeds paper texts with Sentence-BERT → reduces dimensions with UMAP → clusters with HDBSCAN → labels topics with c-TF-IDF. No need to specify the number of topics in advance.

**Key References**
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv*:2203.05794. [arXiv](https://arxiv.org/abs/2203.05794)
- McInnes, L. et al. (2017). HDBSCAN: Hierarchical density based clustering. *JOSS*, 2(11), 205. [DOI](https://doi.org/10.21105/joss.00205)
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*, 3982–3992. [DOI](https://doi.org/10.18653/v1/D19-1410)"""),
            tl("K-means クラスタリング","K-means Clustering"): tl(
                """論文テキストを TF-IDF ベクトルに変換し、K-means アルゴリズムで K 個のクラスターに分割する古典的手法です。
クラスター数 K を事前に指定する必要がありますが、計算が高速で大規模データに向いています。

**主要文献**
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137. [DOI](https://doi.org/10.1109/TIT.1982.1056489)
- Sculley, D. (2010). Web-scale k-means clustering. *WWW 2010*, 1177–1178. [DOI](https://doi.org/10.1145/1772690.1772862)""",
                """Converts paper texts to TF-IDF vectors and partitions them into K clusters using the classic K-means algorithm. Fast and scalable, but requires specifying K in advance.

**Key References**
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137. [DOI](https://doi.org/10.1109/TIT.1982.1056489)
- Sculley, D. (2010). Web-scale k-means clustering. *WWW 2010*, 1177–1178. [DOI](https://doi.org/10.1145/1772690.1772862)"""),
            tl("引用分析","Citation Analysis"): tl(
                """収集した論文間の引用関係をネットワークとして可視化します。2種類のモードがあります。

**① 書誌結合 (Bibliographic Coupling)**
2つの論文が共通して引用している参考文献の数をエッジ強度とします。共通参照が多いほど研究的に近い論文同士です。

**② 直接引用 (Direct Citation)**
収集した論文セット内で「論文Aが論文Bを引用している」関係を直接エッジとして表示します。

**主要文献**
- Kessler, M. M. (1963). Bibliographic coupling between scientific papers. *American Documentation*, 14(1), 10–25. [DOI](https://doi.org/10.1002/asi.5090140103)
- Small, H. (1973). Co-citation in the scientific literature. *JASIST*, 24(4), 265–269. [DOI](https://doi.org/10.1002/asi.4630240406)
- Garfield, E. (1955). Citation indexes for science. *Science*, 122(3159), 108–111. [DOI](https://doi.org/10.1126/science.122.3159.108)""",
                """Visualizes citation relationships among collected papers in two modes.

**① Bibliographic Coupling**
Edge strength = number of shared references between two papers. More shared references means closer research topics.

**② Direct Citation**
Draws a direct edge when paper A cites paper B (both must be in the collected set).

**Key References**
- Kessler, M. M. (1963). Bibliographic coupling between scientific papers. *American Documentation*, 14(1), 10–25. [DOI](https://doi.org/10.1002/asi.5090140103)
- Small, H. (1973). Co-citation in the scientific literature. *JASIST*, 24(4), 265–269. [DOI](https://doi.org/10.1002/asi.4630240406)
- Garfield, E. (1955). Citation indexes for science. *Science*, 122(3159), 108–111. [DOI](https://doi.org/10.1126/science.122.3159.108)"""),
            tl("🌳 引用系譜（3世代）","🌳 Citation Genealogy (3 Generations)"): tl(
                """選択した論文（第0世代）の参考文献リスト（第1世代）→ その参考文献（第2世代）→ さらにその参考文献（第3世代）を辿り、引用の系譜を有向グラフで可視化します。

**世代ごとの色分け**
🔵 第0世代（収集論文） 🟢 第1世代 🟡 第2世代 🔴 第3世代

少数の重要論文・研究者グループの知識系譜把握に最適です。
**注意**: OpenAlexデータが必要です（PubMed・Lens.orgでは利用不可）。論文数が多い場合は処理に時間がかかります。""",
                """Traces backward citations from your selected papers (Gen 0) → their references (Gen 1) → those references' references (Gen 2) → and one level further (Gen 3), visualized as a directed graph.

**Color by generation**
🔵 Gen 0 (collected) 🟢 Gen 1 🟡 Gen 2 🔴 Gen 3

Best suited for a small set of key papers to reveal the intellectual lineage of a research topic.
**Note**: Requires OpenAlex data (not available for PubMed or Lens.org sources)."""),
        }
        if analysis_type in _method_info:
            with st.expander(tl("📖 この手法について","📖 About this method")):
                st.markdown(_method_info[analysis_type])

        # 引用分析サブタイプ
        if tl("引用分析","Citation Analysis") in analysis_type and tl("引用系譜","Citation Genealogy") not in analysis_type:
            citation_type_label = st.radio(
                tl("引用分析の種類","Citation analysis type"),
                [tl("書誌結合 (Bibliographic Coupling)","Bibliographic Coupling"),
                 tl("直接引用 (Direct Citation)","Direct Citation")],
            )
            citation_type = ("bibliographic_coupling"
                             if tl("書誌結合","Bibliographic Coupling") in citation_type_label
                             else "direct_citation")
        else:
            citation_type = "bibliographic_coupling"

        # 引用系譜パラメータ
        if tl("引用系譜","Citation Genealogy") in analysis_type:
            st.markdown("---")
            st.caption(tl(
                "⚠️ OpenAlexデータ専用です。PubMed・Lens.orgでは参考文献データがないため使用できません。",
                "⚠️ OpenAlex data only. PubMed and Lens.org do not provide reference lists."
            ))

            # ── 方向選択 ──
            _dir_backward = tl("⬅ 引用系譜（過去を遡る）","⬅ Backward (traces references)")
            _dir_forward  = tl("➡ 被引用系譜（将来を追う）","➡ Forward (traces citing papers)")
            genealogy_direction_label = st.radio(
                tl("系譜の方向","Genealogy direction"),
                [_dir_backward, _dir_forward],
                key="genealogy_direction",
                help=tl(
                    "⬅ 引用系譜: この論文が引用している論文を遡る（知識の源流）\n"
                    "➡ 被引用系譜: この論文を引用している論文を追う（知識の波及）",
                    "⬅ Backward: traces papers this work references (intellectual origins)\n"
                    "➡ Forward: traces papers that cite this work (influence spread)"
                )
            )
            genealogy_direction = "forward" if _dir_forward in genealogy_direction_label else "backward"

            # ── 論文1本選択 ──
            if works:
                def _paper_label(w):
                    _t = (w.get("title", "") or w.get("id", ""))[:55]
                    _y = w.get("publication_year") or ""
                    return f"[{_y}]  {_t}" if _y else _t

                _paper_options = {
                    w.get("id", ""): _paper_label(w)
                    for w in works if w.get("id")
                }
                _default_id = st.session_state.get("genealogy_seed_id", list(_paper_options.keys())[0])
                if _default_id not in _paper_options:
                    _default_id = list(_paper_options.keys())[0]
                genealogy_seed_id = st.selectbox(
                    tl("📄 起点論文を選択（1本）","📄 Select seed paper (1 paper)"),
                    options=list(_paper_options.keys()),
                    format_func=lambda k: _paper_options[k],
                    index=list(_paper_options.keys()).index(_default_id),
                    key="genealogy_seed_selectbox",
                    help=tl(
                        "この論文を起点として系譜を辿ります。",
                        "Traces the citation genealogy from this paper."
                    )
                )
                st.session_state["genealogy_seed_id"] = genealogy_seed_id
            else:
                genealogy_seed_id = None

            genealogy_generations = st.slider(
                tl("追跡世代数","Generations to trace"), 1, 3, 3,
                key="genealogy_gen",
                help=tl("多いほど深く辿れますが、API呼び出し回数が増加します。",
                        "More generations = deeper trace, but more API calls.")
            )
            _ref_label = (
                tl("1論文あたり最大被引用論文数","Max citing papers per paper")
                if genealogy_direction == "forward"
                else tl("1論文あたり最大参考文献数","Max references per paper")
            )
            genealogy_max_per_gen = st.slider(
                _ref_label, 5, 50, 20,
                key="genealogy_max_per_gen",
                help=tl("各論文から辿る最大数。多いとグラフが巨大になります。",
                        "Max papers to follow per paper. Higher = larger graph.")
            )
        else:
            genealogy_generations  = 3
            genealogy_max_per_gen  = 20
            genealogy_seed_id      = None
            genealogy_direction    = "backward"

        # モデル選択（BERT系手法のとき）
        keybert_types  = [tl("KeyBERT キーワード共起","KeyBERT Keyword Co-occurrence")]
        bertopic_types = [tl("BERTopic クラスタリング","BERTopic Clustering"),
                          tl("K-means クラスタリング","K-means Clustering")]

        if analysis_type in keybert_types:
            model_name = st.radio(
                tl("🔬 モデル選択","🔬 Model"),
                list(KEYBERT_MODELS.keys()),
                format_func=lambda k: k + "  —  " + KEYBERT_MODELS[k][1],
                key="keybert_model_radio",
            )
            model_key = KEYBERT_MODELS[model_name][0]
            if len(works) > 100:
                keybert_max_papers = st.slider(
                    tl("最大処理件数（被引用数順）","Max papers (by citations)"),
                    min_value=100, max_value=min(len(works), 2000),
                    value=min(500, len(works)), step=100,
                    key="keybert_max_papers",
                    help=tl(
                        "被引用数の多い論文を優先処理。件数を減らすと処理が速くなります。",
                        "Prioritizes most-cited papers. Reduce for faster processing."
                    )
                )
            else:
                keybert_max_papers = len(works)
                st.caption(tl(f"全 {len(works)} 件を処理します", f"Processing all {len(works)} papers"))

        elif analysis_type in bertopic_types:
            st.caption(tl(
                "⚠️ BERTopic / K-means は**汎用モデル**を使用してください。"
                "ドメイン特化モデル（BatterySciBERT等）はテーマと無関係な語句がラベルになります。",
                "⚠️ Use a **general-purpose model** for BERTopic / K-means. "
                "Domain-specific models (BatterySciBERT etc.) produce unrelated topic labels."
            ))
            model_name = st.radio(
                tl("🔬 モデル選択","🔬 Model"),
                list(BERTOPIC_MODELS.keys()),
                format_func=lambda k: k + "  —  " + BERTOPIC_MODELS[k][1],
                key="bertopic_model_radio",
            )
            model_key = BERTOPIC_MODELS[model_name][0]

        else:
            model_key = None

        if analysis_type not in keybert_types:
            keybert_max_papers = 500

        if tl("K-means","K-means") in analysis_type:
            n_clusters = st.slider(tl("クラスター数","Number of clusters"), 3, 30, 10)
        else:
            n_clusters = 10

        if tl("BERTopic","BERTopic") in analysis_type:
            bertopic_min_size = st.slider(
                tl("最小トピックサイズ（論文数）","Min topic size (papers)"),
                min_value=3, max_value=50, value=10,
                help=tl(
                    "1トピックを形成するのに必要な最低論文数。小さいほど細かいトピックが生まれます。論文数が多い場合は10〜20が目安。",
                    "Minimum papers required to form one topic. Smaller = more topics. 10–20 is typical for large datasets."
                )
            )

        min_links = st.slider(tl("最小リンク強度","Min link strength"), 1, 10, 2)

        # 可視化方法
        st.markdown("---")
        st.subheader(tl("📊 可視化方法","📊 Visualization"))
        _is_genealogy = tl("引用系譜","Citation Genealogy") in analysis_type
        if _is_genealogy:
            viz_method = tl("アプリ内（インタラクティブ）","In-app (Interactive)")
            st.info(tl(
                "🌳 引用系譜はアプリ内表示のみ対応しています。",
                "🌳 Citation Genealogy supports in-app display only."
            ))
        else:
            viz_method = st.radio(tl("表示方法","Display method"), [
                tl("アプリ内（インタラクティブ）","In-app (Interactive)"),
                "VOSviewer JSON",
                "Retina / Gephi (GEXF)",
            ])

        run_btn = st.button(tl("▶ 解析実行","▶ Run Analysis"),
                            type="primary", use_container_width=True)

    # ── 解析実行 ──
    if run_btn:
        work_keywords = {}
        cluster_map = {}
        cluster_labels_map = {}

        if tl("共著","Co-authorship") in analysis_type:
            with st.spinner(tl("共著ネットワーク構築中...","Building co-authorship network...")):
                vos_data = largest_connected_component(build_coauth(works, min_links))

        elif tl("KeyBERT","KeyBERT") in analysis_type:
            from keybert import KeyBERT
            import html as _html

            _kw_cache_key = f"kw_{selected_ds}_{model_key}_{keybert_max_papers}"

            if _kw_cache_key in st.session_state:
                # ── キャッシュヒット：即座に読み込み ──
                work_keywords = st.session_state[_kw_cache_key]
                st.success(tl(
                    f"✅ キャッシュから読み込みました（{len(work_keywords)}件）",
                    f"✅ Loaded from cache ({len(work_keywords)} papers)"
                ))
            else:
                # ── 被引用数順に上位N件を選択 ──
                _sorted = sorted(works, key=lambda w: w.get("cited_by_count", 0), reverse=True)
                _target = _sorted[:keybert_max_papers]
                _wids, _texts = [], []
                for w in _target:
                    wid = w.get("id", "")
                    title = w.get("title", "") or ""
                    abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}), work=w)
                    text = _html.unescape((title + " " + abstract).strip())[:1000]
                    if text:
                        _wids.append(wid)
                        _texts.append(text)

                with st.spinner(tl(
                    f"KeyBERT でキーワード抽出中... {len(_texts)}件 [{model_name}]（バッチ処理）",
                    f"Extracting keywords... {len(_texts)} papers [{model_name}] (batch)"
                )):
                    kw_model = KeyBERT(model=model_key)
                    _all_kws = kw_model.extract_keywords(
                        _texts, keyphrase_ngram_range=(1, 2), top_n=5
                    )

                work_keywords = {wid: [k for k, _ in kws]
                                 for wid, kws in zip(_wids, _all_kws)}
                st.session_state[_kw_cache_key] = work_keywords
            with st.spinner(tl("ネットワーク構築中...","Building network...")):
                vos_data = largest_connected_component(build_keyword_cooccurrence(works, work_keywords, min_links))
            st.success(tl(f"キーワード抽出完了: {sum(len(v) for v in work_keywords.values())}語 [{model_name}]",
                          f"Keywords extracted: {sum(len(v) for v in work_keywords.values())} [{model_name}]"))

        elif tl("BERTopic","BERTopic") in analysis_type:
            with st.spinner(tl("BERTopicでクラスタリング中...","Running BERTopic clustering...")):
                try:
                    vos_data, cluster_map, cluster_labels_map = build_bertopic_network(works, model_key, min_links, bertopic_min_size)
                    if not vos_data.get("items"):
                        st.warning(tl("論文数が少なすぎてBERTopicを実行できません（最低10件必要）。",
                                      "Not enough papers to run BERTopic (minimum 10 required)."))
                        st.stop()
                    vos_data = largest_connected_component(vos_data)
                except ImportError:
                    st.error(tl("BERTopicをインストールしてください: `pip install bertopic`",
                                "Install BERTopic: `pip install bertopic`"))
                    st.stop()

        elif tl("引用系譜","Citation Genealogy") in analysis_type:
            _is_pubmed_src = any(w.get("_source") == "pubmed" for w in works)
            if _is_pubmed_src:
                st.error(tl(
                    "⚠️ 引用系譜はOpenAlexデータ専用です。PubMedでは参考文献リストが取得できません。",
                    "⚠️ Citation Genealogy requires OpenAlex data. PubMed does not provide reference lists."
                ))
                st.stop()
            # 選択した1本のみを起点にする
            _seed_id = genealogy_seed_id or (works[0].get("id","") if works else "")
            _seed_works = [w for w in works if w.get("id","") == _seed_id]
            if not _seed_works:
                st.error(tl(
                    "起点論文が見つかりませんでした。論文を選択し直してください。",
                    "Seed paper not found. Please re-select a paper."
                ))
                st.stop()
            _seed_title = (_seed_works[0].get("title","") or _seed_id)[:60]

            if genealogy_direction == "forward":
                _dir_ja, _dir_en = "被引用系譜", "forward citation genealogy"
            else:
                _dir_ja, _dir_en = "引用系譜", "backward citation genealogy"

            with st.spinner(tl(
                f"{_dir_ja}を構築中…「{_seed_title}」（最大{genealogy_generations}世代、各{genealogy_max_per_gen}件）",
                f"Building {_dir_en} for '{_seed_title}' (up to {genealogy_generations} gen, {genealogy_max_per_gen}/paper)..."
            )):
                if genealogy_direction == "forward":
                    _g_nodes, _g_edges = build_forward_citation_genealogy(
                        _seed_works,
                        generations=genealogy_generations,
                        max_per_gen=genealogy_max_per_gen,
                        mailto=_oa_email()
                    )
                else:
                    _g_nodes, _g_edges = build_citation_genealogy(
                        _seed_works,
                        generations=genealogy_generations,
                        max_per_gen=genealogy_max_per_gen,
                        mailto=_oa_email()
                    )
            st.session_state["genealogy_nodes"]           = _g_nodes
            st.session_state["genealogy_edges"]           = _g_edges
            st.session_state["genealogy_seed_title"]      = _seed_title
            st.session_state["genealogy_direction_value"] = genealogy_direction
            st.session_state["analysis_type"]             = analysis_type
            vos_data = None  # genealogy uses its own rendering

        elif tl("引用分析","Citation Analysis") in analysis_type:
            _ct_label = tl("書誌結合","Bibliographic Coupling") if citation_type == "bibliographic_coupling" else tl("直接引用","Direct Citation")
            with st.spinner(tl(f"引用分析（{_ct_label}）構築中...", f"Building citation network ({_ct_label})...")):
                vos_data = largest_connected_component(
                    build_citation_network(works, citation_type, min_links))
            if not vos_data.get("items"):
                st.warning(tl(
                    "引用リンクが見つかりませんでした。収集した論文セット内に相互引用がないか、参考文献データが含まれていない可能性があります。",
                    "No citation links found. Papers in the dataset may not cite each other, or reference data may be missing."))
                st.stop()

        else:  # K-means
            with st.spinner(tl(f"K-means（{n_clusters}クラスター）でクラスタリング中...",
                               f"Running K-means ({n_clusters} clusters)...")):
                vos_data, cluster_map, cluster_labels_map = build_kmeans_network(
                    works, model_key, n_clusters, min_links)
                vos_data = largest_connected_component(vos_data)

        if vos_data is not None:
            lcc_size = len(vos_data.get("items", []))
            st.info(tl(f"最大連結成分: {lcc_size} ノードを表示（孤立クラスターを除外）",
                       f"Largest connected component: {lcc_size} nodes (isolated clusters excluded)"))

            with st.spinner(tl("中心性を計算中...","Computing centrality...")):
                centrality = compute_centrality(vos_data)
            st.session_state["vos_data"] = vos_data
            st.session_state["work_keywords"] = work_keywords
            st.session_state["cluster_map"] = cluster_map
            st.session_state["cluster_labels_map"] = cluster_labels_map
            st.session_state["analysis_type"] = analysis_type
            st.session_state["viz_method"] = viz_method
            st.session_state["centrality"] = centrality

    # ── 引用系譜グラフ ──
    _g_nodes = st.session_state.get("genealogy_nodes")
    _g_edges = st.session_state.get("genealogy_edges")
    _g_ana   = st.session_state.get("analysis_type", "")
    if _g_nodes and tl("引用系譜","Citation Genealogy") in _g_ana:
        st.markdown("---")
        _g_direction   = st.session_state.get("genealogy_direction_value", "backward")
        _g_seed_title  = st.session_state.get("genealogy_seed_title", "")

        if _g_direction == "forward":
            st.subheader(tl("🌳 被引用系譜グラフ（Forward Citation Genealogy）",
                            "🌳 Forward Citation Genealogy Graph"))
            st.caption(tl(
                "矢印の向き: 引用している論文 → 引用されている論文（起点論文が中心、外側ほど新しい）",
                "Arrow direction: citing paper → cited paper (seed at center, outer nodes are newer)"
            ))
        else:
            st.subheader(tl("🌳 引用系譜グラフ（Backward Citation Genealogy）",
                            "🌳 Backward Citation Genealogy Graph"))
            st.caption(tl(
                "矢印の向き: 元論文 → 引用されている論文（起点論文が中心、外側ほど古い）",
                "Arrow direction: paper → its reference (seed at center, outer nodes are older)"
            ))

        if _g_seed_title:
            st.caption(tl(f"🔵 起点論文: **{_g_seed_title}**",
                          f"🔵 Seed paper: **{_g_seed_title}**"))

        # 世代ごとの凡例（方向に応じてラベルを変える）
        if _g_direction == "forward":
            _gen_legend = {
                0: ("🔵", tl("第0世代（起点論文）","Gen 0 (seed paper)"),           "#4477CC"),
                1: ("🟢", tl("第1世代（直接引用した論文）","Gen 1 (direct citers)"), "#44AA66"),
                2: ("🟡", tl("第2世代","Gen 2"),                                     "#CC9922"),
                3: ("🔴", tl("第3世代","Gen 3"),                                     "#CC4444"),
            }
        else:
            _gen_legend = {
                0: ("🔵", tl("第0世代（起点論文）","Gen 0 (seed paper)"),          "#4477CC"),
                1: ("🟢", tl("第1世代（直接参照）","Gen 1 (direct references)"),   "#44AA66"),
                2: ("🟡", tl("第2世代","Gen 2"),                                    "#CC9922"),
                3: ("🔴", tl("第3世代","Gen 3"),                                    "#CC4444"),
            }
        _gen_counts = {}
        for nd in _g_nodes.values():
            g = nd.get("gen", 0)
            _gen_counts[g] = _gen_counts.get(g, 0) + 1

        _lcols = st.columns(len(_gen_counts))
        for i, (g, cnt) in enumerate(sorted(_gen_counts.items())):
            icon, label, _ = _gen_legend.get(g, ("⚪", f"Gen {g}", "#aaa"))
            _lcols[i].metric(f"{icon} {label}", f"{cnt} {tl('論文','papers')}")

        # エッジのうちグラフ内ノード間のみカウント
        _valid_edges = [(s, t) for s, t in _g_edges
                        if s in _g_nodes and t in _g_nodes]
        st.caption(tl(
            f"総ノード: {len(_g_nodes)}　有向エッジ: {len(_valid_edges)}　"
            f"（矢印の向き: 元論文 → 引用されている論文）",
            f"Total nodes: {len(_g_nodes)}　Directed edges: {len(_valid_edges)}　"
            f"(Arrow direction: paper → its reference)"
        ))

        render_genealogy_pyvis(_g_nodes, _g_edges)

        # テーブル表示
        with st.expander(tl("📋 系譜ノード一覧","📋 Genealogy Node List"), expanded=True):
            import pandas as pd
            _seed_wid  = st.session_state.get("genealogy_seed_id", "")
            _g_rows = []
            for nid, nd in _g_nodes.items():
                g    = nd.get("gen", 0)
                icon = _gen_legend.get(g, ("⚪","",""))[0]
                is_seed = (nid == _seed_wid)
                _g_rows.append({
                    "_gen_num": g,
                    tl("世代","Gen"):      f"{icon} Gen {g}",
                    tl("種別","Type"):     tl("⭐ 起点論文","⭐ Seed paper") if is_seed else tl(f"第{g}世代引用","Gen {g} reference").format(g=g),
                    tl("タイトル","Title"): nd.get("title", "")[:100],
                    tl("年","Year"):       nd.get("year", ""),
                    "OpenAlex ID":         nid,
                })
            # 世代番号で正しく昇順ソート（Gen 0 → 1 → 2 → 3）
            _g_rows.sort(key=lambda r: (r["_gen_num"], 0 if r[tl("種別","Type")].startswith("⭐") else 1))
            for r in _g_rows:
                del r["_gen_num"]
            st.dataframe(pd.DataFrame(_g_rows), use_container_width=True, hide_index=True)

    # ── DOI引用ネットワーク（ページ上部に表示）──
    _cite_target = st.session_state.get("cite_target")
    if _cite_target and _cite_target.get("id"):
        st.markdown("---")
        _ct_title = _cite_target["title"]
        st.subheader(tl("🔬 引用論文ネットワーク（VOSviewer分析）",
                        "📊 DOI Citation Network Analysis"))
        st.markdown(f"**{_ct_title[:90]}{'...' if len(_ct_title)>90 else ''}**")
        st.caption(tl(
            "この論文を引用している論文群を書誌結合で分析します。"
            "年情報付きVOSviewer JSONをダウンロードしてVOSviewerで開いてください。",
            "Analyzes papers citing this work via bibliographic coupling. "
            "Download the VOSviewer JSON with year scores and open in VOSviewer."
        ))

        _col_a, _col_b = st.columns([3, 1])
        with _col_b:
            if st.button(tl("✕ 閉じる","✕ Close"), key="close_cite"):
                st.session_state["cite_target"] = None
                st.session_state["cite_works"] = []
                st.rerun()

        _cite_max = _col_a.slider(
            tl("取得する引用論文数","Max citing papers to fetch"),
            min_value=50, max_value=500, value=200, step=50,
            key="cite_max_papers"
        )
        _cite_min_links = _col_a.slider(
            tl("最小共有参考文献数（書誌結合の閾値）","Min shared references (coupling threshold)"),
            min_value=1, max_value=10, value=2, key="cite_min_links"
        )

        if st.button(tl("▶ 取得・ネットワーク構築","▶ Fetch & Build Network"),
                     type="primary", key="run_cite_vos"):
            with st.spinner(tl(
                f"引用論文を取得中（最大{_cite_max}件）...",
                f"Fetching citing papers (up to {_cite_max})..."
            )):
                _citing_works = fetch_citing_works_full(
                    _cite_target["id"], max_papers=_cite_max, mailto=_oa_email()
                )
            st.session_state["cite_works"] = _citing_works

        _citing_works = st.session_state.get("cite_works", [])

        if _citing_works:
            _years = [w.get("publication_year") for w in _citing_works
                      if w.get("publication_year")]
            _cm1, _cm2, _cm3, _cm4 = st.columns(4)
            _cm1.metric(tl("引用論文数","Citing papers"), len(_citing_works))
            _cm2.metric(tl("最古","Earliest"), min(_years) if _years else "—")
            _cm3.metric(tl("最新","Latest"),   max(_years) if _years else "—")
            _cm4.metric(tl("年数範囲","Year span"),
                        f"{max(_years)-min(_years)}年" if len(_years)>1 else "—")

            import pandas as pd
            try:
                import plotly.express as px
                _yr_df = (pd.Series(_years).value_counts()
                          .sort_index().reset_index())
                _yr_df.columns = [tl("年","Year"), tl("論文数","Papers")]
                _fig_yr = px.bar(
                    _yr_df, x=tl("年","Year"), y=tl("論文数","Papers"),
                    title=tl("引用論文の年別分布","Citing Papers by Year"),
                    color=tl("論文数","Papers"),
                    color_continuous_scale="Teal"
                )
                _fig_yr.update_layout(coloraxis_showscale=False)
                st.plotly_chart(_fig_yr, use_container_width=True)
            except ImportError:
                pass

            with st.spinner(tl("書誌結合ネットワークを構築中...","Building bibliographic coupling network...")):
                _cite_vos = build_citation_network(
                    _citing_works,
                    citation_type="bibliographic_coupling",
                    min_links=_cite_min_links
                )

            _cn_items = _cite_vos.get("items", [])
            _cn_links = _cite_vos.get("links", [])
            _ci1, _ci2 = st.columns(2)
            _ci1.metric(tl("ノード数","Nodes"), len(_cn_items))
            _ci2.metric(tl("エッジ数","Edges"), len(_cn_links))

            if _cn_items:
                st.success(tl(
                    "✅ DOI引用ネットワーク構築完了。JSONをダウンロードしてください。",
                    "✅ DOI citation network built. Download the JSON."
                ))
                st.info(tl(
                    "💡 VOSviewerで **Scores → Year** を選択すると、論文が出版年ごとに色分けされます。",
                    "💡 In VOSviewer, select **Scores → Year** to color nodes by publication year."
                ))
                _vos_json = json.dumps({"network": _cite_vos}, ensure_ascii=False, indent=2)
                _safe_title = re.sub(r"[^\w]", "_", _ct_title[:30])
                st.download_button(
                    tl("📥 VOSviewer JSON ダウンロード","📥 Download VOSviewer JSON"),
                    data=_vos_json.encode("utf-8"),
                    file_name=f"doi_cite_{_safe_title}.json",
                    mime="application/json",
                    type="primary"
                )
            else:
                st.warning(tl(
                    f"共有参考文献が{_cite_min_links}件以上の論文ペアがありません。"
                    "「最小共有参考文献数」を1に下げてみてください。",
                    f"No paper pairs share ≥{_cite_min_links} references. "
                    "Try lowering 'Min shared references' to 1."
                ))

    # ── 結果表示 ──
    vos_data = st.session_state.get("vos_data")
    if vos_data:
        items = vos_data.get("items",[])
        links = vos_data.get("links",[])

        c1, c2, c3 = st.columns(3)
        c1.metric(tl("ノード","Nodes"), len(items))
        c2.metric(tl("エッジ","Edges"), len(links))
        c3.metric(tl("論文","Papers"), len(works))

        _ana = st.session_state.get("analysis_type","")
        _node_edge_desc = {
            tl("共著ネットワーク","Co-authorship Network"): (
                tl("👤 著者（研究者）1人","👤 One researcher"),
                tl("🤝 共著関係（2人が同じ論文を執筆）。エッジの太さ＝共著回数",
                   "🤝 Co-authorship (two researchers wrote a paper together). Edge weight = frequency")
            ),
            tl("KeyBERT キーワード共起","KeyBERT Keyword Co-occurrence"): (
                tl("🔑 キーワード（抽出された語句）1語","🔑 One keyword (extracted phrase)"),
                tl("🔗 共起関係（2語が同じ論文に登場）。エッジの太さ＝同時出現回数",
                   "🔗 Co-occurrence (two keywords appear in the same paper). Edge weight = frequency")
            ),
            tl("BERTopic クラスタリング","BERTopic Clustering"): (
                tl("📦 トピック（BERTopic が自動生成したクラスター）1群","📦 One topic cluster (auto-generated by BERTopic)"),
                tl("🔗 トピック共起（同じ著者が両トピックの論文を執筆）",
                   "🔗 Topic co-occurrence (same author wrote papers in both topics)")
            ),
            tl("K-means クラスタリング","K-means Clustering"): (
                tl("📦 トピック（K-means が分類したクラスター）1群","📦 One topic cluster (classified by K-means)"),
                tl("🔗 トピック共起（同じ著者が両トピックの論文を執筆）",
                   "🔗 Topic co-occurrence (same author wrote papers in both topics)")
            ),
            tl("引用分析","Citation Analysis"): (
                tl("📄 論文1本（タイトル＋出版年）","📄 One paper (title + year)"),
                tl("🔗 書誌結合＝共通参照数 / 直接引用＝論文Aが論文Bを引用。エッジの太さ＝強度",
                   "🔗 Bibliographic coupling = shared references / Direct citation = paper A cites B. Edge weight = strength")
            ),
        }
        if _ana in _node_edge_desc:
            _nd, _ed = _node_edge_desc[_ana]
            st.caption(f"**{tl('ノード','Node')}**: {_nd}　　**{tl('エッジ','Edge')}**: {_ed}")


        st.markdown("---")
        viz_method = st.session_state.get("viz_method","")

        # ── アプリ内表示（PyVis）──
        if tl("アプリ内","In-app") in viz_method:
            st.subheader(tl("🕸️ インタラクティブネットワーク","🕸️ Interactive Network"))
            st.caption(tl("ノードをドラッグ・ズームで操作できます","Drag nodes and scroll to zoom"))
            render_pyvis(vos_data)

        # ── VOSviewer ──
        elif "VOSviewer" in viz_method:
            st.subheader("VOSviewer")
            json_str = json.dumps({"network": vos_data})
            st.download_button(
                "⬇ Download VOSviewer JSON",
                json_str, file_name=selected_ds+"_vosviewer.json",
                mime="application/json", use_container_width=True
            )
            st.markdown("**手順 / How to:**")
            st.markdown("1. 上のボタンでJSONをダウンロード")
            st.markdown("2. [app.vosviewer.com](https://app.vosviewer.com) を開く")
            st.markdown("3. **Open** → **JSON file** → ダウンロードしたファイルを選択")

        # ── Gephi Lite (GEXF) ──
        else:
            st.subheader("Gephi Lite (GEXF)")
            import base64 as _b64
            gexf_str = to_gexf(vos_data)
            gexf_filename = selected_ds + "_gephi.gexf"
            gexf_b64 = _b64.b64encode(gexf_str.encode("utf-8")).decode()

            # ワンクリック: GEXFダウンロード + Gephi Lite を同時に開く
            _html = f"""
<style>
  .gephi-btn {{
    display:inline-block; width:100%; padding:10px 0;
    background:#0068c9; color:#fff; border:none; border-radius:6px;
    font-size:15px; font-weight:600; cursor:pointer; text-align:center;
  }}
  .gephi-btn:hover {{ background:#0052a3; }}
  .hint {{ margin-top:10px; font-size:13px; color:#555; }}
</style>
<button class="gephi-btn" onclick="openGephiLite()">
  🚀 {tl('Gephi Lite で開く（GEXFを自動ダウンロード）',
         'Open in Gephi Lite (auto-download GEXF)')}
</button>
<div class="hint">
  {tl('ボタンを押すとGEXFファイルが保存され、Gephi Liteが新しいタブで開きます。<br>ファイルをGephi Liteの画面にドラッグ＆ドロップしてください。',
      'The GEXF file will be downloaded and Gephi Lite opens in a new tab.<br>Drag and drop the file onto the Gephi Lite window.')}
</div>
<script>
function openGephiLite() {{
  const b64 = "{gexf_b64}";
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], {{type: "application/xml"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "{gexf_filename}";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  window.open("https://gephi.org/gephi-lite/", "_blank");
}}
</script>
"""
            st.components.v1.html(_html, height=110)
            st.markdown("---")
            st.markdown(tl(
                "💡 **使い方**: ボタンを押す → ダウンロードされた `.gexf` ファイルを Gephi Lite 画面にドラッグ＆ドロップ",
                "💡 **Usage**: Click the button → drag and drop the downloaded `.gexf` file onto Gephi Lite"
            ))

        # ── ホワイトスペース可視化（Feature 4）──
        # BERTopic または K-means クラスタリング実行後のみ表示
        _ws_ana = st.session_state.get("analysis_type", "")
        _cluster_map_ws = st.session_state.get("cluster_map", {})
        _cluster_labels_ws = st.session_state.get("cluster_labels_map", {})
        _is_cluster_ana = (
            tl("BERTopic", "BERTopic") in _ws_ana or
            tl("K-means", "K-means") in _ws_ana
        )
        if _is_cluster_ana and _cluster_map_ws:
            st.markdown("---")
            st.subheader(tl("🗺️ ホワイトスペース可視化", "🗺️ Research White Space"))
            st.caption(tl(
                "各クラスターの論文数（X軸）と平均被引用数（Y軸）を散布図で表示します。"
                "右上＝確立された主流分野　左上＝高インパクトのニッチ（ホワイトスペース候補）",
                "Scatter plot of cluster size (X) vs average citations (Y). "
                "Top-right = established field. Top-left = high-impact niche (white space candidate)."
            ))
            try:
                import plotly.express as px
                import pandas as pd
                # クラスターごとに論文数と平均被引用数を計算
                _ws_cluster_papers = defaultdict(list)
                for w in works:
                    wid = w.get("id", "")
                    clbl = _cluster_map_ws.get(wid)
                    if clbl is not None:
                        _ws_cluster_papers[clbl].append(w.get("cited_by_count", 0) or 0)
                _ws_rows = []
                for c, cits in _ws_cluster_papers.items():
                    label = _cluster_labels_ws.get(c, f"Cluster {c}")
                    _ws_rows.append({
                        tl("論文数", "Papers"): len(cits),
                        tl("平均被引用数", "Avg Citations"): round(sum(cits) / len(cits), 2) if cits else 0,
                        tl("クラスター", "Cluster"): str(label)[:40],
                    })
                _ws_df = pd.DataFrame(_ws_rows)
                if not _ws_df.empty:
                    _ws_fig = px.scatter(
                        _ws_df,
                        x=tl("論文数", "Papers"),
                        y=tl("平均被引用数", "Avg Citations"),
                        size=tl("論文数", "Papers"),
                        color=tl("クラスター", "Cluster"),
                        hover_name=tl("クラスター", "Cluster"),
                        title=tl(
                            "クラスター別：論文数 vs 平均被引用数",
                            "Cluster: Paper Count vs Avg Citation Count"
                        ),
                        size_max=60,
                    )
                    _ws_fig.update_layout(
                        xaxis_title=tl("論文数（クラスター規模）", "Number of Papers (cluster size)"),
                        yaxis_title=tl("平均被引用数", "Average Citation Count"),
                    )
                    st.plotly_chart(_ws_fig, use_container_width=True)
                    st.caption(tl(
                        "📌 左上のバブル（論文少・引用高）が研究のホワイトスペース候補です。"
                        "右下（論文多・引用低）は競争が激しく成熟した分野です。",
                        "📌 Bubbles in the top-left (few papers, high citations) are white space candidates. "
                        "Bottom-right (many papers, low citations) indicates a crowded, mature field."
                    ))
            except ImportError:
                st.warning(tl(
                    "ホワイトスペース可視化には plotly が必要です。`pip install plotly`",
                    "plotly required for White Space plot. Run `pip install plotly`."
                ))

        # ── 国際比較（Feature 5）──
        st.markdown("---")
        st.subheader(tl("🌍 国際比較：国別論文数", "🌍 International Comparison: Papers by Country"))
        st.caption(tl(
            "各論文の著者所属機関から国コードを抽出し、上位10か国の論文数を表示します。",
            "Extracts country codes from author affiliations and shows top 10 countries by paper count."
        ))
        try:
            import plotly.express as px
            import pandas as pd
            _country_count = defaultdict(int)
            for w in works:
                _seen_countries = set()
                for auth in w.get("authorships", []):
                    for inst in auth.get("institutions", []):
                        cc = inst.get("country_code", "")
                        if cc and cc not in _seen_countries:
                            _country_count[cc] += 1
                            _seen_countries.add(cc)
            if _country_count:
                _cc_df = (
                    pd.DataFrame(list(_country_count.items()),
                                 columns=[tl("国コード", "Country"), tl("論文数", "Papers")])
                    .sort_values(tl("論文数", "Papers"), ascending=False)
                    .head(10)
                )
                _cc_fig = px.bar(
                    _cc_df,
                    x=tl("国コード", "Country"),
                    y=tl("論文数", "Papers"),
                    color=tl("論文数", "Papers"),
                    color_continuous_scale="Blues",
                    title=tl("上位10か国の論文数", "Top 10 Countries by Paper Count"),
                )
                _cc_fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(_cc_fig, use_container_width=True)
            else:
                st.info(tl(
                    "所属機関の国情報がデータに含まれていません（PubMedデータでは利用不可）。",
                    "No country information found in affiliations (not available for PubMed data)."
                ))
        except ImportError:
            st.warning(tl(
                "グラフ表示には plotly が必要です。`pip install plotly`",
                "plotly required. Run `pip install plotly`."
            ))

        # ── 中心性ランキング ──
        st.markdown("---")
        centrality = st.session_state.get("centrality", {})
        if centrality:
            st.subheader(tl("📊 中心性ランキング","📊 Centrality Ranking"))
            _metric_opts = {
                tl("PageRank（影響力）","PageRank (influence)"): "pagerank",
                tl("媒介中心性（橋渡し役）","Betweenness (bridge)"):  "betweenness",
                tl("次数中心性（接続数）","Degree (connections)"):  "degree",
            }
            _metric_label = st.radio(
                tl("指標","Metric"), list(_metric_opts.keys()),
                horizontal=True, key="centrality_metric"
            )
            _metric_key = _metric_opts[_metric_label]
            _top_n = st.slider(tl("表示件数","Top N"), 5, 50, 20, key="centrality_topn")
            _ranked = sorted(centrality.items(), key=lambda x: x[1][_metric_key], reverse=True)[:_top_n]

            _metric_descs = {
                "pagerank":    tl(
                    "**PageRank**: 重要なノードからリンクされているほど高スコア。影響力の高い著者・論文を示します。",
                    "**PageRank**: Higher score when linked from important nodes. Identifies highly influential authors/papers."
                ),
                "betweenness": tl(
                    "**媒介中心性**: 他のノード間の最短経路上に多く現れるノード。異なるグループを橋渡しするハブを示します。",
                    "**Betweenness**: Nodes frequently on shortest paths between others. Identifies bridges between research groups."
                ),
                "degree":      tl(
                    "**次数中心性**: 直接つながっているノードの数（正規化）。最も多くの著者・論文と接続しているノードを示します。",
                    "**Degree**: Number of direct connections (normalized). Shows nodes connected to the most authors/papers."
                ),
            }
            st.caption(_metric_descs[_metric_key])

            import pandas as pd
            _df = pd.DataFrame([
                {
                    tl("順位","Rank"):       i + 1,
                    tl("名前","Name"):       label[:50],
                    "PageRank":              vals["pagerank"],
                    tl("媒介中心性","Betweenness"): vals["betweenness"],
                    tl("次数中心性","Degree"):    vals["degree"],
                }
                for i, (label, vals) in enumerate(_ranked)
            ])
            st.dataframe(
                _df.style.highlight_max(
                    subset=["PageRank",
                            tl("媒介中心性","Betweenness"),
                            tl("次数中心性","Degree")],
                    color="#d4edda"
                ),
                use_container_width=True, hide_index=True
            )

            # ── KAKEN助成金分析へ引き渡し ──
            st.markdown("---")
            _ana_label_s2 = st.session_state.get("analysis_type", "")
            _is_author_net = tl("共著","Co-authorship") in _ana_label_s2
            _term_type_ja  = "著者名" if _is_author_net else "キーワード"
            _term_type_en  = "author name" if _is_author_net else "keyword"
            _ranked_for_kaken = sorted(
                centrality.items(), key=lambda x: x[1]["pagerank"], reverse=True
            )[:30]
            _top30_labels = [label for label, _ in _ranked_for_kaken]

            with st.expander(
                tl("📤 ③ KAKEN助成金分析へ引き渡す",
                   "📤 Send to ③ KAKEN Grant Analysis"),
                expanded=False
            ):
                st.caption(tl(
                    f"ランキング上位の{_term_type_ja}をKAKEN助成金検索へ引き渡します。"
                    "KAKENは日本語の研究課題名で登録されているため、**日本語への変換を推奨**します（英語も可）。",
                    f"Transfer top {_term_type_en}s from the ranking to KAKEN grant search. "
                    "Japanese keywords are recommended as KAKEN project titles are in Japanese (English also works)."
                ))

                _sel_item = st.selectbox(
                    tl(f"引き渡す{_term_type_ja}を選択（PageRank上位30）",
                       f"Select {_term_type_en} to transfer (Top 30 by PageRank)"),
                    [tl("（選択してください）","(select one)")] + _top30_labels,
                    key="kaken_transfer_select"
                )
                _default_val = (
                    "" if _sel_item == tl("（選択してください）","(select one)")
                    else _sel_item
                )

                _kaken_send_kw = st.text_input(
                    tl("KAKENキーワード（日本語推奨・編集可）",
                       "KAKEN keyword (Japanese recommended, editable)"),
                    value=_default_val,
                    key="kaken_transfer_input",
                    placeholder=tl(
                        "例: リチウムイオン電池 / 固体電解質",
                        "e.g. リチウムイオン電池 / solid electrolyte"
                    )
                )

                if _is_author_net:
                    st.info(tl(
                        "💡 共著ネットワークの著者名を引き渡す場合、KAKEN検索では"
                        "「研究課題名キーワード」での検索が効果的です。"
                        "必要に応じて著者の研究分野のキーワードに変換してください。",
                        "💡 For co-authorship networks, searching by the researcher's "
                        "topic keyword tends to work better than the author name in KAKEN. "
                        "Consider converting to a research topic keyword if needed."
                    ))

                _send_col, _clear_col = st.columns([3, 1])
                _kw_to_send = _kaken_send_kw.strip() or _default_val
                if _send_col.button(
                    tl("📤 KAKEN分析へ送る","📤 Send to KAKEN Analysis"),
                    type="primary", key="kaken_send_btn",
                    disabled=not bool(_kw_to_send)
                ):
                    st.session_state["kaken_transfer_kw"]        = _kw_to_send
                    st.session_state["kaken_transfer_is_author"]  = _is_author_net
                    st.session_state["kaken_transfer_source"]     = _ana_label_s2
                    st.session_state["kaken_transfer_applied"]    = False
                    st.success(tl(
                        f"✅「{_kw_to_send}」を送信しました。"
                        "サイドバーで「③ KAKEN助成金分析」を選択してください。",
                        f"✅ Sent '{_kw_to_send}' to KAKEN. "
                        "Select '③ KAKEN Grant Analysis' in the sidebar."
                    ))
                if _clear_col.button(tl("クリア","Clear"), key="kaken_send_clear"):
                    for _k in ["kaken_transfer_kw","kaken_transfer_is_author",
                               "kaken_transfer_source","kaken_transfer_applied"]:
                        st.session_state.pop(_k, None)
                    st.rerun()

        # ── 重要論文ランキング（被引用数順） ──
        st.markdown("---")
        st.subheader(tl("📄 重要論文ランキング（被引用数順）",
                        "📄 Key Papers Ranking (by citation count)"))

        # PubMedデータ（被引用数なし）の場合はメッセージのみ表示して終了
        _is_pubmed_src = any(w.get("_source") == "pubmed" for w in works)
        if _is_pubmed_src:
            st.warning(tl(
                "⚠️ PubMedは引用情報を提供していないため、重要論文ランキングは表示できません。",
                "⚠️ PubMed does not provide citation counts, so the key papers ranking is unavailable."
            ))
        else:
            st.caption(tl(
                "ネットワーク分析に依存せず、データセット内の論文を**被引用数**で直接ランキングします。"
                "論文数が少ない場合でも信頼性の高い重要度指標です。",
                "Ranks papers directly by **citation count**, independent of network analysis. "
                "A reliable importance measure even with smaller datasets."
            ))
            st.caption(tl(
                "💰 **研究費対被引用コスト（Funding Cost per Citation）の推定**: "
                "ステップ3のKAKENデータで取得した助成金額をこの表の被引用数で割ることで、"
                "「1被引用あたりの研究費」を概算できます。研究費対効果の評価指標として活用してください。",
                "💰 **Funding Cost per Citation**: Divide the grant amount from Step 3 KAKEN data "
                "by the citation counts in this table to estimate the cost per citation. "
                "Use this as a proxy metric for research funding efficiency."
            ))

        import pandas as pd
        if not _is_pubmed_src:
            _top_n_papers = st.slider(
                tl("表示件数","Top N papers"), 5, 100, 20, key="top_papers_n"
            )
            _paper_rows = []
            for w in works:
                title  = w.get("title","") or "No title"
                year   = w.get("publication_year","") or 0
                cited  = w.get("cited_by_count", 0) or 0
                doi    = w.get("doi","") or ""
                auths  = [a.get("author",{}).get("display_name","")
                          for a in w.get("authorships",[])[:3]]
                topics = [t.get("display_name","") for t in w.get("topics",[])[:2]]
                _paper_rows.append({
                    tl("被引用数","Cited"):   cited,
                    tl("タイトル","Title"):   title,
                    tl("著者","Authors"):     "; ".join(filter(None, auths)),
                    tl("年","Year"):          year,
                    tl("トピック","Topics"):  "; ".join(topics),
                    "DOI":                    doi,
                })

            _papers_df = (
                pd.DataFrame(_paper_rows)
                .sort_values(tl("被引用数","Cited"), ascending=False)
                .reset_index(drop=True)
            )
            _papers_df.index = _papers_df.index + 1  # 1始まり
            _display_cols = None

        # 上位N件をテーブル表示（PubMedは非表示）
        if not _is_pubmed_src:
            _show_df = _papers_df.head(_top_n_papers)
            st.dataframe(_show_df.drop(columns=["DOI"]), use_container_width=True)

        # 詳細エキスパンダー（PubMedは非表示）
        if not _is_pubmed_src:
            for rank, row in _papers_df.head(_top_n_papers).iterrows():
                _cited_label = tl(f"📊 被引用: {row[tl('被引用数','Cited')]}",
                                  f"📊 Cited: {row[tl('被引用数','Cited')]}")
                with st.expander(
                    f"{rank}. {row[tl('タイトル','Title')][:70]}"
                    f"{'...' if len(row[tl('タイトル','Title')])>70 else ''}  "
                    f"({row[tl('年','Year')]})  {_cited_label}"
                ):
                    st.markdown(f"**{row[tl('タイトル','Title')]}**")
                    if row[tl("著者","Authors")]:
                        st.caption("✍️ " + row[tl("著者","Authors")])
                    if row[tl("トピック","Topics")]:
                        st.caption("🏷️ " + row[tl("トピック","Topics")])
                    if row["DOI"]:
                        st.markdown(f"[🔗 DOI]({row['DOI']})")
                # アブストラクト
                _w = next((w for w in works if w.get("title","") == row[tl("タイトル","Title")]), None)
                if _w:
                    _ab = reconstruct_abstract(_w.get("abstract_inverted_index", {}), work=_w)
                    if _ab:
                        st.markdown("**Abstract**")
                        st.write(_ab[:400] + ("..." if len(_ab) > 400 else ""))
                    _wid = _w.get("id", "")
                    if _wid:
                        _cite_key = f"doi_net_{_wid}"
                        if st.button(
                            tl("📊 DOI引用ネットワーク生成","📊 Build DOI Citation Network"),
                            key=f"cite_btn_{rank}"
                        ):
                            with st.spinner(tl("引用論文を取得・ネットワーク構築中...","Fetching & building DOI citation network...")):
                                _cw = fetch_citing_works_full(_wid, max_papers=200, mailto=_oa_email())
                                if _cw:
                                    _vos = build_citation_network(_cw, citation_type="bibliographic_coupling", min_links=1)
                                    st.session_state[_cite_key] = {
                                        "json":    json.dumps({"network": _vos}, ensure_ascii=False, indent=2),
                                        "title":   row[tl("タイトル","Title")],
                                        "n_items": len(_vos.get("items", [])),
                                        "n_links": len(_vos.get("links", [])),
                                    }
                                else:
                                    st.session_state[_cite_key] = {"json": None}
                        _net = st.session_state.get(_cite_key)
                        if _net:
                            if _net.get("json"):
                                st.success(tl(f"✅ ノード(DOI): {_net['n_items']}件 / エッジ: {_net['n_links']}件",
                                              f"✅ Nodes(DOI): {_net['n_items']} / Edges: {_net['n_links']}"))
                                _safe_t = re.sub(r"[^\w]", "_", _net["title"][:30])
                                st.download_button(
                                    tl("📥 DOI引用ネットワーク JSON ダウンロード","📥 Download DOI Citation Network JSON"),
                                    data=_net["json"].encode("utf-8"),
                                    file_name=f"doi_cite_{_safe_t}.json",
                                    mime="application/json",
                                    key=f"dl_cite_{rank}"
                                )
                            else:
                                st.info(tl("引用論文が見つかりませんでした。","No citing papers found."))

        # CSVダウンロード
        _csv_papers = _papers_df.to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            tl("📥 CSVダウンロード（全件）","📥 Download CSV (all papers)"),
            data=_csv_papers,
            file_name="key_papers.csv",
            mime="text/csv"
        )

        # ── 上位ノード & 論文逆引き ──
        st.markdown("---")
        if items:
            st.subheader(tl("🏆 主要ノード（上位20）","🏆 Top 20 Nodes"))
            key = list(items[0]["weights"].keys())[0]
            sorted_items = sorted(items, key=lambda x: x["weights"].get(key,0), reverse=True)[:20]
            work_keywords = st.session_state.get("work_keywords",{})
            ana = st.session_state.get("analysis_type","")

            for i, item in enumerate(sorted_items, 1):
                w_val = round(item["weights"].get(key,0), 2)
                btn_label = f"{i}. {item['label']}  ({key}: {w_val})"
                if st.button(btn_label, key="nd_"+str(i), use_container_width=True):
                    st.session_state["sel_node"] = item["label"]
                    st.session_state["sel_node_id"] = item.get("id", item["label"])

            # 選択ノードの関連論文
            sel = st.session_state.get("sel_node")
            if sel:
                st.markdown("---")
                st.markdown(f"### 📄 **「{sel}」** の関連論文")

                if tl("KeyBERT","KeyBERT") in ana:
                    matched = [w for w in works if sel in work_keywords.get(w.get("id",""),[])]
                elif tl("共著","Co-authorship") in ana:
                    matched = [w for w in works if any(
                        a.get("author",{}).get("display_name","") == sel
                        for a in w.get("authorships",[]))]
                elif tl("BERTopic","BERTopic") in ana or tl("K-means","K-means") in ana:
                    cluster_map = st.session_state.get("cluster_map",{})
                    sel_id = st.session_state.get("sel_node_id","")
                    matched = [w for w in works if str(cluster_map.get(w.get("id",""),"")) == str(sel_id)]
                else:
                    matched = []

                st.caption(tl(f"{len(matched)}件", f"{len(matched)} papers"))
                for w in matched[:20]:
                    title = w.get("title","") or "No title"
                    year  = str(w.get("publication_year",""))
                    doi   = w.get("doi","") or ""
                    auths = [a.get("author",{}).get("display_name","") for a in w.get("authorships",[])[:3]]
                    cited = w.get("cited_by_count",0)
                    with st.expander(f"📄 {title[:70]}{'...' if len(title)>70 else ''}"):
                        st.markdown(f"**{title}**")
                        if auths: st.caption("✍️ " + ", ".join(filter(None,auths)))
                        parts = []
                        if year:  parts.append("📅 " + year)
                        if cited: parts.append(tl(f"📊 被引用: {cited}", f"📊 Cited: {cited}"))
                        if parts: st.caption("  |  ".join(parts))
                        if doi:   st.markdown(f"[🔗 DOI]({doi})")
                        ab = reconstruct_abstract(w.get("abstract_inverted_index",{}), work=w)
                        if ab:
                            st.markdown("**Abstract**")
                            st.write(ab[:500] + ("..." if len(ab)>500 else ""))
                        _wid2 = w.get("id", "")
                        if _wid2:
                            _cite_key2 = f"doi_net_{_wid2}"
                            if st.button(
                                tl("📊 DOI引用ネットワーク生成","📊 Build DOI Citation Network"),
                                key=f"cite_nd_{_wid2}"
                            ):
                                with st.spinner(tl("引用論文を取得・ネットワーク構築中...","Fetching & building DOI citation network...")):
                                    _cw2 = fetch_citing_works_full(_wid2, max_papers=200, mailto=_oa_email())
                                    if _cw2:
                                        _vos2 = build_citation_network(_cw2, citation_type="bibliographic_coupling", min_links=1)
                                        st.session_state[_cite_key2] = {
                                            "json":    json.dumps({"network": _vos2}, ensure_ascii=False, indent=2),
                                            "title":   title,
                                            "n_items": len(_vos2.get("items", [])),
                                            "n_links": len(_vos2.get("links", [])),
                                        }
                                    else:
                                        st.session_state[_cite_key2] = {"json": None}
                            _net2 = st.session_state.get(_cite_key2)
                            if _net2:
                                if _net2.get("json"):
                                    st.success(tl(f"✅ ノード(DOI): {_net2['n_items']}件 / エッジ: {_net2['n_links']}件",
                                                  f"✅ Nodes(DOI): {_net2['n_items']} / Edges: {_net2['n_links']}"))
                                    _safe_t2 = re.sub(r"[^\w]", "_", _net2["title"][:30])
                                    st.download_button(
                                        tl("📥 DOI引用ネットワーク JSON ダウンロード","📥 Download DOI Citation Network JSON"),
                                        data=_net2["json"].encode("utf-8"),
                                        file_name=f"doi_cite_{_safe_t2}.json",
                                        mime="application/json",
                                        key=f"dl_nd_{_wid2}"
                                    )
                                else:
                                    st.info(tl("引用論文が見つかりませんでした。","No citing papers found."))

                if st.button(tl("✕ 選択解除","✕ Clear"), key="clr_nd"):
                    st.session_state["sel_node"] = None
                    st.rerun()

        # ── 特許×論文ネットワーク（Lens.orgデータセットのみ）──
        _patents_in_ds = data.get("patents", [])
        if _patents_in_ds:
            st.markdown("---")
            st.subheader(tl("🔬 特許×論文 ネットワーク", "🔬 Patent × Paper Network"))
            st.caption(tl(
                "特許のNPL引用から照合された論文とのリンクを VOSviewer JSON 形式でエクスポートします。",
                "Export patent-to-paper NPL citation links as VOSviewer JSON."
            ))

            _pat_n = len(_patents_in_ds)
            _paper_n = len(works)
            _npl_total = sum(len(p.get("npl_citations", [])) for p in _patents_in_ds)
            _pm1, _pm2, _pm3 = st.columns(3)
            _pm1.metric(tl("特許数", "Patents"), _pat_n)
            _pm2.metric(tl("照合論文数", "Resolved papers"), _paper_n)
            _pm3.metric(tl("NPL引用（合計）", "NPL citations (total)"), _npl_total)

            if st.button(
                tl("▶ 特許×論文ネットワーク構築 & ダウンロード",
                   "▶ Build Patent×Paper Network & Download"),
                type="primary", key="build_patent_network",
                use_container_width=True,
            ):
                with st.spinner(tl("特許×論文ネットワークを構築中...",
                                   "Building patent×paper network...")):
                    _pp_net = build_patent_paper_network(_patents_in_ds, works)

                _pp_items = _pp_net.get("items", [])
                _pp_links = _pp_net.get("links", [])
                _pp_patents = [i for i in _pp_items if i.get("cluster") == 1]
                _pp_papers  = [i for i in _pp_items if i.get("cluster") == 2]

                st.success(tl(
                    f"✅ ノード: 特許 {len(_pp_patents)}件 / 論文 {len(_pp_papers)}件  "
                    f"/ エッジ: {len(_pp_links)}件",
                    f"✅ Nodes: {len(_pp_patents)} patents / {len(_pp_papers)} papers  "
                    f"/ Edges: {len(_pp_links)}"
                ))

                if _pp_items:
                    _pp_json = json.dumps({"network": _pp_net}, ensure_ascii=False, indent=2)
                    _pp_fname = re.sub(r"[^\w]", "_", selected_ds[:30]) + "_patent_paper.json"
                    st.download_button(
                        tl("📥 VOSviewer JSON ダウンロード（特許×論文）",
                           "📥 Download VOSviewer JSON (Patent×Paper)"),
                        data=_pp_json.encode("utf-8"),
                        file_name=_pp_fname,
                        mime="application/json",
                        type="primary",
                        key="dl_patent_paper_net",
                    )
                    st.info(tl(
                        "💡 VOSviewerで開いて Cluster 1（特許）と Cluster 2（論文）の色分けを確認できます。",
                        "💡 Open in VOSviewer to see Cluster 1 (patents) and Cluster 2 (papers) colored separately."
                    ))
                else:
                    st.warning(tl(
                        "リンクが見つかりませんでした。DOIが抽出されなかった可能性があります。",
                        "No links found. DOIs may not have been extracted from NPL citations."
                    ))

# ════════════════════════════════════════════
# ステップ3: KAKEN助成金分析
# ════════════════════════════════════════════
if tl("③ KAKEN助成金分析","③ KAKEN Grant Analysis") in step:
    st.header(tl("💰 KAKEN助成金分析","💰 KAKEN Grant Analysis"))
    st.caption(tl(
        "OpenAlexが提供するKAKEN（科学研究費助成事業）データを集計・可視化します。",
        "Aggregate and visualize KAKEN (Grants-in-Aid for Scientific Research) data via OpenAlex."
    ))

    # ── ステップ2からの引き継ぎ ──
    _kaken_from_s2     = st.session_state.get("kaken_transfer_kw", "")
    _kaken_from_s2_src = st.session_state.get("kaken_transfer_source", "")
    _kaken_applied     = st.session_state.get("kaken_transfer_applied", False)
    if _kaken_from_s2 and not _kaken_applied:
        st.info(tl(
            f"📤 **ステップ2から引き継ぎ中**: 「{_kaken_from_s2}」"
            f"（分析手法: {_kaken_from_s2_src}）\n\n"
            "「✅ 検索フィールドに適用」を押すとサイドバーのキーワード欄に自動入力されます。",
            f"📤 **Transferred from Step 2**: '{_kaken_from_s2}' "
            f"(Analysis: {_kaken_from_s2_src})\n\n"
            "Click '✅ Apply to search field' to auto-fill the keyword in the sidebar."
        ))
        _apply_col, _dismiss_col = st.columns([2, 1])
        if _apply_col.button(
            tl("✅ 検索フィールドに適用","✅ Apply to search field"),
            type="primary", key="kaken_apply_transfer"
        ):
            st.session_state["kaken_kw"]             = _kaken_from_s2
            st.session_state["kaken_transfer_applied"] = True
            st.rerun()
        if _dismiss_col.button(tl("✕ 閉じる","✕ Dismiss"), key="kaken_dismiss_transfer"):
            for _k in ["kaken_transfer_kw","kaken_transfer_source",
                       "kaken_transfer_is_author","kaken_transfer_applied"]:
                st.session_state.pop(_k, None)
            st.rerun()
    elif _kaken_from_s2 and _kaken_applied:
        st.success(tl(
            f"📤 ステップ2から引き継ぎ適用済み: 「{_kaken_from_s2}」",
            f"📤 Applied from Step 2: '{_kaken_from_s2}'"
        ))

    # ── サイドバーフィルタ ──
    with st.sidebar:
        st.markdown("---")
        st.subheader(tl("🔍 KAKENフィルタ","🔍 KAKEN Filters"))
        kaken_kw = st.text_input(
            tl("キーワード（研究課題名）","Keyword (project title)"),
            key="kaken_kw",
            placeholder=tl("例: 機械学習", "e.g. machine learning")
        )
        _c1, _c2 = st.columns(2)
        kaken_yr_from = _c1.number_input(
            tl("年度（以降）","Year from"), min_value=1965, max_value=2030,
            value=2010, step=1, key="kaken_yr_from"
        )
        kaken_yr_to = _c2.number_input(
            tl("年度（以前）","Year to"), min_value=1965, max_value=2030,
            value=2024, step=1, key="kaken_yr_to"
        )
        kaken_fetch = st.slider(tl("取得件数","Fetch count"), 50, 500, 200, step=50, key="kaken_fetch")
        kaken_sort = st.radio(
            tl("並び順","Sort by"),
            [tl("配分金額（多い順）","Amount (desc)"), tl("論文数（多い順）","Outputs (desc)")],
            key="kaken_sort"
        )
        kaken_btn = st.button(tl("🔍 検索","🔍 Search"), key="kaken_btn", type="primary")

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_kaken_awards(keyword, sort_field, per_page):
        awards, cursor = [], "*"
        base = "https://api.openalex.org/awards"
        filters = ["provenance:kaken"]
        if keyword:
            filters.append(f"display_name.search:{keyword}")
        filt_str = ",".join(filters)
        per_req = min(per_page, 200)
        fetched = 0
        while fetched < per_page:
            params = {
                "filter": filt_str,
                "sort": f"{sort_field}:desc",
                "per_page": per_req,
                "cursor": cursor,
                "mailto": _oa_email(),
            }
            try:
                r = requests.get(base, params=params, timeout=20)
                data = r.json()
                batch = data.get("results", [])
                if not batch:
                    break
                awards.extend(batch)
                fetched += len(batch)
                cursor = data.get("meta", {}).get("next_cursor")
                if not cursor:
                    break
            except Exception:
                break
        return awards

    def _extract_pi(desc):
        if not desc:
            return ""
        m = re.search(r"Principal Investigator[：:](.+?)(?:,|$)", desc)
        return m.group(1).strip() if m else ""

    if kaken_btn:
        _sort_field = "amount" if tl("配分金額","Amount") in kaken_sort else "funded_outputs_count"
        with st.spinner(tl("KAKENデータを取得中...","Fetching KAKEN data...")):
            _awards = fetch_kaken_awards(kaken_kw, _sort_field, kaken_fetch)
        st.session_state["kaken_awards"] = _awards
        st.session_state["kaken_kw_used"] = kaken_kw

    _raw_awards = st.session_state.get("kaken_awards", [])

    if not _raw_awards:
        st.info(tl(
            "左のフィルタでキーワードや年度を指定して「🔍 検索」を押してください。",
            "Set keyword / year filters on the left and click '🔍 Search'."
        ))
    else:
        import pandas as pd

        # 年度フィルタ（クライアント側）
        _awards_filt = [
            a for a in _raw_awards
            if kaken_yr_from <= (a.get("start_year") or 0) <= kaken_yr_to
        ]

        st.caption(tl(f"対象件数: **{len(_awards_filt)}** 件", f"Records: **{len(_awards_filt)}**"))

        if not _awards_filt:
            st.warning(tl("条件に一致する助成金が見つかりませんでした。","No grants matched the filters."))
        else:
            _amount_col  = tl("配分金額（万円）","Amount (10k JPY)")
            _output_col  = tl("論文数","Outputs")
            _cat_col     = tl("研究種目","Category")
            _yr_col      = tl("開始年度","Start Year")
            _title_col   = tl("研究課題名","Project Title")
            _pi_col      = tl("研究代表者","PI")

            _df = pd.DataFrame([{
                _title_col:  (a.get("display_name") or "")[:45],
                _pi_col:     _extract_pi(a.get("description", "")),
                _cat_col:    (a.get("funder_scheme") or tl("不明","Unknown"))[:45],
                _yr_col:     a.get("start_year"),
                tl("終了年度","End Year"): a.get("end_year"),
                _amount_col: round((a.get("amount") or 0) / 10000, 1),
                _output_col: a.get("funded_outputs_count") or 0,
                "URL":       a.get("landing_page_url", ""),
            } for a in _awards_filt])

            _tab_top, _tab_cat, _tab_trend, _tab_scat, _tab_tbl = st.tabs([
                tl("🏆 上位課題","🏆 Top Projects"),
                tl("📂 研究種目別","📂 By Category"),
                tl("📈 年度別トレンド","📈 Year Trend"),
                tl("🔵 金額×論文数","🔵 Amount×Outputs"),
                tl("📋 一覧表","📋 Table"),
            ])

            try:
                import plotly.express as px

                with _tab_top:
                    _sort_col = _amount_col if tl("配分金額","Amount") in kaken_sort else _output_col
                    _top_n = st.slider(tl("表示件数","Show top N"), 5, 50, 20, key="kaken_top_n")
                    _df_top = _df.nlargest(_top_n, _sort_col)
                    _fig1 = px.bar(
                        _df_top, x=_sort_col, y=_title_col, orientation="h",
                        color=_cat_col,
                        hover_data=[_pi_col, _yr_col, _amount_col, _output_col],
                        title=tl(f"上位{_top_n}研究課題（{_sort_col}順）",
                                 f"Top {_top_n} Projects by {_sort_col}"),
                        height=max(400, _top_n * 30),
                    )
                    _fig1.update_layout(yaxis={"autorange": "reversed"}, showlegend=False,
                                        margin={"l": 300})
                    st.plotly_chart(_fig1, use_container_width=True)

                with _tab_cat:
                    _df_cat = (
                        _df.groupby(_cat_col)
                        .agg(
                            **{tl("件数","Count"): (_title_col, "count"),
                               tl("合計金額（万円）","Total Amount"): (_amount_col, "sum"),
                               tl("合計論文数","Total Outputs"): (_output_col, "sum")}
                        )
                        .reset_index()
                        .sort_values(tl("件数","Count"), ascending=False)
                        .head(20)
                    )
                    _cat_metric = st.radio(
                        tl("指標","Metric"),
                        [tl("件数","Count"), tl("合計金額（万円）","Total Amount"), tl("合計論文数","Total Outputs")],
                        horizontal=True, key="kaken_cat_metric"
                    )
                    _fig2 = px.bar(
                        _df_cat, x=_cat_metric, y=_cat_col, orientation="h",
                        title=tl(f"研究種目別 {_cat_metric}", f"By Category: {_cat_metric}"),
                        height=max(350, len(_df_cat) * 30),
                    )
                    _fig2.update_layout(yaxis={"autorange": "reversed"}, margin={"l": 300})
                    st.plotly_chart(_fig2, use_container_width=True)

                with _tab_trend:
                    _df_yr = _df.dropna(subset=[_yr_col]).copy()
                    _df_yr[_yr_col] = _df_yr[_yr_col].astype(int)
                    _df_trend = (
                        _df_yr.groupby(_yr_col)
                        .agg(
                            **{tl("件数","Count"): (_title_col, "count"),
                               tl("合計金額（万円）","Total Amount"): (_amount_col, "sum")}
                        )
                        .reset_index()
                        .sort_values(_yr_col)
                    )
                    _tr_metric = st.radio(
                        tl("指標","Metric"),
                        [tl("件数","Count"), tl("合計金額（万円）","Total Amount")],
                        horizontal=True, key="kaken_tr_metric"
                    )
                    _fig3 = px.bar(
                        _df_trend, x=_yr_col, y=_tr_metric,
                        title=tl(f"年度別 {_tr_metric}", f"Annual {_tr_metric}"),
                    )
                    st.plotly_chart(_fig3, use_container_width=True)

                with _tab_scat:
                    _df_sc = _df[(_df[_amount_col] > 0) & (_df[_output_col] > 0)]
                    if _df_sc.empty:
                        st.info(tl("散布図に表示できるデータがありません。","No data to display in scatter plot."))
                    else:
                        _fig4 = px.scatter(
                            _df_sc, x=_amount_col, y=_output_col,
                            color=_cat_col, hover_name=_title_col,
                            hover_data=[_pi_col, _yr_col],
                            title=tl("配分金額 vs 論文数","Grant Amount vs Publications"),
                            opacity=0.7,
                        )
                        st.plotly_chart(_fig4, use_container_width=True)

            except ImportError:
                st.warning(tl(
                    "グラフ表示には plotly が必要です。`pip install plotly` を実行してください。",
                    "plotly is required for charts. Run `pip install plotly`."
                ))

            with _tab_tbl:
                _df_show = _df.drop(columns=["URL"]).sort_values(_amount_col, ascending=False)
                st.dataframe(_df_show, use_container_width=True, hide_index=True)
                _csv = _df_show.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    tl("📥 CSVダウンロード","📥 Download CSV"),
                    data=_csv, file_name="kaken_grants.csv", mime="text/csv"
                )