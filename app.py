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

# ── KeyBERTモデル定義 ──
KEYBERT_MODELS = {
    "SciBERT (学術論文全般)":       ("allenai/scibert_scivocab_uncased",       "📚 学術論文全般（推奨）"),
    "BioBERT":                      ("dmis-lab/biobert-base-cased-v1.2",       "🧬 生命科学・医療・心理・福祉系"),
    "MiniLM (English)":             ("all-MiniLM-L6-v2",                       "🌐 英語・一般（汎用・高速）"),
    "Multilingual MiniLM (多言語)": ("paraphrase-multilingual-MiniLM-L12-v2",  "🌍 多言語・日本語対応"),
}

# BERTopic / K-means は汎用モデルのみ使用（ドメイン特化モデルは不適）
BERTOPIC_MODELS = {
    "MiniLM (general / English)":       ("all-MiniLM-L6-v2",                          "🌐 英語・汎用（高速）"),
    "Multilingual MiniLM (多言語)":     ("paraphrase-multilingual-MiniLM-L12-v2",     "🌍 多言語対応"),
    "MPNet (general / 高精度)":         ("all-mpnet-base-v2",                          "🎯 英語・高精度（低速）"),
}

# ────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_works(filters, per_page=500):
    base = "https://api.openalex.org/works"
    fields = "id,title,publication_year,doi,authorships,cited_by_count,abstract_inverted_index,topics,concepts,referenced_works"
    works, cursor = [], "*"
    per_req = min(per_page, 200)
    while len(works) < per_page:
        params = {
            "filter": ",".join(filters),
            "per_page": per_req,
            "cursor": cursor,
            "select": fields,
            "mailto": "research@example.com"
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


def reconstruct_abstract(inv_index):
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
def fetch_citing_works_full(work_id: str, max_papers: int = 200):
    """
    指定論文を引用している論文を、書誌結合に必要な referenced_works を含む
    完全データで取得する（被引用数順・最大 max_papers 件）。
    """
    short_id = work_id.rstrip("/").split("/")[-1]
    fields = ("id,title,publication_year,doi,authorships,"
              "cited_by_count,referenced_works,topics")
    results, cursor = [], "*"
    per_req = min(max_papers, 200)
    while len(results) < max_papers:
        params = {
            "filter":   f"cites:{short_id}",
            "sort":     "cited_by_count:desc",
            "per_page": per_req,
            "cursor":   cursor,
            "select":   fields,
            "mailto":   "research@example.com",
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
            aid = a.get("author", {}).get("id")
            name = a.get("author", {}).get("display_name", "")
            if aid and name:
                author_info[aid] = name
                auths.append(aid)
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
    items = []
    for nid in connected:
        w = doi_to_work.get(nid, {})
        year  = w.get("publication_year") or ""
        title = (w.get("title","") or nid)[:50]
        doi_url = w.get("doi", "") or ""
        item = {
            "id":      nid,
            "label":   f"{title} ({year})" if year else title,
            "weights": {"Citations": w.get("cited_by_count", 0)},
        }
        if year:
            item["scores"] = {"Year": int(year)}
        if doi_url:
            item["url"] = doi_url
        items.append(item)
    return {"items": items, "links": link_list}

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
        abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}))
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
        abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}))
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
    try:
        if target in ("Title + Abstract", "タイトル＋抄録"):
            filt = "title.search:" + query
            count_url = "https://api.openalex.org/works?filter=" + filt + "&per_page=1&select=id&mailto=vos@example.com"
            count = requests.get(count_url, timeout=15).json().get("meta",{}).get("count",0)
            url = "https://api.openalex.org/works?filter=" + filt + "&per_page=" + str(max_results) + "&select=id,doi,title,publication_year,authorships&mailto=vos@example.com"
            results = requests.get(url, timeout=20).json().get("results",[])
            return results, count, "works"
        elif target in ("Author Name", "著者名"):
            url = "https://api.openalex.org/authors?search=" + query + "&per_page=" + str(max_results) + "&select=id,display_name,works_count,last_known_institutions&mailto=vos@example.com"
            results = requests.get(url, timeout=15).json().get("results",[])
            return results, len(results), "authors"
        elif target in ("Affiliation Name", "機関名"):
            url = "https://api.openalex.org/institutions?search=" + query + "&per_page=" + str(max_results) + "&select=id,display_name,ror,country_code,type,works_count&mailto=vos@example.com"
            results = requests.get(url, timeout=15).json().get("results",[])
            return results, len(results), "institutions"
        elif target in ("Concept", "コンセプト"):
            c_url = "https://api.openalex.org/concepts?search=" + query + "&per_page=5&select=id,display_name,works_count&mailto=vos@example.com"
            concepts = requests.get(c_url, timeout=15).json().get("results",[])
            if not concepts: return [], 0, "works"
            cid = concepts[0]["id"]
            filt = "concepts.id:" + cid
            count_url = "https://api.openalex.org/works?filter=" + filt + "&per_page=1&select=id&mailto=vos@example.com"
            count = requests.get(count_url, timeout=15).json().get("meta",{}).get("count",0)
            url = "https://api.openalex.org/works?filter=" + filt + "&per_page=" + str(max_results) + "&select=id,doi,title,publication_year,authorships&mailto=vos@example.com"
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
                ("s1_search_result_type","works"), ("s1_search_count",0)]:
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

    tab_kw, tab_author, tab_inst = st.tabs([
        tl("📝 キーワード","📝 Keyword"),
        tl("👤 著者名","👤 Author"),
        tl("🏢 機関名","🏢 Institution"),
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
                        params={"mailto": "research@example.com"}, timeout=10
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
        st.session_state.get("s1_freekw","")
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
                                "select": "id", "mailto": "vos@example.com"},
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
                works = fetch_works(filters, per_page)

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
        }
        if analysis_type in _method_info:
            with st.expander(tl("📖 この手法について","📖 About this method")):
                st.markdown(_method_info[analysis_type])

        # 引用分析サブタイプ
        if tl("引用分析","Citation Analysis") in analysis_type:
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
                    abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}))
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

        # ── 引用論文 VOSviewer分析（ページ上部に表示）──
        _cite_target = st.session_state.get("cite_target")
        if _cite_target and _cite_target.get("id"):
            st.markdown("---")
            _ct_title = _cite_target["title"]
            st.subheader(tl("🔬 引用論文ネットワーク（VOSviewer分析）",
                            "🔬 Citing Papers Network (VOSviewer Analysis)"))
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
                        _cite_target["id"], max_papers=_cite_max
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
                        "✅ ネットワーク構築完了。VOSviewer JSONをダウンロードしてVOSviewerで開いてください。",
                        "✅ Network built. Download the VOSviewer JSON and open it in VOSviewer."
                    ))
                    st.info(tl(
                        "💡 VOSviewerで **Scores → Year** を選択すると、論文が出版年ごとに色分けされます。",
                        "💡 In VOSviewer, select **Scores → Year** to color nodes by publication year."
                    ))
                    _vos_json = json.dumps(_cite_vos, ensure_ascii=False, indent=2)
                    _safe_title = re.sub(r"[^\w]", "_", _ct_title[:30])
                    st.download_button(
                        tl("📥 VOSviewer JSON ダウンロード","📥 Download VOSviewer JSON"),
                        data=_vos_json.encode("utf-8"),
                        file_name=f"citing_{_safe_title}.json",
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

        # ── 重要論文ランキング（被引用数） ──
        st.markdown("---")
        st.subheader(tl("📄 重要論文ランキング（被引用数順）",
                        "📄 Key Papers Ranking (by citation count)"))
        st.caption(tl(
            "ネットワーク分析に依存せず、データセット内の論文を**被引用数**で直接ランキングします。"
            "論文数が少ない場合でも信頼性の高い重要度指標です。",
            "Ranks papers directly by **citation count**, independent of network analysis. "
            "A reliable importance measure even with smaller datasets."
        ))

        import pandas as pd
        _top_n_papers = st.slider(
            tl("表示件数","Top N papers"), 5, 100, 20, key="top_papers_n"
        )
        _paper_rows = []
        for w in works:
            title  = w.get("title","") or "No title"
            year   = w.get("publication_year","")
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

        # 上位N件をテーブル表示
        st.dataframe(
            _papers_df.head(_top_n_papers).drop(columns=["DOI"]),
            use_container_width=True
        )

        # 詳細エキスパンダー
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
                    _ab = reconstruct_abstract(_w.get("abstract_inverted_index", {}))
                    if _ab:
                        st.markdown("**Abstract**")
                        st.write(_ab[:400] + ("..." if len(_ab) > 400 else ""))
                    _wid = _w.get("id", "")
                    if _wid and st.button(
                        tl("📊 DOI引用ネットワーク生成","📊 Build DOI Citation Network"),
                        key=f"cite_btn_{rank}"
                    ):
                        st.session_state["cite_target"] = {
                            "id":    _wid,
                            "title": row[tl("タイトル","Title")],
                        }
                        st.session_state["cite_works"] = []
                        st.toast(tl("↑ ページ上部でDOI引用ネットワークを構築します",
                                    "↑ Building DOI citation network at top of page"), icon="📊")
                        st.rerun()

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
                        ab = reconstruct_abstract(w.get("abstract_inverted_index",{}))
                        if ab:
                            st.markdown("**Abstract**")
                            st.write(ab[:500] + ("..." if len(ab)>500 else ""))
                        _wid2 = w.get("id", "")
                        if _wid2:
                            if st.button(
                                tl("📊 DOI引用ネットワーク生成","📊 Build DOI Citation Network"),
                                key=f"cite_nd_{_wid2}"
                            ):
                                st.session_state["cite_target"] = {"id": _wid2, "title": title}
                                st.session_state["cite_works"] = []
                                st.toast(tl("↑ ページ上部でDOI引用ネットワークを構築します",
                                            "↑ Building DOI citation network at top of page"), icon="📊")
                                st.rerun()

                if st.button(tl("✕ 選択解除","✕ Clear"), key="clr_nd"):
                    st.session_state["sel_node"] = None
                    st.rerun()

# ════════════════════════════════════════════
# ステップ3: KAKEN助成金分析
# ════════════════════════════════════════════
if tl("③ KAKEN助成金分析","③ KAKEN Grant Analysis") in step:
    st.header(tl("💰 KAKEN助成金分析","💰 KAKEN Grant Analysis"))
    st.caption(tl(
        "OpenAlexが提供するKAKEN（科学研究費助成事業）データを集計・可視化します。",
        "Aggregate and visualize KAKEN (Grants-in-Aid for Scientific Research) data via OpenAlex."
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
                "mailto": "research@example.com",
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