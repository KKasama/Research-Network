import streamlit as st
import streamlit.components.v1 as components
import json, requests
from collections import defaultdict
 
st.set_page_config(page_title="Research Network", page_icon="🔬", layout="wide")
 
# ── 言語検出（JavaScriptでブラウザ言語を取得） ──
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"
 
lang_detector = """
<script>
const lang = navigator.language || navigator.userLanguage || "en";
const isJa = lang.startsWith("ja");
const target = window.parent.document.querySelector('input[data-testid="stTextInput"]');
window.parent.postMessage({type: "lang_detect", lang: isJa ? "ja" : "en"}, "*");
</script>
<div id="lang_placeholder" style="display:none"></div>
"""
 
# JavaScript経由で言語を検出してquery_paramsに保存
query_params = st.query_params
if "lang" in query_params:
    st.session_state.ui_lang = query_params["lang"]
 
# 言語切替ボタン（手動オーバーライド）
def set_lang(lang):
    st.session_state.ui_lang = lang
    st.query_params["lang"] = lang
 
lang = st.session_state.ui_lang
 
# ── 翻訳辞書 ──
T = {
    "ja": {
        "title": "## 🔬 OpenAlex → VOSviewer（KeyBERT強化版）",
        "settings": "設定",
        "analysis_type": "分析タイプ",
        "analysis_options": ["Co-authorship", "Co-citation", "Bibliographic coupling",
                             "Keyword Co-occurrence（KeyBERT）", "著者類似度（KeyBERT）",
                             "機関コラボレーション", "国際共著ネットワーク"],
        "topic_select": "### トピック選択",
        "tab_browse": "🗂 階層ブラウズ",
        "tab_keyword": "🔍 キーワード検索",
        "tab_author": "👤 著者名検索",
        "tab_org": "🏢 組織名検索",
        "topic_filter_label": "トピック名で絞り込み",
        "topic_filter_ph": "例: battery, solar",
        "all": "-- すべて --",
        "topics_found": " 件のトピック",
        "kw_search_label": "検索キーワード",
        "kw_search_ph": "例: solid electrolyte, fast charging",
        "kw_search_btn": "検索",
        "kw_searching": "OpenAlexを検索中...",
        "papers_found": " 件の論文が見つかりました",
        "kw_search_target": "検索対象",
        "kw_target_options": ["タイトル", "タイトル＋抄録", "コンセプト", "著者名"],
        "kw_run_btn": "▶ このキーワードで解析実行",
        "author_label": "著者名",
        "author_ph": "例: Komaba Shinichi, Sae Dieb",
        "author_search_btn": "著者を検索",
        "author_searching": "著者を検索中...",
        "author_results": "**検索結果 — 著者を選択してください**",
        "author_selected": "選択中: ",
        "author_clear": "クリア（著者）",
        "org_label": "組織名",
        "org_ph": "例: University of Tokyo, NIMS, RIKEN",
        "org_search_btn": "組織を検索",
        "org_searching": "組織を検索中...",
        "org_results": "**検索結果 — 組織を選択してください**",
        "org_selected": "選択中: ",
        "org_clear": "クリア（組織）",
        "ror_caption": "ROR IDは https://ror.org で検索できます",
        "selected_topics": "### 選択済みトピック (",
        "selected_topics_end": "件)",
        "clear": "クリア",
        "filters": "### フィルタ",
        "kw_filter_label": "キーワード絞り込み（トピックと AND 条件）",
        "kw_filter_ph": "例: solid electrolyte, fast charging",
        "oa_only": "OAのみ",
        "country": "国・地域",
        "countries": ["JP 日本","US 米国","CN 中国","GB 英国","DE ドイツ","FR フランス","KR 韓国","AU オーストラリア","CA カナダ","IT イタリア","IN インド","BR ブラジル","ES スペイン","NL オランダ","SE スウェーデン","CH スイス","TW 台湾"],
        "num_papers": "取得論文数",
        "min_links": "最小リンク強度",
        "min_sim": "最小類似度（KeyBERT用）",
        "no_results": "該当なし",
        "few": " 件 — 少なめ",
        "good": " 件 — 良好",
        "many": " 件 — 多め（取得上限: ",
        "many_end": " 件）",
        "run_btn": "▶ 解析実行",
        "fetching": "論文取得中...",
        "fetched": "取得: ",
        "fetched_end": " 論文",
        "keybert_running": "KeyBERT（BatterySciBERT）でキーワード抽出中...",
        "keybert_done": "キーワード抽出完了: ",
        "keybert_done_end": " キーワード",
        "building": "ネットワーク構築中...",
        "papers": "論文数",
        "nodes": "ノード",
        "edges": "エッジ",
        "download_btn": "⬇ VOSviewer JSON ダウンロード",
        "vos_title": "### VOSviewer で開く手順",
        "vos_step1": "1. 上の **⬇ VOSviewer JSON ダウンロード** を押してJSONを保存",
        "vos_step2": "2. **[app.vosviewer.com](https://app.vosviewer.com)** を開く",
        "vos_step3": "3. 左上の **Open** → **JSON file** → ダウンロードしたファイルを選択",
        "top_nodes": "主要ノード（上位20）",
        "no_topic_warn": "トピックを選択するか、著者名検索または組織名検索タブで対象を選択してください",
        "no_valid_topic": "有効なトピックが選択されていません",
        "ego_info": "EGO起点モード: ",
        "ego_info_end": " の共著ネットワーク（",
        "org_info": "組織起点モード: ",
        "org_info_end": " の論文を対象にネットワークを構築します",
        "filter_label": "**フィルタ**",
        "papers_few": " 件 — 少なめ",
        "papers_good": " 件 — 良好",
        "papers_many_pre": " 件 — 多め（取得上限: ",
        "papers_many_suf": " 件）",
        "lang_btn": "English",
        "expander_label": "ネットワークの見方（",
        "node_label": "ノード",
        "edge_label": "エッジ",
        "cluster_label": "クラスター",
        "usage_label": "活用方法",
        "tab_search": "🔍 検索",
        "search_target": "検索対象",
        "search_options": ["Title + Abstract", "Author Name", "ORCID", "Affiliation Name", "ROR", "Concept"],
        "search_input_label": "検索キーワード / ID",
        "search_input_ph_abstract": "例: solid electrolyte",
        "search_input_ph_author": "例: Komaba Shinichi",
        "search_input_ph_orcid": "例: 0000-0002-1234-5678",
        "search_input_ph_affil": "例: University of Tokyo, NIMS",
        "search_input_ph_ror": "例: 057qpr032",
        "search_input_ph_concept": "例: lithium ion battery, machine learning",
        "search_desc_title_abstract": "📄 タイトルと抄録の両方を全文検索します",
        "search_desc_author": "👤 著者名で検索 → 候補から選択してEGOネットワークを構築",
        "search_desc_orcid": "🔗 ORCID IDで著者を一意に特定（例: 0000-0002-1234-5678）",
        "search_desc_affil": "🏢 機関名で検索 → 候補から選択して機関起点ネットワークを構築",
        "search_desc_ror": "🏷 ROR IDで機関を一意に特定（例: 057qpr032）",
        "search_desc_concept": "💡 OpenAlexのConcept（Wikidata準拠）で分野横断検索",
        "search_btn": "検索",
        "search_running": "OpenAlexを検索中...",
        "search_results_label": "**検索結果 — クリックで選択（著者・機関の場合）**",
        "selected_author": "選択中（著者）: ",
        "selected_org": "選択中（機関）: ",
        "clear_author": "クリア（著者）",
        "clear_org": "クリア（機関）",
        "filter_label_search": "**フィルタ（検索起点モード）**",
        "model_select": "KeyBERTモデル",
        "model_auto": "自動（分野判定）",
        "model_detected": "検出されたドメイン: ",
    },
    "en": {
        "title": "## 🔬 OpenAlex → VOSviewer (KeyBERT Enhanced)",
        "settings": "Settings",
        "analysis_type": "Analysis type",
        "analysis_options": ["Co-authorship", "Co-citation", "Bibliographic coupling",
                             "Keyword Co-occurrence (KeyBERT)", "Author Similarity (KeyBERT)",
                             "Institution Collaboration", "Country Network"],
        "topic_select": "### Topic Selection",
        "tab_browse": "🗂 Browse Hierarchy",
        "tab_keyword": "🔍 Keyword Search",
        "tab_author": "👤 Author Search",
        "tab_org": "🏢 Institution Search",
        "topic_filter_label": "Filter by topic name",
        "topic_filter_ph": "e.g. battery, solar",
        "all": "-- All --",
        "topics_found": " topics",
        "kw_search_label": "Search keyword",
        "kw_search_ph": "e.g. solid electrolyte, fast charging",
        "kw_search_btn": "Search",
        "kw_searching": "Searching OpenAlex...",
        "papers_found": " papers found",
        "kw_search_target": "Search target",
        "kw_target_options": ["Title", "Title + Abstract", "Concept", "Author Name"],
        "kw_run_btn": "▶ Run analysis with this keyword",
        "author_label": "Author name",
        "author_ph": "e.g. Komaba Shinichi, Sae Dieb",
        "author_search_btn": "Search author",
        "author_searching": "Searching for author...",
        "author_results": "**Search results — select an author**",
        "author_selected": "Selected: ",
        "author_clear": "Clear author",
        "org_label": "Institution name",
        "org_ph": "e.g. University of Tokyo, NIMS, RIKEN",
        "org_search_btn": "Search institution",
        "org_searching": "Searching for institution...",
        "org_results": "**Search results — select an institution**",
        "org_selected": "Selected: ",
        "org_clear": "Clear institution",
        "ror_caption": "Search for ROR IDs at https://ror.org",
        "selected_topics": "### Selected Topics (",
        "selected_topics_end": ")",
        "clear": "Clear",
        "filters": "### Filters",
        "kw_filter_label": "Keyword filter (AND with topic)",
        "kw_filter_ph": "e.g. solid electrolyte, fast charging",
        "oa_only": "Open Access only",
        "country": "Country / Region",
        "countries": ["JP Japan","US USA","CN China","GB UK","DE Germany","FR France","KR Korea","AU Australia","CA Canada","IT Italy","IN India","BR Brazil","ES Spain","NL Netherlands","SE Sweden","CH Switzerland","TW Taiwan"],
        "num_papers": "Number of papers",
        "min_links": "Min. link strength",
        "min_sim": "Min. similarity (KeyBERT)",
        "no_results": "No results",
        "few": " papers — few",
        "good": " papers — good",
        "many": " papers — many (fetch limit: ",
        "many_end": ")",
        "run_btn": "▶ Run Analysis",
        "fetching": "Fetching papers...",
        "fetched": "Fetched: ",
        "fetched_end": " papers",
        "keybert_running": "Extracting keywords with KeyBERT (BatterySciBERT)...",
        "keybert_done": "Keyword extraction complete: ",
        "keybert_done_end": " keywords",
        "building": "Building network...",
        "papers": "Papers",
        "nodes": "Nodes",
        "edges": "Edges",
        "download_btn": "⬇ Download VOSviewer JSON",
        "vos_title": "### How to open in VOSviewer",
        "vos_step1": "1. Click **⬇ Download VOSviewer JSON** above to save the file",
        "vos_step2": "2. Open **[app.vosviewer.com](https://app.vosviewer.com)**",
        "vos_step3": "3. Click **Open** → **JSON file** → select the downloaded file",
        "top_nodes": "Top 20 Nodes",
        "no_topic_warn": "Please select a topic, search for an author, or search for an institution.",
        "no_valid_topic": "No valid topics selected.",
        "ego_info": "EGO mode: ",
        "ego_info_end": " (",
        "org_info": "Institution mode: building network for ",
        "org_info_end": "",
        "filter_label": "**Filters**",
        "papers_few": " papers — few",
        "papers_good": " papers — good",
        "papers_many_pre": " papers — many (fetch limit: ",
        "papers_many_suf": ")",
        "lang_btn": "日本語",
        "expander_label": "How to read this network (",
        "node_label": "Node",
        "edge_label": "Edge",
        "cluster_label": "Cluster",
        "usage_label": "How to use",
        "tab_search": "🔍 Search",
        "search_target": "Search target",
        "search_options": ["Title + Abstract", "Author Name", "ORCID", "Affiliation Name", "ROR", "Concept"],
        "search_input_label": "Keyword / ID",
        "search_input_ph_abstract": "e.g. solid electrolyte",
        "search_input_ph_author": "e.g. Komaba Shinichi",
        "search_input_ph_orcid": "e.g. 0000-0002-1234-5678",
        "search_input_ph_affil": "e.g. University of Tokyo, NIMS",
        "search_input_ph_ror": "e.g. 057qpr032",
        "search_input_ph_concept": "e.g. lithium ion battery, machine learning",
        "search_desc_title_abstract": "📄 Full-text search across both title and abstract",
        "search_desc_author": "👤 Search by author name → select from results to build EGO network",
        "search_desc_orcid": "🔗 Identify author precisely by ORCID (e.g. 0000-0002-1234-5678)",
        "search_desc_affil": "🏢 Search by institution name → select to build institution network",
        "search_desc_ror": "🏷 Identify institution precisely by ROR ID (e.g. 057qpr032)",
        "search_desc_concept": "💡 Search across fields using OpenAlex Concepts (Wikidata-based)",
        "search_btn": "Search",
        "search_running": "Searching OpenAlex...",
        "search_results_label": "**Results — click to select (for author / institution)**",
        "selected_author": "Selected (author): ",
        "selected_org": "Selected (institution): ",
        "clear_author": "Clear author",
        "clear_org": "Clear institution",
        "filter_label_search": "**Filters (search mode)**",
        "model_select": "KeyBERT model",
        "model_auto": "Auto (domain detection)",
        "model_detected": "Detected domain: ",
    }
}
 
def t(key):
    return T[lang].get(key, T["en"].get(key, key))
 
# ── キャッシュ・ユーティリティ ──
@st.cache_data
def load_topics():
    with open("/Users/kk/Downloads/topics.json") as f:
        return json.load(f)
 
# 分野別KeyBERTモデル定義
KEYBERT_MODELS = {
    "battery":      ("batterydata/batteryscibert-cased",   "BatterySciBERT",      ["battery", "electrolyte", "cathode", "anode", "lithium", "ion", "electrode", "solid state", "fuel cell", "electrochemical"]),
    "materials":    ("m3rg-iitd/matscibert",               "MatSciBERT",          ["material", "alloy", "crystal", "polymer", "thin film", "nanoparticle", "semiconductor", "ceramic", "composite", "synthesis"]),
    "biomedical":   ("dmis-lab/biobert-base-cased-v1.2",   "BioBERT",             ["cancer", "protein", "gene", "drug", "clinical", "cell", "disease", "therapy", "patient", "diagnosis"]),
    "general":      ("all-MiniLM-L6-v2",                   "SentenceTransformers (general)", []),
}
 
def detect_domain(selected_topics, concept_name=""):
    """トピック名・Concept名からドメインを自動判定"""
    text = " ".join(selected_topics).lower() + " " + concept_name.lower()
    scores = {domain: 0 for domain in KEYBERT_MODELS}
    for domain, (_, _, keywords) in KEYBERT_MODELS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"
 
@st.cache_resource
def load_keybert(model_name="batterydata/batteryscibert-cased"):
    from keybert import KeyBERT
    return KeyBERT(model=model_name)
 
def get_count(filters):
    try:
        r = requests.get("https://api.openalex.org/works?filter=" + ",".join(filters) + "&per_page=1&select=id&mailto=vos@example.com", timeout=15)
        return r.json().get("meta", {}).get("count", 0)
    except:
        return 0
 
def fetch_works(filters, per_page):
    works = []
    for page in range(1, (per_page+99)//100 + 1):
        try:
            url = "https://api.openalex.org/works?filter=" + ",".join(filters) + "&per_page=100&page=" + str(page) + "&select=id,doi,title,publication_year,authorships,cited_by_count,referenced_works,abstract_inverted_index&mailto=vos@example.com"
            batch = requests.get(url, timeout=30).json().get("results", [])
            if not batch: break
            works.extend(batch)
            if len(works) >= per_page: break
        except Exception as e:
            st.error("Error: " + str(e)); break
    return works[:per_page]
 
def search_works_by_keyword(keyword, max_results=50):
    try:
        filt = "title.search:" + keyword + ",abstract.search:" + keyword
        count_url = "https://api.openalex.org/works?filter=" + filt + "&per_page=1&select=id&mailto=vos@example.com"
        count = requests.get(count_url, timeout=15).json().get("meta", {}).get("count", 0)
        url = "https://api.openalex.org/works?filter=" + filt + "&per_page=" + str(max_results) + "&select=id,doi,title,publication_year,authorships&mailto=vos@example.com"
        results = requests.get(url, timeout=20).json().get("results", [])
        return results, count
    except:
        return [], 0
 
def search_openalex(query, target, max_results=50):
    """汎用検索: target = Title / Title+Abstract / Author Name / ORCID / Affiliation Name / ROR"""
    try:
        if target == "Title + Abstract":
            filt = "title.search:" + query + ",abstract.search:" + query
            endpoint = "works"
            select = "id,doi,title,publication_year,authorships"
        elif target == "Author Name":
            url = "https://api.openalex.org/authors?search=" + query + "&per_page=" + str(max_results) + "&select=id,display_name,works_count,last_known_institutions&mailto=vos@example.com"
            results = requests.get(url, timeout=15).json().get("results", [])
            return results, len(results), "authors"
        elif target == "ORCID":
            orcid = query.strip().replace("https://orcid.org/", "")
            url = "https://api.openalex.org/authors?filter=orcid:https://orcid.org/" + orcid + "&select=id,display_name,works_count,last_known_institutions&mailto=vos@example.com"
            results = requests.get(url, timeout=15).json().get("results", [])
            return results, len(results), "authors"
        elif target == "Affiliation Name":
            url = "https://api.openalex.org/institutions?search=" + query + "&per_page=" + str(max_results) + "&select=id,display_name,ror,country_code,type,works_count&mailto=vos@example.com"
            results = requests.get(url, timeout=15).json().get("results", [])
            return results, len(results), "institutions"
        elif target == "ROR":
            ror_id = query.strip().replace("https://ror.org/", "")
            url = "https://api.openalex.org/institutions?filter=ror:https://ror.org/" + ror_id + "&select=id,display_name,ror,country_code,type,works_count&mailto=vos@example.com"
            results = requests.get(url, timeout=15).json().get("results", [])
            return results, len(results), "institutions"
        elif target == "Concept":
            # まずConcept名からIDを検索
            c_url = "https://api.openalex.org/concepts?search=" + query + "&per_page=5&select=id,display_name,works_count&mailto=vos@example.com"
            concepts = requests.get(c_url, timeout=15).json().get("results", [])
            if not concepts:
                return [], 0, "works"
            # 最上位のConceptで論文を検索
            cid = concepts[0]["id"]
            filt = "concepts.id:" + cid
            count_url = "https://api.openalex.org/works?filter=" + filt + "&per_page=1&select=id&mailto=vos@example.com"
            count = requests.get(count_url, timeout=15).json().get("meta", {}).get("count", 0)
            url = "https://api.openalex.org/works?filter=" + filt + "&per_page=50&select=id,doi,title,publication_year,authorships&mailto=vos@example.com"
            results = requests.get(url, timeout=20).json().get("results", [])
            # concept名とIDを付加情報として返す
            for r in results:
                r["_concept_name"] = concepts[0].get("display_name", query)
                r["_concept_id"] = cid
            return results, count, "works_concept"
        else:
            return [], 0, "works"
 
        count_url = "https://api.openalex.org/works?filter=" + filt + "&per_page=1&select=id&mailto=vos@example.com"
        count = requests.get(count_url, timeout=15).json().get("meta", {}).get("count", 0)
        url = "https://api.openalex.org/works?filter=" + filt + "&per_page=" + str(max_results) + "&select=id,doi,title,publication_year,authorships&mailto=vos@example.com"
        results = requests.get(url, timeout=20).json().get("results", [])
        return results, count, "works"
    except:
        return [], 0, "works"
 
def reconstruct_abstract(inverted_index):
    if not inverted_index: return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))
 
def extract_keywords_batch(works, kw_model, max_title=2, max_abstract=10):
    results = {}
    for w in works:
        wid = w.get("id", "")
        title = w.get("title", "") or ""
        abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}))
        kws = set()
        if title.strip():
            try:
                kws_t = kw_model.extract_keywords(title, keyphrase_ngram_range=(1,2), top_n=max_title, stop_words="english")
                kws.update([k for k, s in kws_t if s > 0.3])
            except: pass
        if abstract.strip():
            try:
                kws_a = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1,2), top_n=max_abstract, stop_words="english")
                kws.update([k for k, s in kws_a if s > 0.3])
            except: pass
        results[wid] = list(kws)
    return results
 
def build_author_vectors(works, work_keywords):
    period_weights = {"1990-2000": 0.2, "2001-2010": 0.5, "2011-2025": 1.0}
    author_data, author_names = {}, {}
    for w in works:
        year = w.get("publication_year") or 0
        period = "1990-2000" if year <= 2000 else ("2001-2010" if year <= 2010 else "2011-2025")
        pw = period_weights[period]
        kws = work_keywords.get(w.get("id", ""), [])
        for idx, a in enumerate(w.get("authorships", [])):
            aid = a.get("author", {}).get("id")
            name = a.get("author", {}).get("display_name", "Unknown")
            if not aid: continue
            author_names[aid] = name
            if aid not in author_data: author_data[aid] = {}
            aw = 1.5 if idx == 0 else 0.8
            for kw in kws:
                author_data[aid][kw] = author_data[aid].get(kw, 0) + pw * aw
    vectors = {}
    for aid, vec in author_data.items():
        if vec:
            total = sum(vec.values())
            vectors[aid] = {k: v/total for k, v in vec.items()}
    return vectors, author_names
 
def cosine_similarity(v1, v2):
    common = set(v1.keys()) & set(v2.keys())
    if not common: return 0.0
    dot = sum(v1[k]*v2[k] for k in common)
    n1 = sum(x*x for x in v1.values())**0.5
    n2 = sum(x*x for x in v2.values())**0.5
    return 0.0 if n1==0 or n2==0 else dot/(n1*n2)
 
def build_similarity_network(vectors, author_names, min_sim=0.15):
    aids = list(vectors.keys())
    links = []
    for i in range(len(aids)):
        for j in range(i+1, len(aids)):
            sim = cosine_similarity(vectors[aids[i]], vectors[aids[j]])
            if sim >= min_sim:
                links.append({"source_id": aids[i], "target_id": aids[j], "strength": round(sim, 4)})
    connected = {l["source_id"] for l in links} | {l["target_id"] for l in links}
    items = []
    for aid in connected:
        top_kws = sorted(vectors[aid].items(), key=lambda x: x[1], reverse=True)[:5]
        items.append({"id": aid, "label": author_names.get(aid, "Unknown"), "weights": {"Score": round(sum(vectors[aid].values()), 3)}, "description": ", ".join([k for k, v in top_kws])})
    return {"items": items, "links": links}
 
def build_coauth(works, min_links):
    author_info, work_authors = {}, []
    for w in works:
        aids = []
        for a in w.get("authorships", []):
            aid = a.get("author", {}).get("id")
            name = a.get("author", {}).get("display_name", "Unknown")
            if aid:
                author_info[aid] = name
                aids.append(aid)
        work_authors.append(aids)
    links = defaultdict(int)
    for aids in work_authors:
        for i in range(len(aids)):
            for j in range(i+1, len(aids)):
                links[(min(aids[i], aids[j]), max(aids[i], aids[j]))] += 1
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a, b), s in links.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    doc_count = defaultdict(int)
    for aids in work_authors:
        for a in aids: doc_count[a] += 1
    items = [{"id": aid, "label": name, "weights": {"Documents": doc_count[aid]}} for aid, name in author_info.items() if aid in connected]
    return {"items": items, "links": link_list}
 
def build_cocitation(works, min_links):
    ref_count = defaultdict(int)
    work_refs = []
    for w in works:
        refs = w.get("referenced_works", [])
        work_refs.append(refs)
        for r in refs: ref_count[r] += 1
    cited = {r for r, c in ref_count.items() if c >= 2}
    links = defaultdict(int)
    for refs in work_refs:
        rf = [r for r in refs if r in cited]
        for i in range(len(rf)):
            for j in range(i+1, len(rf)):
                links[(min(rf[i], rf[j]), max(rf[i], rf[j]))] += 1
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a, b), s in links.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    items = [{"id": r, "label": r.split("/")[-1], "weights": {"Citations": ref_count[r]}} for r in connected]
    return {"items": items, "links": link_list}
 
def build_bibcoupling(works, min_links):
    work_refs = [(w.get("id", ""), w.get("referenced_works", [])) for w in works]
    links = defaultdict(int)
    for i in range(len(work_refs)):
        for j in range(i+1, len(work_refs)):
            common = len(set(work_refs[i][1]) & set(work_refs[j][1]))
            if common >= min_links:
                links[(work_refs[i][0], work_refs[j][0])] = common
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a, b), s in links.items()]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    cited_by = {w.get("id", ""): w.get("cited_by_count", 0) for w in works}
    items = [{"id": wid, "label": wid.split("/")[-1], "weights": {"Citations": cited_by.get(wid, 0)}} for wid in connected]
    return {"items": items, "links": link_list}
 
def build_keyword_cooccurrence(works, work_keywords, min_links):
    kw_count = defaultdict(int)
    work_kws = []
    for w in works:
        kws = work_keywords.get(w.get("id", ""), [])
        work_kws.append(kws)
        for k in kws: kw_count[k] += 1
    links = defaultdict(int)
    for kws in work_kws:
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                pair = (min(kws[i], kws[j]), max(kws[i], kws[j]))
                links[pair] += 1
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a, b), s in links.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    items = [{"id": k, "label": k, "weights": {"Occurrences": kw_count[k]}} for k in connected]
    return {"items": items, "links": link_list}
 
def build_institution_collab(works, min_links):
    inst_info, work_insts = {}, []
    for w in works:
        insts = []
        for a in w.get("authorships", []):
            for inst in a.get("institutions", []):
                iid = inst.get("ror") or inst.get("id", "")
                name = inst.get("display_name", "")
                if iid and name:
                    inst_info[iid] = {"name": name, "type": inst.get("type", ""), "country": inst.get("country_code", "")}
                    if iid not in insts: insts.append(iid)
        work_insts.append(insts)
    links = defaultdict(int)
    for insts in work_insts:
        for i in range(len(insts)):
            for j in range(i+1, len(insts)):
                links[(min(insts[i], insts[j]), max(insts[i], insts[j]))] += 1
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a, b), s in links.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    doc_count = defaultdict(int)
    for insts in work_insts:
        for i in insts: doc_count[i] += 1
    items = []
    for iid in connected:
        info = inst_info.get(iid, {})
        desc = info.get("type", "") + (" / " + info.get("country", "") if info.get("country") else "")
        items.append({"id": iid, "label": info.get("name", iid)[:60], "weights": {"Documents": doc_count[iid]}, "description": desc})
    return {"items": items, "links": link_list}
 
def build_country_network(works, min_links):
    country_names = {"JP": "Japan","US": "USA","CN": "China","GB": "UK","DE": "Germany","FR": "France","KR": "Korea","AU": "Australia","CA": "Canada","IT": "Italy","IN": "India","BR": "Brazil","ES": "Spain","NL": "Netherlands","RU": "Russia","SE": "Sweden","CH": "Switzerland","TW": "Taiwan","SG": "Singapore","PL": "Poland","BE": "Belgium","AT": "Austria","DK": "Denmark","FI": "Finland","NO": "Norway","PT": "Portugal","IL": "Israel","ZA": "South Africa","TR": "Turkey"}
    country_count = defaultdict(int)
    work_countries = []
    for w in works:
        countries = []
        for a in w.get("authorships", []):
            for inst in a.get("institutions", []):
                cc = inst.get("country_code", "")
                if cc and cc not in countries: countries.append(cc)
        work_countries.append(countries)
        for cc in countries: country_count[cc] += 1
    links = defaultdict(int)
    for countries in work_countries:
        for i in range(len(countries)):
            for j in range(i+1, len(countries)):
                pair = (min(countries[i], countries[j]), max(countries[i], countries[j]))
                links[pair] += 1
    link_list = [{"source_id": a, "target_id": b, "strength": s} for (a, b), s in links.items() if s >= min_links]
    connected = {l["source_id"] for l in link_list} | {l["target_id"] for l in link_list}
    items = [{"id": cc, "label": country_names.get(cc, cc), "weights": {"Documents": country_count[cc]}} for cc in connected]
    return {"items": items, "links": link_list}
 
# ── 件数プレビューヘルパー ──
def show_count_preview(count, per_page):
    if count == 0:
        st.warning(t("no_results"))
    elif count < 200:
        st.warning(str(count) + t("few"))
    elif count <= 1500:
        st.success(str(count) + t("good"))
    else:
        st.info(str(count) + t("many") + str(per_page) + t("many_end"))
 
# ── 初期化 ──
topics_all = load_topics()
topic_map = {t_["display_name"]: t_["id"].replace("https://openalex.org/T", "") for t_ in topics_all}
 
for key, default in [
    ("selected_topics", []), ("vos_data", None), ("works_count", 0),
    ("json_str", ""), ("work_keywords", {}), ("last_analysis_type", ""),
    ("kw_search_results", []), ("kw_total_count", 0), ("kw_filter", ""), ("kw_mode_active", False),
    ("ego_author_id", None), ("ego_author_name", ""), ("ego_candidates", []),
    ("org_candidates", []), ("org_ror_id", None), ("org_name", ""),
    ("search_results", []), ("search_count", 0),
    ("search_result_type", "works"), ("search_target_used", ""),
    ("concept_id", None), ("concept_name", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default
 
# ── サイドバー ──
with st.sidebar:
    # 言語切替ボタン
    col_lang1, col_lang2 = st.columns([3, 1])
    with col_lang2:
        if st.button(t("lang_btn"), key="lang_toggle"):
            new_lang = "en" if lang == "ja" else "ja"
            set_lang(new_lang)
            st.rerun()
 
    st.header(t("settings"))
    analysis_type = st.radio(t("analysis_type"), t("analysis_options"))
 
    st.markdown(t("topic_select"))
    tab1, tab2 = st.tabs([t("tab_browse"), t("tab_search")])
 
    # タブ1: 階層ブラウズ
    with tab1:
        topic_filter = st.text_input(t("topic_filter_label"), placeholder=t("topic_filter_ph"), key="topic_filter")
        domains = [t("all")] + sorted(set(t_["domain"]["display_name"] for t_ in topics_all if t_.get("domain")))
        sel_domain = st.selectbox("Domain", domains, key="domain")
        filtered = topics_all
        if sel_domain != t("all"):
            filtered = [t_ for t_ in filtered if t_.get("domain", {}).get("display_name") == sel_domain]
        fields = [t("all")] + sorted(set(t_["field"]["display_name"] for t_ in filtered if t_.get("field")))
        sel_field = st.selectbox("Field", fields, key="field")
        if sel_field != t("all"):
            filtered = [t_ for t_ in filtered if t_.get("field", {}).get("display_name") == sel_field]
        subfields = [t("all")] + sorted(set(t_["subfield"]["display_name"] for t_ in filtered if t_.get("subfield")))
        sel_subfield = st.selectbox("Subfield", subfields, key="subfield")
        if sel_subfield != t("all"):
            filtered = [t_ for t_ in filtered if t_.get("subfield", {}).get("display_name") == sel_subfield]
        if topic_filter:
            filtered = [t_ for t_ in filtered if topic_filter.lower() in t_["display_name"].lower()]
        st.markdown("**" + str(len(filtered)) + t("topics_found") + "**")
        for t_ in filtered[:100]:
            tid = t_["id"].replace("https://openalex.org/T", "")
            label = t_["display_name"]
            if st.checkbox(label, value=label in st.session_state.selected_topics, key="b_" + tid):
                if label not in st.session_state.selected_topics:
                    st.session_state.selected_topics.append(label)
            else:
                if label in st.session_state.selected_topics:
                    st.session_state.selected_topics.remove(label)
 
    # タブ2: キーワード検索（独立モード）
    # タブ2: 統合検索
    with tab2:
        search_target = st.radio(t("search_target"), t("search_options"), key="search_target")
        # 検索対象の説明文マップ
        desc_map = {
            "Title + Abstract": t("search_desc_title_abstract"),
            "Author Name": t("search_desc_author"),
            "ORCID": t("search_desc_orcid"),
            "Affiliation Name": t("search_desc_affil"),
            "ROR": t("search_desc_ror"),
            "Concept": t("search_desc_concept"),
        }
        ph_map = {
            "Title + Abstract": t("search_input_ph_abstract"),
            "Author Name": t("search_input_ph_author"),
            "ORCID": t("search_input_ph_orcid"),
            "Affiliation Name": t("search_input_ph_affil"),
            "ROR": t("search_input_ph_ror"),
            "Concept": t("search_input_ph_concept"),
        }
        if search_target in desc_map:
            st.caption(desc_map[search_target])
        search_query = st.text_input(t("search_input_label"), placeholder=ph_map.get(search_target, ""), key="search_query")
 
        if st.button(t("search_btn"), key="search_btn"):
            if search_query:
                with st.spinner(t("search_running")):
                    results, count, result_type = search_openalex(search_query, search_target)
                    st.session_state.search_results = results
                    st.session_state.search_count = count
                    st.session_state.search_result_type = result_type
                    st.session_state.search_target_used = search_target
                    if result_type in ["authors", "institutions"]:
                        st.session_state.ego_author_id = None
                        st.session_state.ego_author_name = ""
                        st.session_state.org_ror_id = None
                        st.session_state.org_name = ""
 
        if st.session_state.get("search_results"):
            result_type = st.session_state.get("search_result_type", "works")
            count = st.session_state.get("search_count", 0)
            paper_word = " 論文" if lang == "ja" else " papers"
 
            if result_type in ("works", "works_concept"):
                shown = len(st.session_state.search_results)
                if count == 0:
                    st.warning(t("no_results"))
                elif count < 200:
                    st.warning(str(count) + t("papers_few"))
                elif count <= 1500:
                    st.success(str(count) + t("papers_good"))
                else:
                    st.info(str(count) + t("papers_many_pre") + str(shown) + t("papers_many_suf"))
                # Conceptの場合はConcept情報を表示してフィルタに使えるよう保存
                if result_type == "works_concept" and st.session_state.search_results:
                    cname = st.session_state.search_results[0].get("_concept_name", "")
                    cid = st.session_state.search_results[0].get("_concept_id", "")
                    if cname:
                        st.info("Concept: **" + cname + "**  [" + cid.split("/")[-1] + "]")
                        st.session_state.concept_id = cid
                        st.session_state.concept_name = cname
                for w in st.session_state.search_results:
                    title = w.get("title", "No title") or "No title"
                    year = str(w.get("publication_year", ""))
                    doi = w.get("doi", "") or ""
                    authors = w.get("authorships", [])
                    first_author = authors[0].get("author", {}).get("display_name", "") if authors else ""
                    author_str = first_author + (" et al." if len(authors) > 1 else "")
                    st.markdown("**" + title[:80] + ("..." if len(title) > 80 else "") + "**")
                    info = year + (" | " + author_str if author_str else "") + (" | [DOI](" + doi + ")" if doi else "")
                    st.caption(info)
                    st.markdown("---")
 
            elif result_type == "authors":
                st.markdown(t("search_results_label"))
                for a in st.session_state.search_results:
                    aid = a.get("id", "")
                    name = a.get("display_name", "")
                    wcount = a.get("works_count", 0)
                    insts = a.get("last_known_institutions", [])
                    inst_name = insts[0].get("display_name", "") if insts else ""
                    label = name + "  (" + str(wcount) + paper_word
                    if inst_name: label += " / " + inst_name[:30]
                    label += ")"
                    if st.button(label, key="sr_author_" + aid.split("/")[-1]):
                        st.session_state.ego_author_id = aid
                        st.session_state.ego_author_name = name
                        st.rerun()
 
            elif result_type == "institutions":
                st.markdown(t("search_results_label"))
                for inst in st.session_state.search_results:
                    iid = inst.get("id", "")
                    name = inst.get("display_name", "")
                    ror = inst.get("ror", "") or ""
                    country = inst.get("country_code", "")
                    itype = inst.get("type", "")
                    wcount = inst.get("works_count", 0)
                    ror_short = ror.replace("https://ror.org/", "") if ror else iid.split("/")[-1]
                    label = name + "  (" + itype + " / " + country + " / " + str(wcount) + paper_word + ")  [ROR: " + ror_short + "]"
                    if st.button(label, key="sr_inst_" + ror_short):
                        st.session_state.org_ror_id = ror or iid
                        st.session_state.org_name = name
                        st.rerun()
 
        # 著者選択済み表示＋フィルタ
        if st.session_state.ego_author_id:
            st.success(t("selected_author") + st.session_state.ego_author_name)
            if st.button(t("clear_author"), key="ego_clear"):
                st.session_state.ego_author_id = None
                st.session_state.ego_author_name = ""
                st.rerun()
            st.markdown("---")
            st.markdown(t("filter_label_search"))
            col1, col2 = st.columns(2)
            ego_year_from = col1.number_input("From", 1990, 2026, 1990, key="ego_year_from")
            ego_year_to = col2.number_input("To", 1990, 2026, 2026, key="ego_year_to")
            ego_oa_only = st.toggle(t("oa_only"), key="ego_oa_only")
            try:
                fp = ["authorships.author.id:" + st.session_state.ego_author_id]
                fp.append("publication_year:" + str(ego_year_from) + "-" + str(ego_year_to))
                if ego_oa_only: fp.append("is_oa:true")
                show_count_preview(get_count(fp), per_page=500)
            except: pass
 
        # 機関選択済み表示＋フィルタ
        if st.session_state.get("org_ror_id"):
            ror_display = st.session_state.org_ror_id.replace("https://ror.org/", "")
            st.success(t("selected_org") + st.session_state.org_name + "  [ROR: " + ror_display + "]")
            if st.button(t("clear_org"), key="org_clear"):
                st.session_state.org_ror_id = None
                st.session_state.org_name = ""
                st.rerun()
            st.markdown("---")
            st.markdown(t("filter_label_search"))
            col1, col2 = st.columns(2)
            inst_year_from = col1.number_input("From", 1990, 2026, 2020, key="inst_year_from")
            inst_year_to = col2.number_input("To", 1990, 2026, 2025, key="inst_year_to")
            inst_oa_only = st.toggle(t("oa_only"), key="inst_oa_only")
            try:
                ror_f = st.session_state.org_ror_id
                if not ror_f.startswith("https://ror.org/"): ror_f = "https://ror.org/" + ror_f
                fp = ["authorships.institutions.ror:" + ror_f]
                fp.append("publication_year:" + str(inst_year_from) + "-" + str(inst_year_to))
                if inst_oa_only: fp.append("is_oa:true")
                show_count_preview(get_count(fp), per_page=500)
            except: pass
 
        st.caption(t("ror_caption"))
 
 
    # フィルタ
    st.markdown(t("filters"))
    keyword_filter = st.text_input(t("kw_filter_label"), placeholder=t("kw_filter_ph"), key="keyword_filter")
    col1, col2 = st.columns(2)
    year_from = col1.number_input("From", 1990, 2026, 2020)
    year_to = col2.number_input("To", 1990, 2026, 2025)
    oa_only = st.toggle(t("oa_only"))
    sel_countries = st.multiselect(t("country"), t("countries"))
    per_page = st.slider(t("num_papers"), 100, 2000, 500, 100)
    min_links = st.slider(t("min_links"), 1, 10, 2)
    min_sim = st.slider(t("min_sim"), 0.05, 0.5, 0.15, 0.05)
 
    # KeyBERT系分析タイプを選んだときだけモデル選択を表示
    kw_analysis_types = ["Keyword Co-occurrence（KeyBERT）", "著者類似度（KeyBERT）",
                         "Keyword Co-occurrence (KeyBERT)", "Author Similarity (KeyBERT)"]
    if analysis_type in kw_analysis_types:
        st.markdown("---")
        model_options = [t("model_auto")] + [v[1] for v in KEYBERT_MODELS.values()]
        selected_model_label = st.selectbox(t("model_select"), model_options, key="keybert_model_label")
        # 現在のモデル説明を表示
        if selected_model_label == t("model_auto"):
            st.caption("🤖 " + ("トピック・Conceptから分野を自動判定してモデルを切替" if lang=="ja" else "Auto-detects domain from topics/concepts and selects the best model"))
        else:
            st.caption("🔬 " + selected_model_label)
    else:
        selected_model_label = t("model_auto")
 
    selected_topics = st.session_state.selected_topics
    if selected_topics:
        tids = ["T" + topic_map[tp] for tp in selected_topics if tp in topic_map]
        if tids:
            fp = ["topics.id:" + tids[0]] if len(tids)==1 else ["topics.id:" + "|".join(tids)]
            fp.append("publication_year:" + str(year_from) + "-" + str(year_to))
            if oa_only: fp.append("is_oa:true")
            if sel_countries:
                codes = [c.split()[0] for c in sel_countries]
                fp.append("authorships.institutions.country_code:" + (codes[0] if len(codes)==1 else "|".join(codes)))
            if keyword_filter: fp.append("title.search:" + keyword_filter)
            show_count_preview(get_count(fp), per_page)
 
    run = st.button(t("run_btn"), type="primary", use_container_width=True)
 
# ── メイン ──
st.markdown(t("title"))
 
# キーワードモードの直接実行
if st.session_state.get("kw_mode_active"):
    st.session_state.kw_mode_active = False
    kw_filt = st.session_state.get("kw_filter", "")
    if kw_filt:
        with st.spinner(t("fetching")):
            works_kw = fetch_works([kw_filt], per_page)
        st.success(t("fetched") + str(len(works_kw)) + t("fetched_end"))
        with st.spinner(t("building")):
            vos_data_kw = build_coauth(works_kw, min_links)
        st.session_state.vos_data = vos_data_kw
        st.session_state.works_count = len(works_kw)
        st.session_state.json_str = json.dumps({"network": vos_data_kw})
        st.session_state.last_analysis_type = "Co-authorship"
 
if run:
    ego_id = st.session_state.ego_author_id
    org_ror = st.session_state.get("org_ror_id")
    concept_id = st.session_state.get("concept_id")
    if not selected_topics and not ego_id and not org_ror and not concept_id:
        st.warning(t("no_topic_warn"))
        st.stop()
    if org_ror:
        ror_filter = org_ror if org_ror.startswith("https://ror.org/") else "https://ror.org/" + org_ror
        filters = ["authorships.institutions.ror:" + ror_filter]
        _yf = st.session_state.get("inst_year_from", year_from)
        _yt = st.session_state.get("inst_year_to", year_to)
        _oa = st.session_state.get("inst_oa_only", False)
        filters.append("publication_year:" + str(_yf) + "-" + str(_yt))
        if _oa: filters.append("is_oa:true")
        st.info(t("org_info") + st.session_state.get("org_name", "") + t("org_info_end"))
    elif ego_id:
        filters = ["authorships.author.id:" + ego_id]
        _yf = st.session_state.get("ego_year_from", 1990)
        _yt = st.session_state.get("ego_year_to", 2026)
        _oa = st.session_state.get("ego_oa_only", False)
        filters.append("publication_year:" + str(_yf) + "-" + str(_yt))
        if _oa: filters.append("is_oa:true")
        st.info(t("ego_info") + st.session_state.ego_author_name + t("ego_info_end") + str(_yf) + "〜" + str(_yt) + "）")
    elif st.session_state.get("concept_id"):
        # Concept起点モード
        filters = ["concepts.id:" + st.session_state.concept_id]
        filters.append("publication_year:" + str(year_from) + "-" + str(year_to))
        if oa_only: filters.append("is_oa:true")
        if sel_countries:
            codes = [c.split()[0] for c in sel_countries]
            filters.append("authorships.institutions.country_code:" + (codes[0] if len(codes)==1 else "|".join(codes)))
        cname = st.session_state.get("concept_name", "")
        st.info("Concept mode: " + cname)
    else:
        tids = ["T" + topic_map[tp] for tp in selected_topics if tp in topic_map]
        if not tids:
            st.warning(t("no_valid_topic"))
            st.stop()
        filters = ["topics.id:" + tids[0]] if len(tids)==1 else ["topics.id:" + "|".join(tids)]
        filters.append("publication_year:" + str(year_from) + "-" + str(year_to))
        if oa_only: filters.append("is_oa:true")
        if sel_countries:
            codes = [c.split()[0] for c in sel_countries]
            filters.append("authorships.institutions.country_code:" + (codes[0] if len(codes)==1 else "|".join(codes)))
        if keyword_filter: filters.append("title.search:" + keyword_filter)
 
    with st.spinner(t("fetching")):
        works = fetch_works(filters, per_page)
    st.success(t("fetched") + str(len(works)) + t("fetched_end"))
 
    work_keywords = {}
    kw_options = ["Keyword Co-occurrence（KeyBERT）", "著者類似度（KeyBERT）", "Keyword Co-occurrence (KeyBERT)", "Author Similarity (KeyBERT)"]
    if analysis_type in kw_options:
        # モデル選択
        model_label = st.session_state.get("keybert_model_label", t("model_auto"))
        if model_label == t("model_auto"):
            domain = detect_domain(
                st.session_state.selected_topics,
                st.session_state.get("concept_name", "")
            )
            model_key, model_display, _ = KEYBERT_MODELS[domain]
            st.info(t("model_detected") + "**" + domain + "** → " + model_display)
        else:
            # ラベルからmodel_keyを逆引き
            model_key = next((v[0] for v in KEYBERT_MODELS.values() if v[1] == model_label), "all-MiniLM-L6-v2")
            model_display = model_label
        with st.spinner(t("keybert_running") + " (" + model_display + ")"):
            kw_model = load_keybert(model_key)
            work_keywords = extract_keywords_batch(works, kw_model)
        st.success(t("keybert_done") + str(sum(len(v) for v in work_keywords.values())) + t("keybert_done_end") + " [" + model_display + "]")
 
    with st.spinner(t("building")):
        if analysis_type in ["Co-authorship"]:
            vos_data = build_coauth(works, min_links)
        elif analysis_type in ["Co-citation"]:
            vos_data = build_cocitation(works, min_links)
        elif analysis_type in ["Bibliographic coupling"]:
            vos_data = build_bibcoupling(works, min_links)
        elif analysis_type in ["Keyword Co-occurrence（KeyBERT）", "Keyword Co-occurrence (KeyBERT)"]:
            vos_data = build_keyword_cooccurrence(works, work_keywords, min_links)
        elif analysis_type in ["著者類似度（KeyBERT）", "Author Similarity (KeyBERT)"]:
            vectors, author_names = build_author_vectors(works, work_keywords)
            vos_data = build_similarity_network(vectors, author_names, min_sim)
        elif analysis_type in ["機関コラボレーション", "Institution Collaboration"]:
            vos_data = build_institution_collab(works, min_links)
        else:
            vos_data = build_country_network(works, min_links)
 
    st.session_state.vos_data = vos_data
    st.session_state.works_count = len(works)
    st.session_state.json_str = json.dumps({"network": vos_data})
    st.session_state.work_keywords = work_keywords
    st.session_state.last_analysis_type = analysis_type
 
if st.session_state.vos_data:
    vos_data = st.session_state.vos_data
    c1, c2, c3 = st.columns(3)
    c1.metric(t("papers"), str(st.session_state.works_count))
    c2.metric(t("nodes"), str(len(vos_data["items"])))
    c3.metric(t("edges"), str(len(vos_data["links"])))
 
    ana = st.session_state.get("last_analysis_type", "")
    explanations = {
        "Co-authorship": [(t("node_label"), "Author (size = papers)" if lang=="en" else "著者（サイズ = 論文数）"), (t("edge_label"), "Co-authorship (thickness = co-authored papers)" if lang=="en" else "共著関係（太さ = 共著論文数）"), (t("cluster_label"), "Group of frequent collaborators" if lang=="en" else "一緒に研究することが多いグループ"), (t("usage_label"), "Dense clusters = core research community" if lang=="en" else "密なクラスターが研究コミュニティの核")],
        "Co-citation": [(t("node_label"), "Cited paper (size = citations)" if lang=="en" else "引用論文（サイズ = 被引用数）"), (t("edge_label"), "Times co-cited" if lang=="en" else "同時引用回数"), (t("cluster_label"), "Foundational paper group" if lang=="en" else "基盤論文グループ"), (t("usage_label"), "Identify key papers in field" if lang=="en" else "重要論文・古典の特定")],
        "Bibliographic coupling": [(t("node_label"), "Target paper (size = citations)" if lang=="en" else "対象論文（サイズ = 被引用数）"), (t("edge_label"), "Shared references" if lang=="en" else "共通参考文献の数"), (t("cluster_label"), "Groups sharing bibliographic base" if lang=="en" else "同じ文献基盤を持つグループ"), (t("usage_label"), "Understand research streams" if lang=="en" else "研究の流れを把握")],
        "Keyword Co-occurrence（KeyBERT）": [(t("node_label"), "キーワード（サイズ = 登場論文数）"), (t("edge_label"), "同じ論文での共起回数"), (t("cluster_label"), "関連研究テーマのグループ"), (t("usage_label"), "クラスター間ノードが複数テーマを橋渡し")],
        "Keyword Co-occurrence (KeyBERT)": [(t("node_label"), "Keyword (size = papers)"), (t("edge_label"), "Co-occurrence in same paper"), (t("cluster_label"), "Related research theme group"), (t("usage_label"), "Bridge nodes connect multiple themes")],
        "著者類似度（KeyBERT）": [(t("node_label"), "著者（サイズ = 研究スコア）"), (t("edge_label"), "研究テーマ類似度"), (t("cluster_label"), "研究テーマが近い著者グループ"), (t("usage_label"), "潜在的な協力者を発見")],
        "Author Similarity (KeyBERT)": [(t("node_label"), "Author (size = score)"), (t("edge_label"), "Research theme similarity"), (t("cluster_label"), "Authors with similar themes"), (t("usage_label"), "Discover potential collaborators")],
        "機関コラボレーション": [(t("node_label"), "機関（サイズ = 論文数）"), (t("edge_label"), "共著論文数"), (t("cluster_label"), "連携機関グループ"), (t("usage_label"), "産学連携・機関間ネットワーク把握")],
        "Institution Collaboration": [(t("node_label"), "Institution (size = papers)"), (t("edge_label"), "Co-authored papers"), (t("cluster_label"), "Collaborative institution group"), (t("usage_label"), "Identify academia-industry partnerships")],
        "国際共著ネットワーク": [(t("node_label"), "国（サイズ = 論文数）"), (t("edge_label"), "2国間の共著論文数"), (t("cluster_label"), "共同研究が多い国グループ"), (t("usage_label"), "国際共同研究のハブ国を特定")],
        "Country Network": [(t("node_label"), "Country (size = papers)"), (t("edge_label"), "Bilateral co-authored papers"), (t("cluster_label"), "Closely collaborating countries"), (t("usage_label"), "Identify international collaboration hubs")],
    }
    if ana in explanations:
        with st.expander(t("expander_label") + ana + ")", expanded=True):
            for label, desc in explanations[ana]:
                st.markdown("- **" + label + "**: " + desc)
 
    st.download_button(t("download_btn"), st.session_state.json_str, file_name="vosviewer_network.json", mime="application/json", use_container_width=True)
    st.markdown(t("vos_title"))
    st.markdown(t("vos_step1"))
    st.markdown(t("vos_step2"))
    st.markdown(t("vos_step3"))
 
    if vos_data["items"]:
        st.subheader(t("top_nodes"))
        key = list(vos_data["items"][0]["weights"].keys())[0]
        for i, a in enumerate(sorted(vos_data["items"], key=lambda x: x["weights"].get(key, 0), reverse=True)[:20], 1):
            st.write(str(i) + ". " + a["label"] + " — " + str(round(a["weights"].get(key, 0), 3)))
            if a.get("description"):
                st.caption(a["description"])