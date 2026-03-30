# 🔬 Research Network Visualizer

**日本語** | [English](#english)

OpenAlex API と VOSviewer を組み合わせた論文ネットワーク可視化ツールです。

---

## 日本語

### 機能一覧

#### 検索方式
- 🗂 **階層ブラウズ** — Domain / Field / Subfield からトピックを選択
- 🔍 **検索タブ** — Title+Abstract / Author Name / ORCID / Affiliation / ROR / Concept
- 🧠 **スマート検索** — 自然言語クエリをClaude APIが条件分解して自動実行

#### 分析タイプ
| タイプ | 説明 |
|---|---|
| Co-authorship | 共著ネットワーク |
| Co-citation | 共引用ネットワーク |
| Bibliographic coupling | 書誌結合 |
| Keyword Co-occurrence (KeyBERT) | キーワード共起 |
| 著者類似度 (KeyBERT / SPECTER2) | 研究テーマ類似度 |
| 機関コラボレーション | 機関間ネットワーク |
| 国際共著ネットワーク | 国別ネットワーク |

#### 特徴
- **EGO起点モード** — 著者名検索→選択→個人ネットワーク構築
- **機関起点モード** — 機関名/ROR検索→選択→機関ネットワーク構築
- **SPECTER2** — aug2023refreshモデルをローカル実行（DOI不要・全論文対応）
- **Topic + Concept統合** — 2018年以前の論文もConceptで補完
- **funders.id対応** — JSPS/MEXT/JST/AMED等の助成機関フィルタ
- **KeyBERTドメイン自動判定** — battery/materials/biomedical/generalを自動選択
- **日本語 / English 切替** — サイドバーボタンまたは `?lang=ja`

#### KeyBERT vs SPECTER2

| 項目 | KeyBERT | SPECTER2 |
|---|---|---|
| エンベディング方式 | キーワード抽出→ベクトル | 768次元の密なベクトル |
| モデル | 分野特化（battery/materials/biomedical/general） | aug2023refresh（全分野） |
| 処理速度 | 遅い | 速い |
| 解釈性 | 高い（キーワード可視化） | 低い（768次元） |
| 2018年以前の論文 | △ | ✅ |
| 得意な用途 | キーワード共起・テーマ分類 | 著者類似度・協力者発見 |

#### Funder対応状況

| Funder | フィルタ方式 |
|---|---|
| JSPS / MEXT / JST / AMED | funders.id（直接フィルタ） |
| NIH / NSF / ERC / DFG / NSFC | funders.id（直接フィルタ） |
| NEDO | abstract.search:NEDO（キーワード代替） |

### インストール
```bash
pip3 install streamlit keybert sentence-transformers requests
pip3 install adapters  # SPECTER2用
```

### 環境変数
```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # スマート検索用（Claude API）
export S2_API_KEY="..."                 # Semantic Scholar API（任意）
```

### 必要なデータ

`topics.json` を `~/Downloads/` に配置してください。
```bash
python3 << 'PYEOF'
import requests, json
topics, page = [], 1
while page <= 30:
    r = requests.get(f"https://api.openalex.org/topics?per_page=200&page={page}&select=id,display_name,subfield,field,domain&mailto=vos@example.com", timeout=30)
    batch = r.json().get("results", [])
    if not batch: break
    topics.extend(batch)
    page += 1
with open("/Users/kk/Downloads/topics.json", "w") as f:
    json.dump(topics, f)
print(f"完了: {len(topics)} トピック")
PYEOF
```

### 起動方法
```bash
python3 -m streamlit run ~/Downloads/app7.py
```

---

## English

### Features

#### Search Methods
- 🗂 **Browse Hierarchy** — Select topics by Domain / Field / Subfield
- 🔍 **Search Tab** — Title+Abstract / Author Name / ORCID / Affiliation / ROR / Concept
- 🧠 **Smart Search** — Claude API decomposes natural language queries automatically

#### Analysis Types
| Type | Description |
|---|---|
| Co-authorship | Author collaboration network |
| Co-citation | Co-citation network |
| Bibliographic coupling | Shared references network |
| Keyword Co-occurrence (KeyBERT) | Keyword co-occurrence |
| Author Similarity (KeyBERT / SPECTER2) | Research theme similarity |
| Institution Collaboration | Institution-level network |
| Country Network | International collaboration network |

#### Highlights
- **EGO mode** — Search author → select → build personal network
- **Institution mode** — Search institution/ROR → select → build network
- **SPECTER2** — Local execution of aug2023refresh model (no DOI required)
- **Topic + Concept integration** — Supplement pre-2018 papers with Concepts
- **funders.id support** — Filter by JSPS / MEXT / JST / AMED / NIH / NSF etc.
- **Auto domain detection** — Automatically selects battery / materials / biomedical / general model
- **Bilingual UI** — Japanese / English toggle or `?lang=en`

#### KeyBERT vs SPECTER2

| Item | KeyBERT | SPECTER2 |
|---|---|---|
| Embedding method | Keyword extraction → vector | Dense 768-dim vector |
| Model | Domain-specific (battery/materials/biomedical/general) | aug2023refresh (all fields) |
| Speed | Slow | Fast |
| Interpretability | High (keywords visible) | Low (768-dim) |
| Pre-2018 papers | △ | ✅ |
| Best for | Keyword co-occurrence, theme clustering | Author similarity, collaborator discovery |

#### Funder Filter Support

| Funder | Method |
|---|---|
| JSPS / MEXT / JST / AMED | funders.id (direct filter) |
| NIH / NSF / ERC / DFG / NSFC | funders.id (direct filter) |
| NEDO | abstract.search:NEDO (keyword fallback) |

### Installation
```bash
pip3 install streamlit keybert sentence-transformers requests
pip3 install adapters  # for SPECTER2
```

### Environment Variables
```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # for Smart Search (Claude API)
export S2_API_KEY="..."                 # Semantic Scholar API (optional)
```

### Required Data

Place `topics.json` in `~/Downloads/`.
```bash
python3 << 'PYEOF'
import requests, json
topics, page = [], 1
while page <= 30:
    r = requests.get(f"https://api.openalex.org/topics?per_page=200&page={page}&select=id,display_name,subfield,field,domain&mailto=vos@example.com", timeout=30)
    batch = r.json().get("results", [])
    if not batch: break
    topics.extend(batch)
    page += 1
with open("/Users/kk/Downloads/topics.json", "w") as f:
    json.dump(topics, f)
print(f"Done: {len(topics)} topics")
PYEOF
```

### Launch
```bash
python3 -m streamlit run ~/Downloads/app7.py
```
