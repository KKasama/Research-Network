# 🔬 Research Network Portal v2

**日本語** | [English](#english)

OpenAlex API を活用した研究ネットワーク収集・分析・可視化ツールです。論文データの収集からクラスタリング、中心性分析、科研費（KAKEN）助成金の可視化まで、3ステップで実行できます。

---

## 日本語

### ワークフロー概要

```
① データ収集・保存
    ↓  著者 / 機関 / テーマで論文を取得 → ローカルJSONに自動保存
② 分析・可視化
    ↓  保存データを読み込み → クラスタリング / ネットワーク分析 → 中心性ランキング
③ KAKEN助成金分析（独立して利用可）
       キーワード・年度・金額でKAKEN採択データを検索 → グラフ表示
```

---

### ステップ1：データ収集・保存

#### 検索方式
| 方式 | 説明 |
|---|---|
| 🗂 階層ブラウズ | Domain / Field / Subfield からトピックを選択 |
| 🔍 著者名 / ORCID | 著者エゴネットワーク向け |
| 🏛 機関名 / ROR | 機関単位の論文収集 |
| 📝 タイトル＋抄録 | キーワードで全文検索 |
| 💡 Concept | OpenAlexの概念タグで絞り込み |

- 収集した論文は `~/research_data/` にJSONとして**自動命名・保存**（例：`Child_Abuse_20260418_1430.json`）

---

### ステップ2：分析・可視化

#### 分析手法

| 手法 | ノード | エッジ | 向いている用途 |
|---|---|---|---|
| 共著ネットワーク | 著者 | 共著関係 | 研究者間の協力関係 |
| 書誌結合 | 論文 | 共通参考文献数 | テーマが近い論文のグループ発見 |
| 直接引用 | 論文 | 引用関係 | 論文の影響力・源流の把握 |
| BERTopic | トピック | トピック共起 | 自動トピック分類（内容類似度） |
| K-means | クラスター | クラスター共起 | クラスター数を指定した分類 |
| KeyBERT | キーワード | キーワード共起 | 頻出テーマ・用語の可視化 |

#### 中心性ランキング（📊）

分析実行後に自動計算され、以下の3指標で上位ノードをランキング表示します。

| 指標 | 意味 |
|---|---|
| PageRank | 重要なノードからリンクされているほど高スコア。影響力の高い著者・論文 |
| 媒介中心性 | 異なるグループを橋渡しするハブ |
| 次数中心性 | 最も多くのノードと直接接続しているノード |

---

### ステップ3：KAKEN助成金分析

科学研究費助成事業（科研費）の採択データをOpenAlex経由で取得・可視化します。ステップ1・2とは独立して利用できます。

#### フィルタ
- キーワード（研究課題名）
- 年度範囲（開始年度）
- 取得件数・並び順（配分金額順 / 論文数順）

#### 表示タブ

| タブ | 内容 |
|---|---|
| 🏆 上位課題 | 配分金額または論文数の上位N件を横棒グラフで表示 |
| 📂 研究種目別 | 基盤研究A/B/C等のカテゴリ別に件数・金額・論文数を集計 |
| 📈 年度別トレンド | 採択件数・総配分金額の年度推移 |
| 🔵 金額×論文数 | 散布図で投資対効果の分布を確認 |
| 📋 一覧表 | 全件表示 + CSVダウンロード |

---

### インストール

```bash
# 基本パッケージ
pip3 install streamlit requests

# 分析系
pip3 install keybert sentence-transformers
pip3 install bertopic
pip3 install scikit-learn          # K-means
pip3 install networkx              # 中心性ランキング

# 可視化系
pip3 install plotly                # KAKENグラフ
pip3 install pyvis                 # ネットワーク図

# SPECTER2（任意）
pip3 install adapters
```

### 環境変数（任意）

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # スマート検索用（Claude API）
```

### 必要なデータ

`topics.json` を `~/Downloads/` に配置してください（階層ブラウズ機能で使用）。

```bash
python3 << 'PYEOF'
import requests, json
topics, page = [], 1
while page <= 30:
    r = requests.get(
        f"https://api.openalex.org/topics?per_page=200&page={page}"
        "&select=id,display_name,subfield,field,domain&mailto=vos@example.com",
        timeout=30
    )
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
python3 -m streamlit run ~/Downloads/app8.py
```

---

## English

### Workflow Overview

```
① Collect & Save
    ↓  Fetch papers by author / institution / topic → auto-save as local JSON
② Analyze & Visualize
    ↓  Load saved data → clustering / network analysis → centrality ranking
③ KAKEN Grant Analysis  (independent — no prior steps required)
       Search KAKEN adoption data by keyword / year / amount → interactive charts
```

---

### Step 1: Collect & Save

#### Search Methods
| Method | Description |
|---|---|
| 🗂 Browse Hierarchy | Select topics by Domain / Field / Subfield |
| 🔍 Author Name / ORCID | For ego-network analysis |
| 🏛 Institution / ROR | Collect papers by institution |
| 📝 Title + Abstract | Full-text keyword search |
| 💡 Concept | Filter by OpenAlex concept tags |

- Collected papers are **auto-named and saved** to `~/research_data/` as JSON  
  (e.g., `Child_Abuse_20260418_1430.json`)

---

### Step 2: Analyze & Visualize

#### Analysis Methods

| Method | Nodes | Edges | Best for |
|---|---|---|---|
| Co-authorship | Authors | Shared papers | Collaboration structure |
| Bibliographic coupling | Papers | Shared references | Thematic grouping |
| Direct citation | Papers | Citation links | Influence & lineage |
| BERTopic | Topics | Topic co-occurrence | Auto topic clustering |
| K-means | Clusters | Cluster co-occurrence | Fixed-count clustering |
| KeyBERT | Keywords | Keyword co-occurrence | Term/theme visualization |

#### Centrality Ranking (📊)

Computed automatically after analysis. Ranks nodes by three metrics:

| Metric | Meaning |
|---|---|
| PageRank | High score when linked from important nodes — most influential authors/papers |
| Betweenness | Nodes bridging different research groups |
| Degree | Nodes with the most direct connections |

---

### Step 3: KAKEN Grant Analysis

Fetch and visualize KAKEN (Grants-in-Aid for Scientific Research) adoption data via OpenAlex. Can be used independently from Steps 1 and 2.

#### Filters
- Keyword (project title)
- Year range (start year)
- Fetch count and sort order (by amount or publication count)

#### Tabs

| Tab | Content |
|---|---|
| 🏆 Top Projects | Horizontal bar chart of top N grants by amount or outputs |
| 📂 By Category | Count / amount / outputs aggregated by grant type (e.g. Grant-in-Aid A/B/C) |
| 📈 Year Trend | Annual adoption count and total grant amount |
| 🔵 Amount × Outputs | Scatter plot of grant amount vs publication count |
| 📋 Table | Full list + CSV download |

---

### Installation

```bash
# Core
pip3 install streamlit requests

# Analysis
pip3 install keybert sentence-transformers
pip3 install bertopic
pip3 install scikit-learn          # K-means
pip3 install networkx              # Centrality ranking

# Visualization
pip3 install plotly                # KAKEN charts
pip3 install pyvis                 # Network graph

# SPECTER2 (optional)
pip3 install adapters
```

### Environment Variables (optional)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # for Smart Search (Claude API)
```

### Required Data

Place `topics.json` in `~/Downloads/` (used by the hierarchy browse feature).

```bash
python3 << 'PYEOF'
import requests, json
topics, page = [], 1
while page <= 30:
    r = requests.get(
        f"https://api.openalex.org/topics?per_page=200&page={page}"
        "&select=id,display_name,subfield,field,domain&mailto=vos@example.com",
        timeout=30
    )
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
python3 -m streamlit run ~/Downloads/app8.py
```
