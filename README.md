# 🔬 Research Network Visualizer
OpenAlex API + VOSviewer を使った論文ネットワーク可視化ツール
## 機能一覧
### 検索方式
- 🗂 **階層ブラウズ** — Domain / Field / Subfield からトピックを選択
- 🔍 **検索タブ** — Title+Abstract / Author Name / ORCID / Affiliation / ROR / Concept
- 🧠 **スマート検索** — 自然言語クエリをClaude APIが条件分解して自動実行

### 分析タイプ
- Co-authorship（共著ネットワーク）
- Co-citation（共引用）
- Bibliographic coupling（書誌結合）
- Keyword Co-occurrence（KeyBERT）
- 著者類似度（KeyBERT / SPECTER2）
- 機関コラボレーション
- 国際共著ネットワーク

### 特徴
- **EGO起点モード** — 著者名検索→選択→個人ネットワーク構築
- **機関起点モード** — 機関名/ROR検索→選択→機関ネットワーク構築
- **SPECTER2** — aug2023refreshモデルをローカル実行（DOI不要・全論文対応）
- **Topic + Concept統合** — 2018年以前の論文もConceptで補完
- **funders.id対応** — JSPS/MEXT/JST/AMED等の助成機関フィルタ
- **日本語 / English 切替** — サイドバーボタンまたは `?lang=ja`

## 必要環境
```bash
pip3 install streamlit keybert sentence-transformers requests
pip3 install adapters  # SPECTER2用
```

## 環境変数
```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # スマート検索用（Claude API）
export S2_API_KEY="..."                 # Semantic Scholar API（任意）
```

## 必要なデータ

topics.json を ~/Downloads/ に配置してください。

取得方法：
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

## 起動方法
```bash
python3 -m streamlit run ~/Downloads/app7.py
```

## KeyBERT vs SPECTER2

| 項目 | KeyBERT | SPECTER2 |
|---|---|---|
| エンベディング方式 | キーワード抽出→ベクトル | 768次元の密なベクトル |
| モデル | BatterySciBERT等（分野特化） | aug2023refresh（全分野） |
| 処理速度 | 遅い | 速い |
| 解釈性 | 高い（キーワード可視化） | 低い（768次元） |
| 2018年以前の論文 | △ | ✅ |
| 得意な用途 | キーワード共起・テーマ分類 | 著者類似度・協力者発見 |

## Funder対応状況

| Funder | フィルタ方式 |
|---|---|
| JSPS / MEXT / JST / AMED | funders.id（直接フィルタ） |
| NIH / NSF / ERC / DFG / NSFC | funders.id（直接フィルタ） |
| NEDO | abstract.search:NEDO（キーワード代替） |
