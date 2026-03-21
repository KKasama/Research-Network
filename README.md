# Research Network Visualizer

OpenAlex API + VOSviewer を使った論文ネットワーク可視化ツール

## 機能
- Co-authorship / Co-citation / Bibliographic coupling
- Keyword Co-occurrence（KeyBERT）
- 著者類似度（KeyBERT）
- 機関コラボレーション / 国際共著ネットワーク
- EGOネットワーク（著者起点）
- 機関・ROR検索
- 日本語 / English 切替

## 必要なライブラリ
```
pip3 install streamlit keybert sentence-transformers requests
```

## 必要なデータ
topics.jsonを~/Downloads/に配置してください。

取得方法:
```
python3 << 'PYEOF'
import requests, json
topics = []
page = 1
while True:
    url = f"https://api.openalex.org/topics?per_page=200&page={page}&select=id,display_name,subfield,field,domain&mailto=vos@example.com"
    batch = requests.get(url, timeout=30).json().get("results", [])
    if not batch: break
    topics.extend(batch)
    page += 1
    if page > 30: break
with open("/Users/kk/Downloads/topics.json", "w") as f:
    json.dump(topics, f)
print(f"完了: {len(topics)} トピック")
PYEOF
```

## 起動方法
```
python3 -m streamlit run app7.py
```
