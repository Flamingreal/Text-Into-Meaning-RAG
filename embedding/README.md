# RAG Pipeline — Step 3: Chunk Embedding

## 策略：natural_language_prefix

在 chunk 正文前拼接一段人类可读的上下文前缀，再送入 embedding 模型。
前缀根据 `page_type` 差异化生成，帮助模型感知文档层级语义（标题 + 章节）。

### 前缀模板

| page_type    | 模板                                              | 示例输出                                                    |
|------------- |---------------------------------------------------|-------------------------------------------------------------|
| `blog`       | `Blog: {title}.`                                  | `Blog: 12. Japan.`                                          |
| `recipe`     | `Recipe for {title}. {sections} section.`         | `Recipe for Broccoli Stir Fry. Ingredients, Procedure section.` |
| `ingredient` | `Ingredient guide: {title}.`                      | `Ingredient guide: Red Bean Paste.`                         |
| `overview`   | `About {title}. {sections}.`                      | `About Chinese cuisine. History.`                           |
| `sub_cuisine`| `About {title}. {sections}.`                      | `About Beijing cuisine. Lead.`                              |
| *(fallback)* | `{title}. {sections}.`                            | —                                                           |

- `{title}` 中会自动去掉 `Cookbook:` 前缀（Wikibooks 条目）
- `{sections}` 会去掉 `(part N)` 后缀并去重，逗号拼接

---

## 用法

```bash
# 默认：chunks_embedding.jsonl + Qwen3-Embedding-4B
python embedding/embed_chunks.py

# 指定 chunk 文件
python embedding/embed_chunks.py --input artifacts/chunks/chunks_tfidf.jsonl
python embedding/embed_chunks.py --input artifacts/chunks/corpus_fixed.jsonl

# 指定更小的模型（0.6B，更快）
python embedding/embed_chunks.py --model model_checkpoints/Qwen__Qwen3-Embedding-0.6B

# 组合参数
python embedding/embed_chunks.py \
    --input      artifacts/chunks/chunks_tfidf.jsonl \
    --model      model_checkpoints/Qwen__Qwen3-Embedding-0.6B \
    --batch-size 16 \
    --max-length 512

# 只验证前缀输出，不实际编码（快速 debug）
python embedding/embed_chunks.py --dry-run
```

---

## 输出文件

所有文件输出到 `artifacts/faiss/`，命名格式：

```
{input_stem}.{model_name}.nlp_prefix.{index_type}.{ext}
```

| 文件                        | 描述                                       |
|-----------------------------|--------------------------------------------|
| `.index`                    | FAISS 二进制索引（FlatIP，支持余弦检索）   |
| `.meta.jsonl`               | 每行对应一个向量，含原始 chunk 元数据 + `embed_text` |
| `.manifest.json`            | 运行参数记录（模型、维度、策略、路径等）   |

### meta.jsonl 字段

```json
{
  "vector_id": 0,
  "chunk_id": 0,
  "source": "80Cuisine",
  "page_type": "blog",
  "title": "12. Japan",
  "sections": ["Lead (part 1)", "Lead (part 2)"],
  "url": "https://...",
  "text": "Japan has the most Michelin star restaurants...",
  "embed_text": "Blog: 12. Japan.\n\nJapan has the most Michelin star..."
}
```

---

## 查询端对齐

检索时，query 应使用与文档相同的 instruction 进行编码，以保持 embedding 空间对齐：

```python
from sentence_transformers import SentenceTransformer
import faiss, json

model = SentenceTransformer("model_checkpoints/Qwen__Qwen3-Embedding-4B", trust_remote_code=True)

# query 编码（使用 retrieval.query prompt）
query = "What is bento box?"
q_emb = model.encode(
    [query],
    prompt_name="retrieval.query",   # 与文档侧 retrieval.passage 对应
    normalize_embeddings=True,
    convert_to_numpy=True,
)

# 检索
index = faiss.read_index("artifacts/faiss/chunks_embedding.Qwen__Qwen3-Embedding-4B.nlp_prefix.FlatIP.index")
scores, ids = index.search(q_emb, k=5)

# 读取元数据
meta = []
with open("artifacts/faiss/chunks_embedding.Qwen__Qwen3-Embedding-4B.nlp_prefix.FlatIP.meta.jsonl") as f:
    for line in f:
        meta.append(json.loads(line))

for score, idx in zip(scores[0], ids[0]):
    print(f"score={score:.4f}  {meta[idx]['title']} / {meta[idx]['sections']}")
    print(f"  {meta[idx]['text'][:120]}…\n")
```
