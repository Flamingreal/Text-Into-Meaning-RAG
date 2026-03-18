# Directory Structure

rag_baseline/
├── data/
│   ├── corpus.jsonl
│   ├── benchmark.json
│   └── test_queries.json
├── index_store/
├── outputs/
├── config.py
├── utils.py
├── build_index.py
├── run_inference.py
├── run_evaluation.py
├── requirements.txt
└── README.md



# RAG Baseline

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Build index

python build_index.py

## 3. Run inference

<pre class="overflow-visible! px-0!" data-start="12949" data-end="12984"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python run_inference.py</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

## 4. Run evaluation

<pre class="overflow-visible! px-0!" data-start="13007" data-end="13043"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python run_evaluation.py</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

## Files

* `data/corpus.jsonl`: background corpus
* `data/benchmark.json`: labelled benchmark for evaluation
* `data/test_queries.json`: inference input
* `outputs/inference_output.json`: inference outputs
* `outputs/evaluation_results.json`: evaluation results


# 怎么跑

按顺序执行：

```bash
pip install -r requirements.txt
python build_index.py
python inference.py
python evaluation.py
```
