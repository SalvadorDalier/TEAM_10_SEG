import os
import sys
import time
import json
import csv
import pandas as pd
from flask import Flask, jsonify, request, render_template

# Fix stdout encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data import DOCS, QUERIES
from embedding import Embedder
from learned_index import LearnedIndex, LearnedIndexTrainer
from vector_db import VectorDB, METHOD_INVERTED, METHOD_HNSW, METHOD_LEARNED
from benchmark import benchmark_all, plot_results

app = Flask(__name__)
app.json.ensure_ascii = False

# Global variables for databases and data
embedder = None
doc_vectors = None
query_vectors = None
db_inverted = None
db_hnsw = None
db_learned = None

def init_dbs():
    global embedder, doc_vectors, query_vectors
    global db_inverted, db_hnsw, db_learned

    print("[INFO] Initializing Embedder...")
    embedder = Embedder()
    
    print(f"[INFO] Encoding {len(DOCS)} documents...")
    doc_vectors = embedder.encode(DOCS)
    
    print(f"[INFO] Encoding {len(QUERIES)} queries...")
    query_vectors = embedder.encode(QUERIES)

    print("[INFO] Building Inverted Index...")
    db_inverted = VectorDB(method=METHOD_INVERTED)
    db_inverted.build(doc_vectors)

    print("[INFO] Building HNSW Index...")
    db_hnsw = VectorDB(method=METHOD_HNSW)
    db_hnsw.build(doc_vectors)

    print("[INFO] Building Learned Index...")
    trainer = LearnedIndexTrainer(doc_vectors=doc_vectors, n_clusters=4)
    X_train, y_train = trainer.create_training_data(queries=query_vectors, doc_vectors=doc_vectors)
    learned_model = LearnedIndex()
    learned_model.train_model(X_train, y_train, epochs=100, lr=1e-3)
    db_learned = VectorDB(method=METHOD_LEARNED, learned_model=learned_model)
    db_learned.build(doc_vectors)
    print("[INFO] Initialization complete.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_path = os.path.join("results", "metrics.csv")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics file not found"}), 404
    
    try:
        df = pd.read_csv(metrics_path)
        # Normalize column names for the frontend
        results = []
        for _, row in df.iterrows():
            recall_col = [c for c in df.columns if c.startswith("Recall")][0]
            results.append({
                "method": row["Method"],
                "recall": float(row[recall_col]),
                "qps": float(row["QPS"]),
                "memory": float(row["Memory (MB)"])
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('query', '')
    k = data.get('k', 3)

    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    try:
        embed_start = time.perf_counter()
        q_vec = embedder.encode([query_text])[0]
        embed_time_ms = (time.perf_counter() - embed_start) * 1000
    except Exception as e:
        return jsonify({"error": f"Failed to encode query: {e}"}), 500

    results = {}
    
    methods = {
        'inverted': db_inverted,
        'hnsw': db_hnsw,
        'learned': db_learned
    }

    for name, db in methods.items():
        start_time = time.perf_counter()
        doc_ids = db.search(q_vec, k=k)
        search_time_ms = (time.perf_counter() - start_time) * 1000

        docs = []
        for d_id in doc_ids:
            if 0 <= d_id < len(DOCS):
                docs.append({
                    "doc_id": d_id,
                    "text": DOCS[d_id]
                })

        results[name] = {
            "search_time_ms": round(search_time_ms, 4),
            "docs": docs
        }

    return jsonify({"embed_time_ms": round(embed_time_ms, 2), "results": results})

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark_api():
    try:
        dbs = {
            METHOD_INVERTED: db_inverted,
            METHOD_HNSW: db_hnsw,
            METHOD_LEARNED: db_learned,
        }
        metrics_df = benchmark_all(
            dbs=dbs,
            doc_vectors=doc_vectors,
            query_vectors=query_vectors,
            k=3,
            n_repeat=1000,
        )
        os.makedirs("results", exist_ok=True)
        metrics_df.to_csv(os.path.join("results", "metrics.csv"), index=False)
        plot_results(metrics_df, os.path.join("results", "plots"))
        
        # Normalize column names for the frontend
        recall_col = [c for c in metrics_df.columns if c.startswith("Recall")][0]
        results = []
        for _, row in metrics_df.iterrows():
            results.append({
                "method": row["Method"],
                "recall": float(row[recall_col]),
                "qps": float(row["QPS"]),
                "memory": float(row["Memory (MB)"])
            })
        return jsonify({"metrics": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize databases before the first request or at startup
# Using a manual call instead of before_first_request (deprecated)
with app.app_context():
    init_dbs()

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
