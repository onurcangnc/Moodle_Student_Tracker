"""
RAG Quality Eval — iterative optimization.
Loads vector store, runs test queries, scores results.
Run on server: cd /opt/moodle-bot && source venv/bin/activate && python tests/test_rag_quality.py
"""
import sys, json, time
sys.path.insert(0, ".")
from core.vector_store import VectorStore

vs = VectorStore()
vs.initialize()

# ─── Test Queries ─────────────────────────────────────────────────────────────
# Each course has 2-3 queries with expected keywords
TEST_QUERIES = [
    # CTIS 363 — Ethical and Social Issues
    {
        "query": "privacy and surveillance",
        "expected_keywords": ["privacy", "surveillance", "data", "personal", "monitor"],
        "course": "CTIS 363",
    },
    {
        "query": "intellectual property copyright",
        "expected_keywords": ["intellectual", "property", "copyright", "patent", "software"],
        "course": "CTIS 363",
    },
    {
        "query": "computer ethics history",
        "expected_keywords": ["ethics", "computer", "history", "moral", "technology"],
        "course": "CTIS 363",
    },

    # CTIS 465 — Microservice Development
    {
        "query": "microservice architecture",
        "expected_keywords": ["microservice", "service", "architecture", "api", "deploy"],
        "course": "CTIS 465",
    },
    {
        "query": "docker container",
        "expected_keywords": ["docker", "container", "image", "deploy"],
        "course": "CTIS 465",
    },

    # EDEB 201 — Turkish Fiction
    {
        "query": "Felatun Bey ile Rakım Efendi",
        "expected_keywords": ["felatun", "rakım", "ahmet", "mithat"],
        "course": "EDEB 201",
    },
    {
        "query": "Kiralık Konak romanı",
        "expected_keywords": ["kiralık", "konak", "roman"],
        "course": "EDEB 201",
    },
    {
        "query": "Tanzimat dönemi edebiyat",
        "expected_keywords": ["tanzimat", "edebiyat", "batı", "roman"],
        "course": "EDEB 201",
    },
    {
        "query": "Doğu Batı çatışması Türk romanı",
        "expected_keywords": ["doğu", "batı", "roman", "çatışma"],
        "course": "EDEB 201",
    },

    # HCIV 102 — History of Civilization II
    {
        "query": "Columbian Exchange disease",
        "expected_keywords": ["columbian", "exchange", "disease", "food", "america"],
        "course": "HCIV 102",
    },
    {
        "query": "Ottoman Empire discovery",
        "expected_keywords": ["ottoman", "discovery", "columbus", "new world"],
        "course": "HCIV 102",
    },
    {
        "query": "what is history Carr",
        "expected_keywords": ["history", "carr", "historian", "fact", "interpretation"],
        "course": "HCIV 102",
    },
    {
        "query": "note taking strategies",
        "expected_keywords": ["note", "taking", "strategy", "lecture"],
        "course": "HCIV 102",
    },

    # Cross-course / general
    {
        "query": "Şerif Mardin modernleşme",
        "expected_keywords": ["mardin", "modernleşme", "batılılaşma", "tanzimat"],
        "course": None,
    },
    {
        "query": "Nurdan Gürbilek Kör Ayna",
        "expected_keywords": ["gürbilek", "kör", "ayna", "şark"],
        "course": None,
    },
]


def eval_single(query_info, n_results=15, threshold=0.30, max_chunks=7, course_filter=None):
    """Evaluate a single query."""
    results = vs.query(
        query_text=query_info["query"],
        n_results=n_results,
        course_filter=course_filter,
    )

    if not results:
        return {
            "query": query_info["query"],
            "total_results": 0,
            "after_filter": 0,
            "top_score": 0,
            "avg_score": 0,
            "keyword_precision": 0,
            "found_keywords": [],
            "missing_keywords": query_info["expected_keywords"],
            "unique_files": 0,
            "filenames": [],
        }

    scores = [(1 - r["distance"]) for r in results]
    top_score = scores[0]

    # Threshold filter
    filtered = [r for r, s in zip(results, scores) if s > threshold][:max_chunks]

    # Keyword precision
    all_text = " ".join(r.get("text", "").lower() for r in filtered)
    expected = query_info["expected_keywords"]
    found_keywords = [kw for kw in expected if kw.lower() in all_text]
    keyword_precision = len(found_keywords) / len(expected) if expected else 0

    # File diversity
    filenames = set()
    for r in filtered:
        meta = r.get("metadata", {})
        fname = meta.get("filename", "")
        if fname:
            filenames.add(fname)

    return {
        "query": query_info["query"],
        "total_results": len(results),
        "after_filter": len(filtered),
        "top_score": round(top_score, 4),
        "avg_score": round(sum(scores[:len(filtered)]) / max(len(filtered), 1), 4),
        "keyword_precision": round(keyword_precision, 2),
        "found_keywords": found_keywords,
        "missing_keywords": [k for k in expected if k.lower() not in all_text],
        "unique_files": len(filenames),
        "filenames": list(filenames),
    }


def eval_all(n_results=15, threshold=0.30, max_chunks=7, verbose=True):
    """Run all queries, return aggregate scores."""
    results = []
    for q in TEST_QUERIES:
        r = eval_single(q, n_results, threshold, max_chunks)
        results.append(r)
        if verbose:
            status = "✅" if r["keyword_precision"] >= 0.6 else "⚠️" if r["keyword_precision"] >= 0.3 else "❌"
            print(
                f"{status} {r['query']:<40} "
                f"top={r['top_score']:.3f} "
                f"kw={r['keyword_precision']:.0%} "
                f"chunks={r['after_filter']} "
                f"files={r['unique_files']}"
            )
            if r["missing_keywords"] and verbose:
                print(f"   missing: {r['missing_keywords']}")

    avg_precision = sum(r["keyword_precision"] for r in results) / len(results)
    avg_top = sum(r["top_score"] for r in results) / len(results)
    avg_chunks = sum(r["after_filter"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["keyword_precision"] >= 0.6) / len(results)

    print(f"\n{'='*70}")
    print(f"AGGREGATE: kw_precision={avg_precision:.0%}  top_score={avg_top:.3f}  "
          f"chunks={avg_chunks:.1f}  pass_rate={pass_rate:.0%}")
    print(f"PARAMS: n_results={n_results}  threshold={threshold}  max_chunks={max_chunks}")

    return {
        "results": results,
        "avg_precision": avg_precision,
        "avg_top": avg_top,
        "avg_chunks": avg_chunks,
        "pass_rate": pass_rate,
    }


def param_sweep():
    """Try multiple parameter combinations, find the best."""
    combos = []
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for max_chunks in [5, 7, 10]:
            for n_results in [10, 15, 20, 25]:
                combos.append((n_results, threshold, max_chunks))

    print(f"Running {len(combos)} combinations...\n")
    best = None
    best_score = -1
    all_results = []

    for n_results, threshold, max_chunks in combos:
        r = eval_all(n_results, threshold, max_chunks, verbose=False)
        score = r["avg_precision"] * 0.6 + r["pass_rate"] * 0.3 + r["avg_top"] * 0.1
        all_results.append({
            "n_results": n_results,
            "threshold": threshold,
            "max_chunks": max_chunks,
            "avg_precision": round(r["avg_precision"], 3),
            "pass_rate": round(r["pass_rate"], 3),
            "avg_top": round(r["avg_top"], 3),
            "avg_chunks": round(r["avg_chunks"], 1),
            "combined_score": round(score, 4),
        })
        if score > best_score:
            best_score = score
            best = all_results[-1]

    # Sort by combined score
    all_results.sort(key=lambda x: x["combined_score"], reverse=True)

    print(f"\n{'='*70}")
    print("TOP 10 COMBINATIONS:")
    print(f"{'n_res':>5} {'thresh':>6} {'max_ch':>6} | {'kw_prec':>7} {'pass%':>6} {'top_sc':>6} {'chunks':>6} | {'score':>6}")
    print("-" * 70)
    for r in all_results[:10]:
        print(
            f"{r['n_results']:>5} {r['threshold']:>6.2f} {r['max_chunks']:>6} | "
            f"{r['avg_precision']:>6.0%} {r['pass_rate']:>6.0%} {r['avg_top']:>6.3f} "
            f"{r['avg_chunks']:>6.1f} | {r['combined_score']:>6.4f}"
        )

    print(f"\n🏆 BEST: n_results={best['n_results']} threshold={best['threshold']} "
          f"max_chunks={best['max_chunks']} → score={best['combined_score']:.4f}")

    return best, all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--max-chunks", type=int, default=7)
    parser.add_argument("--n-results", type=int, default=15)
    args = parser.parse_args()

    if args.sweep:
        param_sweep()
    else:
        print(f"=== Eval: n_results={args.n_results} threshold={args.threshold} max_chunks={args.max_chunks} ===\n")
        eval_all(args.n_results, args.threshold, args.max_chunks)
