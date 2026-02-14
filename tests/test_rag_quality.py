"""
RAG Quality Eval — semantic vs hybrid comparison.
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


def eval_single(query_info, n_results=15, threshold=0.25, max_chunks=10, search_fn=None):
    """Evaluate a single query using the given search function."""
    if search_fn is None:
        search_fn = vs.query

    # Determine the right call signature
    if search_fn == vs.query:
        results = search_fn(
            query_text=query_info["query"],
            n_results=n_results,
            course_filter=query_info.get("course"),
        )
    else:
        results = search_fn(
            query=query_info["query"],
            n_results=n_results,
            course_filter=query_info.get("course"),
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


def eval_all(n_results=15, threshold=0.25, max_chunks=10, verbose=True, search_fn=None):
    """Run all queries, return aggregate scores."""
    results = []
    for q in TEST_QUERIES:
        r = eval_single(q, n_results, threshold, max_chunks, search_fn=search_fn)
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

    return {
        "results": results,
        "avg_precision": avg_precision,
        "avg_top": avg_top,
        "avg_chunks": avg_chunks,
        "pass_rate": pass_rate,
    }


def compare():
    """Run semantic-only vs hybrid, print side-by-side comparison."""
    print("=" * 70)
    print("=== SEMANTIC ONLY (FAISS) ===")
    print("=" * 70)
    sem = eval_all(search_fn=vs.query)

    print(f"\n\n{'=' * 70}")
    print("=== HYBRID (BM25 + FAISS via RRF) ===")
    print("=" * 70)
    hyb = eval_all(search_fn=vs.hybrid_search)

    # Per-query delta
    print(f"\n\n{'=' * 70}")
    print("=== PER-QUERY DELTA (hybrid - semantic) ===")
    print(f"{'query':<40} {'sem_kw':>6} {'hyb_kw':>6} {'delta':>6}")
    print("-" * 60)
    for s, h in zip(sem["results"], hyb["results"]):
        delta = h["keyword_precision"] - s["keyword_precision"]
        marker = "⬆️" if delta > 0 else "⬇️" if delta < 0 else "  "
        print(
            f"{s['query']:<40} "
            f"{s['keyword_precision']:>5.0%} "
            f"{h['keyword_precision']:>5.0%} "
            f"{delta:>+5.0%} {marker}"
        )

    print(f"\n{'=' * 70}")
    dp = hyb["avg_precision"] - sem["avg_precision"]
    dr = hyb["pass_rate"] - sem["pass_rate"]
    print(f"SUMMARY: kw_precision {sem['avg_precision']:.0%} → {hyb['avg_precision']:.0%} ({dp:+.0%})")
    print(f"         pass_rate   {sem['pass_rate']:.0%} → {hyb['pass_rate']:.0%} ({dr:+.0%})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare semantic vs hybrid")
    parser.add_argument("--hybrid", action="store_true", help="Run eval with hybrid search")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--n-results", type=int, default=15)
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.hybrid:
        print(f"=== Hybrid Eval: n={args.n_results} t={args.threshold} mc={args.max_chunks} ===\n")
        eval_all(args.n_results, args.threshold, args.max_chunks, search_fn=vs.hybrid_search)
    else:
        print(f"=== Semantic Eval: n={args.n_results} t={args.threshold} mc={args.max_chunks} ===\n")
        eval_all(args.n_results, args.threshold, args.max_chunks, search_fn=vs.query)
