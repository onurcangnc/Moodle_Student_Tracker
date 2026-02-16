"""
RAG Quality Eval — static + auto-generated queries with baseline tracking.
Run: cd /opt/moodle-bot && source venv/bin/activate && python tests/test_rag_quality.py
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

# ─── Static Test Queries ──────────────────────────────────────────────────────
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
    {"query": "Kiralık Konak romanı", "expected_keywords": ["kiralık", "konak", "roman"], "course": "EDEB 201"},
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
    {"query": "Nurdan Gürbilek Kör Ayna", "expected_keywords": ["gürbilek", "kör", "ayna", "şark"], "course": None},
]


# ─── Auto Query Generation ────────────────────────────────────────────────────
STOPWORDS = {
    "bu",
    "bir",
    "ile",
    "için",
    "olan",
    "gibi",
    "daha",
    "kadar",
    "this",
    "that",
    "with",
    "from",
    "have",
    "will",
    "been",
    "which",
    "their",
    "about",
    "would",
    "there",
    "could",
    "other",
    "into",
    "çok",
    "ama",
    "veya",
    "hem",
    "sonra",
    "önce",
    "arasında",
    "yani",
    "zaten",
    "ancak",
    "böyle",
    "şekilde",
    "olarak",
    "olduğu",
    "olduğunu",
    "olmak",
    "değil",
    "yapan",
    "eden",
    "üzerinde",
    "ayrıca",
    "bkz",
    "sayfa",
    "ders",
    "hafta",
    "page",
    "chapter",
    "section",
    "figure",
    "table",
    "also",
    "oldu",
    "çünkü",
    "dolayısıyla",
    "böylece",
    "fakat",
}


def generate_auto_queries(vs) -> list[dict]:
    """Generate 1 test query per indexed file from actual chunk content."""
    auto = []
    seen_files = set()

    for i, text in enumerate(vs._texts):
        meta = vs._metadatas[i]
        fname = meta.get("filename", "")
        course = meta.get("course", "")

        if fname in seen_files or not fname or fname.endswith("_structure.md"):
            continue
        seen_files.add(fname)

        # Combine first 2-3 chunks for richer keyword pool
        combined = text
        for j in range(i + 1, min(i + 3, len(vs._texts))):
            if vs._metadatas[j].get("filename") == fname:
                combined += " " + vs._texts[j]

        words = [w.lower() for w in re.split(r"\W+", combined) if len(w) >= 4]
        words = [w for w in words if w not in STOPWORDS]
        freq = Counter(words)
        top_kw = [w for w, c in freq.most_common(8) if c >= 2]

        if len(top_kw) >= 3:
            auto.append(
                {
                    "query": " ".join(top_kw[:3]),
                    "expected_keywords": top_kw[:5],
                    "course": course or None,
                    "source_file": fname,
                    "auto_generated": True,
                }
            )

    return auto


# ─── Eval Functions ───────────────────────────────────────────────────────────
def eval_single(q, vs, search_fn=None, threshold=0.25, max_chunks=10):
    if search_fn is None:
        search_fn = vs.hybrid_search

    if search_fn == vs.query:
        results = search_fn(query_text=q["query"], n_results=15, course_filter=q.get("course"))
    else:
        results = search_fn(query=q["query"], n_results=15, course_filter=q.get("course"))

    if not results:
        return {
            "query": q["query"],
            "total_results": 0,
            "after_filter": 0,
            "top_score": 0,
            "avg_score": 0,
            "keyword_precision": 0,
            "found_keywords": [],
            "missing_keywords": q["expected_keywords"],
            "unique_files": 0,
            "filenames": [],
        }

    scores = [(1 - r["distance"]) for r in results]
    top_score = scores[0]
    filtered = [r for r, s in zip(results, scores, strict=False) if s > threshold][:max_chunks]

    all_text = " ".join(r.get("text", "").lower() for r in filtered)
    expected = q["expected_keywords"]
    found = [kw for kw in expected if kw.lower() in all_text]
    precision = len(found) / len(expected) if expected else 0

    filenames = {r.get("metadata", {}).get("filename", "") for r in filtered} - {""}

    return {
        "query": q["query"],
        "total_results": len(results),
        "after_filter": len(filtered),
        "top_score": round(top_score, 4),
        "avg_score": round(sum(scores[: len(filtered)]) / max(len(filtered), 1), 4),
        "keyword_precision": round(precision, 2),
        "found_keywords": found,
        "missing_keywords": [k for k in expected if k.lower() not in all_text],
        "unique_files": len(filenames),
        "filenames": list(filenames),
    }


def eval_all(vs, search_fn=None, queries=None, threshold=0.25, max_chunks=10, verbose=True):
    if queries is None:
        static = TEST_QUERIES
        auto = generate_auto_queries(vs)
        queries = static + auto
        print(f"Queries: {len(static)} static + {len(auto)} auto = {len(queries)} total\n")

    results = []
    for q in queries:
        r = eval_single(q, vs, search_fn, threshold, max_chunks)
        results.append(r)
        if verbose:
            status = "✅" if r["keyword_precision"] >= 0.6 else "⚠️" if r["keyword_precision"] >= 0.3 else "❌"
            tag = " [auto]" if q.get("auto_generated") else ""
            print(
                f"{status} {r['query']:<40} kw={r['keyword_precision']:.0%} top={r['top_score']:.3f} chunks={r['after_filter']}{tag}"
            )
            if r["missing_keywords"] and verbose:
                print(f"   missing: {r['missing_keywords']}")

    avg_prec = sum(r["keyword_precision"] for r in results) / len(results)
    avg_top = sum(r["top_score"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["keyword_precision"] >= 0.6) / len(results)

    print(f"\n{'='*70}")
    print(
        f"AGGREGATE: kw_precision={avg_prec:.0%}  top_score={avg_top:.3f}  pass_rate={pass_rate:.0%}  total={len(results)}"
    )

    # Breakdown
    static_r = [r for r, q in zip(results, queries, strict=False) if not q.get("auto_generated")]
    auto_r = [r for r, q in zip(results, queries, strict=False) if q.get("auto_generated")]
    if static_r:
        sp = sum(r["keyword_precision"] for r in static_r) / len(static_r)
        print(f"  Static: {sp:.0%} ({len(static_r)} queries)")
    if auto_r:
        ap = sum(r["keyword_precision"] for r in auto_r) / len(auto_r)
        print(f"  Auto:   {ap:.0%} ({len(auto_r)} queries)")

    return {
        "results": results,
        "queries": queries,
        "avg_precision": avg_prec,
        "avg_top": avg_top,
        "pass_rate": pass_rate,
    }


def compare_search(vs, verbose=True):
    """Side-by-side semantic vs hybrid comparison."""
    queries = TEST_QUERIES + generate_auto_queries(vs)
    print(f"Queries: {len(TEST_QUERIES)} static + {len(queries) - len(TEST_QUERIES)} auto = {len(queries)} total\n")

    print(f"{'='*70}")
    print("=== SEMANTIC ONLY ===")
    print(f"{'='*70}")
    sem = eval_all(vs, search_fn=vs.query, queries=queries, verbose=verbose)

    print(f"\n{'='*70}")
    print("=== HYBRID (BM25 + FAISS) ===")
    print(f"{'='*70}")
    hyb = eval_all(vs, search_fn=vs.hybrid_search, queries=queries, verbose=verbose)

    # Per-query delta
    print(f"\n{'='*70}")
    print("=== DELTA ===")
    print(f"{'query':<40} {'sem':>5} {'hyb':>5} {'delta':>6}")
    print("-" * 60)
    for s, h, q in zip(sem["results"], hyb["results"], queries, strict=False):
        d = h["keyword_precision"] - s["keyword_precision"]
        if d != 0 or verbose:
            m = "+" if d > 0 else "-" if d < 0 else " "
            tag = " [a]" if q.get("auto_generated") else ""
            print(f"{s['query']:<40} {s['keyword_precision']:>4.0%} {h['keyword_precision']:>4.0%} {d:>+5.0%} {m}{tag}")

    dp = hyb["avg_precision"] - sem["avg_precision"]
    dr = hyb["pass_rate"] - sem["pass_rate"]
    print(f"\n{'='*70}")
    print(f"SUMMARY: precision {sem['avg_precision']:.0%} → {hyb['avg_precision']:.0%} ({dp:+.0%})")
    print(f"         pass_rate {sem['pass_rate']:.0%} → {hyb['pass_rate']:.0%} ({dr:+.0%})")

    return sem, hyb


# ─── Baseline ─────────────────────────────────────────────────────────────────
BASELINE_PATH = Path("tests/rag_baseline.json")


def save_baseline(result: dict):
    data = {
        "timestamp": datetime.now().isoformat(),
        "avg_precision": result["avg_precision"],
        "pass_rate": result["pass_rate"],
        "total_queries": len(result["results"]),
        "details": [
            {"query": r["query"], "kw_precision": r["keyword_precision"], "top_score": r["top_score"]}
            for r in result["results"]
        ],
    }
    BASELINE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nBaseline saved: {BASELINE_PATH}")


def compare_baseline(current: dict):
    if not BASELINE_PATH.exists():
        print("No baseline found. Run with --save-baseline first.")
        return
    bl = json.loads(BASELINE_PATH.read_text())
    dp = current["avg_precision"] - bl["avg_precision"]
    dr = current["pass_rate"] - bl["pass_rate"]
    print(f"\nvs Baseline ({bl['timestamp'][:10]}):")
    print(f"  Precision: {bl['avg_precision']:.0%} → {current['avg_precision']:.0%} ({dp:+.0%})")
    print(f"  Pass rate: {bl['pass_rate']:.0%} → {current['pass_rate']:.0%} ({dr:+.0%})")
    print(f"  Queries:   {bl['total_queries']} → {len(current['results'])}")
    if dp < -0.05:
        print("  REGRESSION DETECTED — precision dropped >5%")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from core.vector_store import VectorStore

    parser = argparse.ArgumentParser(description="RAG Quality Eval")
    parser.add_argument("--compare", action="store_true", help="Semantic vs hybrid comparison")
    parser.add_argument("--auto-only", action="store_true", help="Only auto-generated queries")
    parser.add_argument("--static-only", action="store_true", help="Only static queries")
    parser.add_argument("--save-baseline", action="store_true", help="Save current as baseline")
    parser.add_argument("--vs-baseline", action="store_true", help="Compare with saved baseline")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--max-chunks", type=int, default=10)
    args = parser.parse_args()

    vs = VectorStore()
    vs.initialize()

    if args.compare:
        compare_search(vs, verbose=args.verbose)
    else:
        queries = None
        if args.auto_only:
            queries = generate_auto_queries(vs)
            print(f"Auto-generated queries: {len(queries)}\n")
        elif args.static_only:
            queries = TEST_QUERIES
            print(f"Static queries: {len(queries)}\n")

        result = eval_all(
            vs, queries=queries, threshold=args.threshold, max_chunks=args.max_chunks, verbose=args.verbose
        )

        if args.save_baseline:
            save_baseline(result)
        if args.vs_baseline:
            compare_baseline(result)
