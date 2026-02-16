"""
Debug script: compare semantic / BM25 / hybrid for specific queries.
Run: cd /opt/moodle-bot && source venv/bin/activate && python tests/debug_hybrid.py
"""

import sys

sys.path.insert(0, ".")
from core.vector_store import VectorStore

vs = VectorStore()
vs.initialize()


def show_results(label, results, max_show=10):
    print(f"\n--- {label} ({len(results)} results) ---")
    for i, r in enumerate(results[:max_show]):
        score = 1 - r["distance"]
        fname = r.get("metadata", {}).get("filename", "?")
        text_preview = r["text"][:120].replace("\n", " ")
        bm25 = f"  bm25={r['bm25_score']:.1f}" if "bm25_score" in r else ""
        print(f"  [{i+1}] score={score:.3f}{bm25}  file={fname}")
        print(f"       {text_preview}")
    print()


def debug_query(query, course=None):
    print(f"\n{'='*80}")
    print(f'QUERY: "{query}"  course_filter={course}')
    print(f"{'='*80}")

    sem = vs.query(query_text=query, n_results=15, course_filter=course)
    bm25 = vs.bm25_search(query, n_results=15, course_filter=course)
    hyb = vs.hybrid_search(query=query, n_results=15, course_filter=course)

    show_results("SEMANTIC (FAISS)", sem)
    show_results("BM25", bm25)
    show_results("HYBRID (RRF)", hyb)

    # Find chunks in semantic but dropped from hybrid top 10
    sem_keys = {r["text"][:150] for r in sem[:10]}
    hyb_keys = {r["text"][:150] for r in hyb[:10]}
    dropped = sem_keys - hyb_keys
    added = hyb_keys - sem_keys

    if dropped:
        print(f"DROPPED from semantic→hybrid ({len(dropped)}):")
        for r in sem[:10]:
            if r["text"][:150] in dropped:
                fname = r.get("metadata", {}).get("filename", "?")
                print(f"  - [{1-r['distance']:.3f}] {fname}: {r['text'][:100].replace(chr(10),' ')}")
    if added:
        print(f"\nADDED by BM25→hybrid ({len(added)}):")
        for r in hyb[:10]:
            if r["text"][:150] in added:
                fname = r.get("metadata", {}).get("filename", "?")
                bm25s = f" bm25={r.get('bm25_score', '?')}" if "bm25_score" in r else ""
                print(f"  + [{1-r['distance']:.3f}]{bm25s} {fname}: {r['text'][:100].replace(chr(10),' ')}")
    print()


def check_keyword_in_corpus(keyword, course=None):
    """Check if keyword exists anywhere in the indexed chunks."""
    count = 0
    files = set()
    for i, text in enumerate(vs._texts):
        meta = vs._metadatas[i]
        if course and course.lower() not in meta.get("course", "").lower():
            continue
        if keyword.lower() in text.lower():
            count += 1
            files.add(meta.get("filename", "?"))
    return count, files


def analyze_failing():
    """Analyze queries that fail in hybrid mode."""
    from tests.test_rag_quality import TEST_QUERIES, eval_single

    print(f"\n{'='*80}")
    print("FAILING QUERIES ANALYSIS (hybrid mode)")
    print(f"{'='*80}\n")

    for q in TEST_QUERIES:
        r = eval_single(q, search_fn=vs.hybrid_search)
        if r["keyword_precision"] < 1.0:
            print(f"QUERY: \"{q['query']}\"  (course={q.get('course')})")
            print(f"  kw_precision={r['keyword_precision']:.0%}  chunks={r['after_filter']}  top={r['top_score']:.3f}")
            print(f"  found:   {r['found_keywords']}")
            print(f"  missing: {r['missing_keywords']}")

            # Check if missing keywords exist in corpus
            for kw in r["missing_keywords"]:
                cnt, fnames = check_keyword_in_corpus(kw, q.get("course"))
                if cnt > 0:
                    print(f'    "{kw}": EXISTS in corpus — {cnt} chunks across {fnames}')
                else:
                    # Also check without course filter
                    cnt2, fnames2 = check_keyword_in_corpus(kw)
                    if cnt2 > 0:
                        print(f'    "{kw}": NOT in course, but {cnt2} chunks globally in {fnames2}')
                    else:
                        print(f'    "{kw}": NOT IN CORPUS AT ALL')
            print()


if __name__ == "__main__":
    # 1. Debug the regression query
    debug_query("Doğu Batı çatışması Türk romanı", course="EDEB 201")

    # 2. Debug other weak queries
    debug_query("docker container", course="CTIS 465")
    debug_query("microservice architecture", course="CTIS 465")

    # 3. Analyze all imperfect queries
    analyze_failing()
