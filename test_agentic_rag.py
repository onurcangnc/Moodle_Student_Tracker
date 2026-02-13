#!/usr/bin/env python3
"""Comprehensive Agentic RAG Test Suite
Tests: query rewrite quality, RAG retrieval accuracy, source attribution, edge cases.
"""
import sys, os, time, json
sys.path.insert(0, "/opt/moodle-bot")
os.chdir("/opt/moodle-bot")

from dotenv import load_dotenv
load_dotenv()

from core.vector_store import VectorStore
from core.llm_providers import MultiProviderEngine

vs = VectorStore()
vs.initialize()
engine = MultiProviderEngine()

stats = vs.get_stats()
print(f"=== Vector Store: {stats.get('total_chunks',0)} chunks, {stats.get('unique_files',0)} files, {stats.get('unique_courses',0)} courses ===\n")

# List available courses and file counts
course_files = {}
for m in vs._metadatas:
    c = m.get("course", "unknown")
    f = m.get("filename", "")
    if c not in course_files:
        course_files[c] = set()
    course_files[c].add(f)
print("Courses & files:")
for c, files in sorted(course_files.items()):
    print(f"  {c}: {len(files)} files")
print()

# ─── Test 1: Query Rewrite Quality ───────────────────────────────────────────
print("=" * 60)
print("TEST 1: QUERY REWRITE QUALITY")
print("=" * 60)

rewrite_cases = [
    # (user_msg, context, description)
    ("audit öğret", "CTIS 474 çalışacağım", "Short + course context"),
    ("bunu açıkla", "hegemoni nedir", "Pronoun reference"),
    ("devam", "Tanzimat edebiyatı anlatıyorduk", "Continue command"),
    ("ne demek bu", "binary search tree", "Vague + tech context"),
    ("sınava hazırla", "CTIS 363 ethics", "Exam prep + course"),
    ("örnekler ver", "sorting algorithms quicksort", "Examples request"),
    ("karşılaştır", "stack ve queue", "Compare request"),
    ("Kiralık Konak", "", "Direct topic, no context"),
    ("midterm konuları", "HCIV 300", "Exam topics + course"),
    ("formüller", "fizik mekanikte enerji", "Formulas + physics"),
    ("bu konu sınavda çıkar mı", "Berna Moran edebiyat analizi", "Exam relevance"),
    ("özet ver", "Ahmet Mithat Efendi Felâtun Bey ile Râkım Efendi", "Summary + specific work"),
]

rewrite_results = []
for user_msg, context, desc in rewrite_cases:
    history = [{"role": "user", "content": context}] if context else []
    context_str = " | ".join([context]) if context else ""

    start = time.time()
    try:
        rewritten = engine.complete(
            task="topic_detect",
            system=(
                "Görevi: Öğrencinin mesajını vektör veritabanı araması için optimize et.\n"
                "Kurallar:\n"
                "- Kısa/belirsiz mesajı anahtar kelimelerle genişlet\n"
                "- Türkçe ve İngilizce terimleri karıştır (embedding daha iyi çalışır)\n"
                "- SADECE arama sorgusunu yaz, açıklama yapma\n"
                "- Bağlamdan konuyu çıkar, genel kelime ekleme\n"
                "Örnek: 'audit öğret' → 'IS auditing bilgi sistemleri denetimi audit standartları'\n"
                "Örnek: 'bunu açıkla' + bağlam:'hegemoni' → 'hegemoni kavramı hegemonya tanım örnekleri'"
            ),
            messages=[{"role": "user", "content": f"Bağlam: {context_str}\nMesaj: {user_msg}"}],
            max_tokens=60,
        )
        elapsed = (time.time() - start) * 1000
        rewritten = rewritten.strip() if rewritten else ""
    except Exception as e:
        rewritten = f"ERROR: {e}"
        elapsed = 0

    rewrite_results.append({
        "input": user_msg, "context": context, "output": rewritten, "ms": elapsed
    })
    print(f"  [{desc}]")
    print(f"    Input: '{user_msg}' + ctx:'{context}'")
    print(f"    → '{rewritten}' ({elapsed:.0f}ms)")
    print()

avg_rewrite_ms = sum(r["ms"] for r in rewrite_results) / len(rewrite_results)
print(f"Avg rewrite latency: {avg_rewrite_ms:.0f}ms\n")

# ─── Test 2: RAG Retrieval — Original vs Rewritten Query ─────────────────────
print("=" * 60)
print("TEST 2: RAG RETRIEVAL — ORIGINAL vs REWRITTEN")
print("=" * 60)

retrieval_cases = [
    # (original_query, expected_topic, course_filter)
    ("audit öğret", "auditing/denetim", None),
    ("Kiralık Konak kimin eseri", "Yakup Kadri", None),
    ("sorting algorithms", "sorting/sıralama", None),
    ("hegemoni nedir", "hegemoni/hegemonya", None),
    ("binary search", "binary search tree/ikili arama", None),
    ("Tanzimat edebiyatı", "Tanzimat dönemi", None),
    ("ethics in IT", "etik/bilişim etiği", None),
    ("integral hesapla", "integral/calculus", None),
]

print(f"\n{'Query':<30} {'Orig Score':>10} {'Rewrite Score':>13} {'Delta':>7} {'Better?':>8}")
print("─" * 75)

improvements = 0
regressions = 0
for orig_query, expected, course_filter in retrieval_cases:
    # Original query
    orig_results = vs.query(query_text=orig_query, n_results=5, course_filter=course_filter)
    orig_top = (1 - orig_results[0]["distance"]) if orig_results else 0

    # Rewrite the query
    try:
        rewritten = engine.complete(
            task="topic_detect",
            system=(
                "Görevi: Öğrencinin mesajını vektör veritabanı araması için optimize et.\n"
                "Kurallar:\n"
                "- Kısa/belirsiz mesajı anahtar kelimelerle genişlet\n"
                "- Türkçe ve İngilizce terimleri karıştır\n"
                "- SADECE arama sorgusunu yaz\n"
                "- Bağlamdan konuyu çıkar, genel kelime ekleme"
            ),
            messages=[{"role": "user", "content": f"Bağlam: \nMesaj: {orig_query}"}],
            max_tokens=60,
        )
        rewritten = rewritten.strip() if rewritten else orig_query
    except:
        rewritten = orig_query

    # Rewritten query
    rw_results = vs.query(query_text=rewritten, n_results=5, course_filter=course_filter)
    rw_top = (1 - rw_results[0]["distance"]) if rw_results else 0

    delta = rw_top - orig_top
    better = "✅" if delta > 0.02 else ("❌" if delta < -0.02 else "➖")
    if delta > 0.02: improvements += 1
    if delta < -0.02: regressions += 1

    print(f"  {orig_query:<28} {orig_top:>10.3f} {rw_top:>13.3f} {delta:>+7.3f} {better:>8}")
    print(f"    Rewritten: '{rewritten}'")
    if orig_results:
        print(f"    Orig top: {orig_results[0]['metadata'].get('filename','')[:40]}")
    if rw_results:
        print(f"    RW top:   {rw_results[0]['metadata'].get('filename','')[:40]}")
    print()

print(f"\nSummary: {improvements} improved, {regressions} regressed, {len(retrieval_cases)-improvements-regressions} neutral")

# ─── Test 3: Source Attribution ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: SOURCE ATTRIBUTION")
print("=" * 60)

attribution_cases = [
    ("Kiralık Konak romanını anlat", None, True, "Should show source files"),
    ("Python nedir", None, True, "General topic — may have weak match"),
    ("audit nedir", "CTIS 474-1 IS Auditing & Assurance", False, "Course with NO materials"),
    ("ethics öğret", None, True, "Should find ethics materials"),
]

for query, course_filter, expect_sources, desc in attribution_cases:
    results = vs.query(query_text=query, n_results=7, course_filter=course_filter)
    top_score = (1 - results[0]["distance"]) if results else 0

    # Check if course has materials
    has_materials = True
    if course_filter:
        files = vs.get_files_for_course(course_name=course_filter)
        has_materials = len(files) > 0

    low_relevance = not results or top_score < 0.3

    # Extract source files
    source_files = []
    seen = set()
    for r in results[:7]:
        fname = r.get("metadata", {}).get("filename", "")
        if fname and fname not in seen:
            source_files.append(fname)
            seen.add(fname)

    print(f"\n  [{desc}]")
    print(f"    Query: '{query}' | Course: {course_filter or 'None'}")
    print(f"    Top score: {top_score:.3f} | Has materials: {has_materials} | Low relevance: {low_relevance}")

    if course_filter and not has_materials:
        print(f"    → ℹ️ '{course_filter}' materyali yok → LLM genel bilgi + uyarı")
    elif low_relevance:
        print(f"    → ⚠️ Zayıf eşleşme → Genel bilgi uyarısı")
    elif source_files:
        sources = ", ".join(source_files[:4])
        print(f"    → 📚 Kaynak: {sources}")

    status = "✅" if (expect_sources and source_files and not low_relevance) or (not expect_sources and not has_materials) else "⚠️"
    print(f"    Status: {status}")

# ─── Test 4: Iterative RAG Evaluation ────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: ITERATIVE RAG EVALUATION — Would a 2nd search help?")
print("=" * 60)
print("For each query: 1st search → if top_score < 0.35, try LLM-refined 2nd search\n")

iterative_cases = [
    "Berna Moran'ın edebiyat eleştirisi yaklaşımı",
    "quicksort ve mergesort karşılaştırması",
    "Tanzimat'ta Batılılaşma eleştirisi",
    "bilgi sistemleri güvenlik standartları",
    "stack overflow hatası nedir",
    "Ahmet Mithat Efendi romanları",
    "veritabanı normalizasyonu",
    "Namık Kemal'in tiyatroları",
    "machine learning temel kavramlar",
    "Halit Ziya Uşaklıgil realizm",
]

needs_iterative = 0
for query in iterative_cases:
    # First search
    r1 = vs.query(query_text=query, n_results=5)
    s1 = (1 - r1[0]["distance"]) if r1 else 0
    f1 = r1[0]["metadata"].get("filename", "")[:35] if r1 else "none"

    # If weak, try LLM-refined query
    if s1 < 0.35:
        try:
            refined = engine.complete(
                task="topic_detect",
                system=(
                    "İlk arama başarısız. Sorguyu yeniden yaz, farklı anahtar kelimeler kullan.\n"
                    "Eş anlamlılar, İngilizce karşılıklar, alternatif terimler ekle.\n"
                    "SADECE sorguyu yaz."
                ),
                messages=[{"role": "user", "content": f"Başarısız sorgu: {query}\nİlk sonuç: {f1}"}],
                max_tokens=60,
            )
            refined = refined.strip() if refined else query
        except:
            refined = query

        r2 = vs.query(query_text=refined, n_results=5)
        s2 = (1 - r2[0]["distance"]) if r2 else 0
        f2 = r2[0]["metadata"].get("filename", "")[:35] if r2 else "none"

        improved = s2 > s1 + 0.03
        if improved:
            needs_iterative += 1

        print(f"  ⚠️ '{query}'")
        print(f"    1st: {s1:.3f} ({f1})")
        print(f"    Refined: '{refined}'")
        print(f"    2nd: {s2:.3f} ({f2}) {'✅ IMPROVED' if improved else '➖ no gain'}")
    else:
        print(f"  ✅ '{query}' → {s1:.3f} ({f1}) — 1st search sufficient")

print(f"\nVerdict: {needs_iterative}/{len(iterative_cases)} cases benefited from 2nd search")
if needs_iterative <= 2:
    print("→ Full iterative RAG NOT needed — single rewrite is sufficient")
elif needs_iterative <= 4:
    print("→ Marginal benefit — consider iterative RAG for edge cases only")
else:
    print("→ Strong benefit — full iterative RAG recommended")

# ─── Test 5: End-to-End Latency Budget ──────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: END-TO-END LATENCY BUDGET")
print("=" * 60)

query = "Tanzimat edebiyatında Batılılaşma eleştirisi"
history = [{"role": "user", "content": "edebiyat çalışacağım"}]

# Step 1: Query rewrite
t0 = time.time()
try:
    rw = engine.complete(
        task="topic_detect",
        system="Öğrencinin mesajını vektör araması için optimize et. SADECE sorguyu yaz.",
        messages=[{"role": "user", "content": f"Bağlam: edebiyat çalışacağım\nMesaj: {query}"}],
        max_tokens=60,
    )
    rw = rw.strip() if rw else query
except:
    rw = query
t1 = time.time()

# Step 2: FAISS search
results = vs.query(query_text=rw, n_results=15)
t2 = time.time()

# Step 3: LLM response (simulate with short prompt)
context = "\n".join([r["text"][:200] for r in results[:5]])
try:
    resp = engine.complete(
        task="chat",
        system="Kısa bir cevap ver.",
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nSORU: {query}"}],
        max_tokens=200,
    )
except:
    resp = "error"
t3 = time.time()

print(f"  Query rewrite:  {(t1-t0)*1000:>7.0f}ms  (nano)")
print(f"  FAISS search:   {(t2-t1)*1000:>7.0f}ms  (local)")
print(f"  LLM response:   {(t3-t2)*1000:>7.0f}ms  (Gemini Flash)")
print(f"  ────────────────────────")
print(f"  TOTAL:           {(t3-t0)*1000:>7.0f}ms")
print(f"\n  Rewritten query: '{rw}'")
print(f"  Top result: {results[0]['metadata'].get('filename','')}" if results else "  No results")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
