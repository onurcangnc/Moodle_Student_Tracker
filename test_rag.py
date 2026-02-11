import sys, os
sys.path.insert(0, '/opt/moodle-bot')
os.chdir('/opt/moodle-bot')
from dotenv import load_dotenv
load_dotenv()
from core.vector_store import VectorStore

vs = VectorStore()
vs.initialize()

print("=" * 60)
print("RAG KALİTE TESTİ")
print("=" * 60)

# Show available courses first
stats = vs.get_stats()
print(f"\nIndeksli: {stats['total_chunks']} chunk, {stats['unique_courses']} kurs")
print(f"Kurslar: {stats['courses']}")
print()

tests = [
    # TEST 1: Dogru kurs esleme
    {"query": "Felatun Bey ile Rakim Efendi", "expected_course": "EDEB", "desc": "EDEB sorgusu EDEB chunk dondurmeli"},
    {"query": "Columbian Exchange hastalik yayilimi", "expected_course": "HCIV", "desc": "HCIV sorgusu HCIV chunk dondurmeli"},
    {"query": "COBIT framework audit", "expected_course": "CTIS 474", "desc": "CTIS 474 sorgusu CTIS 474 chunk dondurmeli"},
    {"query": "microservice docker kubernetes", "expected_course": "CTIS 465", "desc": "CTIS 465 sorgusu CTIS 465 chunk dondurmeli"},
    {"query": "ethical issues privacy", "expected_course": "CTIS 363", "desc": "CTIS 363 sorgusu CTIS 363 chunk dondurmeli"},
    # TEST 2: Kurs karisma testi
    {"query": "Osmanli Imparatorlugu", "expected_course": "HCIV", "desc": "Osmanli -> HCIV gelmeli, EDEB degil"},
    {"query": "Tanzimat donemi", "expected_course": "EDEB", "desc": "Tanzimat -> EDEB gelmeli"},
    # TEST 3: Olmayan bilgi testi
    {"query": "Bitcoin cryptocurrency blockchain", "expected_course": None, "desc": "Kapsam disi -> dusuk relevance"},
    {"query": "machine learning neural network", "expected_course": None, "desc": "Kapsam disi -> dusuk relevance"},
]

passed = 0
failed = 0

for i, test in enumerate(tests):
    print(f"\n--- Test {i+1}: {test['desc']} ---")
    print(f"Query: {test['query']}")

    results = vs.query(query_text=test['query'], n_results=5)

    if not results:
        print("SONUC: Hic chunk gelmedi")
        if test['expected_course'] is None:
            passed += 1
            print("  -> ✅ Beklenen (kapsam disi)")
        else:
            failed += 1
            print("  -> ❌ Chunk gelmesi gerekiyordu")
        continue

    for j, r in enumerate(results[:3]):
        meta = r.get('metadata', {})
        course = meta.get('course', '?')
        filename = meta.get('filename', '?')
        distance = r.get('distance', 0)
        similarity = max(0, 1 - distance)
        text_preview = r.get('text', '')[:100].replace('\n', ' ')
        print(f"  #{j+1} Kurs: {course}")
        print(f"       Skor: {similarity:.3f} | Dosya: {filename}")
        print(f"       Text: {text_preview}")

    # Beklenen kurs kontrolu
    top_courses = [r.get('metadata', {}).get('course', '') for r in results[:3]]

    if test['expected_course']:
        match = any(test['expected_course'].lower() in c.lower() for c in top_courses)
        if match:
            passed += 1
            print(f"  BEKLENEN: {test['expected_course']} | ✅ DOGRU")
        else:
            failed += 1
            print(f"  BEKLENEN: {test['expected_course']} | ❌ YANLIS")
            print(f"  Gelen kurslar: {top_courses}")
    else:
        # Kapsam disi test: top skor < 0.3 ise basarili
        top_sim = max(0, 1 - results[0].get('distance', 0))
        if top_sim < 0.3:
            passed += 1
            print(f"  Kapsam disi: Top skor {top_sim:.3f} < 0.3 -> ✅ DOGRU")
        else:
            failed += 1
            print(f"  Kapsam disi: Top skor {top_sim:.3f} >= 0.3 -> ❌ Beklenenden yuksek")

print(f"\n{'=' * 60}")
print(f"SONUC: {passed} basarili / {passed + failed} toplam")
if failed:
    print(f"❌ {failed} test basarisiz")
else:
    print("✅ Tum testler gecti!")
print("=" * 60)
