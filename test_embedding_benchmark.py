#!/usr/bin/env python3
"""Benchmark: all-MiniLM-L6-v2 vs paraphrase-multilingual-MiniLM-L12-v2
Tests Turkish academic content retrieval quality.
"""
import sys, os, time, json
sys.path.insert(0, "/opt/moodle-bot")
os.chdir("/opt/moodle-bot")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer

# Load both models
print("Loading models...")
t0 = time.time()
model_en = SentenceTransformer("all-MiniLM-L6-v2")
t1 = time.time()
model_ml = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
t2 = time.time()
print(f"  all-MiniLM-L6-v2: {(t1-t0):.1f}s, dim={model_en.get_sentence_embedding_dimension()}")
print(f"  multilingual-L12: {(t2-t1):.1f}s, dim={model_ml.get_sentence_embedding_dimension()}")

# Load actual chunks from the vector store
meta_path = "/opt/moodle-bot/data/chromadb/metadata.json"
with open(meta_path, "r", encoding="utf-8") as f:
    saved = json.load(f)
texts = saved["texts"]
metadatas = saved["metadatas"]
print(f"\nLoaded {len(texts)} chunks from vector store\n")

# Test queries — Turkish academic content
test_cases = [
    # (query, expected_keyword_in_result, description)
    ("Kiralık Konak romanı kimin eseri", "Kiralık Konak", "Turkish novel attribution"),
    ("Tanzimat edebiyatında Batılılaşma eleştirisi", "Tanzimat", "Turkish literary period"),
    ("hegemoni kavramı nedir", "hegemoni", "Turkish social science term"),
    ("Berna Moran edebiyat eleştirisi", "Berna Moran", "Turkish literary critic"),
    ("bilişim etiği temel ilkeleri", "ethic", "CS ethics in Turkish"),
    ("Yakup Kadri Karaosmanoğlu romanları", "Yakup Kadri", "Turkish author"),
    ("Doğu-Batı çatışması Türk edebiyatında", "Doğu", "East-West conflict theme"),
    ("audit denetim bilgi sistemleri", "audit", "IS auditing"),
    ("Ahmet Mithat Efendi Felâtun Bey", "Felâtun", "Specific Turkish novel"),
    ("ethical issues in computing", "ethical", "English CS ethics query"),
    ("Şerif Mardin modernleşme", "Mardin", "Turkish sociologist"),
    ("information age social issues", "information", "English query"),
    ("Osmanlı toplumunda değişim", "Osmanlı", "Ottoman social change"),
    ("privacy and surveillance technology", "privacy", "English tech ethics"),
    ("Namık Kemal vatan edebiyatı", "Namık Kemal", "Turkish patriotic literature"),
]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Encode all chunks with both models
print("Encoding all chunks with both models...")
t0 = time.time()
emb_en = model_en.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
t1 = time.time()
emb_ml = model_ml.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
t2 = time.time()
print(f"  English model encode: {(t1-t0):.1f}s")
print(f"  Multilingual model encode: {(t2-t1):.1f}s")

# Benchmark
print(f"\n{'='*70}")
print(f"{'Query':<45} {'EN Score':>9} {'ML Score':>9} {'Delta':>7} {'Winner':>7}")
print(f"{'─'*70}")

en_wins = 0
ml_wins = 0
ties = 0
en_total = 0
ml_total = 0

for query, expected_kw, desc in test_cases:
    # Encode query with both models
    q_en = model_en.encode([query], normalize_embeddings=True)[0]
    q_ml = model_ml.encode([query], normalize_embeddings=True)[0]

    # Find best match
    scores_en = emb_en @ q_en
    scores_ml = emb_ml @ q_ml

    top_en_idx = np.argmax(scores_en)
    top_ml_idx = np.argmax(scores_ml)

    top_en = scores_en[top_en_idx]
    top_ml = scores_ml[top_ml_idx]

    en_total += top_en
    ml_total += top_ml

    delta = top_ml - top_en
    if delta > 0.02:
        winner = "ML ✅"
        ml_wins += 1
    elif delta < -0.02:
        winner = "EN ✅"
        en_wins += 1
    else:
        winner = "TIE"
        ties += 1

    print(f"  {desc:<43} {top_en:>9.3f} {top_ml:>9.3f} {delta:>+7.3f} {winner:>7}")

    # Show which chunk was found
    en_meta = metadatas[top_en_idx]
    ml_meta = metadatas[top_ml_idx]
    en_file = en_meta.get("filename", "")[:40]
    ml_file = ml_meta.get("filename", "")[:40]

    # Check if expected keyword found
    en_text = texts[top_en_idx][:100]
    ml_text = texts[top_ml_idx][:100]
    en_found = expected_kw.lower() in en_text.lower() or expected_kw.lower() in en_file.lower()
    ml_found = expected_kw.lower() in ml_text.lower() or expected_kw.lower() in ml_file.lower()

    hit_en = "HIT" if en_found else "MISS"
    hit_ml = "HIT" if ml_found else "MISS"
    print(f"    EN: [{hit_en}] {en_file}")
    print(f"    ML: [{hit_ml}] {ml_file}")

print(f"\n{'='*70}")
print(f"RESULTS:")
print(f"  Multilingual wins: {ml_wins}/{len(test_cases)}")
print(f"  English wins:      {en_wins}/{len(test_cases)}")
print(f"  Ties:              {ties}/{len(test_cases)}")
print(f"  Avg EN score:      {en_total/len(test_cases):.3f}")
print(f"  Avg ML score:      {ml_total/len(test_cases):.3f}")
print(f"  Avg delta:         {(ml_total-en_total)/len(test_cases):+.3f}")

if ml_wins > en_wins:
    print(f"\n→ MULTILINGUAL MODEL IS BETTER for this corpus")
    print(f"  Recommendation: Switch to paraphrase-multilingual-MiniLM-L12-v2")
elif en_wins > ml_wins:
    print(f"\n→ ENGLISH MODEL IS BETTER — keep all-MiniLM-L6-v2")
else:
    print(f"\n→ TIE — marginal difference, keep current model")

# Keyword hit rate
en_hits = sum(1 for q, kw, _ in test_cases
              for i in [np.argmax(model_en.encode([q], normalize_embeddings=True)[0] @ emb_en.T)]
              if kw.lower() in texts[i][:200].lower() or kw.lower() in metadatas[i].get("filename","").lower())
ml_hits = sum(1 for q, kw, _ in test_cases
              for i in [np.argmax(model_ml.encode([q], normalize_embeddings=True)[0] @ emb_ml.T)]
              if kw.lower() in texts[i][:200].lower() or kw.lower() in metadatas[i].get("filename","").lower())
print(f"\n  Keyword hit rate — EN: {en_hits}/{len(test_cases)}, ML: {ml_hits}/{len(test_cases)}")
