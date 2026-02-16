"""
E2E Test Scenarios — 60 scenarios across 6 courses.
Automated RAG retrieval + course detection checks, manual response quality checks.

Run:
  cd /opt/moodle-bot && source venv/bin/activate
  python tests/test_e2e_scenarios.py                   # full auto eval
  python tests/test_e2e_scenarios.py --course CTIS363   # single course
  python tests/test_e2e_scenarios.py --level hard        # filter by level
  python tests/test_e2e_scenarios.py --live              # live LLM eval (costs API credits)
  python tests/test_e2e_scenarios.py --report            # markdown report
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

# ─── Scenario Data Structure ─────────────────────────────────────────────────


@dataclass
class Scenario:
    id: str
    course_code: str
    level: str  # easy / medium / hard
    user_message: str
    skill: str  # course_detection / rag_retrieval / response_quality / context_tracking / quiz_mode / synthesis
    expected_behavior: str
    pass_criteria: str
    fail_criteria: str
    # Automated check fields
    expected_course: str = ""  # fullname for course detection check
    expected_keywords: list[str] = field(default_factory=list)  # RAG keyword check
    max_words: int = 0  # 0 = no limit
    requires_question: bool = False  # should response contain '?'
    requires_context: bool = False  # needs prior message context (manual only)
    prior_topic: str = ""  # what topic should be in context


# ─── All 60 Scenarios ────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    # ══════════════════════════════════════════════════════════════════════════
    # CTIS 363 — Ethical and Social Issues in Information Systems
    # ══════════════════════════════════════════════════════════════════════════
    Scenario(
        id="CTIS363-01",
        course_code="CTIS 363",
        level="easy",
        user_message="CTIS 363 çalışacağım",
        skill="course_detection",
        expected_behavior="Kursu tanır, ana konuları özetler, 'hangi konudan başlayalım?' sorar",
        pass_criteria="Doğru kurs tespiti + Socratic açılış + konu listesi",
        fail_criteria="Yanlış kurs veya direkt bilgi kusması",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        requires_question=True,
    ),
    Scenario(
        id="CTIS363-02",
        course_code="CTIS 363",
        level="easy",
        user_message="privacy nedir kısaca açıkla",
        skill="rag_retrieval",
        expected_behavior="Mahremiyet kavramını 3-5 cümleyle açıklar, kontrol sorusu sorar",
        pass_criteria="Kısa açıklama + soru + chunk copy-paste yok",
        fail_criteria=">150 kelime veya materyalden direkt yapıştırma",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        expected_keywords=["privacy", "mahremiyet", "personal", "data", "kişisel"],
        max_words=150,
        requires_question=True,
    ),
    Scenario(
        id="CTIS363-03",
        course_code="CTIS 363",
        level="easy",
        user_message="ACM etik kodları nelerdir",
        skill="rag_retrieval",
        expected_behavior="ACM Code of Ethics ana maddelerini özetler",
        pass_criteria="Doğru chunk retrieval + özet formatında",
        fail_criteria="IEEE ile karıştırma veya yanlış bilgi",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        expected_keywords=["acm", "ethics", "code", "etik"],
    ),
    Scenario(
        id="CTIS363-04",
        course_code="CTIS 363",
        level="medium",
        user_message="KVKK ile GDPR arasındaki farklar neler",
        skill="rag_retrieval",
        expected_behavior="İki mevzuatı karşılaştırır, temel farkları listeler",
        pass_criteria="Her iki kaynaktan chunk + karşılaştırmalı format",
        fail_criteria="Sadece birini açıklama veya hallüsinasyon",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        expected_keywords=["kvkk", "gdpr", "veri", "koruma"],
    ),
    Scenario(
        id="CTIS363-05",
        course_code="CTIS 363",
        level="medium",
        user_message="whistleblowing etik mi yoksa ihanet mi?",
        skill="response_quality",
        expected_behavior="İki perspektifi sunar, 'sen ne düşünüyorsun?' sorar",
        pass_criteria="Dengeli yaklaşım + Socratic soru + düşündürme",
        fail_criteria="Tek taraflı cevap veya soru sormama",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        expected_keywords=["whistleblowing", "etik", "ihbar"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS363-06",
        course_code="CTIS 363",
        level="medium",
        user_message="devam et",
        skill="context_tracking",
        expected_behavior="Önceki konuyu (whistleblowing) derinleştirir — gerçek vakalar, yasal korumalar",
        pass_criteria="Bağlamı koruyor + yeni bilgi katmanı",
        fail_criteria="Bağlam kaybı veya tekrar",
        requires_context=True,
        prior_topic="whistleblowing",
    ),
    Scenario(
        id="CTIS363-07",
        course_code="CTIS 363",
        level="medium",
        user_message="anlamadım",
        skill="context_tracking",
        expected_behavior="Aynı konuyu daha basit dille, analoji ile açıklar",
        pass_criteria="Dil seviyesi düşer + analoji/örnek eklenir",
        fail_criteria="Aynı açıklamayı tekrar veya daha teknik hale getirme",
        requires_context=True,
        prior_topic="whistleblowing",
    ),
    Scenario(
        id="CTIS363-08",
        course_code="CTIS 363",
        level="hard",
        user_message="Bir şirket çalışanlarını AI ile izlerse bu etik midir? Hangi framework ile değerlendirirsin?",
        skill="synthesis",
        expected_behavior="Surveillance + AI etiği + etik frameworkler birleştirilerek analiz",
        pass_criteria="Birden fazla konu sentezi + framework referansı + Socratic",
        fail_criteria="Tek boyutlu cevap veya framework belirtmeme",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        expected_keywords=["surveillance", "izleme", "ethics", "framework", "etik"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS363-09",
        course_code="CTIS 363",
        level="hard",
        user_message="beni test et",
        skill="quiz_mode",
        expected_behavior="CTIS 363 konularından tek bir soru sorar, cevabı bekler",
        pass_criteria="Konuya uygun tek soru + cevap beklemesi",
        fail_criteria="Birden fazla soru veya cevabı da vermesi",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        requires_question=True,
    ),
    Scenario(
        id="CTIS363-10",
        course_code="CTIS 363",
        level="hard",
        user_message="Dijital uçurum ile yapay zeka etiği arasında nasıl bir bağlantı var?",
        skill="synthesis",
        expected_behavior="Digital divide ve AI ethics konularını birleştiren analiz",
        pass_criteria="İki ayrı chunk setinden sentez + orijinal analiz",
        fail_criteria="Konuları ayrı ayrı açıklama, bağlantı kuramama",
        expected_course="CTIS 363-1 Ethical and Social Issues in Information Systems",
        expected_keywords=["dijital", "uçurum", "divide", "yapay", "zeka", "ethics"],
    ),
    # ══════════════════════════════════════════════════════════════════════════
    # HCIV 102 — History of Civilization II
    # ══════════════════════════════════════════════════════════════════════════
    Scenario(
        id="HCIV102-01",
        course_code="HCIV 102",
        level="easy",
        user_message="HCIV 102 çalışmam lazım",
        skill="course_detection",
        expected_behavior="Kursu tanır, dönemleri/konuları özetler, başlangıç noktası sorar",
        pass_criteria="Doğru kurs + konu haritası + Socratic açılış",
        fail_criteria="History of Civilization I ile karıştırma",
        expected_course="HCIV 102-4 History of Civilization II",
        requires_question=True,
    ),
    Scenario(
        id="HCIV102-02",
        course_code="HCIV 102",
        level="easy",
        user_message="Fransız Devrimi neden çıktı",
        skill="rag_retrieval",
        expected_behavior="Temel sebepleri kısa özetler, 'sence en belirleyici faktör hangisi?' sorar",
        pass_criteria="3-5 cümle + Socratic soru",
        fail_criteria="Ders notu yapıştırması veya kronolojik bilgi kusması",
        expected_keywords=["fransız", "french", "revolution", "devrim"],
        requires_question=True,
    ),
    Scenario(
        id="HCIV102-03",
        course_code="HCIV 102",
        level="easy",
        user_message="Sanayi Devrimi ne zaman başladı",
        skill="rag_retrieval",
        expected_behavior="Tarih ve yer bilgisi + kısa bağlam",
        pass_criteria="Doğru tarih + kısa açıklama",
        fail_criteria="Yanlış tarih veya aşırı detay",
        expected_keywords=["industrial", "sanayi", "revolution", "devrim", "england", "ingiltere"],
    ),
    Scenario(
        id="HCIV102-04",
        course_code="HCIV 102",
        level="medium",
        user_message="Fransız Devrimi ile Amerikan Devrimi arasındaki farklar",
        skill="rag_retrieval",
        expected_behavior="İki devrimi karşılaştırır, benzerlik ve farkları sunar",
        pass_criteria="Yapılandırılmış karşılaştırma + birden fazla chunk kullanımı",
        fail_criteria="Sadece birini açıklama",
        expected_keywords=["fransız", "amerikan", "french", "american", "revolution", "devrim"],
    ),
    Scenario(
        id="HCIV102-05",
        course_code="HCIV 102",
        level="medium",
        user_message="emperyalizm ve kolonizasyon aynı şey mi",
        skill="response_quality",
        expected_behavior="İki kavramı ayırır, örneklerle açıklar, düşündürücü soru sorar",
        pass_criteria="Net kavram ayrımı + tarihsel örnek + Socratic",
        fail_criteria="Kavramları eş anlamlı gösterme",
        expected_keywords=["emperyalizm", "imperialism", "kolonizasyon", "colonization"],
        requires_question=True,
    ),
    Scenario(
        id="HCIV102-06",
        course_code="HCIV 102",
        level="medium",
        user_message="Soğuk Savaş'ın ana cepheleri nelerdi",
        skill="rag_retrieval",
        expected_behavior="Proxy wars, ideolojik rekabet, silahlanma yarışı vb. özetler",
        pass_criteria="Çok boyutlu cevap + chunk'lardan sentez",
        fail_criteria="Sadece askeri boyut veya eksik bilgi",
        expected_keywords=["soğuk", "cold", "war", "savaş"],
    ),
    Scenario(
        id="HCIV102-07",
        course_code="HCIV 102",
        level="medium",
        user_message="anlamadım basit anlat",
        skill="context_tracking",
        expected_behavior="Soğuk Savaş'ı günlük dil ve analoji ile yeniden açıklar",
        pass_criteria="Basitleştirme + analoji",
        fail_criteria="Aynı cevabı tekrar",
        requires_context=True,
        prior_topic="Soğuk Savaş",
    ),
    Scenario(
        id="HCIV102-08",
        course_code="HCIV 102",
        level="hard",
        user_message="I. Dünya Savaşı olmasaydı II. Dünya Savaşı olur muydu? Tartış.",
        skill="synthesis",
        expected_behavior="Nedensellik zincirini analiz eder, farklı perspektifler sunar",
        pass_criteria="Çok perspektifli analiz + Socratic + materyalden kanıt",
        fail_criteria="Tek cümlelik evet/hayır veya yüzeysel cevap",
        expected_keywords=["dünya", "savaş", "world", "war"],
        requires_question=True,
    ),
    Scenario(
        id="HCIV102-09",
        course_code="HCIV 102",
        level="hard",
        user_message="beni test et",
        skill="quiz_mode",
        expected_behavior="HCIV 102 konularından tek soru sorar",
        pass_criteria="Ders kapsamında tek soru + cevap beklemesi",
        fail_criteria="Çoklu soru veya cevabı verme",
        expected_course="HCIV 102-4 History of Civilization II",
        requires_question=True,
    ),
    Scenario(
        id="HCIV102-10",
        course_code="HCIV 102",
        level="hard",
        user_message="Sanayi Devrimi'nin emperyalizme etkisini ve bunun bugünkü küresel eşitsizliğe yansımasını açıkla",
        skill="synthesis",
        expected_behavior="Sanayi Devrimi → emperyalizm → modern eşitsizlik zincirini kurar",
        pass_criteria="3 konuyu bağlayan tutarlı analiz + chunk sentezi",
        fail_criteria="Konuları izole açıklama, bağ kuramama",
        expected_keywords=["sanayi", "industrial", "emperyalizm", "imperialism", "eşitsizlik"],
    ),
    # ══════════════════════════════════════════════════════════════════════════
    # CTIS 474 — Information Systems Auditing
    # ══════════════════════════════════════════════════════════════════════════
    Scenario(
        id="CTIS474-01",
        course_code="CTIS 474",
        level="easy",
        user_message="CTIS 474 çalışacağım",
        skill="course_detection",
        expected_behavior="IS Auditing kursunu tanır, ana konuları listeler",
        pass_criteria="Doğru kurs + Socratic başlangıç",
        fail_criteria="Yanlış kurs tespiti",
        expected_course="CTIS 474-1 Information Systems Auditing",
        requires_question=True,
    ),
    Scenario(
        id="CTIS474-02",
        course_code="CTIS 474",
        level="easy",
        user_message="COBIT nedir",
        skill="rag_retrieval",
        expected_behavior="COBIT framework'ünü kısa özetler",
        pass_criteria="Doğru tanım + kısa format",
        fail_criteria="ITIL ile karıştırma veya chunk yapıştırma",
        expected_course="CTIS 474-1 Information Systems Auditing",
        expected_keywords=["cobit", "framework", "governance", "control"],
    ),
    Scenario(
        id="CTIS474-03",
        course_code="CTIS 474",
        level="easy",
        user_message="IT audit süreci nasıl işler",
        skill="rag_retrieval",
        expected_behavior="Planlama → yürütme → raporlama adımlarını özetler",
        pass_criteria="Süreç adımları + kısa açıklama + soru",
        fail_criteria="Detaylı prosedür dökmesi",
        expected_course="CTIS 474-1 Information Systems Auditing",
        expected_keywords=["audit", "denetim", "planlama", "plan"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS474-04",
        course_code="CTIS 474",
        level="medium",
        user_message="COBIT ile ITIL arasındaki fark ne",
        skill="rag_retrieval",
        expected_behavior="İki framework'ü karşılaştırır, kullanım alanlarını belirtir",
        pass_criteria="Net karşılaştırma + doğru chunk retrieval",
        fail_criteria="Tek framework açıklama veya yanlış bilgi",
        expected_course="CTIS 474-1 Information Systems Auditing",
        expected_keywords=["cobit", "itil"],
    ),
    Scenario(
        id="CTIS474-05",
        course_code="CTIS 474",
        level="medium",
        user_message="risk değerlendirme nasıl yapılır adım adım",
        skill="rag_retrieval",
        expected_behavior="Risk assessment metodolojisini adım adım özetler",
        pass_criteria="Yapılandırılmış cevap + ders materyaliyle uyumlu",
        fail_criteria="Genel bilgi (ders materyali dışı) veya eksik adımlar",
        expected_course="CTIS 474-1 Information Systems Auditing",
        expected_keywords=["risk", "değerlendirme", "assessment"],
    ),
    Scenario(
        id="CTIS474-06",
        course_code="CTIS 474",
        level="medium",
        user_message="penetrasyon testi ile IT audit arasındaki ilişki ne",
        skill="synthesis",
        expected_behavior="Pentest'in audit sürecindeki rolünü açıklar",
        pass_criteria="İlişkiyi doğru kurar + her iki kavramdan chunk",
        fail_criteria="İlişki kuramama veya pentest'i bağımsız açıklama",
        expected_course="CTIS 474-1 Information Systems Auditing",
        expected_keywords=["penetrasyon", "pentest", "audit", "denetim"],
    ),
    Scenario(
        id="CTIS474-07",
        course_code="CTIS 474",
        level="medium",
        user_message="devam et, daha detay ver",
        skill="context_tracking",
        expected_behavior="Pentest-audit ilişkisini derinleştirir (raporlama, bulgu sınıflandırma)",
        pass_criteria="Önceki bağlamı korur + yeni bilgi ekler",
        fail_criteria="Bağlam kaybı veya tekrar",
        requires_context=True,
        prior_topic="penetrasyon testi ve audit",
    ),
    Scenario(
        id="CTIS474-08",
        course_code="CTIS 474",
        level="hard",
        user_message="Bir şirketin ERP sistemi hacklenirse auditor olarak ilk ne yaparsın?",
        skill="synthesis",
        expected_behavior="Incident response + audit prosedürü + BCP perspektifinden yaklaşır",
        pass_criteria="Multi-domain yaklaşım + Socratic + materyalden referans",
        fail_criteria="Tek boyutlu cevap veya gerçekçi olmayan adımlar",
        expected_course="CTIS 474-1 Information Systems Auditing",
        expected_keywords=["erp", "audit", "incident", "response"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS474-09",
        course_code="CTIS 474",
        level="hard",
        user_message="beni test et",
        skill="quiz_mode",
        expected_behavior="CTIS 474 konusundan tek soru",
        pass_criteria="Ders kapsamında soru + cevap bekler",
        fail_criteria="Çoklu soru veya cevap verme",
        expected_course="CTIS 474-1 Information Systems Auditing",
        requires_question=True,
    ),
    Scenario(
        id="CTIS474-10",
        course_code="CTIS 474",
        level="hard",
        user_message="SOX compliance'ı sağlamak için IT auditor hangi kontrolleri test etmeli? Bunu COBIT ile nasıl eşlersin?",
        skill="synthesis",
        expected_behavior="SOX gereksinimleri + COBIT kontrolleri eşleştirmesi",
        pass_criteria="Framework cross-reference + pratik örnek + derinlik",
        fail_criteria="Genel bilgi veya eşleştirme yapamama",
        expected_course="CTIS 474-1 Information Systems Auditing",
        expected_keywords=["sox", "cobit", "compliance", "control"],
    ),
    # ══════════════════════════════════════════════════════════════════════════
    # EDEB 201 — Introduction to Turkish Fiction
    # ══════════════════════════════════════════════════════════════════════════
    Scenario(
        id="EDEB201-01",
        course_code="EDEB 201",
        level="easy",
        user_message="EDEB 201 çalışacağım",
        skill="course_detection",
        expected_behavior="Kursu tanır, dönemleri/yazarları özetler",
        pass_criteria="Doğru kurs + edebiyat dönemleri + Socratic",
        fail_criteria="Yanlış kurs veya edebiyat tarihine dalmak",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        requires_question=True,
    ),
    Scenario(
        id="EDEB201-02",
        course_code="EDEB 201",
        level="easy",
        user_message="Tanzimat edebiyatı ne zaman başladı",
        skill="rag_retrieval",
        expected_behavior="Tarih + kısa bağlam + 'bu dönemin hangi özelliği seni ilgilendiriyor?' sorusu",
        pass_criteria="Doğru tarih + Socratic",
        fail_criteria="Yanlış bilgi veya aşırı detay",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        expected_keywords=["tanzimat", "1839", "edebiyat"],
        requires_question=True,
    ),
    Scenario(
        id="EDEB201-03",
        course_code="EDEB 201",
        level="easy",
        user_message="Servet-i Fünun döneminin özellikleri",
        skill="rag_retrieval",
        expected_behavior="Ana özellikleri özetler, temsilci yazarları belirtir",
        pass_criteria="Kısa + doğru + soru",
        fail_criteria="Chunk copy-paste",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        expected_keywords=["servet", "fünun"],
    ),
    Scenario(
        id="EDEB201-04",
        course_code="EDEB 201",
        level="medium",
        user_message="Tanzimat ile Servet-i Fünun edebiyatı arasındaki farklar",
        skill="rag_retrieval",
        expected_behavior="İki dönemi tematik, stilistik ve toplumsal açıdan karşılaştırır",
        pass_criteria="Yapılandırılmış karşılaştırma + chunk sentezi",
        fail_criteria="Sadece birini açıklama",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        expected_keywords=["tanzimat", "servet", "fünun"],
    ),
    Scenario(
        id="EDEB201-05",
        course_code="EDEB 201",
        level="medium",
        user_message="Türk edebiyatında doğu-batı çatışması nasıl işlenmiş",
        skill="rag_retrieval",
        expected_behavior="Farklı dönemlerden örneklerle temayı analiz eder",
        pass_criteria="Çoklu dönem referansı + eser örneği + Socratic",
        fail_criteria="Tek dönem veya tek eser",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        expected_keywords=["doğu", "batı", "çatışma", "roman"],
        requires_question=True,
    ),
    Scenario(
        id="EDEB201-06",
        course_code="EDEB 201",
        level="medium",
        user_message="köy romanı nedir, önemli örnekleri",
        skill="rag_retrieval",
        expected_behavior="Köy romanı tanımı, temsilcileri, toplumsal bağlamı",
        pass_criteria="Tanım + yazarlar + eserler + bağlam",
        fail_criteria="Şehir romanı ile karıştırma",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        expected_keywords=["köy", "roman"],
    ),
    Scenario(
        id="EDEB201-07",
        course_code="EDEB 201",
        level="medium",
        user_message="anlamadım, daha basit anlat",
        skill="context_tracking",
        expected_behavior="Köy romanını günlük dille, somut örnekle yeniden açıklar",
        pass_criteria="Basit dil + güncel analoji",
        fail_criteria="Akademik dil tekrarı",
        requires_context=True,
        prior_topic="köy romanı",
    ),
    Scenario(
        id="EDEB201-08",
        course_code="EDEB 201",
        level="hard",
        user_message="Oğuz Atay'ın Tutunamayanlar'daki anlatı tekniği geleneksel Türk romanından nasıl ayrılıyor?",
        skill="synthesis",
        expected_behavior="Postmodern teknikler vs. geleneksel anlatı karşılaştırması",
        pass_criteria="Teknik terim kullanımı + karşılaştırma + materyalden destek",
        fail_criteria="Yüzeysel cevap veya yanlış eser bilgisi",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        expected_keywords=["oğuz", "atay", "tutunamayanlar"],
    ),
    Scenario(
        id="EDEB201-09",
        course_code="EDEB 201",
        level="hard",
        user_message="beni test et",
        skill="quiz_mode",
        expected_behavior="Türk edebiyatı konusundan tek soru",
        pass_criteria="Ders kapsamında soru + cevap bekleme",
        fail_criteria="Birden fazla soru",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        requires_question=True,
    ),
    Scenario(
        id="EDEB201-10",
        course_code="EDEB 201",
        level="hard",
        user_message="Cumhuriyet dönemi edebiyatında kadın temsilinin değişimini Tanzimat'tan bugüne karşılaştır",
        skill="synthesis",
        expected_behavior="Dönemler arası karşılaştırma, eser ve karakter örnekleriyle",
        pass_criteria="Çoklu dönem + eser referansı + toplumsal bağlam",
        fail_criteria="Tek dönem veya yüzeysel",
        expected_course="EDEB 201-2 Introduction to Turkish Fiction",
        expected_keywords=["kadın", "tanzimat", "cumhuriyet", "roman"],
    ),
    # ══════════════════════════════════════════════════════════════════════════
    # CTIS 465 — Microservice Development
    # ══════════════════════════════════════════════════════════════════════════
    Scenario(
        id="CTIS465-01",
        course_code="CTIS 465",
        level="easy",
        user_message="CTIS 465 çalışacağım",
        skill="course_detection",
        expected_behavior="Microservice Development kursunu tanır, konuları listeler",
        pass_criteria="Doğru kurs + Socratic açılış",
        fail_criteria="Yanlış kurs tespiti",
        expected_course="CTIS 465-1 Microservice Development",
        requires_question=True,
    ),
    Scenario(
        id="CTIS465-02",
        course_code="CTIS 465",
        level="easy",
        user_message="microservice nedir monolith'ten farkı ne",
        skill="rag_retrieval",
        expected_behavior="Kısa tanım, temel fark, kontrol sorusu",
        pass_criteria="3-5 cümle + karşılaştırma + soru",
        fail_criteria="Wall-of-text veya sadece tanım",
        expected_course="CTIS 465-1 Microservice Development",
        expected_keywords=["microservice", "monolith", "service"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS465-03",
        course_code="CTIS 465",
        level="easy",
        user_message="API gateway ne işe yarar",
        skill="rag_retrieval",
        expected_behavior="API gateway'in rolünü kısa açıklar",
        pass_criteria="Doğru açıklama + kısa format",
        fail_criteria="Yanlış bilgi veya aşırı detay",
        expected_course="CTIS 465-1 Microservice Development",
        expected_keywords=["api", "gateway"],
    ),
    Scenario(
        id="CTIS465-04",
        course_code="CTIS 465",
        level="medium",
        user_message="circuit breaker pattern nedir ne zaman kullanılır",
        skill="rag_retrieval",
        expected_behavior="Pattern'ı açıklar, kullanım senaryosu verir, Socratic soru",
        pass_criteria="Doğru açıklama + senaryo + soru",
        fail_criteria="Sadece tanım veya yanlış kullanım senaryosu",
        expected_course="CTIS 465-1 Microservice Development",
        expected_keywords=["circuit", "breaker", "pattern"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS465-05",
        course_code="CTIS 465",
        level="medium",
        user_message="Kafka ile RabbitMQ arasındaki fark ne, hangisini ne zaman kullanırım",
        skill="rag_retrieval",
        expected_behavior="İki message broker'ı karşılaştırır, kullanım senaryolarını belirtir",
        pass_criteria="Net karşılaştırma + use-case örnekleri",
        fail_criteria="Sadece birini açıklama",
        expected_course="CTIS 465-1 Microservice Development",
        expected_keywords=["kafka", "rabbitmq", "message"],
    ),
    Scenario(
        id="CTIS465-06",
        course_code="CTIS 465",
        level="medium",
        user_message="saga pattern nedir",
        skill="rag_retrieval",
        expected_behavior="Distributed transaction yönetimini saga ile açıklar",
        pass_criteria="Kısa + doğru + choreography vs orchestration ayrımı",
        fail_criteria="CQRS ile karıştırma",
        expected_course="CTIS 465-1 Microservice Development",
        expected_keywords=["saga", "pattern", "transaction"],
    ),
    Scenario(
        id="CTIS465-07",
        course_code="CTIS 465",
        level="medium",
        user_message="devam et daha detay ver",
        skill="context_tracking",
        expected_behavior="Saga pattern'ı derinleştirir — compensating transactions, failure handling",
        pass_criteria="Bağlamı korur + derinleştirir",
        fail_criteria="Bağlam kaybı",
        requires_context=True,
        prior_topic="saga pattern",
    ),
    Scenario(
        id="CTIS465-08",
        course_code="CTIS 465",
        level="hard",
        user_message="E-ticaret uygulaması için microservice mimarisi tasarla: hangi servisler, hangi iletişim pattern'ları, hangi veritabanı stratejisi?",
        skill="synthesis",
        expected_behavior="Servis decomposition + iletişim + DB strategy birlikte tasarlar",
        pass_criteria="Tutarlı mimari + birden fazla pattern referansı + Socratic",
        fail_criteria="Genel bilgi veya tutarsız tasarım",
        expected_course="CTIS 465-1 Microservice Development",
        expected_keywords=["microservice", "service", "database"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS465-09",
        course_code="CTIS 465",
        level="hard",
        user_message="beni test et",
        skill="quiz_mode",
        expected_behavior="Microservice konusundan tek soru",
        pass_criteria="Ders kapsamında soru + cevap bekleme",
        fail_criteria="Birden fazla soru",
        expected_course="CTIS 465-1 Microservice Development",
        requires_question=True,
    ),
    Scenario(
        id="CTIS465-10",
        course_code="CTIS 465",
        level="hard",
        user_message="Kubernetes pod'u crash loop'a girdi, service mesh üzerinden distributed tracing ile debug et — adımları anlat",
        skill="synthesis",
        expected_behavior="K8s + service mesh + tracing birleştirilerek debug süreci",
        pass_criteria="Gerçekçi debug akışı + birden fazla konu sentezi",
        fail_criteria="Yüzeysel veya tek boyutlu cevap",
        expected_course="CTIS 465-1 Microservice Development",
        expected_keywords=["kubernetes", "pod", "service", "mesh", "tracing"],
    ),
    # ══════════════════════════════════════════════════════════════════════════
    # CTIS 456 — Senior Project II
    # ══════════════════════════════════════════════════════════════════════════
    Scenario(
        id="CTIS456-01",
        course_code="CTIS 456",
        level="easy",
        user_message="CTIS 456 çalışacağım",
        skill="course_detection",
        expected_behavior="Senior Project II tanır, proje süreç aşamalarını listeler",
        pass_criteria="Doğru kurs + Socratic",
        fail_criteria="Senior Project I ile karıştırma",
        expected_course="CTIS 456-4 Senior Project II",
        requires_question=True,
    ),
    Scenario(
        id="CTIS456-02",
        course_code="CTIS 456",
        level="easy",
        user_message="SRS dokümanı nedir ne içerir",
        skill="rag_retrieval",
        expected_behavior="Software Requirements Specification'ı kısaca özetler",
        pass_criteria="Doğru tanım + ana bölümler",
        fail_criteria="SRS (Student Registration System) ile karıştırma",
        expected_course="CTIS 456-4 Senior Project II",
        expected_keywords=["srs", "requirements", "specification", "gereksinim"],
    ),
    Scenario(
        id="CTIS456-03",
        course_code="CTIS 456",
        level="easy",
        user_message="Agile ile Waterfall farkı ne",
        skill="rag_retrieval",
        expected_behavior="Temel farkları özetler, 'projenizde hangisini kullanıyorsunuz?' sorar",
        pass_criteria="Kısa karşılaştırma + Socratic",
        fail_criteria="Wall-of-text",
        expected_course="CTIS 456-4 Senior Project II",
        expected_keywords=["agile", "waterfall"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS456-04",
        course_code="CTIS 456",
        level="medium",
        user_message="sprint planning nasıl yapılır best practice'ler ne",
        skill="rag_retrieval",
        expected_behavior="Sprint planning sürecini adım adım, best practice'lerle açıklar",
        pass_criteria="Pratik tavsiyeler + ders materyalinden referans",
        fail_criteria="Genel bilgi (ders dışı)",
        expected_course="CTIS 456-4 Senior Project II",
        expected_keywords=["sprint", "planning"],
    ),
    Scenario(
        id="CTIS456-05",
        course_code="CTIS 456",
        level="medium",
        user_message="UML diyagramlarından hangilerini projemde kullanmalıyım",
        skill="response_quality",
        expected_behavior="Senior project bağlamında en faydalı UML diyagramlarını önerir",
        pass_criteria="Proje bağlamında öneriler + gerekçe + Socratic",
        fail_criteria="Tüm UML diyagramlarını listeleme",
        expected_course="CTIS 456-4 Senior Project II",
        expected_keywords=["uml", "diyagram", "diagram"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS456-06",
        course_code="CTIS 456",
        level="medium",
        user_message="test planı nasıl yazılır",
        skill="rag_retrieval",
        expected_behavior="Test planı bileşenlerini ve yazım sürecini özetler",
        pass_criteria="Yapılandırılmış cevap + ders materyali referansı",
        fail_criteria="Genel bilgi veya chunk yapıştırma",
        expected_course="CTIS 456-4 Senior Project II",
        expected_keywords=["test", "plan"],
    ),
    Scenario(
        id="CTIS456-07",
        course_code="CTIS 456",
        level="medium",
        user_message="anlamadım, UAT ile unit test farkını basit anlat",
        skill="context_tracking",
        expected_behavior="Test türlerini basit dille, analoji ile ayırır",
        pass_criteria="Basit dil + analoji + net ayrım",
        fail_criteria="Teknik jargon tekrarı",
        requires_context=True,
        prior_topic="test planı",
    ),
    Scenario(
        id="CTIS456-08",
        course_code="CTIS 456",
        level="hard",
        user_message="Projemizin deadline'ı yaklaşıyor ama 3 sprint'lik iş var, 1 sprint'imiz kaldı. Nasıl önceliklendirme yapmalıyım?",
        skill="synthesis",
        expected_behavior="MoSCoW veya benzeri prioritization + risk yönetimi + pratik tavsiye",
        pass_criteria="Gerçekçi strateji + framework referansı + Socratic",
        fail_criteria="'Daha çok çalış' gibi yüzeysel tavsiye",
        expected_course="CTIS 456-4 Senior Project II",
        expected_keywords=["sprint", "önceliklendirme", "priorit"],
        requires_question=True,
    ),
    Scenario(
        id="CTIS456-09",
        course_code="CTIS 456",
        level="hard",
        user_message="beni test et",
        skill="quiz_mode",
        expected_behavior="Senior Project konusundan tek soru",
        pass_criteria="Ders kapsamında soru + cevap bekleme",
        fail_criteria="Birden fazla soru",
        expected_course="CTIS 456-4 Senior Project II",
        requires_question=True,
    ),
    Scenario(
        id="CTIS456-10",
        course_code="CTIS 456",
        level="hard",
        user_message="Final sunumunda jüri 'sisteminiz ölçeklenebilir mi?' diye sorarsa nasıl cevap vermeliyim? Teknik ve sunum açısından hazırla.",
        skill="synthesis",
        expected_behavior="Scalability açıklaması + sunum tekniği + ikna stratejisi",
        pass_criteria="Teknik cevap + sunum tavsiyesi + Socratic",
        fail_criteria="Sadece teknik veya sadece sunum tavsiyesi",
        expected_course="CTIS 456-4 Senior Project II",
        expected_keywords=["ölçeklenebilir", "scalab", "sunum"],
    ),
]

assert len(SCENARIOS) == 60, f"Expected 60 scenarios, got {len(SCENARIOS)}"


# ─── Automated Evaluation ────────────────────────────────────────────────────


def eval_course_detection(scenarios: list[Scenario], detect_fn) -> list[dict]:
    """Test course detection accuracy. detect_fn(msg) → fullname|None."""
    results = []
    for s in scenarios:
        if not s.expected_course:
            continue
        detected = detect_fn(s.user_message)
        match = detected and s.expected_course.lower() in detected.lower()
        results.append(
            {
                "id": s.id,
                "message": s.user_message,
                "expected": s.expected_course,
                "detected": detected or "(None)",
                "pass": match,
            }
        )
    return results


def eval_rag_retrieval(scenarios: list[Scenario], vs, search_fn=None, threshold=0.25) -> list[dict]:
    """Test RAG retrieval — keyword precision per scenario."""
    if search_fn is None:
        search_fn = vs.hybrid_search

    results = []
    for s in scenarios:
        if not s.expected_keywords:
            continue

        # Determine search params
        course_filter = s.expected_course if s.expected_course else None
        if search_fn == vs.query:
            hits = search_fn(query_text=s.user_message, n_results=15, course_filter=course_filter)
        else:
            hits = search_fn(query=s.user_message, n_results=15, course_filter=course_filter)

        if not hits:
            results.append(
                {
                    "id": s.id,
                    "message": s.user_message,
                    "total_hits": 0,
                    "keyword_precision": 0,
                    "found": [],
                    "missing": s.expected_keywords,
                    "pass": False,
                }
            )
            continue

        # Apply threshold
        scores = [(1 - r["distance"]) for r in hits]
        top_score = scores[0]
        adaptive = max(top_score * 0.60, 0.20)
        filtered = [r for r, sc in zip(hits, scores, strict=False) if sc > adaptive][:10]

        all_text = " ".join(r.get("text", "").lower() for r in filtered)
        found = [kw for kw in s.expected_keywords if kw.lower() in all_text]
        missing = [kw for kw in s.expected_keywords if kw.lower() not in all_text]
        precision = len(found) / len(s.expected_keywords) if s.expected_keywords else 0

        results.append(
            {
                "id": s.id,
                "message": s.user_message,
                "total_hits": len(hits),
                "after_filter": len(filtered),
                "top_score": round(top_score, 3),
                "keyword_precision": round(precision, 2),
                "found": found,
                "missing": missing,
                "pass": precision >= 0.6,
            }
        )

    return results


def eval_live_response(scenarios: list[Scenario], llm, vs, detect_fn) -> list[dict]:
    """
    Live LLM eval — actually calls LLM and checks response format.
    WARNING: Costs API credits. Use --live flag.
    """
    results = []

    for s in scenarios:
        if s.requires_context:
            # Context-dependent scenarios need sequential sending — skip in auto
            results.append(
                {
                    "id": s.id,
                    "message": s.user_message,
                    "skipped": True,
                    "reason": "requires_context",
                }
            )
            continue

        # Detect course
        course = detect_fn(s.user_message) if s.expected_course else None

        # Get RAG chunks
        chunks = []
        if course or len(s.user_message.split()) > 2:
            chunks = vs.hybrid_search(query=s.user_message, n_results=15, course_filter=course)
            if chunks:
                top = 1 - chunks[0]["distance"]
                adaptive = max(top * 0.60, 0.20)
                chunks = [r for r in chunks if (1 - r["distance"]) > adaptive][:10]

        # Build messages
        messages = [{"role": "user", "content": s.user_message}]

        # Call LLM
        try:
            response = llm.chat_with_history(
                messages=messages,
                context_chunks=chunks if chunks else None,
            )
        except Exception as e:
            results.append(
                {
                    "id": s.id,
                    "message": s.user_message,
                    "error": str(e),
                    "pass": False,
                }
            )
            continue

        # Automated checks
        word_count = len(response.split())
        has_question = "?" in response
        checks = {}

        if s.max_words > 0:
            checks["word_count"] = word_count <= s.max_words
            checks["actual_words"] = word_count

        if s.requires_question:
            checks["has_question"] = has_question

        # Keyword check in response
        if s.expected_keywords:
            resp_lower = response.lower()
            found_kw = [kw for kw in s.expected_keywords if kw.lower() in resp_lower]
            checks["keyword_match"] = len(found_kw) / len(s.expected_keywords)

        all_pass = all(v if isinstance(v, bool) else v >= 0.4 for v in checks.values() if isinstance(v, (bool, float)))

        results.append(
            {
                "id": s.id,
                "message": s.user_message,
                "response_preview": response[:200],
                "word_count": word_count,
                "checks": checks,
                "auto_pass": all_pass,
                "needs_manual": s.skill in ("response_quality", "synthesis", "context_tracking"),
            }
        )

        # Rate limit
        time.sleep(1)

    return results


# ─── Report Generation ───────────────────────────────────────────────────────

REPORT_PATH = Path("tests/e2e_report.json")
REPORT_MD_PATH = Path("tests/e2e_report.md")


def generate_markdown_report(
    course_results: list[dict],
    rag_results: list[dict],
    live_results: list[dict] | None = None,
) -> str:
    """Generate a markdown report table."""
    lines = [
        f"# E2E Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Course detection
    if course_results:
        passed = sum(1 for r in course_results if r["pass"])
        lines.append(f"## Course Detection: {passed}/{len(course_results)}")
        lines.append("")
        lines.append("| ID | Message | Expected | Detected | Pass |")
        lines.append("|---|---|---|---|---|")
        for r in course_results:
            icon = "✅" if r["pass"] else "❌"
            lines.append(f"| {r['id']} | {r['message'][:40]} | {r['expected'][:30]} | {r['detected'][:30]} | {icon} |")
        lines.append("")

    # RAG retrieval
    if rag_results:
        passed = sum(1 for r in rag_results if r["pass"])
        avg_prec = sum(r["keyword_precision"] for r in rag_results) / len(rag_results)
        lines.append(f"## RAG Retrieval: {passed}/{len(rag_results)} pass, avg precision={avg_prec:.0%}")
        lines.append("")
        lines.append("| ID | Message | Precision | Top Score | Missing | Pass |")
        lines.append("|---|---|---|---|---|---|")
        for r in rag_results:
            icon = "✅" if r["pass"] else "❌"
            top = r.get("top_score", 0)
            missing = ", ".join(r.get("missing", [])[:3])
            lines.append(
                f"| {r['id']} | {r['message'][:35]} | {r['keyword_precision']:.0%} | {top:.3f} | {missing} | {icon} |"
            )
        lines.append("")

    # Live LLM (if ran)
    if live_results:
        auto_pass = sum(1 for r in live_results if r.get("auto_pass"))
        skipped = sum(1 for r in live_results if r.get("skipped"))
        total = len(live_results) - skipped
        lines.append(f"## Live LLM: {auto_pass}/{total} auto-pass ({skipped} skipped)")
        lines.append("")
        lines.append("| ID | Message | Words | Has ? | KW Match | Auto | Manual? |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in live_results:
            if r.get("skipped"):
                lines.append(f"| {r['id']} | {r['message'][:30]} | — | — | — | ⏭️ | {r['reason']} |")
                continue
            if r.get("error"):
                lines.append(f"| {r['id']} | {r['message'][:30]} | ERR | — | — | ❌ | {r['error'][:20]} |")
                continue
            checks = r.get("checks", {})
            wc = r.get("word_count", 0)
            hq = "✅" if checks.get("has_question", True) else "❌"
            kw = f"{checks.get('keyword_match', 0):.0%}" if "keyword_match" in checks else "—"
            ap = "✅" if r.get("auto_pass") else "❌"
            mn = "🔍" if r.get("needs_manual") else "—"
            lines.append(f"| {r['id']} | {r['message'][:30]} | {wc} | {hq} | {kw} | {ap} | {mn} |")
        lines.append("")

    # Summary by course
    lines.append("## Summary by Course")
    lines.append("")
    courses = {}
    for r in rag_results:
        cid = r["id"].rsplit("-", 1)[0]
        courses.setdefault(cid, []).append(r)
    lines.append("| Course | Queries | Precision | Pass Rate |")
    lines.append("|---|---|---|---|")
    for cid, rs in sorted(courses.items()):
        avg = sum(r["keyword_precision"] for r in rs) / len(rs)
        pr = sum(1 for r in rs if r["pass"]) / len(rs)
        lines.append(f"| {cid} | {len(rs)} | {avg:.0%} | {pr:.0%} |")
    lines.append("")

    # Summary by skill
    lines.append("## Summary by Skill")
    lines.append("")
    skills = {}
    for r in rag_results:
        sc = next((s for s in SCENARIOS if s.id == r["id"]), None)
        if sc:
            skills.setdefault(sc.skill, []).append(r)
    lines.append("| Skill | Queries | Avg Precision | Pass Rate |")
    lines.append("|---|---|---|---|")
    for sk, rs in sorted(skills.items()):
        avg = sum(r["keyword_precision"] for r in rs) / len(rs)
        pr = sum(1 for r in rs if r["pass"]) / len(rs)
        lines.append(f"| {sk} | {len(rs)} | {avg:.0%} | {pr:.0%} |")
    lines.append("")

    # Summary by level
    lines.append("## Summary by Level")
    lines.append("")
    levels = {}
    for r in rag_results:
        sc = next((s for s in SCENARIOS if s.id == r["id"]), None)
        if sc:
            levels.setdefault(sc.level, []).append(r)
    lines.append("| Level | Queries | Avg Precision | Pass Rate |")
    lines.append("|---|---|---|---|")
    for lv in ["easy", "medium", "hard"]:
        rs = levels.get(lv, [])
        if rs:
            avg = sum(r["keyword_precision"] for r in rs) / len(rs)
            pr = sum(1 for r in rs if r["pass"]) / len(rs)
            lines.append(f"| {lv} | {len(rs)} | {avg:.0%} | {pr:.0%} |")
    lines.append("")

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2E Test Scenarios")
    parser.add_argument("--course", type=str, help="Filter by course code (e.g. CTIS363)")
    parser.add_argument("--level", type=str, choices=["easy", "medium", "hard"], help="Filter by level")
    parser.add_argument("--skill", type=str, help="Filter by skill (rag_retrieval, course_detection, ...)")
    parser.add_argument("--live", action="store_true", help="Run live LLM evaluation (costs API credits)")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from core.vector_store import VectorStore

    vs = VectorStore()
    vs.initialize()

    # Filter scenarios
    filtered = SCENARIOS
    if args.course:
        code = args.course.upper().replace("-", " ")
        if " " not in code and len(code) > 4:
            # e.g. "CTIS363" → "CTIS 363"
            code = code[:-3] + " " + code[-3:]
        filtered = [s for s in filtered if code in s.course_code.upper()]
    if args.level:
        filtered = [s for s in filtered if s.level == args.level]
    if args.skill:
        filtered = [s for s in filtered if s.skill == args.skill]

    print(f"Running {len(filtered)} scenarios\n")

    # ── Course Detection ──────────────────────────────────────────────────
    # We need a mock detect function that works without Telegram context
    # Use the same logic as detect_active_course but with static course list
    course_list = [
        {"shortname": "CTIS 363-1", "fullname": "CTIS 363-1 Ethical and Social Issues in Information Systems"},
        {"shortname": "HCIV 102-4", "fullname": "HCIV 102-4 History of Civilization II"},
        {"shortname": "CTIS 474-1", "fullname": "CTIS 474-1 Information Systems Auditing"},
        {"shortname": "EDEB 201-2", "fullname": "EDEB 201-2 Introduction to Turkish Fiction"},
        {"shortname": "CTIS 465-1", "fullname": "CTIS 465-1 Microservice Development"},
        {"shortname": "CTIS 456-4", "fullname": "CTIS 456-4 Senior Project II"},
    ]

    def mock_detect(msg: str) -> str | None:
        msg_upper = msg.upper().replace("-", " ").replace("_", " ")
        for c in course_list:
            code = c["shortname"].split("-")[0].strip().upper()
            if code in msg_upper:
                return c["fullname"]
            dept = code.split()[0] if " " in code else code
            if len(dept) >= 3 and dept in msg_upper.split():
                return c["fullname"]
        msg_words = msg.split()
        for c in course_list:
            nums = [p for p in c["shortname"].split() if p.replace("-", "").isdigit() and len(p) >= 3]
            for num in nums:
                num_clean = num.split("-")[0]
                if num_clean in msg_words:
                    return c["fullname"]
        return None

    print("=" * 70)
    print("COURSE DETECTION")
    print("=" * 70)
    course_scenarios = [s for s in filtered if s.expected_course]
    course_results = eval_course_detection(course_scenarios, mock_detect)
    for r in course_results:
        icon = "✅" if r["pass"] else "❌"
        print(f"  {icon} {r['id']:<14} {r['message']:<40} → {r['detected'][:40]}")
    if course_results:
        passed = sum(1 for r in course_results if r["pass"])
        print(f"\n  Result: {passed}/{len(course_results)} passed\n")

    # ── RAG Retrieval ─────────────────────────────────────────────────────
    print("=" * 70)
    print("RAG RETRIEVAL")
    print("=" * 70)
    rag_scenarios = [s for s in filtered if s.expected_keywords]
    rag_results = eval_rag_retrieval(rag_scenarios, vs)
    for r in rag_results:
        icon = "✅" if r["pass"] else "⚠️" if r["keyword_precision"] >= 0.3 else "❌"
        missing = f"  missing: {r['missing']}" if r["missing"] and args.verbose else ""
        print(
            f"  {icon} {r['id']:<14} {r['message']:<40} prec={r['keyword_precision']:.0%} top={r.get('top_score', 0):.3f}{missing}"
        )
    if rag_results:
        avg_prec = sum(r["keyword_precision"] for r in rag_results) / len(rag_results)
        pass_rate = sum(1 for r in rag_results if r["pass"]) / len(rag_results)
        print(f"\n  Result: precision={avg_prec:.0%}  pass_rate={pass_rate:.0%}  ({len(rag_results)} queries)\n")

    # ── Live LLM ──────────────────────────────────────────────────────────
    live_results = None
    if args.live:
        from core.llm_engine import LLMEngine
        from core.memory_manager import MemoryManager

        mem = MemoryManager()
        llm = LLMEngine(vs)
        llm.mem_manager = mem

        print("=" * 70)
        print("LIVE LLM EVALUATION")
        print("=" * 70)
        live_results = eval_live_response(filtered, llm, vs, mock_detect)
        for r in live_results:
            if r.get("skipped"):
                print(f"  ⏭️ {r['id']:<14} {r['message']:<40} (skipped: {r['reason']})")
            elif r.get("error"):
                print(f"  ❌ {r['id']:<14} {r['message']:<40} ERROR: {r['error'][:50]}")
            else:
                ap = "✅" if r.get("auto_pass") else "❌"
                mn = " [manual]" if r.get("needs_manual") else ""
                print(f"  {ap} {r['id']:<14} {r['message']:<40} words={r['word_count']}{mn}")
                if args.verbose:
                    print(f"     {r['response_preview'][:120]}")
        auto_pass = sum(1 for r in live_results if r.get("auto_pass"))
        skipped = sum(1 for r in live_results if r.get("skipped"))
        print(f"\n  Result: {auto_pass}/{len(live_results) - skipped} auto-pass ({skipped} skipped)\n")

    # ── Report ────────────────────────────────────────────────────────────
    if args.report:
        md = generate_markdown_report(course_results, rag_results, live_results)
        REPORT_MD_PATH.write_text(md, encoding="utf-8")
        print(f"Report saved: {REPORT_MD_PATH}")

        # Also save JSON
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(filtered),
            "course_detection": course_results,
            "rag_retrieval": rag_results,
            "live_llm": live_results,
        }
        REPORT_PATH.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON saved: {REPORT_PATH}")
