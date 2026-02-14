"""
Moodle AI Assistant - Main Entry Point
=======================================
CLI interface for syncing, chatting, generating summaries, and launching web UI.
"""

import sys
import logging
import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from core import config
from core.moodle_client import MoodleClient
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.llm_engine import LLMEngine
from core.sync_engine import SyncEngine

# ─── Setup ───────────────────────────────────────────────────────────────────

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("moodle-ai")


def build_components():
    """Initialize all system components."""
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for e in errors:
            console.print(f"  ✗ {e}")
        console.print("\n[dim]Copy .env.example to .env and fill in the values.[/dim]")
        sys.exit(1)

    moodle = MoodleClient()
    processor = DocumentProcessor()
    vector_store = VectorStore()
    vector_store.initialize()

    llm = LLMEngine(vector_store)
    sync = SyncEngine(moodle, processor, vector_store)

    return moodle, processor, vector_store, llm, sync


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_sync(args):
    """Sync Moodle content to local vector store."""
    from core.memory import StaticProfile
    
    moodle, processor, vector_store, llm, sync = build_components()

    console.print(Panel("🔄 Moodle Sync", style="bold blue"))

    if not moodle.connect():
        console.print("[red]Moodle connection failed![/red]")
        return

    # Auto-populate profile from Moodle
    courses = moodle.get_courses()
    profile = StaticProfile()
    profile.auto_populate_from_moodle(
        site_info=moodle.site_info,
        courses=[c.fullname for c in courses],
    )

    sync.sync_all(force=args.force)

    # Print stats
    stats = vector_store.get_stats()
    table = Table(title="Sync Sonuçları")
    table.add_column("Metrik", style="cyan")
    table.add_column("Değer", style="green")
    table.add_row("Toplam Chunk", str(stats.get("total_chunks", 0)))
    table.add_row("Kurs Sayısı", str(stats.get("unique_courses", 0)))
    table.add_row("Dosya Sayısı", str(stats.get("unique_files", 0)))
    console.print(table)


def cmd_courses(args):
    """List enrolled courses and their content."""
    moodle, *_ = build_components()

    if not moodle.connect():
        return

    courses = moodle.get_courses()

    console.print(Panel("📚 Kayıtlı Kurslar", style="bold blue"))

    for course in courses:
        console.print(f"\n[bold cyan]{course.fullname}[/bold cyan] ({course.shortname})")

        if args.detailed:
            sections = moodle.get_course_content(course.id)
            for section in sections:
                if section.name.lower() == "general":
                    continue
                console.print(f"  📂 {section.name}")
                for mod in section.modules:
                    icon = {"resource": "📄", "assign": "📝", "quiz": "❓",
                            "forum": "💬", "url": "🔗", "page": "📃"}.get(
                        mod.get("modname", ""), "•"
                    )
                    console.print(f"     {icon} {mod.get('name', '')}")


def cmd_chat(args):
    """Interactive chat mode with RAG."""
    moodle, processor, vector_store, llm, sync = build_components()

    stats = vector_store.get_stats()
    if stats.get("total_chunks", 0) == 0:
        console.print("[yellow]⚠ Vector store is empty. Run 'python main.py sync' first.[/yellow]")
        console.print("[dim]Continuing without RAG context...[/dim]\n")

    console.print(Panel.fit(
        "[bold green]🎓 Moodle AI Assistant[/bold green]\n"
        f"[dim]Chat: {llm.engine.router.chat} | Extract: {llm.engine.router.extraction} | "
        f"Chunks: {stats.get('total_chunks', 0)} | "
        f"Kurslar: {stats.get('unique_courses', 0)}[/dim]\n\n"
        "Komutlar:\n"
        "  /kurs <isim>  — Belirli bir kursa odaklan\n"
        "  /kurslar      — Kayıtlı kursları listele\n"
        "  /özet <kurs>  — Haftalık özet oluştur\n"
        "  /sorular <konu> — Pratik sorular oluştur\n"
        "  /hafıza       — Kayıtlı anıları göster\n"
        "  /ilerleme     — Konu bazlı öğrenme ilerlemesi\n"
        "  /hatırla <bilgi> — Manuel hafıza ekle\n"
        "  /unut <id>    — Belirli bir hafızayı sil\n"
        "  /profil       — Profil dosyasını düzenle\n"
        "  /maliyet     — Tahmini aylık maliyet\n"
        "  /modeller    — Model routing tablosu\n"
        "  /stats        — İndeks & hafıza istatistikleri\n"
        "  /temizle      — Sohbet geçmişini sil\n"
        "  /çıkış        — Çıkış",
        border_style="green",
    ))

    available_courses = stats.get("courses", [])
    chat_history: list[dict] = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Sen[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Görüşürüz! 👋[/dim]")
            break

        if not user_input.strip():
            continue

        # Handle commands
        if user_input.startswith("/"):
            _handle_command(user_input, llm, vector_store, available_courses)
            continue

        # Regular chat with RAG
        chat_history.append({"role": "user", "content": user_input})
        with console.status("[bold green]Düşünüyorum...[/bold green]"):
            response = llm.chat_with_history(messages=chat_history[-10:])
        chat_history.append({"role": "assistant", "content": response})

        console.print(f"\n[bold green]Asistan[/bold green]")
        console.print(Markdown(response))


def _handle_command(cmd: str, llm: LLMEngine, vs: VectorStore, courses: list):
    """Process slash commands in chat mode."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/çıkış", "/exit", "/quit"):
        console.print("[dim]Görüşürüz! 👋[/dim]")
        sys.exit(0)

    elif command in ("/kurs", "/course"):
        if not arg:
            console.print("[yellow]Kullanım: /kurs <kurs adı>[/yellow]")
            return
        # Fuzzy match
        matched = [c for c in courses if arg.lower() in c.lower()]
        if matched:
            llm.set_active_course(matched[0])
            console.print(f"[green]✓ Aktif kurs: {matched[0]}[/green]")
        else:
            console.print(f"[red]Kurs bulunamadı: {arg}[/red]")
            console.print(f"[dim]Mevcut kurslar: {', '.join(courses)}[/dim]")

    elif command in ("/kurslar", "/courses"):
        for c in courses:
            indicator = " ← aktif" if c == llm.active_course else ""
            console.print(f"  📚 {c}{indicator}")

    elif command in ("/özet", "/summary"):
        if not arg:
            course = llm.active_course or (courses[0] if courses else None)
        else:
            matched = [c for c in courses if arg.lower() in c.lower()]
            course = matched[0] if matched else None

        if not course:
            console.print("[red]Kurs belirtilmedi veya bulunamadı.[/red]")
            return

        console.print(f"[bold]Generating weekly summary for: {course}[/bold]")
        with console.status("[bold green]Özet oluşturuluyor...[/bold green]"):
            chunks = vs.query(f"{course} weekly topic overview", n_results=15, course_filter=course)
            context = "\n\n".join(c["text"] for c in chunks)
            summary = llm.generate_weekly_summary(course, "All Sections", context)

        console.print(Markdown(summary))

    elif command in ("/sorular", "/questions"):
        if not arg:
            console.print("[yellow]Kullanım: /sorular <konu>[/yellow]")
            return
        with console.status("[bold green]Sorular oluşturuluyor...[/bold green]"):
            questions = llm.generate_practice_questions(arg, llm.active_course)
        console.print(Markdown(questions))

    elif command == "/stats":
        stats = vs.get_stats()
        mem_stats = llm.get_memory_stats()

        table = Table(title="📊 İstatistikler")
        table.add_column("Metrik", style="cyan")
        table.add_column("Değer", style="green")
        table.add_row("Toplam Chunk", str(stats.get("total_chunks", 0)))
        table.add_row("Kurs Sayısı", str(stats.get("unique_courses", 0)))
        table.add_row("Dosya Sayısı", str(stats.get("unique_files", 0)))
        table.add_row("─" * 20, "─" * 10)
        table.add_row("💾 Oturumlar", str(mem_stats.get("total_sessions", 0)))
        table.add_row("💬 Mesajlar", str(mem_stats.get("total_messages", 0)))
        table.add_row("🧠 Anılar", str(mem_stats.get("semantic_memories", 0)))
        table.add_row("📈 Takip Edilen Konular", str(mem_stats.get("tracked_topics", 0)))
        if llm.active_course:
            table.add_row("📚 Aktif Kurs", llm.active_course)
        console.print(table)

    elif command in ("/maliyet", "/cost"):
        estimates = llm.engine.estimate_costs(turns_per_day=20)
        table = Table(title="💰 Tahmini Aylık Maliyet (20 tur/gün)")
        table.add_column("Görev", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("Aylık Maliyet", style="green")

        for task, info in estimates.items():
            if task == "total_monthly":
                continue
            table.add_row(task, info["model"], f"${info['monthly_cost']:.3f}")

        table.add_row("─" * 15, "─" * 15, "─" * 10)
        table.add_row("[bold]TOPLAM[/bold]", "", f"[bold]${estimates['total_monthly']:.2f}/ay[/bold]")
        console.print(table)

    elif command in ("/modeller", "/models"):
        models = llm.engine.get_available_models()
        table = Table(title="🤖 Kullanılabilir Modeller")
        table.add_column("Key", style="cyan")
        table.add_column("Provider", style="yellow")
        table.add_column("API Key", style="green")
        table.add_column("Maliyet (In/Out)", style="white")
        for m in models:
            status = "✅" if m["has_key"] else "❌"
            table.add_row(m["key"], m["provider"], status, m["cost"])
        console.print(table)

        console.print(f"\n[dim]Routing:[/dim]")
        router = llm.engine.router
        for task in ["chat", "extraction", "topic_detect", "summary", "questions"]:
            console.print(f"  {task:20s} → {getattr(router, task)}")

    elif command in ("/hafıza", "/memory", "/memories"):
        memories = llm.list_memories(llm.active_course)
        if not memories:
            console.print("[dim]Henüz kayıtlı hafıza yok. Konuştukça öğreneceğim![/dim]")
            return

        table = Table(title="🧠 Semantic Hafıza")
        table.add_column("ID", style="dim")
        table.add_column("Kategori", style="cyan")
        table.add_column("İçerik", style="white", max_width=60)
        table.add_column("Kurs", style="green")
        table.add_column("Güven", style="yellow")

        for m in memories:
            table.add_row(
                str(m.id),
                m.category,
                m.content[:60] + ("..." if len(m.content) > 60 else ""),
                m.course[:20] if m.course else "-",
                f"{m.confidence:.0%}",
            )
        console.print(table)

    elif command in ("/ilerleme", "/progress"):
        course = llm.active_course
        progress = llm.get_learning_progress(course)
        if not progress:
            console.print("[dim]Henüz öğrenme ilerlemesi kaydedilmedi.[/dim]")
            return

        table = Table(title="📈 Öğrenme İlerlemesi")
        table.add_column("Konu", style="white", max_width=40)
        table.add_column("Kurs", style="cyan", max_width=30)
        table.add_column("Seviye", style="green")
        table.add_column("Sorulma", style="yellow")

        for p in progress:
            bar = "█" * int(p.mastery_level * 10) + "░" * (10 - int(p.mastery_level * 10))
            level_color = "red" if p.mastery_level < 0.3 else "yellow" if p.mastery_level < 0.6 else "green"
            table.add_row(
                p.topic,
                p.course[:30],
                f"[{level_color}]{bar} {p.mastery_level:.0%}[/{level_color}]",
                str(p.times_asked),
            )
        console.print(table)

    elif command in ("/hatırla", "/remember"):
        if not arg:
            console.print("[yellow]Kullanım: /hatırla <bilgi>[/yellow]")
            console.print("[dim]Örnek: /hatırla Midterm sınavı 15 Mart'ta[/dim]")
            return
        llm.add_memory("fact", arg, llm.active_course or "")
        console.print(f"[green]✓ Hafızaya kaydedildi: {arg}[/green]")

    elif command in ("/unut", "/forget"):
        if not arg or not arg.isdigit():
            console.print("[yellow]Kullanım: /unut <hafıza_id>[/yellow]")
            console.print("[dim]/hafıza komutuyla ID'leri görebilirsin[/dim]")
            return
        llm.forget_memory(int(arg))
        console.print(f"[green]✓ Hafıza #{arg} silindi.[/green]")

    elif command in ("/profil", "/profile"):
        profile_path = llm.get_profile_path()
        console.print(f"[cyan]Profil dosyası: {profile_path}[/cyan]")
        console.print("[dim]Bu dosyayı herhangi bir editörle düzenleyebilirsin.[/dim]")
        console.print("[dim]İçindeki bilgiler her chat turunda system prompt'a eklenir.[/dim]")
        console.print(f"\n[dim]Hızlı görüntüleme:[/dim]")
        try:
            from pathlib import Path
            content = Path(profile_path).read_text()
            console.print(Markdown(content))
        except Exception:
            pass

    elif command in ("/ara", "/search"):
        if not arg:
            console.print("[yellow]Kullanım: /ara <aranacak kelime>[/yellow]")
            return
        results = llm.mem_manager.db.search_messages(arg, limit=10)
        if not results:
            console.print(f"[dim]'{arg}' için sonuç bulunamadı.[/dim]")
            return
        for r in results:
            role = "🧑" if r["role"] == "user" else "🤖"
            ts = r["timestamp"][:16]
            preview = r["content"][:100].replace("\n", " ")
            console.print(f"  {role} [{ts}] {preview}...")

    elif command in ("/temizle", "/clear"):
        llm.reset_conversation()
        console.print("[green]✓ Sohbet geçmişi temizlendi.[/green]")

    else:
        console.print(f"[yellow]Bilinmeyen komut: {command}[/yellow]")


def cmd_summary(args):
    """Generate weekly summaries for all or specific courses."""
    moodle, processor, vector_store, llm, sync = build_components()

    if not moodle.connect():
        return

    courses = moodle.get_courses()

    # Filter to specific course if provided
    if args.course:
        courses = [c for c in courses if args.course.lower() in c.fullname.lower()]
        if not courses:
            console.print(f"[red]Course not found: {args.course}[/red]")
            return

    for course in courses:
        console.print(Panel(f"📝 {course.fullname}", style="bold blue"))

        sections = moodle.get_course_content(course.id)
        topics_text = moodle.get_course_topics_text(course)

        with console.status("[bold green]Özet oluşturuluyor...[/bold green]"):
            summary = llm.generate_weekly_summary(
                course_name=course.fullname,
                section_name="Full Course",
                section_content=topics_text,
            )

        console.print(Markdown(summary))
        console.print()


def cmd_web(args):
    """Launch Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        console.print("[red]Gradio is not installed. Run: pip install gradio[/red]")
        return

    _, _, vector_store, llm, _ = build_components()

    stats = vector_store.get_stats()
    courses = stats.get("courses", [])

    def respond(message, history, course_filter):
        """Gradio chat handler."""
        if course_filter and course_filter != "Tümü":
            llm.set_active_course(course_filter)
        else:
            llm.clear_course_filter()

        messages = []
        for user_msg, bot_msg in (history or []):
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})
        return llm.chat_with_history(messages=messages[-10:])

    def generate_summary(course):
        if not course or course == "Tümü":
            return "Lütfen bir kurs seçin."
        chunks = vector_store.query(f"{course} overview", n_results=15, course_filter=course)
        context = "\n\n".join(c["text"] for c in chunks)
        return llm.generate_weekly_summary(course, "All Sections", context)

    def generate_questions(topic, course):
        c = course if course != "Tümü" else None
        return llm.generate_practice_questions(topic, c)

    # Build Gradio UI
    with gr.Blocks(title="🎓 Moodle AI Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎓 Moodle AI Assistant")
        gr.Markdown(f"*{stats.get('total_chunks', 0)} chunks indexed from {stats.get('unique_courses', 0)} courses*")

        with gr.Tab("💬 Sohbet"):
            course_dd = gr.Dropdown(
                choices=["Tümü"] + courses,
                value="Tümü",
                label="Kurs Filtresi",
            )
            chatbot = gr.ChatInterface(
                fn=respond,
                additional_inputs=[course_dd],
                title="",
                retry_btn="Tekrar Dene",
                undo_btn="Geri Al",
                clear_btn="Temizle",
            )

        with gr.Tab("📝 Haftalık Özet"):
            summary_course = gr.Dropdown(choices=["Tümü"] + courses, label="Kurs")
            summary_btn = gr.Button("Özet Oluştur", variant="primary")
            summary_output = gr.Markdown()
            summary_btn.click(generate_summary, summary_course, summary_output)

        with gr.Tab("❓ Pratik Sorular"):
            q_topic = gr.Textbox(label="Konu", placeholder="Buffer overflow, SQL injection, etc.")
            q_course = gr.Dropdown(choices=["Tümü"] + courses, label="Kurs")
            q_btn = gr.Button("Soru Oluştur", variant="primary")
            q_output = gr.Markdown()
            q_btn.click(generate_questions, [q_topic, q_course], q_output)

    console.print(f"[green]🌐 Web UI starting on http://localhost:{args.port}[/green]")
    demo.launch(server_port=args.port, share=args.share)


# ─── CLI Parser ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎓 Moodle AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # sync
    p_sync = sub.add_parser("sync", help="Sync Moodle content to vector store")
    p_sync.add_argument("--force", action="store_true", help="Force re-sync all files")
    p_sync.set_defaults(func=cmd_sync)

    # courses
    p_courses = sub.add_parser("courses", help="List enrolled courses")
    p_courses.add_argument("-d", "--detailed", action="store_true", help="Show course content details")
    p_courses.set_defaults(func=cmd_courses)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat with RAG")
    p_chat.set_defaults(func=cmd_chat)

    # summary
    p_summary = sub.add_parser("summary", help="Generate weekly summaries")
    p_summary.add_argument("-c", "--course", help="Specific course name (partial match)")
    p_summary.set_defaults(func=cmd_summary)

    # web
    p_web = sub.add_parser("web", help="Launch Gradio web UI")
    p_web.add_argument("-p", "--port", type=int, default=7860)
    p_web.add_argument("--share", action="store_true", help="Create public link")
    p_web.set_defaults(func=cmd_web)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
