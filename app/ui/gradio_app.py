# app/ui/gradio_app.py
import json
from pathlib import Path
import gradio as gr

from app.graph.rag_graph import build_rag_graph
from app.rag.ingest import check_and_ingest_stream, list_tracked_files, save_uploaded_pdfs
from app.extract.structured import extract_to_json
from app.extract.cv_detect import detect_cv

# Build graph once
_rag = build_rag_graph()


# ---------- Helpers ----------
def _format_sources(sources):
    if not sources:
        return ""
    parts = [f"{s['source']} (p.{s['page']})" for s in sources]
    return "\n\nSources:\n- " + "\n- ".join(parts)


def chat_reply(message, history, selected_sources):
    """
    Invoke the graph with the user message + current history + selected PDFs list.
    """
    pairs = [(u or "", a or "") for (u, a) in history]
    out = _rag.invoke({"query": message, "history": pairs, "allowed_sources": selected_sources})
    ans = (out.get("answer") or "").strip()
    srcs = _format_sources(out.get("sources", []))
    return (ans + ("\n\n" + srcs if srcs else "")).strip()


def _bar_html(i: int, n: int) -> str:
    i = max(0, int(i))
    n = max(1, int(n))
    pct = int(round(100 * i / n))
    return (
        f"<progress max='{n}' value='{i}' style='width:100%'></progress>"
        f"<div style='text-align:right; font-size:0.9em'>{pct}%</div>"
    )


# ---------- Load (stream re-indexing with progress, then enable input) ----------
def on_load():
    """
    On app load, (re)ingest PDFs if needed and stream progress lines while keeping controls locked.
    After finishing, populate the checkbox with ALL + files, and enable the textbox.
    """
    raw_dir = Path("data/raw_pdfs")

    # Stream progress: keep controls locked during ingestion
    for upd in check_and_ingest_stream(raw_dir):
        # Support both: dict progress ({msg, file, i, n}) or plain str lines
        if isinstance(upd, dict):
            msg = upd.get("msg", "")
            cur_file = upd.get("file") or ""
            i = upd.get("i", 0)
            n = upd.get("n", 1)
            bar = _bar_html(i, n)
            file_line = f"**Ingesting:** `{cur_file}` ({i}/{n})" if cur_file else f"({i}/{n})"
            yield (
                gr.update(value=f"🔄 {msg}"),        # status
                gr.update(interactive=False),        # txt
                gr.update(interactive=False),        # checkbox
                gr.update(interactive=False),        # manage button
                [],                                  # selected_state
                [],                                  # files_state
                gr.update(value=bar, visible=True),  # ingest progress bar
                gr.update(value=file_line, visible=True),  # ingest current file
            )
        else:
            # Fallback for string-only updates
            yield (
                gr.update(value=str(upd)),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                [],
                [],
                gr.update(),   # leave bar as-is
                gr.update(),   # leave file label as-is
            )

    # Ready → populate checkbox with ALL + current files and enable typing
    files = list_tracked_files()
    choices = ["ALL"] + files
    visible_val = ["ALL"] + files

    selected_state = files[:]   # filenames only (no 'ALL' in state)
    files_state = files[:]

    # Set bar to 100% and freeze
    bar_done = _bar_html(1, 1)
    final_file_line = "Done."

    yield (
        gr.update(value="✅ Ready. You can start chatting."),
        gr.update(interactive=True),
        gr.update(choices=choices, value=visible_val, interactive=True),
        gr.update(interactive=True),
        selected_state,
        files_state,
        gr.update(value=bar_done, visible=True),
        gr.update(value=final_file_line, visible=True),
    )


# ---------- Manager helpers ----------
def refresh_file_list():
    files = list_tracked_files()
    text = "### Loaded PDFs\n" + ("\n".join(f"- {f}" for f in files) if files else "_No PDFs found_")
    return text, files


def toggle_manager(current_visible):
    """
    Toggle visibility of the manager subpanel (list + refresh + upload).
    """
    new_visible = not current_visible
    return new_visible, gr.update(visible=new_visible), gr.update(visible=new_visible), gr.update(visible=new_visible)


# Streaming upload + re-index with progress
def do_upload_stream(files):
    """
    Save uploaded PDFs, stream re-index progress, then reset selection to ALL and enable typing.
    Yields updates so progress bar and current file are live during ingestion.
    """
    if not files:
        # No changes
        yield gr.update(), gr.update(), gr.update(), [], [], gr.update(), gr.update(), gr.update()
        return

    # Save temp files to data/raw_pdfs
    tmp_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    _ = save_uploaded_pdfs(tmp_paths)

    raw_dir = Path("data/raw_pdfs")

    # Stream re-index
    msgs = []
    for upd in check_and_ingest_stream(raw_dir):
        if isinstance(upd, dict):
            msg = upd.get("msg", "")
            cur_file = upd.get("file") or ""
            i = upd.get("i", 0)
            n = upd.get("n", 1)
            bar = _bar_html(i, n)
            file_line = f"**Ingesting:** `{cur_file}` ({i}/{n})" if cur_file else f"({i}/{n})"
            msgs.append(msg)
            # Partial updates: status, list (unchanged yet), checkbox (unchanged), files_state (unchanged),
            # selected_state (unchanged), txt (disabled), bar, file label
            yield (
                gr.update(value=f"🔄 {msg}"),
                gr.update(), gr.update(), [], [], gr.update(interactive=False),
                gr.update(value=bar, visible=True),
                gr.update(value=file_line, visible=True),
            )
        else:
            msgs.append(str(upd))
            yield (
                gr.update(value=f"🔄 {upd}"),
                gr.update(), gr.update(), [], [], gr.update(interactive=False),
                gr.update(), gr.update(),
            )

    # After indexing, refresh lists & defaults
    files_now = list_tracked_files()
    choices = ["ALL"] + files_now
    visible_val = ["ALL"] + files_now
    list_text = "### Loaded PDFs\n" + ("\n".join(f"- {f}" for f in files_now) if files_now else "_No PDFs found_")
    status_msg = ("\n".join(msgs) + "\n✅ Upload + indexing complete.") if msgs else "✅ Upload + indexing complete."
    bar_done = _bar_html(1, 1)

    # Final state: defaults to ALL, enable textbox, bar 100%
    yield (
        gr.update(value=status_msg),
        gr.update(value=list_text),
        gr.update(choices=choices, value=visible_val),
        files_now,              # files_state
        files_now,              # selected_state (filenames)
        gr.update(interactive=True),  # txt enable
        gr.update(value=bar_done, visible=True),
        gr.update(value="Done.", visible=True),
    )


# ---------- ALL logic with previous state ----------
def handle_select_change(visible_vals, files, prev_had_all):
    """
    visible_vals: items checked in the CheckboxGroup (may include 'ALL')
    files: list of filenames
    prev_had_all: bool (was ALL checked previously?)
    Returns: (checkbox_visible_value, selected_files_state, new_had_all_bool)
    """
    files_set = set(files or [])
    sel = set([s for s in (visible_vals or []) if s != "ALL"])
    has_all = "ALL" in (visible_vals or [])

    if has_all:
        # User checked ALL -> select everything and keep ALL checked
        return ["ALL"] + files, files[:], True

    # If ALL was previously on and user just unticked it -> deselect everything
    if prev_had_all and not has_all:
        return [], [], False

    # If user manually selected all PDFs -> auto-check ALL
    if sel == files_set and files:
        return ["ALL"] + files, files[:], True

    # Partial selection
    visible = list(sel)
    return visible, list(sel), False


# ---------- Structured extraction (UI-driven) ----------
def run_extract(selected_files, history):
    """
    Disable input, process selected PDFs, stream progress, then post a chat summary.
    Yields (chatbot_history, status_markdown, textbox_update, progress_markdown).
    """
    raw_dir = Path("data/raw_pdfs")
    out_dir = Path("data/structured/cv")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nothing selected → keep chat, show warning, keep input disabled (until user selects)
    if not selected_files:
        yield history, gr.update(value="⚠️ Select at least one document to extract."), gr.update(interactive=False), gr.update(value="Idle.")
        return

    total = len(selected_files)
    done = 0
    hits = 0
    lines = []

    # Lock input
    yield history, gr.update(value="⏳ Extracting…"), gr.update(interactive=False), gr.update(value="Starting…")

    for fname in selected_files:
        done += 1
        pdf = raw_dir / fname
        if not pdf.exists():
            yield history, gr.update(value=f"⚠️ Missing file: {fname}"), gr.update(interactive=False), gr.update(value=f"{done}/{total}")
            continue

        det = detect_cv(pdf)
        if not det.get("is_cv", False):
            yield history, gr.update(value=f"⏭️ Not a CV: {fname} ({done}/{total})"), gr.update(interactive=False), gr.update(value=f"{done}/{total}")
            continue

        out = extract_to_json(pdf, schema_name="cv_standard", out_dir=out_dir, debug=False)
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        name = data.get("name") or "N/A"
        email = data.get("email") or "N/A"
        edu_count = len(data.get("education") or [])
        lines.append(f"- **{fname}** — {name} — {email} — education entries: {edu_count}")
        hits += 1

        yield history, gr.update(value=f"✅ {fname} ({done}/{total})"), gr.update(interactive=False), gr.update(value=f"{done}/{total}")

    # Build a bot message with the summary
    summary = (
        f"Structured extraction finished.\n\n"
        f"Processed: {total}\n"
        f"Extracted CVs: {hits}\n"
        f"Output folder: data/structured/cv\n\n"
        + ("\n".join(lines) if lines else "(No CVs detected among the selected files.)")
    )
    new_hist = history + [["[extract]", summary]]

    # Unlock input
    yield new_hist, gr.update(value="✅ Extraction complete."), gr.update(interactive=True), gr.update(value="Done.")


# ---------- App ----------
def build_app():
    with gr.Blocks(title="RAGBot — LangGraph", theme="soft") as demo:
        gr.Markdown("## RAGBot — LangGraph\nGrounded answers over your PDFs. The app checks and (re)indexes on first load.")

        status = gr.Markdown("⏳ Preparing...", elem_id="status")

        # NEW: Ingestion progress bar + current file
        ingest_bar = gr.HTML(_bar_html(0, 100), visible=True)
        ingest_file = gr.Markdown("", visible=True)

        # Selection row (always visible)
        select_all_and_files = gr.CheckboxGroup(
            label="Use PDFs",
            choices=["ALL"],
            value=["ALL"],
            interactive=False,
        )

        # Manager (hidden panel toggled by a button)
        manage_visible = gr.State(False)
        manage_toggle = gr.Button("📂 Manage PDFs (show/hide)", interactive=False)
        pdf_list = gr.Markdown(visible=False)
        refresh_btn = gr.Button("🔄 Refresh list", visible=False)
        upload = gr.Files(label="➕ Add PDF(s)", file_types=[".pdf"], file_count="multiple", visible=False)

        # --- Structured extraction (CVs) panel ---
        with gr.Accordion("Structured extraction (CVs)", open=False):
            gr.Markdown("Run CV detection + JSON extraction on the currently selected PDFs.")
            extract_btn = gr.Button("Extract to JSON", variant="primary")
            extract_progress = gr.Markdown("Idle.")
            gr.Markdown("**Output folder:** `data/structured/cv`")

        # Chat
        chatbot = gr.Chatbot(height=460, type="tuples")  # explicit format to avoid future default changes
        txt = gr.Textbox(placeholder="Ask me about your PDFs...", interactive=False)

        # Hidden states
        selected_state = gr.State([])    # filenames only (no 'ALL')
        files_state = gr.State([])       # available filenames
        had_all_state = gr.State(True)   # tracks if ALL was checked previously

        # Load: stream progress then enable and set defaults
        demo.load(
            fn=on_load,
            outputs=[
                status, txt, select_all_and_files, manage_toggle,
                selected_state, files_state, ingest_bar, ingest_file
            ],
        )

        # Toggle manager visibility
        manage_toggle.click(
            fn=toggle_manager,
            inputs=[manage_visible],
            outputs=[manage_visible, pdf_list, refresh_btn, upload],
        )

        # Refresh list → defaults to ALL and enables textbox
        def _refresh():
            text, files = refresh_file_list()
            return (
                gr.update(value=text),                                  # pdf_list
                files,                                                  # files_state
                gr.update(choices=["ALL"] + files, value=["ALL"] + files),  # checkbox
                files,                                                  # selected_state
                True,                                                   # had_all_state
                gr.update(interactive=True),                            # txt enable
                gr.update(value="✅ Ready. You can start chatting."),    # status
            )

        refresh_btn.click(
            _refresh,
            outputs=[pdf_list, files_state, select_all_and_files, selected_state, had_all_state, txt, status],
        )

        # Checkbox change → enforce ALL logic + gate textbox
        def _on_select_change(visible_vals, files, prev_had_all):
            checkbox_val, selected_files, new_had_all = handle_select_change(visible_vals, files, prev_had_all)
            has_selection = bool(selected_files)
            txt_update = gr.update(interactive=has_selection)
            status_update = gr.update(value=("✅ Ready. You can start chatting." if has_selection else "⚠️ Select at least one document to ask."))
            return gr.update(value=checkbox_val), selected_files, new_had_all, txt_update, status_update

        select_all_and_files.change(
            _on_select_change,
            inputs=[select_all_and_files, files_state, had_all_state],
            outputs=[select_all_and_files, selected_state, had_all_state, txt, status],
        )

        # Upload → stream re-index progress and reset selection to ALL (also enable textbox)
        upload.upload(
            fn=do_upload_stream,
            inputs=[upload],
            outputs=[status, pdf_list, select_all_and_files, files_state, selected_state, txt, ingest_bar, ingest_file],
        )

        # Enter to submit (no Send button) — block when no selection
        def on_submit(user_msg, history, selected_files):
            if not selected_files:
                # Block send + warn
                return "", history, gr.update(value="⚠️ Select at least one document to ask.")
            history = history + [[user_msg, None]]
            answer = chat_reply(user_msg, history, selected_files)
            history[-1][1] = answer
            return "", history, gr.update(value="✅ Ready. You can start chatting.")

        txt.submit(on_submit, [txt, chatbot, selected_state], [txt, chatbot, status])

        # Extraction button (streams progress; disables txt during run)
        extract_btn.click(
            fn=run_extract,
            inputs=[selected_state, chatbot],
            outputs=[chatbot, status, txt, extract_progress],
        )

    return demo
