# app/ui/gradio_app.py  (replace the whole file with this version)
import gradio as gr
from pathlib import Path

from app.graph.rag_graph import build_rag_graph
from app.rag.ingest import check_and_ingest_stream, list_tracked_files, save_uploaded_pdfs

_rag = build_rag_graph()

def _format_sources(sources):
    if not sources:
        return ""
    parts = [f"{s['source']} (p.{s['page']})" for s in sources]
    return "\n\nSources:\n- " + "\n- ".join(parts)

def chat_reply(message, history, selected_sources):
    pairs = [(u or "", a or "") for (u, a) in history]
    out = _rag.invoke({"query": message, "history": pairs, "allowed_sources": selected_sources})
    ans = out.get("answer", "").strip()
    srcs = _format_sources(out.get("sources", []))
    return (ans + ("\n\n" + srcs if srcs else "")).strip()

# ---------- Load (progress + final enable) ----------
def on_load():
    raw_dir = Path("data/raw_pdfs")
    # Stream progress: keep controls locked
    for line in check_and_ingest_stream(raw_dir):
        yield (
            gr.update(value=line),        # status
            gr.update(interactive=False), # textbox
            gr.update(interactive=False), # checkbox
            gr.update(interactive=False), # manage button
            [],                           # selected_state (filenames)
            [],                           # files_state
        )
    # Ready → populate checkbox with ALL + files, default select-all
    files = list_tracked_files()
    choices = ["ALL"] + files
    visible_val = ["ALL"] + files
    selected_state = files[:]   # filenames only
    files_state = files[:]
    yield (
        gr.update(value="✅ Ready. You can start chatting."),
        gr.update(interactive=True),
        gr.update(choices=choices, value=visible_val, interactive=True),
        gr.update(interactive=True),
        selected_state,
        files_state,
    )

# ---------- Manager helpers ----------
def refresh_file_list():
    files = list_tracked_files()
    text = "### Loaded PDFs\n" + ("\n".join(f"- {f}" for f in files) if files else "_No PDFs found_")
    return text, files

def toggle_manager(current_visible):
    new_visible = not current_visible
    # Control visibility of list + refresh + upload together
    return new_visible, gr.update(visible=new_visible), gr.update(visible=new_visible), gr.update(visible=new_visible)

def do_upload(files):
    if not files:
        # no change
        return gr.update(), gr.update(), gr.update(), [], [], gr.update()
    tmp_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    _ = save_uploaded_pdfs(tmp_paths)

    raw_dir = Path("data/raw_pdfs")
    msgs = []
    for line in check_and_ingest_stream(raw_dir):
        msgs.append(line)

    files_now = list_tracked_files()
    choices = ["ALL"] + files_now
    visible_val = ["ALL"] + files_now
    list_text = "### Loaded PDFs\n" + ("\n".join(f"- {f}" for f in files_now) if files_now else "_No PDFs found_")
    status_msg = ("\n".join(msgs) + "\n✅ Upload + indexing complete.") if msgs else "✅ Upload + indexing complete."
    # After upload, default back to ALL → enable textbox
    return (
        gr.update(value=status_msg),
        gr.update(value=list_text),
        gr.update(choices=choices, value=visible_val),
        files_now,              # files_state
        files_now,              # selected_state (filenames)
        gr.update(interactive=True),  # txt enable
    )

# ---------- ALL logic with previous state ----------
def handle_select_change(visible_vals, files, prev_had_all):
    """
    visible_vals: items checked in the CheckboxGroup (may include 'ALL')
    files: list of filenames
    prev_had_all: bool (was ALL checked previously?)
    Returns: (checkbox_visible_value, selected_files_state, new_had_all_bool)
    """
    files_set = set(files)
    sel = set([s for s in (visible_vals or []) if s != "ALL"])
    has_all = "ALL" in (visible_vals or [])

    if has_all:
        # user checked ALL -> select everything and keep ALL checked
        return ["ALL"] + files, files[:], True

    # If ALL was previously on and user just unticked it -> deselect everything
    if prev_had_all and not has_all:
        return [], [], False

    # If user manually selected all PDFs -> auto-check ALL
    if sel == files_set and files:
        return ["ALL"] + files, files[:], True

    # Partial selection
    return list(sel), list(sel), False

def build_app():
    with gr.Blocks(title="RAGBot — LangGraph", theme="soft") as demo:
        gr.Markdown("## RAGBot — LangGraph\nGrounded answers over your PDFs. The app checks and (re)indexes on first load.")

        status = gr.Markdown("⏳ Preparing...", elem_id="status")

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

        # Chat
        chatbot = gr.Chatbot(height=460, type="tuples")  # explicit format
        txt = gr.Textbox(placeholder="Ask me about your PDFs...", interactive=False)

        # Hidden states
        selected_state = gr.State([])    # filenames only (no 'ALL')
        files_state = gr.State([])       # available filenames
        had_all_state = gr.State(True)   # tracks if ALL was checked previously

        # Load: stream progress then enable and set defaults
        demo.load(
            fn=on_load,
            outputs=[status, txt, select_all_and_files, manage_toggle, selected_state, files_state],
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
                gr.update(value=text),          # pdf_list
                files,                          # files_state
                gr.update(choices=["ALL"] + files, value=["ALL"] + files),  # checkbox
                files,                          # selected_state
                True,                           # had_all_state
                gr.update(interactive=True),    # txt enable
                gr.update(value="✅ Ready. You can start chatting."),
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

        # Upload → re-index and reset selection to ALL (also enable textbox)
        upload.upload(
            fn=do_upload,
            inputs=[upload],
            outputs=[status, pdf_list, select_all_and_files, files_state, selected_state, txt],
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

    return demo
