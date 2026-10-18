import html
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tasks",
    page_icon="✅",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

.block-container {
    max-width: 720px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 20px;
    padding: 28px 32px;
    color: #ffffff;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    margin-bottom: 20px;
}
.app-header h1 {
    margin: 0;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    padding: 0;
}
.app-header p {
    margin: 4px 0 0 0;
    opacity: 0.85;
    font-size: 0.95rem;
}

/* Stat cards */
.stat-row { display: flex; gap: 12px; margin-bottom: 18px; }
.stat-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.stat-card .num { font-size: 1.6rem; font-weight: 700; color: #111827; line-height: 1.1; }
.stat-card .lbl { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; margin-top: 2px; }

/* Progress bar */
.progress-wrap { margin-bottom: 22px; }
.progress-meta { display: flex; justify-content: space-between; font-size: 0.8rem; color: #6b7280; margin-bottom: 6px; font-weight: 500; }
.progress-track { height: 8px; background: #e0e7ff; border-radius: 999px; overflow: hidden; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #4f46e5, #7c3aed); border-radius: 999px; transition: width 0.4s ease; }

/* Add form */
div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px 16px 8px 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    margin-bottom: 18px;
}
div[data-testid="stTextInput"] input {
    border-radius: 10px;
    border: 1.5px solid #e5e7eb;
    padding: 10px 14px;
    font-size: 0.95rem;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}

/* Buttons */
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1.5px solid #e5e7eb;
    transition: all 0.15s ease;
}
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
    border: none;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.3);
    color: #ffffff;
}
.stButton > button:hover {
    border-color: #6366f1;
    color: #4f46e5;
}

/* Task cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border-radius: 14px;
    border-color: #e5e7eb;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 6px 18px rgba(17, 24, 39, 0.08);
}
.task-text { font-size: 1rem; font-weight: 500; color: #111827; padding-top: 2px; word-break: break-word; }
.task-text.done { color: #9ca3af; text-decoration: line-through; }
.task-meta { font-size: 0.72rem; color: #9ca3af; margin-top: 2px; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 48px 20px;
    background: #ffffff;
    border: 2px dashed #c7d2fe;
    border-radius: 16px;
    color: #6b7280;
}
.empty-state .emoji { font-size: 2.4rem; }
.empty-state .title { font-weight: 700; color: #374151; margin-top: 6px; }

.section-label {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #6b7280; margin: 6px 0 10px 2px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "todos": [],  # list of {id, text, done, created}
        "next_id": 1,  # auto-incrementing id
        "confirm_clear": False,  # two-step "clear all" confirmation
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------------------------------------------------------------------------
# Actions (used as callbacks so state updates before the rerun renders)
# ---------------------------------------------------------------------------
def add_todo() -> None:
    text = st.session_state.get("new_todo", "").strip()
    if not text:
        st.session_state["flash"] = ("warning", "Please enter a task first.")
        return
    st.session_state.todos.insert(
        0,
        {
            "id": st.session_state.next_id,
            "text": text,
            "done": False,
            "created": datetime.now(),
        },
    )
    st.session_state.next_id += 1
    st.session_state["flash"] = ("success", "Task added.")


def toggle_todo(todo_id: int) -> None:
    for todo in st.session_state.todos:
        if todo["id"] == todo_id:
            todo["done"] = st.session_state[f"chk_{todo_id}"]
            break


def delete_todo(todo_id: int) -> None:
    st.session_state.todos = [t for t in st.session_state.todos if t["id"] != todo_id]
    st.session_state.pop(f"chk_{todo_id}", None)


def clear_completed() -> None:
    for t in st.session_state.todos:
        if t["done"]:
            st.session_state.pop(f"chk_{t['id']}", None)
    st.session_state.todos = [t for t in st.session_state.todos if not t["done"]]


def request_clear_all() -> None:
    st.session_state.confirm_clear = True


def cancel_clear_all() -> None:
    st.session_state.confirm_clear = False


def clear_all() -> None:
    for t in st.session_state.todos:
        st.session_state.pop(f"chk_{t['id']}", None)
    st.session_state.todos = []
    st.session_state.confirm_clear = False


# ---------------------------------------------------------------------------
# Derived data
# ---------------------------------------------------------------------------
todos = st.session_state.todos
total = len(todos)
completed = sum(1 for t in todos if t["done"])
remaining = total - completed
pct = int(completed / total * 100) if total else 0

# ---------------------------------------------------------------------------
# Header + stats + progress
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>✅ My Tasks</h1>
        <p>{datetime.now().strftime('%A, %B %d')} &middot; Stay focused, get things done.</p>
    </div>

    <div class="stat-row">
        <div class="stat-card"><div class="num">{total}</div><div class="lbl">Total</div></div>
        <div class="stat-card"><div class="num">{remaining}</div><div class="lbl">Remaining</div></div>
        <div class="stat-card"><div class="num">{completed}</div><div class="lbl">Completed</div></div>
    </div>

    <div class="progress-wrap">
        <div class="progress-meta"><span>Progress</span><span>{pct}%</span></div>
        <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Add task form
# ---------------------------------------------------------------------------
with st.form("add_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1.2], vertical_alignment="bottom")
    with col_input:
        st.text_input(
            "New task",
            key="new_todo",
            placeholder="What needs to be done?",
            label_visibility="collapsed",
        )
    with col_btn:
        st.form_submit_button("Add", on_click=add_todo, use_container_width=True)

flash = st.session_state.pop("flash", None)
if flash:
    kind, message = flash
    (st.success if kind == "success" else st.warning)(
        message, icon="✅" if kind == "success" else "⚠️"
    )

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
view = st.radio(
    "Filter",
    ["All", "Active", "Completed"],
    horizontal=True,
    label_visibility="collapsed",
)

if view == "Active":
    visible = [t for t in todos if not t["done"]]
elif view == "Completed":
    visible = [t for t in todos if t["done"]]
else:
    visible = todos

# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="section-label">{view} &middot; {len(visible)}</div>',
    unsafe_allow_html=True,
)

if not visible:
    if total == 0:
        title, sub, emoji = (
            "No tasks yet",
            "Add your first task above to get started.",
            "📝",
        )
    elif view == "Active":
        title, sub, emoji = "All caught up!", "Nothing left to do. Nice work.", "🎉"
    else:
        title, sub, emoji = (
            "Nothing completed yet",
            "Finish a task and it will show up here.",
            "⏳",
        )
    st.markdown(
        f'<div class="empty-state"><div class="emoji">{emoji}</div>'
        f'<div class="title">{title}</div><div>{sub}</div></div>',
        unsafe_allow_html=True,
    )
else:
    for todo in visible:
        with st.container(border=True):
            c_check, c_text, c_del = st.columns(
                [0.6, 6, 1], vertical_alignment="center"
            )

            with c_check:
                st.checkbox(
                    "Done",
                    value=todo["done"],
                    key=f"chk_{todo['id']}",
                    on_change=toggle_todo,
                    args=(todo["id"],),
                    label_visibility="collapsed",
                )

            with c_text:
                safe_text = html.escape(todo["text"])
                css_class = "task-text done" if todo["done"] else "task-text"
                created = todo["created"].strftime("%b %d, %H:%M")
                st.markdown(
                    f'<div class="{css_class}">{safe_text}</div>'
                    f'<div class="task-meta">Added {created}</div>',
                    unsafe_allow_html=True,
                )

            with c_del:
                st.button(
                    "🗑️",
                    key=f"del_{todo['id']}",
                    on_click=delete_todo,
                    args=(todo["id"],),
                    help="Delete task",
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------
if total:
    st.write("")
    if st.session_state.confirm_clear:
        st.warning(f"Delete all {total} tasks? This can't be undone.", icon="⚠️")
        b1, b2 = st.columns(2)
        b1.button(
            "Yes, clear everything",
            on_click=clear_all,
            use_container_width=True,
            type="primary",
        )
        b2.button("Cancel", on_click=cancel_clear_all, use_container_width=True)
    else:
        b1, b2 = st.columns(2)
        b1.button(
            "Clear completed",
            on_click=clear_completed,
            disabled=completed == 0,
            use_container_width=True,
        )
        b2.button("Clear all", on_click=request_clear_all, use_container_width=True)
