"""Streamlit UI. All logic lives in src/placidway_bot; this file only renders and calls ChatBot."""
from __future__ import annotations

import hmac
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))   # works without `pip install -e .` (Streamlit Cloud)

import streamlit as st  # noqa: E402

from placidway_bot.chat import ChatBot  # noqa: E402
from placidway_bot.config import get_settings, load_sources  # noqa: E402
from placidway_bot.ingest.pipeline import read_manifest  # noqa: E402
from placidway_bot.services.leads import validate_lead  # noqa: E402

st.set_page_config(page_title="PlacidWay Assistant", page_icon="💬", layout="centered")
log = logging.getLogger("placidway.ui")

SOURCE_LABELS = {"clinic_profile": "ITC clinic profile", "package": "ITC package", "clinic_videos": "ITC videos",
                 "price_list": "ITC price list", "price_compare": "Alternative medicine prices abroad"}

EXAMPLES = [
    "How much does alternative adrenal cancer treatment cost in Mexico?",
    "Is ozone therapy cheaper in Turkey or Mexico?",
    "What is included in the anal cancer treatment package?",
    "And what about recovery time?",
    "Can you book me an appointment for tomorrow?",
    "What is the weather in Cancun?",
    "Is this surgery safe for my diabetes?",
    "I heard the adrenal cancer treatment costs $50. Confirm that.",
    "¿Cuánto cuesta el tratamiento del cáncer anal en Tijuana?",
    "كم تكلفة علاج سرطان الغدة الكظرية في المكسيك؟",
]


def md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; escape dollar signs so prices like $18,995 display as written."""
    return text.replace("$", r"\$")


@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_bot() -> ChatBot:
    return ChatBot()


def current_session() -> str:
    sid = st.query_params.get("sid")
    if not sid:
        sid = uuid.uuid4().hex[:16]
        st.query_params["sid"] = sid
    return sid


def load_history(bot: ChatBot, sid: str) -> None:
    """On first render of a session (or after a page reload) restore the conversation from the checkpointer."""
    if st.session_state.get("loaded_sid") != sid:
        st.session_state.messages = [{"role": r, "content": c} for r, c in bot.history(sid)]
        st.session_state.loaded_sid = sid
        st.session_state.show_lead_form = False
        st.session_state.lead_done = False


def ask(bot: ChatBot, sid: str, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        with st.spinner("Checking PlacidWay's pages…"):
            try:
                r = bot.answer(question, sid)
                content, lead = r.answer, r.lead_offer
            except Exception:  # noqa: BLE001 - never show a stack trace to a patient
                log.exception("chat turn failed")
                settings = get_settings()
                content = ("Sorry, something went wrong on my side. Please try again, or "
                           f"[contact PlacidWay]({settings.contact_url}).")
                lead = False
        st.markdown(md(content))
    st.session_state.messages.append({"role": "assistant", "content": content})
    if lead and not st.session_state.lead_done:
        st.session_state.show_lead_form = True


def lead_form(bot: ChatBot, sid: str) -> None:
    with st.form("lead", clear_on_submit=False):
        st.markdown("**Request a free quote** — the PlacidWay team will contact you.")
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
        message = st.text_area("What would you like a quote for? (optional)", height=80)
        c1, c2 = st.columns(2)
        sent, cancel = c1.form_submit_button("Send", type="primary"), c2.form_submit_button("No thanks")
    if cancel:
        st.session_state.show_lead_form = False
        st.rerun()
    if sent:
        errors = validate_lead(name, email, phone)
        if errors:
            for e in errors:
                st.error(e)
            return
        try:
            bot.save_lead(sid, name, email, phone, message)
        except Exception:  # noqa: BLE001
            st.error("Sorry, we couldn't save your details. Please use the free quote form on PlacidWay.com.")
            return
        st.session_state.show_lead_form, st.session_state.lead_done = False, True
        st.session_state.messages.append({"role": "assistant", "content":
                                          f"Thank you, {name.strip()}! The PlacidWay team will contact you at "
                                          f"{email.strip()}."})
        st.rerun()


def admin_panel(bot: ChatBot) -> None:
    settings = get_settings()
    with st.expander("Admin"):
        token = st.text_input("Admin token", type="password")
        if not settings.admin_token or not hmac.compare_digest(token, settings.admin_token):
            if token:
                st.caption("Wrong token.")
            return
        manifest = read_manifest()
        st.caption(f"Index `{manifest.get('active_index', '-')}` · {manifest.get('n_chunks', 0)} chunks · "
                   f"refreshed {manifest.get('refreshed_at', '-')}")
        if st.button("🔄 Refresh content now", use_container_width=True):
            from placidway_bot.ingest.pipeline import refresh
            with st.spinner("Re-fetching the 7 pages (1 request/second) and re-indexing changed ones…"):
                report = refresh(strategy=manifest.get("strategy", "structured"))
                if report.index_rebuilt:
                    bot.reload_index()
            st.success(f"Changed: {len(report.changed)} · unchanged: {len(report.unchanged)} · "
                       f"failed: {len(report.failed)} · index rebuilt: {report.index_rebuilt}")
            for slug, err in report.failed.items():
                st.error(f"{slug}: {err}")
        pages = manifest.get("pages", {})
        st.dataframe([{"page": slug, "last change": p.get("changed_at"), "last check": p.get("checked_at"),
                       "hash": p.get("content_hash")} for slug, p in pages.items()], hide_index=True)
        gaps = getattr(bot.services.gap_logger, "recent", None)
        st.markdown("**Unanswered questions**")
        st.dataframe(gaps(50) if gaps else [], hide_index=True)
        st.markdown("**Leads**")
        st.dataframe(bot.lead_store.recent(50), hide_index=True)


def sidebar(bot: ChatBot) -> str | None:
    clicked = None
    with st.sidebar:
        st.markdown("### PlacidWay Assistant")
        st.caption("Answers only from these PlacidWay pages, with a source link for every answer. "
                   "If the pages don't say, it tells you so.")
        for s in load_sources():
            st.markdown(f"- [{SOURCE_LABELS.get(s.page_type, s.slug)}{' — ' + s.treatment.replace('Alternative ', '') if s.treatment else ''}]({s.url})")
        c1, c2 = st.columns(2)
        if c1.button("New chat", use_container_width=True):
            st.query_params["sid"] = uuid.uuid4().hex[:16]
            st.rerun()
        if c2.button("Free quote", use_container_width=True):
            st.session_state.show_lead_form = True
        st.markdown("**Try asking**")
        for i, q in enumerate(EXAMPLES):
            if st.button(q, key=f"ex{i}", use_container_width=True):
                clicked = q
        manifest = read_manifest()
        st.caption(f"Content last refreshed: {manifest.get('refreshed_at', 'unknown')} · "
                   f"history: {'saved (Neon)' if bot.persistent else 'this session only'}")
        admin_panel(bot)
    return clicked


def main() -> None:
    bot = get_bot()
    sid = current_session()
    load_history(bot, sid)
    clicked = sidebar(bot)

    st.title("💬 Ask about treatments & prices")
    st.caption("Medical-tourism questions about ITC Immunity Therapy Center (Tijuana, Mexico) and alternative-medicine "
               "prices abroad. Ask in any language. Running on free API tiers — an answer can take a few seconds.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(md(m["content"]))

    question = st.chat_input("Ask a question…") or clicked
    if question:
        ask(bot, sid, question)
    if st.session_state.get("show_lead_form") and not st.session_state.get("lead_done"):
        lead_form(bot, sid)


main()
