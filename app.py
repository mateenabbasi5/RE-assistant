import streamlit as st
import uuid
import time
import re
import datetime
import json
import hashlib
from io import BytesIO
from dataclasses import dataclass, asdict

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import requests
import google.generativeai as genai
from openai import OpenAI
from github import Github

from rag import retrieve_context

# INITIAL SETUP

load_dotenv()
st.set_page_config(
    page_title="RE Assistant (Gemini + OpenAI + LLaMA + Flan-T5 + RAG)",
    layout="wide",
)

device_info = "GPU" if torch.cuda.is_available() else "CPU"

# Research Settings (Versioned)

PROMPT_VERSION = "AC_PROMPT_V4"
RAG_CONTEXT_MAX_CHARS = 6000

# Session State

def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def now_utc_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def log_event(event_type: str, payload: dict | None = None):
    """Research-grade event logging to support correctness + reproducibility."""
    if payload is None:
        payload = {}
    st.session_state.events.append(
        {"ts_utc": now_utc_iso(), "event": event_type, "payload": payload}
    )


ss_init("feedback_log", [])
ss_init("session_id", str(uuid.uuid4()))
ss_init("user_info_submitted", False)
ss_init("admin_authenticated", False)

# User-story-related state
ss_init("user_story_text", "")
ss_init("user_story_manual", "")
ss_init("candidate_user_stories", [])
ss_init("selected_user_story_index", None)

# Acceptance-criteria-related state
ss_init("criteria_by_model", {})          # model -> list[str] normalized criteria
ss_init("generated", {})                  # model -> raw output
ss_init("per_model_artifacts", {})        # model -> {raw/parsed/normalized/final_display/metadata}
ss_init("selected_criteria_entries", [])  # list[dict]
ss_init("final_selected_ac", "")          # Step 5 text_area key

# Feedback/editing state
ss_init("action", "Accept")
ss_init("edited_outputs", {})             # model -> edited text

# Research config locking + timeline
ss_init("events", [])
ss_init("study_config", None)
ss_init("config_locked", False)

# Reset helper (forces stable key-space when needed)
ss_init("reset_nonce", 0)

ss_init("pending_regen", False)
ss_init("final_auto_source", "")

# Secrets Helper

def require_secret(key: str):
    """Stop the app with a friendly message if a secret is missing."""
    try:
        return st.secrets[key]
    except KeyError:
        st.error(f"Missing secret: {key}. Please add `{key}` in Streamlit Secrets.")
        st.stop()


# GitHub helper (single-token, private submissions repo)

def get_submissions_repo():
    """
    Uses ONE token (GITHUB_TOKEN) to access the private submissions repo.
    Required secrets:
      - GITHUB_TOKEN
      - SUBMISSIONS_REPO  (e.g., "mateenabbasi5/RE-assistant-submissions")
    """
    token = require_secret("GITHUB_TOKEN")
    repo_fullname = require_secret("SUBMISSIONS_REPO")
    g = Github(token)
    return g.get_repo(repo_fullname)


# RAG Cache

@st.cache_data(show_spinner=False)
def cached_retrieve_context(query: str) -> str:
    """Cache RAG context for consistency + performance; truncate deterministically."""
    ctx = retrieve_context(query) or ""
    if len(ctx) > RAG_CONTEXT_MAX_CHARS:
        ctx = ctx[:RAG_CONTEXT_MAX_CHARS] + "\n\n[TRUNCATED CONTEXT]"
    return ctx


# Research Config (Frozen at first generation)

@dataclass
class StudyConfig:
    prompt_version: str
    device: str
    gemini_model: str
    openai_model: str
    together_model: str
    flan_model_id: str
    temperatures: dict
    max_tokens: dict
    rag_context_max_chars: int
    selected_models: list
    rag_for_model: dict
    locked_at_utc: str


# Parsing + Normalization (Shared)

def extract_numbered_items(text: str) -> list[str]:
    """Extract 1..4 items from model output with a robust regex + fallback."""
    if not text:
        return []
    matches = re.findall(
        r"\b([1-4])[\.\)]\s*(.+?)(?=\s*[1-4][\.\)]|$)",
        text.strip(),
        flags=re.DOTALL,
    )
    items = [m[1].strip() for m in matches if m[1].strip()]
    if items:
        return items

    # fallback: line-based
    out = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        l = re.sub(r"^\s*\d+[\.\)]\s*", "", l).strip()
        if l:
            out.append(l)
    return out


def normalize_criterion(c: str) -> str:
    """Normalize to: The system shall ... ."""
    c = (c or "").strip()
    c = re.sub(r"^\s*[1-4][\.\)\:\-]\s*", "", c).strip()
    c = re.sub(r"\s+", " ", c).strip()

    if not re.match(r"(?i)^the system\s+(shall|must|should|can|will)\b", c):
        if c and c[0].isupper():
            c = c[0].lower() + c[1:]
        c = "The system shall " + c

    if not c.endswith("."):
        c += "."
    return c


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Remove near-duplicates while preserving order (research consistency)."""
    seen = set()
    out = []
    for it in items:
        key = re.sub(r"[^a-z0-9 ]", " ", it.lower())
        key = re.sub(
            r"\b(the|a|an|of|to|and|for|in|on|with|shall|must|should|can|will)\b",
            " ",
            key,
        )
        key = " ".join(key.split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def postprocess_criteria(raw_output: str, user_story: str) -> dict:
    """
    Shared postprocessor for ALL models:
    - Extract criteria
    - Normalize format
    - Deduplicate
    - Enforce exactly 4
    - Produce final display string
    """
    parsed = extract_numbered_items(raw_output)
    normalized = [normalize_criterion(x) for x in parsed if x and x.strip()]
    normalized = dedupe_preserve_order(normalized)

    # Simple heuristic: drop obvious story restatement
    us = (user_story or "").strip().lower()
    us_snip = us[:60] if us else ""
    if us_snip:
        normalized = [c for c in normalized if us_snip not in c.lower()]

    fallback_pool = [
        "The system shall allow the user to complete the main task successfully.",
        "The system shall provide clear and helpful feedback after each action.",
        "The system shall validate user inputs and present meaningful error messages.",
        "The system shall handle exceptional cases gracefully and maintain stability.",
        "The system shall log relevant user actions for auditing and traceability.",
        "The system shall guide the user through the steps required to complete the task.",
    ]

    i = 0
    while len(normalized) < 4 and i < len(fallback_pool):
        if fallback_pool[i] not in normalized:
            normalized.append(fallback_pool[i])
        i += 1

    normalized = normalized[:4]
    final_display = "\n".join([f"{i+1}. {c}" for i, c in enumerate(normalized)])

    return {"parsed": parsed, "normalized": normalized, "final_display": final_display}


def criterion_id(model_name: str, criterion_text: str) -> str:
    """Stable ID for checkbox keys + research logs."""
    base = f"{model_name}||{criterion_text}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:16]


# USER FORM (SAVED TO PRIVATE SUBMISSIONS REPO)

if not st.session_state.user_info_submitted:
    st.header("👤 Research Participation Form")
    st.markdown(
        "Please fill out this short form before using the RE Assistant.\n\n"
        "**Purpose:** This data is collected only for research purposes and will not be shared publicly."
    )

    with st.form("user_info_form"):
        name = st.text_input("Full Name (optional)")
        email = st.text_input("Email (optional)")
        affiliation = st.text_input("Affiliation / Organization (optional)")

        experience_it = st.selectbox(
            "Your experience level in IT:",
            [
                "No experience",
                "Beginner",
                "Intermediate",
                "Advanced",
                "Expert / Professional",
            ],
        )

        experience_years = st.number_input(
            "How many years of experience do you have in IT or related fields?",
            min_value=0,
            max_value=50,
            step=1,
            help="Enter a whole number (e.g., 2, 5, 10).",
        )

        experience_re_years = st.number_input(
            "How many years of experience do you have with software requirements / Requirements Engineering?",
            min_value=0,
            max_value=50,
            step=1,
            help="Enter a whole number (e.g., 1, 3, 10).",
        )

        consent = st.checkbox("I consent to my data being collected for research purposes")
        submitted = st.form_submit_button("Submit Information")

    if submitted:
        if not consent:
            st.error("❌ You must consent before proceeding.")
            st.stop()
        else:
            try:
                repo = get_submissions_repo()

                timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                path = f"submissions/{timestamp}_{st.session_state.session_id}.json"

                data = {
                    "timestamp": timestamp,
                    "session_id": st.session_state.session_id,
                    "name": name,
                    "email": email,
                    "affiliation": affiliation,
                    "experience_it": experience_it,
                    "experience_years": experience_years,
                    "experience_re_years": experience_re_years,
                    "device": device_info,
                }

                repo.create_file(
                    path=path,
                    message=f"User submission {timestamp}",
                    content=json.dumps(data, indent=2),
                )

                st.success(
                    "✅ Thank you! Your information has been securely saved. "
                    "You can now use the app below."
                )
                st.session_state.user_info_submitted = True
                log_event("user_info_submitted", {"device": device_info})
                st.rerun()

            except Exception as e:
                st.error("⚠️ Could not save your information to the private GitHub repo.")
                st.exception(e)
                st.stop()


# MAIN APP (after user info submitted)

if st.session_state.user_info_submitted:
    # Secrets / API keys
    GEMINI_KEY = require_secret("GEMINI_KEY")
    OPENAI_API_KEY = require_secret("OPENAI_API_KEY")
    TOGETHER_API_KEY = require_secret("TOGETHER_API_KEY")
    ADMIN_PASSWORD = require_secret("ADMIN_PASSWORD")

    genai.configure(api_key=GEMINI_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    GEMINI_MODEL = "gemini-2.5-flash"
    OPENAI_MODEL = "gpt-4o-mini"
    TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    FLAN_MODEL_ID = "google/flan-t5-large"

    # FLAN-T5-LARGE
 
    @st.cache_resource(show_spinner="⏳ Loading FLAN-T5-Large (first run only)…")
    def load_flan():
        tok = AutoTokenizer.from_pretrained(FLAN_MODEL_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_ID,
            torch_dtype=torch.float32,
        ).cpu()
        return tok, mdl

    tokenizer, flan_model = load_flan()

    # SHARED PROMPT
  
    def build_prompt(user_story: str, context: str | None = None) -> str:
        """
        Build main prompt for Gemini / OpenAI / LLaMA.
        Optionally include RAG context.
        """
        context_block = ""
        if context:
            context_block = (
                "Use the following project documentation as context:\n"
                f"{context}\n\n"
            )

        return f"""{context_block}You are a professional requirements engineer.

Given the user story below, write **exactly 4** acceptance criteria.

Rules:
- Do NOT repeat or rephrase the user story itself.
- Each acceptance criterion must be about the system behavior.
- Prefer phrasing like "The system shall ...".
- Start each criterion with a number and a period like "1." on its own line.

User Story:
{user_story}

Now write the 4 acceptance criteria:

1."""

    # FLAN-T5 HELPERS (kept polishing with OpenAI)
 
    def polish_criteria_with_openai(criteria: list[str], user_story: str) -> list[str] | None:
        try:
            draft = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
            prompt = f"""
You are a senior requirements engineer.

User story:
{user_story}

Draft acceptance criteria:
{draft}

Task:
- Rewrite these into exactly 4 clear, non-overlapping acceptance criteria.
- Do NOT introduce new requirements beyond the user story.
- Each criterion MUST start with "The system shall".
- Each criterion MUST be on its own line and numbered 1., 2., 3., 4.
- Keep them concise but specific and testable.

Now respond with ONLY the 4 numbered criteria:
1.
2.
3.
4.
"""
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a rigorous requirements engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=400,
            )
            text = response.choices[0].message.content or ""

            matches = re.findall(
                r"\b[1-4][\.\)]\s*(.+?)(?=\s*[1-4][\.\)]|$)",
                text,
                flags=re.DOTALL,
            )
            cleaned = [m.strip() for m in matches if m.strip()]
            if len(cleaned) >= 4:
                return cleaned[:4]
            return None
        except Exception:
            return None

    def generate_flan_output(user_story: str, use_rag: bool = False) -> str:
        """
        Generate acceptance criteria using Flan-T5-Large
        + optional RAG context + OpenAI polishing.
        """
        context = cached_retrieve_context(user_story) if use_rag else None

        flan_prompt = """
You are a requirements engineer.

Write exactly four meaningful acceptance criteria for the user story below.

Rules:
- Each criterion MUST begin with "1.", "2.", "3.", or "4."
- Do NOT repeat the user story wording.
- Each criterion must describe system behavior.
- Prefer phrasing like "The system shall ...".
- Do NOT leave any item empty.
"""
        if context:
            flan_prompt = (
                "Project documentation (for context):\n"
                f"{context}\n\n"
                + flan_prompt
            )

        flan_prompt += f"\nUser Story:\n{user_story}\n\nAcceptance Criteria:\n1.\n2.\n3.\n4.\n"

        inputs = tokenizer(flan_prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = flan_model.generate(
                inputs.input_ids,
                max_length=260,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

        base_items = [normalize_criterion(x) for x in extract_numbered_items(raw)]
        base_items = dedupe_preserve_order(base_items)
        while len(base_items) < 4:
            base_items.append("The system shall provide clear and helpful feedback after each action.")
        base_items = base_items[:4]

        polished = polish_criteria_with_openai(base_items, user_story)
        if polished and len(polished) == 4:
            return "\n".join([f"{i+1}. {normalize_criterion(polished[i])}" for i in range(4)])

        return "\n".join([f"{i+1}. {base_items[i]}" for i in range(4)])
 
    # GEMINI / OPENAI / LLAMA HELPERS

    def try_gemini_output(user_story: str, use_rag: bool) -> str:
        try:
            context = cached_retrieve_context(user_story) if use_rag else None
            model_g = genai.GenerativeModel(GEMINI_MODEL)
            response = model_g.generate_content(build_prompt(user_story, context))
            return (response.text or "").strip()
        except Exception as e:
            return f"❌ Gemini error: {e}"

    def try_openai_output(user_story: str, use_rag: bool) -> str:
        try:
            context = cached_retrieve_context(user_story) if use_rag else None
            prompt = build_prompt(user_story, context)
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful requirements engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            return f"❌ OpenAI error: {e}"

    def try_llama3_together(user_story: str, use_rag: bool) -> str:
        try:
            context = cached_retrieve_context(user_story) if use_rag else None
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": TOGETHER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a professional requirements engineer."},
                    {"role": "user", "content": build_prompt(user_story, context)},
                ],
                "temperature": 0.7,
                "max_tokens": 512,
            }
            response = requests.post(
                "https://api.together.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=45,
            )
            response.raise_for_status()
            return (response.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            return f"❌ Together.ai (LLaMA) error: {e}"

    def run_generation(user_story: str, selected_models: list[str], rag_for_model: dict[str, bool]):
        st.session_state.generated = {}
        st.session_state.criteria_by_model = {}
        st.session_state.per_model_artifacts = {}
        st.session_state.selected_criteria_entries = []
        st.session_state.final_selected_ac = ""
        st.session_state.edited_outputs = {}
        st.session_state.final_auto_source = ""

        cols_out = st.columns(len(selected_models))
        for i, model_name in enumerate(selected_models):
            with cols_out[i]:
                st.subheader(f"{model_name} Output")
                start = time.time()

                use_rag_flag = rag_for_model.get(model_name, False)

                if model_name == "Flan-T5":
                    raw_output = generate_flan_output(user_story, use_rag=use_rag_flag)
                elif model_name == "Gemini":
                    raw_output = try_gemini_output(user_story, use_rag=use_rag_flag)
                elif model_name == "OpenAI":
                    raw_output = try_openai_output(user_story, use_rag=use_rag_flag)
                elif model_name == "LLaMA-3 (Together)":
                    raw_output = try_llama3_together(user_story, use_rag=use_rag_flag)
                else:
                    raw_output = "❌ Unsupported model"

                duration = time.time() - start
                st.caption(f"⏱️ Time taken: {duration:.2f} sec")

                pp = postprocess_criteria(raw_output, user_story)

                st.session_state.generated[model_name] = raw_output
                st.session_state.per_model_artifacts[model_name] = {
                    "raw": raw_output,
                    "parsed": pp["parsed"],
                    "normalized": pp["normalized"],
                    "final_display": pp["final_display"],
                    "rag_enabled": bool(use_rag_flag),
                    "duration_sec": duration,
                }

                st.text_area(
                    "Generated acceptance criteria",
                    value=pp["final_display"],
                    height=200,
                    key=f"out_{model_name}_{st.session_state.reset_nonce}",
                    label_visibility="collapsed",
                    disabled=True,
                )

                st.session_state.criteria_by_model[model_name] = pp["normalized"]

                log_event(
                    "model_generation_complete",
                    {
                        "model": model_name,
                        "duration_sec": duration,
                        "rag_enabled": bool(use_rag_flag),
                        "criteria_count": len(pp["normalized"]),
                    },
                )

        st.success("✅ Generation complete. Proceed to selection below.")

    def build_final_from_selection(entries: list[dict]) -> str:
        lines = [f"{i+1}. {e['text']}" for i, e in enumerate(entries)]
        return "\n".join(lines)

    def build_final_from_edited_outputs(edited_outputs: dict, user_story: str) -> tuple[str, list[str]]:
        combined = []
        for model_name, txt in edited_outputs.items():
            pp = postprocess_criteria(txt, user_story)
            combined.extend(pp["normalized"])
        combined = dedupe_preserve_order(combined)
        lines = [f"{i+1}. {c}" for i, c in enumerate(combined)]
        return "\n".join(lines), combined

    def mark_final_manual():
        st.session_state.final_auto_source = "manual"

    # MAIN INTERFACE

    st.title("📚 Human-in-the-Loop Acceptance Criteria Assistant")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Device Used:** `{device_info}`")

    # Research Controls (Sidebar)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Research Controls")

    if st.sidebar.button("Reset story", use_container_width=True):
        st.session_state.user_story_text = ""
        st.session_state.user_story_manual = ""
        st.session_state.candidate_user_stories = []
        st.session_state.selected_user_story_index = None
        log_event("reset_story")
        st.rerun()

    if st.sidebar.button("Reset run", use_container_width=True):
        st.session_state.generated = {}
        st.session_state.criteria_by_model = {}
        st.session_state.per_model_artifacts = {}
        st.session_state.selected_criteria_entries = []
        st.session_state.final_selected_ac = ""
        st.session_state.action = "Accept"
        st.session_state.edited_outputs = {}
        st.session_state.pending_regen = False
        st.session_state.final_auto_source = ""
        log_event("reset_run")
        st.rerun()

    if st.sidebar.button("Reset all", use_container_width=True):
        sid = st.session_state.session_id
        user_ok = st.session_state.user_info_submitted
        st.session_state.clear()
        st.session_state.session_id = sid
        st.session_state.user_info_submitted = user_ok

        ss_init("feedback_log", [])
        ss_init("admin_authenticated", False)

        ss_init("user_story_text", "")
        ss_init("user_story_manual", "")
        ss_init("candidate_user_stories", [])
        ss_init("selected_user_story_index", None)

        ss_init("criteria_by_model", {})
        ss_init("generated", {})
        ss_init("per_model_artifacts", {})
        ss_init("selected_criteria_entries", [])
        ss_init("final_selected_ac", "")

        ss_init("action", "Accept")
        ss_init("edited_outputs", {})

        ss_init("events", [])
        ss_init("study_config", None)
        ss_init("config_locked", False)

        ss_init("reset_nonce", 0)
        ss_init("pending_regen", False)
        ss_init("final_auto_source", "")

        log_event("reset_all")
        st.rerun()

    if st.session_state.admin_authenticated:
        with st.sidebar.expander("Debug (admin use)", expanded=False):
            st.write("Config locked:", st.session_state.config_locked)
            st.write("Study config:", st.session_state.study_config)
            st.write("User story length:", len(st.session_state.user_story_text or ""))
            st.write("Generated models:", list(st.session_state.generated.keys()))
            st.write("Selected criteria count:", len(st.session_state.selected_criteria_entries))
            st.write("Final AC length:", len(st.session_state.final_selected_ac or ""))
            st.write("Final auto source:", st.session_state.final_auto_source)

    # Describe what you need

    st.markdown("## 1️⃣ Describe what you need")

    tab_manual, tab_form, tab_llm = st.tabs(
        [
            "✍️ I already have a user story",
            "🧩 Help me build a user story",
            "🤖 Convert my requirement into user stories",
        ]
    )

    def sync_manual_to_canonical():
        st.session_state.user_story_text = st.session_state.user_story_manual
        log_event("user_story_manual_changed", {"len": len(st.session_state.user_story_text or "")})

    with tab_form:
        st.markdown(
            """
Use this guided form if you're not familiar with user stories.  
We'll build something like:  
**As a `<role>`, I want `<goal>` so that `<reason>`.**
"""
        )

        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("Who is the user? (role)", placeholder="e.g., online banking customer")
        with col2:
            goal = st.text_input("What do they want to do?", placeholder="e.g., view my transaction history")

        reason = st.text_input(
            "Why do they want this? (benefit / reason)",
            placeholder="e.g., so that I can track my spending",
        )

        context_extra = st.text_area(
            "Optional context (constraints, system name, etc.)",
            placeholder="e.g., within the mobile app, only for accounts the customer owns.",
            height=80,
        )

        if st.button("✨ Build User Story", key="build_user_story"):
            if role and goal:
                base_story = f"As a {role}, I want to {goal}"
                if reason:
                    base_story += f" so that {reason}"
                base_story += "."

                if context_extra.strip():
                    base_story += f"\n\nAdditional context: {context_extra.strip()}"

                st.session_state.user_story_text = base_story
                st.session_state.user_story_manual = base_story
                st.session_state.candidate_user_stories = []
                st.session_state.selected_user_story_index = None

                log_event("user_story_built_form", {"role": role, "goal": goal, "has_reason": bool(reason)})
                st.success("✅ User story created. You can review/edit it in the first tab.")
            else:
                st.warning("Please fill at least the *role* and the *goal*.")

    with tab_llm:
        st.markdown(
            """
Describe what you need in plain language and I'll generate several alternative user stories.  
For example: *"I need a feature for managers to approve timesheets from a dashboard."*
"""
        )
        raw_req = st.text_area("Describe your requirement", height=120, key="raw_requirement_text")

        if st.button("🤖 Generate User Stories", key="llm_generate_user_story"):
            if not raw_req.strip():
                st.warning("Please describe your requirement first.")
            else:
                try:
                    prompt = f"""
You are a requirements engineer.

Convert the following plain-language requirement into **3 alternative user stories**
using the classic template:

"As a <type of user>, I want <some goal> so that <some reason>."

Requirement:
\"\"\"{raw_req}\"\"\"


Rules:
- Generate **3 distinct user stories** that are all reasonable interpretations.
- Number them as 1., 2., 3.
- Each story should be on a separate line.

Now write the 3 user stories:
1.
2.
3.
"""
                    response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You write clear, concise user stories."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.5,
                        max_tokens=300,
                    )
                    stories_text = (response.choices[0].message.content or "").strip()

                    candidates = []
                    for line in stories_text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
                        if len(line) < 15:
                            continue
                        candidates.append(line)

                    if not candidates:
                        candidates = [stories_text]

                    st.session_state.candidate_user_stories = candidates
                    st.session_state.selected_user_story_index = 0
                    st.session_state.user_story_text = candidates[0]
                    st.session_state.user_story_manual = candidates[0]

                    log_event("user_story_generated_llm", {"count": len(candidates)})
                    st.success("✅ User stories generated. Select your preferred one below.")
                    for i, s in enumerate(candidates, start=1):
                        st.markdown(f"**{i}.** {s}")

                except Exception as e:
                    st.error("⚠️ Could not generate user stories automatically.")
                    st.exception(e)

    if st.session_state.candidate_user_stories:
        st.markdown("### 1️⃣ Select which user story to use")
        options = [f"{i+1}. {s}" for i, s in enumerate(st.session_state.candidate_user_stories)]
        default_index = st.session_state.selected_user_story_index or 0

        choice = st.radio("Candidate user stories:", options, index=default_index, key="user_story_choice")
        chosen_index = options.index(choice)
        st.session_state.selected_user_story_index = chosen_index

        chosen_story = st.session_state.candidate_user_stories[chosen_index]
        if chosen_story != st.session_state.user_story_text:
            st.session_state.user_story_text = chosen_story
            st.session_state.user_story_manual = chosen_story
            log_event("user_story_candidate_selected", {"index": chosen_index})

    with tab_manual:
        st.markdown("Write or paste your user story here (e.g., `As a customer, I want ... so that ...`).")

        if not st.session_state.user_story_manual and st.session_state.user_story_text:
            st.session_state.user_story_manual = st.session_state.user_story_text

        st.text_area(
            "User Story",
            height=150,
            key="user_story_manual",
            on_change=sync_manual_to_canonical,
        )

    user_story = st.session_state.user_story_text

    # Generate acceptance criteria

    st.markdown("## 2️⃣ Generate acceptance criteria from this user story")
    st.text_area(
        "Final user story used as input",
        value=user_story,
        height=120,
        disabled=True,
    )

    st.markdown("### Choose Models and Retrieval Options")

    available_models = ["Flan-T5", "Gemini", "OpenAI", "LLaMA-3 (Together)"]
    model_icons = {
        "Flan-T5": "🧠",
        "Gemini": "✨",
        "OpenAI": "⚙️",
        "LLaMA-3 (Together)": "🦙",
    }
    model_descriptions = {
        "Flan-T5": "Local sequence-to-sequence model (FLAN-T5-Large).",
        "Gemini": "Google Gemini 2.5 Flash via Generative AI API.",
        "OpenAI": "OpenAI gpt-4o-mini chat-completion model.",
        "LLaMA-3 (Together)": "Meta LLaMA-3.1 via Together.ai API.",
    }

    if st.session_state.config_locked and st.session_state.study_config:
        locked_models = st.session_state.study_config["selected_models"]
        locked_rag = st.session_state.study_config["rag_for_model"]
    else:
        locked_models = None
        locked_rag = None

    selected_models: list[str] = []
    rag_for_model: dict[str, bool] = {}

    cols = st.columns(len(available_models))
    for col, model_name in zip(cols, available_models):
        safe_key = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        default_enabled = (model_name == "Flan-T5")
        default_rag = False

        if locked_models is not None:
            default_enabled = model_name in locked_models
            default_rag = bool(locked_rag.get(model_name, False))

        with col:
            with st.container(border=True):
                st.markdown(f"#### {model_icons.get(model_name, '🤖')} {model_name}")
                st.caption(model_descriptions.get(model_name, ""))

                use_model = st.checkbox(
                    "Enable Model",
                    key=f"use_{safe_key}_{st.session_state.reset_nonce}",
                    value=default_enabled,
                    disabled=st.session_state.config_locked,
                )

                use_rag_flag = st.checkbox(
                    "Apply RAG (project documents)",
                    key=f"rag_{safe_key}_{st.session_state.reset_nonce}",
                    value=default_rag,
                    disabled=st.session_state.config_locked,
                )

                if use_model:
                    selected_models.append(model_name)
                rag_for_model[model_name] = use_rag_flag

    can_generate = bool(user_story and user_story.strip()) and bool(selected_models)

    generate = st.button(
        "Generate Acceptance Criteria",
        disabled=not can_generate,
        help="Enter/select a user story and enable at least one model." if not can_generate else None,
    )

    if generate:
        if not st.session_state.config_locked:
            cfg = StudyConfig(
                prompt_version=PROMPT_VERSION,
                device=device_info,
                gemini_model=GEMINI_MODEL,
                openai_model=OPENAI_MODEL,
                together_model=TOGETHER_MODEL,
                flan_model_id=FLAN_MODEL_ID,
                temperatures={
                    "Flan-T5": None,
                    "Gemini": None,
                    "OpenAI": 0.7,
                    "LLaMA-3 (Together)": 0.7,
                    "Flan_Polish_OpenAI": 0.4,
                },
                max_tokens={
                    "OpenAI": 512,
                    "LLaMA-3 (Together)": 512,
                    "Gemini": None,
                    "Flan-T5": 260,
                    "Flan_Polish_OpenAI": 400,
                },
                rag_context_max_chars=RAG_CONTEXT_MAX_CHARS,
                selected_models=selected_models,
                rag_for_model=rag_for_model,
                locked_at_utc=now_utc_iso(),
            )

            st.session_state.study_config = asdict(cfg)
            st.session_state.config_locked = True
            log_event("study_config_locked", {"config": st.session_state.study_config})

        log_event("generate_clicked", {"selected_models": selected_models, "rag_for_model": rag_for_model})
        run_generation(user_story, selected_models, rag_for_model)

    if st.session_state.pending_regen and st.session_state.study_config and (st.session_state.user_story_text or "").strip():
        st.session_state.pending_regen = False
        cfg = st.session_state.study_config
        selected_models_regen = list(cfg.get("selected_models", []))
        rag_for_model_regen = dict(cfg.get("rag_for_model", {}))
        log_event("auto_regenerate_started", {"selected_models": selected_models_regen, "rag_for_model": rag_for_model_regen})
        run_generation(st.session_state.user_story_text, selected_models_regen, rag_for_model_regen)
        st.session_state.action = "Accept"
        log_event("auto_regenerate_finished", {})
        st.info("✅ Regenerated outputs. You can now review them.")
        st.rerun()

    # Select individual acceptance criteria across models

    if st.session_state.criteria_by_model:
        st.markdown("## 3️⃣ Select the most relevant acceptance criteria")
        st.caption("Your selections below are automatically combined into the final list in step 5.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Clear selections"):
                st.session_state.selected_criteria_entries = []
                st.session_state.final_selected_ac = ""
                st.session_state.final_auto_source = ""
                log_event("selections_cleared")
                st.rerun()

        with col_s2:
            show_context = st.checkbox("Show RAG context used (transparency)", value=False)

        selected_entries_temp = []
        current_selected_ids = {e["id"] for e in st.session_state.selected_criteria_entries}

        for model_name, crit_list in st.session_state.criteria_by_model.items():
            if not crit_list:
                continue

            st.markdown(f"**{model_name}**")

            if show_context:
                rag_used = bool(st.session_state.per_model_artifacts.get(model_name, {}).get("rag_enabled"))

                with st.expander(f"Context for {model_name}", expanded=False):
                    if not rag_used:
                        st.info("RAG was NOT enabled for this model during generation, so no context was provided.")
                    else:
                        ctx = cached_retrieve_context(user_story).strip()
                        if not ctx:
                            st.warning("RAG was enabled, but no context was retrieved for this user story.")
                        else:
                            st.write(ctx)

            for crit in crit_list:
                cid = criterion_id(model_name, crit)
                key = f"crit_{cid}"

                checked = st.checkbox(
                    crit,
                    key=key,
                    value=(cid in current_selected_ids),
                )

                if checked:
                    selected_entries_temp.append({"id": cid, "model": model_name, "text": crit})

        if selected_entries_temp != st.session_state.selected_criteria_entries:
            st.session_state.selected_criteria_entries = selected_entries_temp
            st.session_state.final_selected_ac = build_final_from_selection(selected_entries_temp)
            st.session_state.final_auto_source = "selection"
            log_event("selections_updated", {"count": len(selected_entries_temp)})

    # Provide feedback on the generated criteria

    if st.session_state.generated:
        st.markdown("## 4️⃣ Provide feedback on the generated criteria")

        action = st.radio(
            "Overall, how did you handle the model outputs?",
            ("Accept", "Edit", "Regenerate"),
            index=("Accept", "Edit", "Regenerate").index(st.session_state.action)
            if st.session_state.action in ("Accept", "Edit", "Regenerate")
            else 0,
        )

        if action != st.session_state.action:
            st.session_state.action = action
            log_event("feedback_action_changed", {"action": action})
            if action == "Regenerate":
                st.session_state.pending_regen = True
                st.rerun()

        edited_outputs = {}
        for model_name, raw in st.session_state.generated.items():
            if st.session_state.action == "Edit":
                edited_text = st.text_area(
                    f"Edit {model_name} Output:",
                    value=st.session_state.per_model_artifacts.get(model_name, {}).get("final_display", raw),
                    height=200,
                    key=f"edit_{model_name}_{st.session_state.reset_nonce}",
                )
                edited_outputs[model_name] = edited_text
            else:
                edited_outputs[model_name] = st.session_state.per_model_artifacts.get(model_name, {}).get("final_display", raw)

        if st.session_state.action == "Edit":
            if edited_outputs != st.session_state.edited_outputs:
                st.session_state.edited_outputs = edited_outputs
                combined_text, combined_list = build_final_from_edited_outputs(edited_outputs, st.session_state.user_story_text)
                if (not st.session_state.final_selected_ac.strip()) or (st.session_state.final_auto_source in ("edited", "")):
                    st.session_state.final_selected_ac = combined_text
                    st.session_state.final_auto_source = "edited"
                log_event("edited_outputs_updated", {"models": list(edited_outputs.keys()), "combined_count": len(combined_list)})
        else:
            st.session_state.edited_outputs = edited_outputs

    #  Final acceptance criteria (your agreed version)

    if st.session_state.generated:
        st.markdown("## 5️⃣ Final acceptance criteria (your agreed version)")

        enable_editing = st.checkbox("Enable manual editing of final criteria", value=True)

        st.text_area(
            "Final acceptance criteria",
            height=220,
            key="final_selected_ac",
            disabled=not enable_editing,
            on_change=mark_final_manual if enable_editing else None,
        )

        def validate_before_save() -> tuple[bool, str]:
            if not (st.session_state.user_story_text or "").strip():
                return False, "User story is empty."
            if not st.session_state.generated:
                return False, "No generated outputs found."
            if st.session_state.study_config is None:
                return False, "Study config not locked."
            for m in st.session_state.study_config.get("selected_models", []):
                if m not in st.session_state.per_model_artifacts:
                    return False, f"Missing artifacts for model: {m}"
            return True, "OK"

        if st.button("Submit Feedback"):
            ok, msg = validate_before_save()
            if not ok:
                st.error(f"❌ Cannot submit: {msg}")
            else:
                log_event("submit_clicked", {"action": st.session_state.action})

                timestamp_local = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                utc_slug = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

                payload = {
                    "timestamp_local": timestamp_local,
                    "timestamp_utc": now_utc_iso(),
                    "session_id": st.session_state.session_id,
                    "device": device_info,
                    "study_config": st.session_state.study_config,
                    "user_story": st.session_state.user_story_text,
                    "models": list(st.session_state.generated.keys()),
                    "per_model_artifacts": st.session_state.per_model_artifacts,
                    "selection": st.session_state.selected_criteria_entries,
                    "final_ac": st.session_state.final_selected_ac,
                    "human_action": st.session_state.action,
                    "edited_outputs": st.session_state.edited_outputs,
                    "events": st.session_state.events,
                }

                try:
                    repo = get_submissions_repo()
                    feedback_path = f"feedback/{utc_slug}_{st.session_state.session_id}.json"
                    repo.create_file(
                        path=feedback_path,
                        message=f"Feedback batch {utc_slug}",
                        content=json.dumps(payload, indent=2),
                    )
                    st.success("✅ Feedback saved (private repo)!")
                    log_event("feedback_saved", {"path": feedback_path})
                except Exception as e:
                    st.error("⚠️ Could not save feedback to the private GitHub repo.")
                    st.exception(e)
                    log_event("feedback_save_failed", {"error": str(e)})

    # ADMIN Section (READS FROM PRIVATE SUBMISSIONS REPO)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Research Admin")

    if not st.session_state.admin_authenticated:
        password_input = st.sidebar.text_input("Enter Admin Password:", type="password")
        if st.sidebar.button("Login", use_container_width=True):
            if password_input == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.sidebar.success("✅ Access granted.")
            else:
                st.sidebar.error("❌ Incorrect password.")
    else:
        st.sidebar.success("🔓 Admin Mode Active")

        if st.sidebar.button("Download Feedback Log (Excel)", use_container_width=True):
            try:
                repo = get_submissions_repo()

                files = repo.get_contents("feedback")
                all_feedback = []

                for file in files:
                    if file.path.endswith(".json"):
                        content = file.decoded_content.decode("utf-8")
                        entry = json.loads(content)
                        all_feedback.append(entry)

                if all_feedback:
                    df = pd.json_normalize(all_feedback)

                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False)
                    excel_buffer.seek(0)

                    st.sidebar.download_button(
                        label="⬇️ Download Feedback Log (Excel)",
                        data=excel_buffer,
                        file_name=f"feedback_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                    st.sidebar.success(f"✅ {len(all_feedback)} feedback entries found.")
                else:
                    st.sidebar.warning("⚠️ No feedback files found in private repo.")

            except Exception as e:
                st.sidebar.error("❌ Failed to fetch feedback log from private repo.")
                st.sidebar.exception(e)

        if st.sidebar.button("📥 Download All Submissions", use_container_width=True):
            try:
                repo = get_submissions_repo()

                files = repo.get_contents("submissions")
                all_data = []

                for file in files:
                    if file.path.endswith(".json"):
                        content = file.decoded_content.decode("utf-8")
                        entry = json.loads(content)
                        all_data.append(entry)

                if all_data:
                    df = pd.DataFrame(all_data)
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    st.sidebar.download_button(
                        label="⬇️ Download CSV File",
                        data=csv_buffer.getvalue(),
                        file_name=f"research_submissions_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.sidebar.success(f"✅ {len(all_data)} submissions found.")
                else:
                    st.sidebar.warning("⚠️ No submissions found in private repo.")
            except Exception as e:
                st.sidebar.error("❌ Failed to fetch submissions from private repo.")
                st.sidebar.exception(e)
