import streamlit as st
import uuid
import time
import re
import datetime
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests
import google.generativeai as genai
from openai import OpenAI
from github import Github
import json
from io import BytesIO

# -------------------------------
# INITIAL SETUP
# -------------------------------
load_dotenv()
st.set_page_config(
    page_title="RE Assistant (Gemini + OpenAI + LLaMA + Flan-T5)",
    layout="wide",
)

device_info = "GPU" if torch.cuda.is_available() else "CPU"

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_info_submitted" not in st.session_state:
    st.session_state.user_info_submitted = False
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False


def require_secret(key: str):
    """Stop the app with a friendly message if a secret is missing."""
    try:
        return st.secrets[key]
    except KeyError:
        st.error(f"Missing secret: {key}. Please add `{key}` in Streamlit Secrets.")
        st.stop()


# -------------------------------
# USER FORM (MANDATORY)
# -------------------------------
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
        purpose = st.text_area("How do you plan to use this tool?")
        consent = st.checkbox("I consent to my data being collected for research purposes")
        submitted = st.form_submit_button("Submit Information")

    if submitted:
        if not consent:
            st.error("❌ You must consent before proceeding.")
            st.stop()
        else:
            try:
                token = require_secret("GITHUB_TOKEN")
                repo_name = require_secret("GITHUB_REPO")
                g = Github(token)
                repo = g.get_repo(repo_name)

                timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                path = f"submissions/{timestamp}_{st.session_state.session_id}.json"

                data = {
                    "timestamp": timestamp,
                    "session_id": st.session_state.session_id,
                    "name": name,
                    "email": email,
                    "affiliation": affiliation,
                    "purpose": purpose,
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
                st.rerun()

            except Exception as e:
                st.error("⚠️ Could not save your information to GitHub. Please contact the developer.")
                st.exception(e)
                st.stop()

# -------------------------------
# MAIN APP (after form)
# -------------------------------
if st.session_state.user_info_submitted:
    # Secrets
    GEMINI_KEY = require_secret("GEMINI_KEY")
    OPENAI_API_KEY = require_secret("OPENAI_API_KEY")
    TOGETHER_API_KEY = require_secret("TOGETHER_API_KEY")

    GITHUB_TOKEN = require_secret("GITHUB_TOKEN")
    GITHUB_REPO = require_secret("GITHUB_REPO")
    ADMIN_PASSWORD = require_secret("ADMIN_PASSWORD")

    # API clients
    genai.configure(api_key=GEMINI_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Model names
    GEMINI_MODEL = "gemini-2.5-flash"
    TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    # ---------------------------
    # Load Flan-T5-Large once
    # ---------------------------
    @st.cache_resource(show_spinner="⏳ Loading FLAN-T5-Large (first run only)…")
    def load_flan():
        model_id = "google/flan-t5-large"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).cpu()
        return tok, mdl

    tokenizer, model = load_flan()

    # ---------------------------
    # Shared prompt (Gemini / OpenAI / LLaMA)
    # ---------------------------
    def build_prompt(user_story: str) -> str:
        return f"""You are a professional requirements engineer.

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

    # Flan-T5-Large

    def deduplicate_criteria(criteria: list[str]) -> list[str]:
        """Remove near-duplicate criteria based on a normalized key."""
        seen = set()
        unique = []
        for c in criteria:
            text = c.strip()
            if not text:
                continue
            # Normalize: lower, remove punctuation, remove very common words
            key = re.sub(r"[^a-z0-9 ]", " ", text.lower())
            key = re.sub(
                r"\b(the|a|an|of|to|and|for|in|on|with|shall|must|should|can|will)\b",
                " ",
                key,
            )
            key = " ".join(key.split())
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return unique

    def polish_criteria_with_openai(criteria: list[str], user_story: str) -> list[str] | None:
        """
        Use OpenAI to rewrite criteria into 4 clear, non-redundant acceptance criteria.
        Returns a list of 4 strings or None if polishing fails.
        """
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
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a rigorous requirements engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=400,
            )
            text = response.choices[0].message.content

            # Extract 1.–4. from OpenAI output
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
            # If anything goes wrong, just skip polishing
            return None

    def generate_flan_output(user_story: str) -> str:
        """Generate 4 high-quality acceptance criteria using Flan-T5-Large + OpenAI polishing."""

        flan_prompt = f"""
You are a requirements engineer.

Write exactly four meaningful acceptance criteria for the user story below.

Rules:
- Each criterion MUST begin with "1.", "2.", "3.", or "4."
- Do NOT repeat the user story wording.
- Each criterion must describe system behavior.
- Prefer phrasing like "The system shall ...".
- Do NOT leave any item empty.

User Story:
{user_story}

Acceptance Criteria:
1.
2.
3.
4.
"""

        inputs = tokenizer(flan_prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=260,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ---- Step 1: extract raw criteria from FLAN output ----
        matches = re.findall(
            r"\b([1-4])[\.\)\:\-]\s*(.*?)\s*(?=(?:[1-4][\.\)\:\-])|$)",
            raw,
            flags=re.DOTALL,
        )
        extracted = [m[1].strip() for m in matches]

        # Remove blanks / very short fragments
        cleaned = []
        for item in extracted:
            if not item:
                continue
            if item in [".", ":", "-", " "]:
                continue
            if len(item) < 5:
                continue
            cleaned.append(item)

        # Remove items that clearly repeat the user story
        us_l = user_story.lower()
        cleaned = [c for c in cleaned if us_l[:40] not in c.lower()]

        # Deduplicate semantically similar criteria
        cleaned = deduplicate_criteria(cleaned)

        # ---- Step 2: fallback pool if FLAN gave too few good items ----
        fallback_pool = [
            "The system shall allow the user to complete the main task successfully.",
            "The system shall provide clear and helpful feedback after each action.",
            "The system shall validate user inputs and present meaningful error messages.",
            "The system shall handle exceptional cases gracefully and maintain stability.",
            "The system shall log relevant user actions for auditing and traceability.",
            "The system shall guide the user through the steps required to complete the task.",
        ]

        i_fb = 0
        while len(cleaned) < 4 and i_fb < len(fallback_pool):
            cleaned.append(fallback_pool[i_fb])
            i_fb += 1

        cleaned = cleaned[:4]

        # ---- Step 3: optional polishing with OpenAI ----
        polished = polish_criteria_with_openai(cleaned, user_story)
        if polished and len(polished) == 4:
            cleaned = polished

        # ---- Step 4: enforce "The system shall ..." style & punctuation ----
        final = []
        for c in cleaned:
            text = c.strip()
            if not text:
                continue

            # Remove any leftover numbering or bullets at the start
            text = re.sub(r"^\s*[1-4][\.\)\:\-]\s*", "", text).strip()

            # If it doesn't already start with "The system ..." then wrap it
            if not re.match(r"(?i)^the system\s+(shall|must|should|can|will)\b", text):
                # Lowercase first letter of remaining text for nice flow
                if text and text[0].isupper():
                    text = text[0].lower() + text[1:]
                text = "The system shall " + text

            # Ensure it ends with a period
            if not text.endswith("."):
                text += "."

            final.append(text)

        # If somehow fewer than 4 after cleanup, top up with fallback_pool
        i_fb2 = 0
        while len(final) < 4 and i_fb2 < len(fallback_pool):
            final.append(fallback_pool[i_fb2])
            i_fb2 += 1

        final = final[:4]

        return "\n".join(f"{i+1}. {c}" for i, c in enumerate(final))

    # ---------------------------
    # Gemini helper
    # ---------------------------
    def try_gemini_output(user_story: str) -> str:
        try:
            model_g = genai.GenerativeModel(GEMINI_MODEL)
            response = model_g.generate_content(build_prompt(user_story))
            return response.text.strip()
        except Exception as e:
            return f"❌ Gemini error: {e}"

    # ---------------------------
    # OpenAI helper
    # ---------------------------
    def try_openai_output(user_story: str) -> str:
        try:
            prompt = build_prompt(user_story)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful requirements engineer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI error: {e}"

    # ---------------------------
    # LLaMA-3 (Together) helper
    # ---------------------------
    def try_llama3_together(user_story: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": TOGETHER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional requirements engineer.",
                    },
                    {"role": "user", "content": build_prompt(user_story)},
                ],
                "temperature": 0.7,
                "max_tokens": 512,
            }
            response = requests.post(
                "https://api.together.ai/v1/chat/completions",
                headers=headers,
                json=data,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"❌ Together.ai (LLaMA) error: {e}"

    # ---------------------------
    # MAIN INTERFACE
    # ---------------------------
    st.title("📚 Human-in-the-Loop Acceptance Criteria Assistant")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Device Used:** `{device_info}`")

    user_story = st.text_area("Enter your User Story:", height=150)
    model_choices = st.multiselect(
        "Choose Models:",
        ["Gemini", "OpenAI", "Flan-T5", "LLaMA-3 (Together)"],
        default=["Flan-T5"],
    )
    generate = st.button("Generate Acceptance Criteria")

    if generate and user_story:
        st.session_state.generated = {}
        cols = st.columns(len(model_choices))

        for i, model_name in enumerate(model_choices):
            with cols[i]:
                st.subheader(f"{model_name} Output")
                start = time.time()

                if model_name == "Flan-T5":
                    output = generate_flan_output(user_story)
                elif model_name == "Gemini":
                    output = try_gemini_output(user_story)
                elif model_name == "OpenAI":
                    output = try_openai_output(user_story)
                elif model_name == "LLaMA-3 (Together)":
                    output = try_llama3_together(user_story)
                else:
                    output = "❌ Unsupported model"

                st.session_state.generated[model_name] = output
                st.text_area("", value=output, height=200, key=f"out_{model_name}")
                st.caption(f"⏱️ Time taken: {time.time() - start:.2f} sec")

    # ---------------------------
    # FEEDBACK SECTION
    # ---------------------------
    if "generated" in st.session_state and st.session_state.generated:
        action = st.radio(
            "What would you like to do?", ("Accept", "Edit", "Regenerate")
        )
        edited = {}

        for model_name in st.session_state.generated:
            if action == "Edit":
                edited[model_name] = st.text_area(
                    f"Edit {model_name} Output:",
                    value=st.session_state.generated[model_name],
                    height=200,
                )
            else:
                edited[model_name] = st.session_state.generated[model_name]

        if st.button("Submit Feedback"):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for model_name, final_text in edited.items():
                st.session_state.feedback_log.append(
                    {
                        "timestamp": timestamp,
                        "session_id": st.session_state.session_id,
                        "model": model_name,
                        "user_story": user_story,
                        "generated_ac": st.session_state.generated[model_name],
                        "human_action": action,
                        "edited_ac": final_text,
                    }
                )
            st.success("✅ Feedback saved!")

    # ---------------------------
    # FEEDBACK DOWNLOAD
    # ---------------------------
    if st.sidebar.button("Download Feedback Log"):
        if st.session_state.feedback_log:
            df = pd.DataFrame(st.session_state.feedback_log)
            filename = (
                f"feedback_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            df.to_excel(filename, index=False, engine="openpyxl")
            st.sidebar.success(f"✅ Feedback log saved as '{filename}'!")
        else:
            st.sidebar.warning("⚠️ No feedback to download.")

    # ---------------------------
    # ADMIN SECTION (password protected)
    # ---------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Research Admin")

    if not st.session_state.admin_authenticated:
        password_input = st.sidebar.text_input("Enter Admin Password:", type="password")
        if st.sidebar.button("Login"):
            if password_input == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.sidebar.success("✅ Access granted.")
            else:
                st.sidebar.error("❌ Incorrect password.")
    else:
        st.sidebar.success("🔓 Admin Mode Active")

        if st.sidebar.button("📥 Download All Submissions"):
            try:
                g = Github(GITHUB_TOKEN)
                repo = g.get_repo(GITHUB_REPO)

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
                    )
                    st.sidebar.success(f"✅ {len(all_data)} submissions found.")
                else:
                    st.sidebar.warning("⚠️ No submissions found in GitHub repo.")
            except Exception as e:
                st.sidebar.error("❌ Failed to fetch submissions.")
                st.sidebar.exception(e)
