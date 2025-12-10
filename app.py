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
from rag import retrieve_context

load_dotenv()
st.set_page_config(
    page_title="RE Assistant (Gemini + OpenAI + LLaMA + Flan-T5 + RAG)",
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


# USER FORM
if not st.session_state.user_info_submitted:
    st.header("👤 Research Participation Form")
    st.markdown(
        "Please fill out this short form before using the RE Assistant.\n\n"
        "**Purpose:** This data is collected only for research purposes and will not be shared publicly."
    )

    with st.form("user_info_form"):
        name = st.text_input("Full Name")
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

        experience_software = st.selectbox(
            "Your experience level with computer software:",
            [
                "Very uncomfortable",
                "Somewhat comfortable",
                "Comfortable",
                "Very comfortable",
                "Expert user",
            ],
        )

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
                    "experience_it": experience_it,
                    "experience_years": experience_years,
                    "experience_software": experience_software,
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

# MAIN APP after user info submitted
if st.session_state.user_info_submitted:
    # Secrets / API keys
    GEMINI_KEY = require_secret("GEMINI_KEY")
    OPENAI_API_KEY = require_secret("OPENAI_API_KEY")
    TOGETHER_API_KEY = require_secret("TOGETHER_API_KEY")

    GITHUB_TOKEN = require_secret("GITHUB_TOKEN")
    GITHUB_REPO = require_secret("GITHUB_REPO")
    ADMIN_PASSWORD = require_secret("ADMIN_PASSWORD")

    genai.configure(api_key=GEMINI_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    GEMINI_MODEL = "gemini-2.5-flash"
    TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    # FLAN-T5-LARGE

    @st.cache_resource(show_spinner="⏳ Loading FLAN-T5-Large (first run only)…")
    def load_flan():
        model_id = "google/flan-t5-large"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
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

    # FLAN-T5 HELPERS
 
    def deduplicate_criteria(criteria: list[str]) -> list[str]:
        """Remove near-duplicate criteria based on a normalized key."""
        seen = set()
        unique = []
        for c in criteria:
            text = c.strip()
            if not text:
                continue
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
        Generate 4 high-quality acceptance criteria using Flan-T5-Large
        + optional RAG context + optional OpenAI polishing.
        """
        context = retrieve_context(user_story) if use_rag else None

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

        matches = re.findall(
            r"\b([1-4])[\.\)\:\-]\s*(.*?)\s*(?=(?:[1-4][\.\)\:\-])|$)",
            raw,
            flags=re.DOTALL,
        )
        extracted = [m[1].strip() for m in matches]

        cleaned = []
        for item in extracted:
            if not item:
                continue
            if item in [".", ":", "-", " "]:
                continue
            if len(item) < 5:
                continue
            cleaned.append(item)

        us_l = user_story.lower()
        cleaned = [c for c in cleaned if us_l[:40] not in c.lower()]

        cleaned = deduplicate_criteria(cleaned)

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

        polished = polish_criteria_with_openai(cleaned, user_story)
        if polished and len(polished) == 4:
            cleaned = polished

        final = []
        for c in cleaned:
            text = c.strip()
            if not text:
                continue

            text = re.sub(r"^\s*[1-4][\.\)\:\-]\s*", "", text).strip()

            if not re.match(r"(?i)^the system\s+(shall|must|should|can|will)\b", text):
                if text and text[0].isupper():
                    text = text[0].lower() + text[1:]
                text = "The system shall " + text

            if not text.endswith("."):
                text += "."

            final.append(text)

        i_fb2 = 0
        while len(final) < 4 and i_fb2 < len(fallback_pool):
            final.append(fallback_pool[i_fb2])
            i_fb2 += 1

        final = final[:4]

        return "\n".join(f"{i+1}. {c}" for i, c in enumerate(final))

    # GEMINI / OPENAI / LLAMA HELPERS

    def try_gemini_output(user_story: str, use_rag: bool) -> str:
        try:
            context = retrieve_context(user_story) if use_rag else None
            model_g = genai.GenerativeModel(GEMINI_MODEL)
            response = model_g.generate_content(build_prompt(user_story, context))
            return response.text.strip()
        except Exception as e:
            return f"❌ Gemini error: {e}"

    def try_openai_output(user_story: str, use_rag: bool) -> str:
        try:
            context = retrieve_context(user_story) if use_rag else None
            prompt = build_prompt(user_story, context)
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

    def try_llama3_together(user_story: str, use_rag: bool) -> str:
        try:
            context = retrieve_context(user_story) if use_rag else None
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
                    {"role": "user", "content": build_prompt(user_story, context)},
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

    # MAIN INTERFACE

    st.title("📚 Human-in-the-Loop Acceptance Criteria Assistant")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Device Used:** `{device_info}`")

    user_story = st.text_area("Enter your User Story:", height=150)

    #  updated model selection after Nov meeting 
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

        with col:
            # Card container with border
            with st.container(border=True):
                st.markdown(
                    f"#### {model_icons.get(model_name, '🤖')} {model_name}"
                )
                st.caption(model_descriptions.get(model_name, ""))

                use_model = st.checkbox(
                    "Enable Model",
                    key=f"use_{safe_key}",
                    value=(model_name == "Flan-T5"),  # Flan-T5 enabled by default
                )
                use_rag_flag = st.checkbox(
                    "Apply RAG (project documents)", key=f"rag_{safe_key}"
                )

                if use_model:
                    selected_models.append(model_name)
                rag_for_model[model_name] = use_rag_flag

    generate = st.button("Generate Acceptance Criteria")

    if generate and user_story:
        if not selected_models:
            st.warning("⚠️ Please select at least one model.")
        else:
            st.session_state.generated = {}
            cols = st.columns(len(selected_models))

            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    st.subheader(f"{model_name} Output")
                    start = time.time()

                    use_rag_flag = rag_for_model.get(model_name, False)

                    if model_name == "Flan-T5":
                        output = generate_flan_output(user_story, use_rag=use_rag_flag)
                    elif model_name == "Gemini":
                        output = try_gemini_output(user_story, use_rag=use_rag_flag)
                    elif model_name == "OpenAI":
                        output = try_openai_output(user_story, use_rag=use_rag_flag)
                    elif model_name == "LLaMA-3 (Together)":
                        output = try_llama3_together(user_story, use_rag=use_rag_flag)
                    else:
                        output = "❌ Unsupported model"

                    st.session_state.generated[model_name] = output
                    st.text_area("", value=output, height=200, key=f"out_{model_name}")
                    st.caption(f"⏱️ Time taken: {time.time() - start:.2f} sec")

    # FEEDBACK SECTION

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
            utc_slug = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

            g = Github(GITHUB_TOKEN)
            repo = g.get_repo(GITHUB_REPO)

            for model_name, final_text in edited.items():
                entry = {
                    "timestamp": timestamp,
                    "session_id": st.session_state.session_id,
                    "model": model_name,
                    "user_story": user_story,
                    "generated_ac": st.session_state.generated[model_name],
                    "human_action": action,
                    "edited_ac": final_text,
                    "device": device_info,
                }

                # Keep also in local session
                st.session_state.feedback_log.append(entry)

                # Save each feedback entry to GitHub
                try:
                    feedback_path = (
                        f"feedback/{utc_slug}_{st.session_state.session_id}_{model_name}.json"
                    )
                    repo.create_file(
                        path=feedback_path,
                        message=f"Feedback {utc_slug} ({model_name})",
                        content=json.dumps(entry, indent=2),
                    )
                except Exception as e:
                    st.error("⚠️ Could not save feedback to GitHub.")
                    st.exception(e)

            st.success("✅ Feedback saved!")

    # Feedback download (from GITHUB)

    if st.sidebar.button("Download Feedback Log"):
        try:
            g = Github(GITHUB_TOKEN)
            repo = g.get_repo(GITHUB_REPO)

            files = repo.get_contents("feedback")
            all_feedback = []

            for file in files:
                if file.path.endswith(".json"):
                    content = file.decoded_content.decode("utf-8")
                    entry = json.loads(content)
                    all_feedback.append(entry)

            if all_feedback:
                df = pd.DataFrame(all_feedback)

                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False)
                excel_buffer.seek(0)

                st.sidebar.download_button(
                    label="⬇️ Download Feedback Log (Excel)",
                    data=excel_buffer,
                    file_name=f"feedback_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.sidebar.success(f"✅ {len(all_feedback)} feedback entries found.")
            else:
                st.sidebar.warning("⚠️ No feedback files found in GitHub repo.")

        except Exception as e:
            st.sidebar.error("❌ Failed to fetch feedback log from GitHub.")
            st.sidebar.exception(e)

    # ADMIN Section

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