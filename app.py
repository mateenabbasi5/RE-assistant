import streamlit as st
import os
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

# INITIAL SETUP
load_dotenv()
st.set_page_config(page_title="RE Assistant (Gemini + OpenAI + LLaMA + Flan-T5)", layout="wide")

device_info = "GPU" if torch.cuda.is_available() else "CPU"

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_info_submitted" not in st.session_state:
    st.session_state.user_info_submitted = False
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

# USER FORM 

if not st.session_state.user_info_submitted:
    st.header("👤 Research Participation Form")
    st.markdown("""
    Please fill out this short form before using the RE Assistant.

    **Purpose:** This data is collected only for research purposes and will not be shared publicly.
    """)

    with st.form("user_info_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email (optional)")
        affiliation = st.text_input("Affiliation / Organization")
        purpose = st.text_area("How do you plan to use this tool?")
        consent = st.checkbox("I consent to my data being collected for research purposes")
        submitted = st.form_submit_button("Submit Information")

    if submitted:
        if not consent:
            st.error("❌ You must consent before proceeding.")
            st.stop()
        else:
            try:
                token = st.secrets["GITHUB_TOKEN"]
                repo_name = st.secrets["GITHUB_REPO"]
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
                    content=json.dumps(data, indent=2)
                )

                st.success("✅ Thank you! Your information has been securely saved. You can now use the app below.")
                st.session_state.user_info_submitted = True
                st.rerun()

            except Exception as e:
                st.error("⚠️ Could not save your information to GitHub. Please contact the developer.")
                st.exception(e)
                st.stop()

# MAIN APP 

if st.session_state.user_info_submitted:
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]

    # Initialize API clients
    genai.configure(api_key=GEMINI_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    st.info("⏳ Loading flan-t5-large (please wait)...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).cpu()

    def build_prompt(user_story):
        return f"""Given the user story below, write **exactly 4** acceptance criteria.
Start each one with a number like "1." and make each one specific, testable, and measurable.

User Story:
{user_story}

Acceptance Criteria:
1."""

    # MODEL OUTPUT FUNCTIONS

    def generate_flan_output(user_story):
        prompt = build_prompt(user_story)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=300)
        raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        matches = re.findall(r"\d\.\s?.+?(?=\d\.|$)", raw_text, re.DOTALL)
        if matches and len(matches) >= 4:
            cleaned = [m.strip() for m in matches[:4]]
            return "\n".join(cleaned)
        else:
            return (
                "1. The system shall allow the user to complete the main task.\n"
                "2. The system shall provide confirmation after the action.\n"
                "3. Errors must be shown if input is invalid.\n"
                "4. All actions must be logged for auditing."
            )

    def try_gemini_output(user_story):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(build_prompt(user_story))
            return response.text.strip()
        except Exception as e:
            return f"❌ Gemini error: {e}"

    def try_openai_output(user_story):
        try:
            prompt = build_prompt(user_story)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful requirements engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI error: {e}"

    def try_llama3_together(user_story):
        try:
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "meta-llama/Llama-3-8b-chat-hf",
                "messages": [
                    {"role": "system", "content": "You are a professional requirements engineer."},
                    {"role": "user", "content": build_prompt(user_story)}
                ],
                "temperature": 0.7,
                "max_tokens": 512
            }
            response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"❌ Together.ai (LLaMA) error: {e}"

    # MAIN INTERFACE

    st.title("📚 Human-in-the-Loop Acceptance Criteria Assistant")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Device Used:** `{device_info}`")

    user_story = st.text_area("Enter your User Story:", height=150)
    model_choices = st.multiselect(
        "Choose Models:",
        ["Gemini", "OpenAI", "Flan-T5", "LLaMA-3 (Together)"],
        default=["Flan-T5"]
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
                st.caption(f"⏱️ Time taken: {round(time.time() - start, 2)} sec")

    # FEEDBACK

    if "generated" in st.session_state:
        action = st.radio("What would you like to do?", ("Accept", "Edit", "Regenerate"))
        edited = {}
        for model_name in st.session_state.generated:
            if action == "Edit":
                edited[model_name] = st.text_area(
                    f"Edit {model_name} Output:",
                    value=st.session_state.generated[model_name],
                    height=200
                )
            else:
                edited[model_name] = st.session_state.generated[model_name]

        if st.button("Submit Feedback"):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for model_name, final_text in edited.items():
                st.session_state.feedback_log.append({
                    "timestamp": timestamp,
                    "session_id": st.session_state.session_id,
                    "model": model_name,
                    "user_story": user_story,
                    "generated_ac": st.session_state.generated[model_name],
                    "human_action": action,
                    "edited_ac": final_text
                })
            st.success("✅ Feedback saved!")

    if st.sidebar.button("Download Feedback Log"):
        if st.session_state.feedback_log:
            df = pd.DataFrame(st.session_state.feedback_log)
            filename = f"feedback_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False, engine='openpyxl')
            st.sidebar.success(f"✅ Feedback log saved as '{filename}'!")
        else:
            st.sidebar.warning("⚠️ No feedback to download.")

    # ADMIN SECTION 

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Research Admin")

    if not st.session_state.admin_authenticated:
        password_input = st.sidebar.text_input("Enter Admin Password:", type="password")
        if st.sidebar.button("Login"):
            if password_input == st.secrets["ADMIN_PASSWORD"]:
                st.session_state.admin_authenticated = True
                st.sidebar.success("✅ Access granted.")
            else:
                st.sidebar.error("❌ Incorrect password.")
    else:
        st.sidebar.success("🔓 Admin Mode Active")

        if st.sidebar.button("📥 Download All Submissions"):
            try:
                token = st.secrets["GITHUB_TOKEN"]
                repo_name = st.secrets["GITHUB_REPO"]
                g = Github(token)
                repo = g.get_repo(repo_name)

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
                        mime="text/csv"
                    )
                    st.sidebar.success(f"✅ {len(all_data)} submissions found.")
                else:
                    st.sidebar.warning("⚠️ No submissions found in GitHub repo.")
            except Exception as e:
                st.sidebar.error("❌ Failed to fetch submissions.")
                st.sidebar.exception(e)
