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
import cohere

load_dotenv()

st.set_page_config(page_title="RE Assistant (Gemini + Flan-T5 + LLaMA-3 + Cohere + OpenRouter)", layout="wide")

device_info = "GPU" if torch.cuda.is_available() else "CPU"

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

GEMINI_API_KEYS = [
    os.getenv("GEMINI_KEY1"),
    os.getenv("GEMINI_KEY2"),
    os.getenv("GEMINI_KEY3"),
]
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def try_gemini_key():
    for key in GEMINI_API_KEYS:
        if not key:
            continue
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-pro")
            if model.generate_content("Hello, world!").text:
                return key
        except Exception:
            continue
    return None

working_key = try_gemini_key()

st.info("⏳ Loading flan-t5-large (please wait)...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).cpu()


def build_prompt(user_story):
    return f"""Given the user story below, write **exactly 4** acceptance criteria.
Start each one with a number like \"1.\" and make each one specific, testable, and measurable.

User Story:
{user_story}

Acceptance Criteria:
1."""


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
        if not working_key:
            return "❌ Gemini API keys are missing or invalid."
        genai.configure(api_key=working_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(build_prompt(user_story))
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"

def try_llama3_together(user_story):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
            "Content-Type": "application/json"
        }
        prompt = build_prompt(user_story)
        data = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [
                {"role": "system", "content": "You are a professional requirements engineer."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Together.ai (LLaMA-3) error: {e}"

def try_cohere(prompt):
    try:
        co = cohere.Client(COHERE_API_KEY)
        response = co.chat(
            model="command-r",
            message=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ Cohere error: {e}"

def try_openrouter(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ OpenRouter error: {e}"

st.title("📚 Human-in-the-Loop Acceptance Criteria Assistant")
st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
st.markdown(f"**Device Used:** `{device_info}`")

user_story = st.text_area("Enter your User Story:", height=150)
model_choices = st.multiselect("Choose Models:", ["Gemini", "Flan-T5", "LLaMA-3 (Together)", "Cohere", "OpenRouter"], default=["Flan-T5"])
generate = st.button("Generate Acceptance Criteria")

if generate and user_story:
    st.session_state.generated = {}
    prompt = build_prompt(user_story)
    cols = st.columns(len(model_choices))
    for i, model_name in enumerate(model_choices):
        with cols[i]:
            st.subheader(f"{model_name} Output")
            start = time.time()
            if model_name == "Flan-T5":
                output = generate_flan_output(user_story)
            elif model_name == "Gemini":
                output = try_gemini_output(user_story)
            elif model_name == "LLaMA-3 (Together)":
                output = try_llama3_together(user_story)
            elif model_name == "Cohere":
                output = try_cohere(prompt)
            elif model_name == "OpenRouter":
                output = try_openrouter(prompt)
            else:
                output = "❌ Unsupported model"
            st.session_state.generated[model_name] = output
            st.text_area("", value=output, height=200, key=f"out_{model_name}")
            st.caption(f"⏱️ Time taken: {round(time.time() - start, 2)} sec")

if "generated" in st.session_state:
    action = st.radio("What would you like to do?", ("Accept", "Edit", "Regenerate"))
    edited = {}
    for model_name in st.session_state.generated:
        if action == "Edit":
            edited[model_name] = st.text_area(f"Edit {model_name} Output:", value=st.session_state.generated[model_name], height=200)
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
