import google.generativeai as genai
#change
GEMINI_KEY = ""  # change

genai.configure(api_key=GEMINI_KEY)

models = genai.list_models()

for m in models:
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
