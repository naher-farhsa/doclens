import os, time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

keys = [os.getenv(f"GOOGLE_API_KEY_{i}") for i in range(1, 5) if os.getenv(f"GOOGLE_API_KEY_{i}")]
print(f"Found {len(keys)} keys")

for i, key in enumerate(keys):
    try:
        m = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=key, temperature=0)
        r = m.invoke("say hi")
        print(f"Key {i+1}: ✅ WORKS — {r.content[:30]}")
    except Exception as e:
        print(f"Key {i+1}: ❌ FAILED — {str(e)[:80]}")
    time.sleep(5)