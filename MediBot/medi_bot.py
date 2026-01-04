import os
import json
import base64
import datetime
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv(override=True)


class EvidenceManager:
    def __init__(self, storage_file="medical_records.json"):
        self.storage_file = storage_file
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w') as f:
                json.dump([], f)

    def save_evidence(self, category: str, content: str, metadata: dict = None):
        entry = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "category": category,
            "content": content,
            "metadata": metadata or {}
        }
        with open(self.storage_file, 'r+') as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=4)

    def get_recent_history(self, limit=5) -> str:
        with open(self.storage_file, 'r') as f:
            data = json.load(f)
            last_entries = data[-limit:]
            return "\n".join([f"[{e['date']}] {e['category']}: {e['content']}" for e in last_entries])


class MedicalEvidenceBot:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.model = os.getenv("OPENAI_DEPLOYMENT_NAME", "google/gemini-2.0-flash-exp:free")
        
        self.evidence_db = EvidenceManager()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            default_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "MedEvidence-Bot"}
        )

    def _encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_request(self, text_input: str, image_path: str = None):
       
        past_context = self.evidence_db.get_recent_history()
        
        system_msg = (
            "You are a Medical Assistant. Help users organize symptoms and images. "
            "Refer to the following history:\n" + past_context + 
            "\nDISCLAIMER: You are an AI, not a doctor. Advise seeing a doctor."
        )

        messages = [{"role": "system", "content": system_msg}]

       
        content_list = [{"type": "text", "text": text_input}]
        
        if image_path:
            base64_image = self._encode_image(image_path)
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        messages.append({"role": "user", "content": content_list})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            bot_reply = response.choices[0].message.content
            
           
            category = "Screenshot Analysis" if image_path else "Symptom Log"
            self.evidence_db.save_evidence(category, f"User: {text_input} | Bot: {bot_reply}")
            
            return bot_reply
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    bot = MedicalEvidenceBot()
    print(" Medical Bot Active ")
    print("Type your message here.")

    while True:
        user_in = input("\nYou: ")
        if user_in.lower() == 'exit': break

        if user_in.startswith('/img '):
            path = user_in.replace('/img ', '').strip()
            if os.path.exists(path):
                desc = input("Describe this screenshot : ")
                print("Pitching image...")
                print(f"Bot: {bot.process_request(desc, image_path=path)}")
            else:
                print("File not found.")
        else:
            print(f"Bot: {bot.process_request(user_in)}")