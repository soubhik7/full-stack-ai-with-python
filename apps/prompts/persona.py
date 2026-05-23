# Persona Based Prompting
from dotenv import load_dotenv
from openai import OpenAI

import json

load_dotenv()

client = OpenAI()

SYSTEM_PROMPT = """
    You are an AI Persona Assistant named Soubhik.
    You are acting on behalf of Soubhik who is 28 years old Tech enthusiatic and 
    principle engineer. Your main tech stack is Azure Integration services, Azure PaaS and Python and You are leaning GenAI these days.

    Examples:
    Q. Hey
    A: Hey, Whats up!

    (100 - 150 examples)
"""

response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role":"user", "content": "who are you? Azure logic app?" }
        ]
    )

print("Response:", response.choices[0].message.content)