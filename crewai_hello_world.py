"""
CrewAI minimal hello-world test with Groq via OpenAI-compatible endpoint.
Run this BEFORE crewai_diagnosis_test.py to confirm the LLM connection works.

CrewAI version: 1.15.22
Model: openai/gpt-oss-120b (Groq's OpenAI-compat endpoint)

Key insight from source inspection:
  - LLM.__new__ routes "openai/<model>" to the native OpenAI provider.
  - When base_url is supplied and the model is not in CrewAI's known OpenAI list,
    custom_openai_route is set to True — behaves as a custom OpenAI-compat endpoint.
  - api_key must be passed explicitly (CrewAI does not auto-read GROQ_API_KEY;
    it looks for OPENAI_API_KEY by default for the OpenAI provider).
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()  # loads .env from the project root if present

# --- Sanity check: require GROQ_API_KEY ---
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    sys.exit("ERROR: GROQ_API_KEY environment variable not set.")

from crewai import Agent, Task, Crew, Process, LLM

print("=" * 60)
print("CrewAI Hello-World Smoke Test")
print(f"CrewAI version: 1.15.22")
print(f"Model: openai/gpt-oss-120b via Groq")
print("=" * 60)

# Configure LLM using the "groq/" prefix so CrewAI routes via LiteLLM
# (which has native Groq support). The full model name after the prefix
# is what LiteLLM forwards to Groq — "openai/gpt-oss-120b" is Groq's
# actual model identifier and must be passed through verbatim.
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key,
    max_tokens=200,
)

print("\n[1] LLM object created successfully.")
print(f"    llm.model  = {llm.model}")
print(f"    llm type   = {type(llm).__name__}")

# --- Minimal Agent ---
hello_agent = Agent(
    role="Greeter",
    goal="Respond to a simple greeting with a one-sentence reply.",
    backstory="You are a polite assistant that replies briefly.",
    llm=llm,
    verbose=False,
)

print("\n[2] Agent created.")

# --- Minimal Task ---
hello_task = Task(
    description="Say hello back in exactly one sentence. No JSON, just a plain sentence.",
    expected_output="A single sentence greeting response.",
    agent=hello_agent,
)

print("[3] Task created.")

# --- Minimal Crew ---
hello_crew = Crew(
    agents=[hello_agent],
    tasks=[hello_task],
    process=Process.sequential,
    verbose=False,
)

print("[4] Crew created. Kicking off...\n")

result = hello_crew.kickoff()

print("\n[5] Kickoff complete.")
print(f"\nRaw result:\n{result}")
print("\n" + "=" * 60)
print("Hello-world test PASSED — CrewAI + Groq connection is working.")
print("=" * 60)
