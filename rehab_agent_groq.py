#!/usr/bin/env python3
"""
rehab_agent_groq.py

MVP script:
- Takes an athlete ACL risk profile (CMAS + deficits)
- Calls Groq's LLM (llama-3.3-70b-versatile)
- Gets back a structured rehab/training suggestion JSON
"""

import os
import json
import textwrap

from groq import Groq
import groq  # for error types


# ==========================
# 1. CONFIG
# ==========================

# Set this environment variable before running:
# export GROQ_API_KEY="your_api_key_here"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY environment variable. "
        "Set it with: export GROQ_API_KEY='your_api_key_here'"
    )

# Init Groq client
client = Groq(api_key=GROQ_API_KEY)

# Use a current Groq model (see console.groq.com/docs/models)
MODEL_NAME = "llama-3.3-70b-versatile"
# If that’s too heavy, you can try:
# MODEL_NAME = "llama-3.1-8b-instant"


# ==========================
# 2. PROMPTS
# ==========================

SYSTEM_PROMPT = """
You are a sports physical therapy assistant.
You provide general, non-diagnostic guidance for training and rehabilitation
based on athlete movement deficits and risk profiles.

Rules:
- You do NOT provide medical diagnoses.
- You do NOT prescribe exact sets, reps, or loads.
- You keep suggestions high-level and evidence-informed.
- You ALWAYS recommend that a licensed clinician review your suggestions.
- You MUST return ONLY valid JSON with these keys:
  ["summary", "focus_areas", "example_exercises", "disclaimer"].
"""

USER_PROMPT_TEMPLATE = """
Given the following athlete risk profile and movement deficits:

{risk_profile_json}

1. Summarize the main risk factors and what they imply.
2. Suggest 3–5 general training/rehab focus areas for a PT or strength coach.
3. Provide 3–5 example exercise TYPES or categories (not specific programming).
4. Add a short disclaimer reminding that a licensed clinician must review.

Return ONLY a JSON object with keys:
- "summary": string
- "focus_areas": list of strings
- "example_exercises": list of strings
- "disclaimer": string

Do not include any extra text outside the JSON.
"""


# ==========================
# 3. CORE CALL FUNCTION
# ==========================

def call_rehab_agent_groq(risk_profile: dict) -> dict:
    """
    Send the risk_profile to Groq LLM and get back a rehab plan dict.

    risk_profile: Python dict with fields like:
        {
          "sport": "soccer",
          "position": "defender",
          "age": 20,
          "sex": "female",
          "injury_history": [...],
          "cmas_score": 8,
          "cmas_risk_band": "high",
          "deficits": [...]
        }
    """

    risk_profile_json = json.dumps(risk_profile, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(risk_profile_json=risk_profile_json)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.3,
            max_tokens=512,  # enough for our JSON
        )
    except groq.APIStatusError as e:
        # Groq returned a non-2xx status code
        print("❌ Groq APIStatusError:")
        print(f"Status: {e.status_code}")
        print(f"Response: {e.response}")
        raise
    except groq.APIConnectionError as e:
        print("❌ Groq APIConnectionError (network issue):")
        print(str(e))
        raise
    except Exception as e:
        print("❌ Unexpected error calling Groq:")
        print(str(e))
        raise

    content = resp.choices[0].message.content.strip()

    # content SHOULD be JSON text; parse it
    try:
        rehab_plan = json.loads(content)
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON from model. Raw content:")
        print(content)
        raise e

    # Basic sanity check
    for key in ["summary", "focus_areas", "example_exercises", "disclaimer"]:
        if key not in rehab_plan:
            raise ValueError(f"Missing key '{key}' in model response JSON.")

    return rehab_plan


# ==========================
# 4. EXAMPLE RISK PROFILE & DEMO
# ==========================

def get_example_risk_profile() -> dict:
    """Your example from the conversation."""
    return {
        "sport": "soccer",
        "position": "defender",
        "age": 20,
        "sex": "female",
        "injury_history": ["left ACL reconstruction 14 months ago"],
        "cmas_score": 8,
        "cmas_risk_band": "high",
        "deficits": [
            "excessive knee valgus during cutting",
            "limited knee flexion on landing",
            "poor deceleration control",
            "mild trunk control deficit",
        ],
    }
#mediccal, conuterfactual

def pretty_print_rehab_plan(rehab_plan: dict) -> None:
    """Nicely print the resulting plan for console demo."""
    print("\n=== SUMMARY ===")
    print(textwrap.fill(rehab_plan["summary"], width=80))

    print("\n=== FOCUS AREAS ===")
    for i, area in enumerate(rehab_plan["focus_areas"], start=1):
        print(f"{i}. {area}")

    print("\n=== EXAMPLE EXERCISES ===")
    for i, ex in enumerate(rehab_plan["example_exercises"], start=1):
        print(f"{i}. {ex}")

    print("\n=== DISCLAIMER ===")
    print(textwrap.fill(rehab_plan["disclaimer"], width=80))
    print()


def main():
    risk_profile = get_example_risk_profile()
    print("Sending this risk profile to Groq LLM:\n")
    print(json.dumps(risk_profile, indent=2))

    print("\nCalling rehab agent...\n")
    rehab_plan = call_rehab_agent_groq(risk_profile)

    print("Raw JSON response:\n")
    print(json.dumps(rehab_plan, indent=2))

    print("\nPretty-printed response:")
    pretty_print_rehab_plan(rehab_plan)



if __name__ == "__main__":
    ####groq free key: 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
    main()
