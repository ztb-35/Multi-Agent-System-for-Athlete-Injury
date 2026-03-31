import json
import requests
import os
from typing import Optional, Tuple, Dict, Any

class RehabAgent:
    """
    Athlete rehabilitation recommendation agent.
    Uses a free LLM API (OpenRouter) to generate rehab plans.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the agent.
        api_key: API key for the chosen provider.
        provider: 'openai' or 'openrouter' (default from env LLM_PROVIDER, else 'openrouter').
        model: model id for the provider (default from env LLM_MODEL).
        """
        # Provider selection
        self.provider = (provider or os.getenv('LLM_PROVIDER', 'openrouter')).lower()

        # Key selection per provider
        if self.provider == 'openai':
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            # OpenAI chat completions endpoint
            self.api_url = "https://api.openai.com/v1/chat/completions"
            # Default model: user asked for ChatGPT 5; allow override via env
            self.model = model or os.getenv("LLM_MODEL", "gpt-5")
        else:
            # Default to OpenRouter
            self.provider = 'openrouter'
            self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            # Choose a free model by default on OpenRouter; allow override via env
            self.model = model or os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
    
    def assess_athlete(self, athlete_profile: dict) -> dict:
        """
        Generate rehab recommendations for an athlete.
        
        Args:
            athlete_profile: dict with keys:
                - sport, position, sex, age, injury_history, cmas_score, cmas_risk_band, deficits
        
        Returns:
            dict with keys: summary, focus_areas, example_exercises, disclaimer
        """
        prompt = self._build_prompt(athlete_profile)
        
        response = self._call_llm(prompt)
        
        # Parse LLM response into structured format
        result = self._parse_response(response, athlete_profile)
        
        return result
    
    def _build_prompt(self, profile: dict) -> str:
        """Build a detailed prompt for the LLM."""
        return f"""You are an expert sports rehabilitation specialist. Analyze this athlete profile and provide rehab recommendations:

**Athlete Profile:**
- Sport: {profile.get('sport', 'N/A')}
- Position: {profile.get('position', 'N/A')}
- Sex: {profile.get('sex', 'N/A')}
- Age: {profile.get('age', 'N/A')}
- Injury History: {', '.join(profile.get('injury_history', []))}
- CMAS Score: {profile.get('cmas_score', 'N/A')}
- Risk Band: {profile.get('cmas_risk_band', 'N/A')}
- Identified Deficits: {', '.join(profile.get('deficits', []))}

Provide your response in this exact JSON format:
{{
  "summary": "Brief 1-2 sentence assessment of risk and primary concerns",
  "focus_areas": ["area 1", "area 2", "area 3", "area 4"],
  "example_exercises": ["exercise 1", "exercise 2", "exercise 3", "exercise 4"],
  "disclaimer": "Standard clinical disclaimer about consulting licensed professionals"
}}

Only return valid JSON, no additional text."""
    
    def _build_request(self, prompt: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Build URL, headers, and payload for the selected provider."""
        if self.provider == 'openai':
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500,
            }
            return self.api_url, headers, payload
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            # Optional but recommended by OpenRouter; customize if you have a domain/app name
            referer = os.getenv("APP_REFERER")
            app_title = os.getenv("APP_TITLE", "PoseAngle Rehab Agent")
            if referer:
                headers["HTTP-Referer"] = referer
            headers["X-Title"] = app_title

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500,
            }
            return self.api_url, headers, payload

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if not self.api_key:
            # Fallback: return example response for testing
            print("WARNING: No API key provided. Returning example response.")
            return self._example_response()

        url, headers, payload = self._build_request(prompt)

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            # Helpful diagnostics for common auth/model errors
            status = getattr(e, 'response', None).status_code if hasattr(e, 'response') and e.response is not None else None
            if status in (401, 403):
                print(f"API call failed (auth): {status}. Check API key and provider permissions. Provider={self.provider}, Model={self.model}")
            else:
                print(f"API call failed: {e}")
            return self._example_response()
    
    def _parse_response(self, llm_output: str, profile: dict) -> dict:
        """Parse LLM JSON response."""
        try:
            result = json.loads(llm_output)
            result.setdefault("disclaimer", 
                "These are general suggestions and must be reviewed by a licensed clinician.")
            return result
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return {
                "summary": "Unable to generate assessment at this time.",
                "focus_areas": [],
                "example_exercises": [],
                "disclaimer": "Consult a licensed clinician before starting any rehabilitation program."
            }
    
    def _example_response(self) -> str:
        """Return example response for offline testing."""
        return json.dumps({
            "summary": "This athlete has high ACL re-injury risk due to valgus alignment, stiff landing mechanics, and limited deceleration control post-reconstruction.",
            "focus_areas": [
                "Hip and trunk control to reduce dynamic knee valgus",
                "Softer, deeper landings with increased knee flexion",
                "Eccentric lower-body strength for deceleration",
                "Unilateral strength work on the reconstructed side"
            ],
            "example_exercises": [
                "Lateral band walks and single-leg squats with valgus feedback",
                "Drop jumps emphasizing soft landings and trunk alignment",
                "Eccentric split squats and deceleration sprints",
                "Single-leg RDLs and step-downs on the weaker leg"
            ],
            "disclaimer": "These are general suggestions and must be reviewed by a licensed clinician."
        })


if __name__ == "__main__":
    # Example usage. Set OPENAI_API_KEY (for provider=openai) or OPENROUTER_API_KEY (for provider=openrouter)
    athlete = {
        "sport": "soccer",
        "position": "defender",
        "sex": "female",
        "age": 20,
        "injury_history": ["left ACL reconstruction 14 months ago"],
        "cmas_score": 8,
        "cmas_risk_band": "high",
        "deficits": [
            "excessive knee valgus during cutting",
            "limited knee flexion at landing",
            "poor deceleration control",
            "mild trunk control deficit"
        ]
    }

    # Provider and secrets from env
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    model = os.getenv("LLM_MODEL")
    if provider == 'openai':
        key = os.getenv("OPENAI_API_KEY")
    else:
        key = os.getenv("OPENROUTER_API_KEY")

    if model:
        print(f"Using model: {model}")
    print(f"Provider: {provider}")
    if key:
        print(f"{('OPENAI' if provider=='openai' else 'OPENROUTER')}_API_KEY detected; making a live API call...")
    else:
        print("No API key found; running in demo mode (offline example).")

    agent = RehabAgent(api_key=key, provider=provider, model=model)
    result = agent.assess_athlete(athlete)
    print(json.dumps(result, indent=2))




    ####openai router key:      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
    ####groq free key: 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