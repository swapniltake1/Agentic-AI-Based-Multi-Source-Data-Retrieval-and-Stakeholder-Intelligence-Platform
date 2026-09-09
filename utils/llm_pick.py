from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted, TooManyRequests

def pick_llm(level: str):
    level = level.lower().strip()

    model_map = {
        "low": "gemini-2.5-flash-lite",
        "medium": "gemini-2.5-flash",
        "high": "gemini-2.5-pro",
    }

    if level not in model_map:
        raise ValueError("Invalid level. Choose from 'low', 'medium', or 'high'.")

    base_llm = ChatGoogleGenerativeAI(
        model=model_map[level],
        temperature=0,
        max_retries=3
    )

    # Attach LCEL retry with jitter and targeted 429/quota exceptions
    return base_llm.with_retry(
        retry_if_exception_type=(ResourceExhausted, TooManyRequests),
        stop_after_attempt=4,
        wait_exponential_jitter=True
    )