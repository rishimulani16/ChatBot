from langchain_ollama import OllamaLLM
from prompts import classify_prompt, pii_prompt, mask_prompt

llm = OllamaLLM(model="llama3")


def classify_topic_with_llm(question: str) -> bool:
    formatted = classify_prompt.format(question=question)
    result = llm.invoke(formatted)
    cleaned = result.strip().lower()
    return cleaned.startswith("medical")


def is_pii_request_with_llm(question: str) -> bool:
    formatted = pii_prompt.format(question=question)
    result = llm.invoke(formatted)
    cleaned = result.strip().lower()
    return cleaned.startswith("pii")


def mask_pii_with_llm(user_message: str, answer: str) -> str:
    formatted = mask_prompt.format(user_message=user_message, answer=answer)
    result = llm.invoke(formatted)
    return result.strip()