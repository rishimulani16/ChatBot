from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from datetime import datetime
from zoneinfo import ZoneInfo
from prompts import answer_prompt, classify_prompt

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    session_id: str

llm = OllamaLLM(model="llama3")

session_histories = {}

def get_india_time():
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    return now_ist.strftime("The current time in India (IST) is %H:%M on %Y-%m-%d.")

def classify_topic_with_llm(question: str) -> bool:
    formatted = classify_prompt.format(question=question)
    result = llm.invoke(formatted)
    cleaned = result.strip().lower()
    return cleaned.startswith("medical")

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    user_msg_lower = payload.message.lower()

    if "time" in user_msg_lower and "india" in user_msg_lower:
        answer = get_india_time()
        history = session_histories.get(payload.session_id, "")
        new_history = history + f"\nUser: {payload.message}\nAssistant: {answer}"
        session_histories[payload.session_id] = new_history
        return JSONResponse({"answer": answer})

    is_medical = classify_topic_with_llm(payload.message)
    if not is_medical:
        answer = "You are going off topic of medical or healthcare questions. Please ask only health-related questions."
        history = session_histories.get(payload.session_id, "")
        new_history = history + f"\nUser: {payload.message}\nAssistant: {answer}"
        session_histories[payload.session_id] = new_history
        return JSONResponse({"answer": answer})

    history = session_histories.get(payload.session_id, "")
    chain_input = {"history": history, "input": payload.message}
    formatted_prompt = answer_prompt.format(**chain_input)
    answer = llm.invoke(formatted_prompt)
    new_history = history + f"\nUser: {payload.message}\nAssistant: {answer}"
    session_histories[payload.session_id] = new_history
    return JSONResponse({"answer": answer})