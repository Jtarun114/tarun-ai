from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
import os

# Load env
load_dotenv()

# FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database setup
engine = create_engine("sqlite:///chat.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String)
    bot_reply = Column(String)

Base.metadata.create_all(bind=engine)

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chat endpoint
@app.post("/chat")
async def chat(message: str = Form(...)):
    db = SessionLocal()

    # Load last 5 messages for memory
    history = db.query(ChatHistory).order_by(ChatHistory.id.desc()).limit(5).all()
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for chat in reversed(history):
        messages.append({"role": "user", "content": chat.user_message})
        messages.append({"role": "assistant", "content": chat.bot_reply})
    messages.append({"role": "user", "content": message})

    # Call OpenAI GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = response.choices[0].message.content

    # Save chat
    new_chat = ChatHistory(user_message=message, bot_reply=reply)
    db.add(new_chat)
    db.commit()
    db.close()

    return {"reply": reply}

# Get chat history
@app.get("/history")
def get_history():
    db = SessionLocal()
    chats = db.query(ChatHistory).all()
    db.close()
    return [{"user_message": c.user_message, "bot_reply": c.bot_reply} for c in chats]

# PDF upload + summarize
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        return {"summary": f"PDF read error: {e}"}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this PDF clearly."},
            {"role": "user", "content": text[:4000]}
        ]
    )
    summary = response.choices[0].message.content
    return {"summary": summary}
