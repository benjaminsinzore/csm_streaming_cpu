import asyncio
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import platform
import sqlite3
import time
import threading
import json
import queue
from fastapi.websockets import WebSocketState
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Optional
from generator import Segment, load_csm_1b_local
from llm_interface import LLMInterface
from rag_system import RAGSystem
from vad import AudioStreamProcessor
from pydantic import BaseModel
import logging
from config import ConfigManager
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from pathlib import Path

# JWT Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ... [your existing imports and global variables remain the same] ...

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    session_token = Column(String, unique=True, index=True)
    expires_at = Column(String)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

# ... [your existing Conversation model stays the same] ...

# JWT Token Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# User authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_user(db, email: str, password: str):
    hashed_password = get_password_hash(password)
    user = User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    db = SessionLocal()
    user = get_user_by_email(db, email=email)
    db.close()
    
    if user is None:
        raise credentials_exception
    return user

# Session management for WebSocket connections
class SessionManager:
    def __init__(self):
        self.sessions = {}  # {token: {'connections': [], 'user_data': {}}}
    
    def add_connection(self, token: str, websocket, user_data: dict):
        if token not in self.sessions:
            self.sessions[token] = {'connections': [], 'user_data': user_data}
        self.sessions[token]['connections'].append(websocket)
    
    def remove_connection(self, token: str, websocket):
        if token in self.sessions:
            self.sessions[token]['connections'].remove(websocket)
            if not self.sessions[token]['connections']:
                del self.sessions[token]
    
    def get_user_connections(self, token: str):
        return self.sessions.get(token, {}).get('connections', [])

session_manager = SessionManager()

# ... [your existing global variables remain the same] ...

# Authentication routes
@app.post("/token")
async def login_for_access_token(form_data: UserLogin):
    db = SessionLocal()
    user = authenticate_user(db, form_data.email, form_data.password)
    db.close()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(user_data: UserCreate):
    db = SessionLocal()
    
    # Check if user already exists
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        db.close()
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists"
        )
    
    # Create new user
    user = create_user(db, user_data.email, user_data.password)
    db.close()
    
    # Create access token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# Protected routes
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("chat.html", {"request": request, "user": current_user.email})

@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("setup.html", {"request": request, "user": current_user.email})

# WebSocket with authentication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    if not token:
        await websocket.close(code=1008)  # Policy violation
        return
    
    # Verify token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            await websocket.close(code=1008)
            return
    except JWTError:
        await websocket.close(code=1008)
        return
    
    await websocket.accept()
    
    # Add to session manager with user-specific data
    user_data = {"email": email, "conversation_history": []}
    session_manager.add_connection(token, websocket, user_data)
    
    # Load saved config for this user
    saved = config_manager.load_config()
    if saved:
        await websocket.send_json({"type": "saved_config", "config": saved})
    
    try:
        while True:
            data = await websocket.receive_json()
            # Process messages for this specific session
            if data["type"] == "config":
                try:
                    config_data = data["config"]
                    conf = CompanionConfig(**config_data)
                    saved = config_manager.save_config(config_data)
                    if saved:
                        initialize_models(conf)
                        await websocket.send_json({"type": "status", "message": "Models initialized and configuration saved"})
                    else:
                        await websocket.send_json({"type": "error", "message": "Failed to save configuration"})
                except Exception as e:
                    logger.error(f"Error processing config: {str(e)}")
                    await websocket.send_json({"type": "error", "message": f"Configuration error: {str(e)}"})
            
            # ... [rest of your message handling logic with session-specific data] ...
            
            elif data["type"] == "text_message":
                user_text = data["text"]
                session_id = data.get("session_id", "default")
                
                # Get user-specific conversation history
                user_connections = session_manager.get_user_connections(token)
                # Process for this user's session only
                # You'll need to modify process_user_input to accept session-specific data
                process_user_input_for_user_session(token, user_text, session_id)
            
    except WebSocketDisconnect:
        session_manager.remove_connection(token, websocket)

def process_user_input_for_user_session(token: str, user_text: str, session_id: str):
    # This function should handle user input for a specific user session
    # You'll need to modify your existing process_user_input to work with sessions
    pass

# ... [your existing functions remain the same, but modify them to work with user sessions] ...

# Update your startup event
@app.on_event("startup")
async def startup_event():
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    os.makedirs("static", exist_ok=True)
    os.makedirs("audio/user", exist_ok=True)
    os.makedirs("audio/ai", exist_ok=True)
    os.makedirs("embeddings_cache", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Create login page
    with open("templates/login.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input[type="email"], input[type="password"] { width: 100%; padding: 8px; box-sizing: border-box; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .register-link { margin-top: 10px; }
    </style>
</head>
<body>
    <h2>Login</h2>
    <form id="loginForm">
        <div class="form-group">
            <label for="email">Email:</label>
            <input type="email" id="email" required>
        </div>
        <div class="form-group">
            <label for="password">Password:</label>
            <input type="password" id="password" required>
        </div>
        <button type="submit">Login</button>
    </form>
    <div class="register-link">
        <p>Don't have an account? <a href="/register">Register here</a></p>
    </div>
    <div id="message"></div>

    <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch('/token', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email, password })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    localStorage.setItem('access_token', data.access_token);
                    window.location.href = '/chat';
                } else {
                    const error = await response.json();
                    document.getElementById('message').innerHTML = `<p style="color: red;">${error.detail}</p>`;
                }
            } catch (error) {
                document.getElementById('message').innerHTML = '<p style="color: red;">Login failed</p>';
            }
        });
    </script>
</body>
</html>
        """)
    
    # Create register page
    with open("templates/register.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Register</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input[type="email"], input[type="password"] { width: 100%; padding: 8px; box-sizing: border-box; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .login-link { margin-top: 10px; }
    </style>
</head>
<body>
    <h2>Register</h2>
    <form id="registerForm">
        <div class="form-group">
            <label for="email">Email:</label>
            <input type="email" id="email" required>
        </div>
        <div class="form-group">
            <label for="password">Password:</label>
            <input type="password" id="password" required>
        </div>
        <button type="submit">Register</button>
    </form>
    <div class="login-link">
        <p>Already have an account? <a href="/login">Login here</a></p>
    </div>
    <div id="message"></div>

    <script>
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email, password })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    localStorage.setItem('access_token', data.access_token);
                    window.location.href = '/chat';
                } else {
                    const error = await response.json();
                    document.getElementById('message').innerHTML = `<p style="color: red;">${error.detail}</p>`;
                }
            } catch (error) {
                document.getElementById('message').innerHTML = '<p style="color: red;">Registration failed</p>';
            }
        });
    </script>
</body>
</html>
        """)

    # Create chat page with token handling
    with open("templates/chat.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI Companion - Chat</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        #messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        .message { margin: 5px 0; padding: 5px; }
        .user { background-color: #e3f2fd; }
        .ai { background-color: #f5f5f5; }
        #input-form { display: flex; gap: 10px; }
        #message-input { flex: 1; padding: 10px; }
        button { padding: 10px 20px; }
        #status { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>AI Companion Chat</h1>
    <div id="status">Connecting...</div>
    <div id="messages"></div>
    <form id="input-form">
        <input type="text" id="message-input" placeholder="Type your message..." required>
        <button type="submit">Send</button>
    </form>
    <button id="interrupt-btn">Interrupt</button>

    <script>
        const token = localStorage.getItem('access_token');
        if (!token) {
            window.location.href = '/login';
            exit;
        }

        const ws = new WebSocket(`ws://localhost:8000/ws?token=${token}`);
        
        ws.onopen = () => {
            document.getElementById('status').textContent = 'Connected';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messagesDiv = document.getElementById('messages');
            
            if (data.type === 'transcription') {
                messagesDiv.innerHTML += `<div class="message user"><strong>You:</strong> ${data.text}</div>`;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } else if (data.type === 'response') {
                messagesDiv.innerHTML += `<div class="message ai"><strong>AI:</strong> ${data.text}</div>`;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        };
        
        document.getElementById('input-form').addEventListener('submit', (e) => {
            e.preventDefault();
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (message) {
                ws.send(JSON.stringify({
                    type: 'text_message',
                    text: message,
                    session_id: 'default'
                }));
                input.value = '';
            }
        });
        
        document.getElementById('interrupt-btn').addEventListener('click', () => {
            ws.send(JSON.stringify({ type: 'interrupt' }));
        });
    </script>
</body>
</html>
        """)

    # Redirect root to login
    with open("templates/index.html", "w") as f:
        f.write("""<meta http-equiv="refresh" content="0; url=/login" />""")

    try:
        torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    except:
        pass
    asyncio.create_task(process_message_queue())

# Root route redirects to login
@app.get("/")
async def root():
    return RedirectResponse(url="/login")

if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=lambda: asyncio.run(loop.run_forever()), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)