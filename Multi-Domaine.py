import re
import io
import os
import base64
import mimetypes
from pathlib import Path
import hmac
import hashlib
import sqlite3
import tempfile
import subprocess
from datetime import datetime

import faiss
import pymupdf
import streamlit as st

from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-DomaineAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DB_PATH = str(BASE_DIR / "multidomaine.db")


def trouver_background():
    """Trouve automatiquement l'image de background dans assets/."""
    if not ASSETS_DIR.exists():
        return None

    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = [
        path for path in ASSETS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]

    if not images:
        return None

    # Priorité aux fichiers dont le nom contient 'background'.
    images.sort(
        key=lambda path: (
            "background" not in path.stem.lower(),
            path.name.lower(),
        )
    )

    return images[0]


def encoder_background():
    """Encode l'image en data URI pour l'utiliser directement dans le CSS."""
    image_path = trouver_background()

    if image_path is None:
        return None

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


BACKGROUND_IMAGE = encoder_background()

# ============================================================
# TRANSLATIONS
# ============================================================

TEXT = {
    "fr": {
        "app_name": "Multi-DomaineAI",
        "tagline": "Votre assistant documentaire intelligent",
        "pdf": "Documents PDF",
        "upload_pdf": "Charger un ou plusieurs fichiers PDF",
        "image": "Image",
        "upload_image": "Charger une image",
        "build_rag": "Construire le RAG",
        "rag_ready": "RAG prêt",
        "analyze_image": "Analyser l'image",
        "image_understanding": "Compréhension de l'image",
        "question": "Votre question",
        "question_placeholder": "Posez votre question...",
        "send": "Envoyer",
        "assistant": "Assistant",
        "sources": "Sources",
        "login": "Connexion",
        "register": "Créer un compte",
        "logout": "Déconnexion",
        "username": "Nom d'utilisateur",
        "password": "Mot de passe",
        "confirm_password": "Confirmer le mot de passe",
        "login_button": "Se connecter",
        "register_button": "Créer mon compte",
        "guest": "Mode invité",
        "history": "Historique",
        "new_chat": "Nouvelle conversation",
        "no_history": "Aucune conversation enregistrée.",
        "theme": "Thème",
        "light": "Clair",
        "dark": "Sombre",
        "soft": "Soft",
        "language": "Langue",
        "welcome": "Bienvenue",
        "login_to_save": "Connectez-vous pour sauvegarder vos conversations.",
        "guest_not_saved": "Mode invité : cette conversation ne sera pas sauvegardée.",
        "pdf_or_image": "Chargez au moins un PDF ou une image.",
        "empty_question": "Veuillez entrer une question.",
        "rag_success": "RAG construit avec succès.",
        "image_success": "Image analysée avec succès.",
        "login_success": "Connexion réussie.",
        "register_success": "Compte créé avec succès.",
        "logout_success": "Vous êtes déconnecté.",
        "invalid_login": "Nom d'utilisateur ou mot de passe incorrect.",
        "user_exists": "Ce nom d'utilisateur existe déjà.",
        "password_mismatch": "Les mots de passe ne correspondent pas.",
        "password_short": "Le mot de passe doit contenir au moins 8 caractères.",
        "username_short": "Le nom d'utilisateur doit contenir au moins 3 caractères.",
        "processing": "Traitement en cours...",
        "searching": "Recherche dans les documents...",
        "analyzing": "Analyse de l'image...",
        "no_image_analysis": "Analysez d'abord l'image.",
        "no_rag": "Construisez d'abord le RAG.",
        "delete_history": "Supprimer l'historique",
        "history_deleted": "Historique supprimé.",
        "conversation": "Conversation",
        "guest_label": "Invité",
    },
    "en": {
        "app_name": "Multi-DomaineAI",
        "tagline": "Your intelligent document assistant",
        "pdf": "PDF Documents",
        "upload_pdf": "Upload one or more PDF files",
        "image": "Image",
        "upload_image": "Upload an image",
        "build_rag": "Build RAG",
        "rag_ready": "RAG ready",
        "analyze_image": "Analyze image",
        "image_understanding": "Image understanding",
        "question": "Your question",
        "question_placeholder": "Ask your question...",
        "send": "Send",
        "assistant": "Assistant",
        "sources": "Sources",
        "login": "Login",
        "register": "Create account",
        "logout": "Logout",
        "username": "Username",
        "password": "Password",
        "confirm_password": "Confirm password",
        "login_button": "Sign in",
        "register_button": "Create account",
        "guest": "Guest mode",
        "history": "History",
        "new_chat": "New conversation",
        "no_history": "No saved conversations.",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "soft": "Soft",
        "language": "Language",
        "welcome": "Welcome",
        "login_to_save": "Log in to save your conversations.",
        "guest_not_saved": "Guest mode: this conversation will not be saved.",
        "pdf_or_image": "Upload at least one PDF or an image.",
        "empty_question": "Please enter a question.",
        "rag_success": "RAG built successfully.",
        "image_success": "Image analyzed successfully.",
        "login_success": "Login successful.",
        "register_success": "Account created successfully.",
        "logout_success": "You are logged out.",
        "invalid_login": "Invalid username or password.",
        "user_exists": "This username already exists.",
        "password_mismatch": "Passwords do not match.",
        "password_short": "Password must contain at least 8 characters.",
        "username_short": "Username must contain at least 3 characters.",
        "processing": "Processing...",
        "searching": "Searching documents...",
        "analyzing": "Analyzing image...",
        "no_image_analysis": "Analyze the image first.",
        "no_rag": "Build the RAG first.",
        "delete_history": "Delete history",
        "history_deleted": "History deleted.",
        "conversation": "Conversation",
        "guest_label": "Guest",
    },
    "ar": {
        "app_name": "Multi-DomaineAI",
        "tagline": "مساعدك الذكي للوثائق",
        "pdf": "وثائق PDF",
        "upload_pdf": "حمّل ملف PDF واحداً أو أكثر",
        "image": "الصورة",
        "upload_image": "حمّل صورة",
        "build_rag": "بناء RAG",
        "rag_ready": "تم تجهيز RAG",
        "analyze_image": "تحليل الصورة",
        "image_understanding": "فهم الصورة",
        "question": "سؤالك",
        "question_placeholder": "اكتب سؤالك...",
        "send": "إرسال",
        "assistant": "المساعد",
        "sources": "المصادر",
        "login": "تسجيل الدخول",
        "register": "إنشاء حساب",
        "logout": "تسجيل الخروج",
        "username": "اسم المستخدم",
        "password": "كلمة المرور",
        "confirm_password": "تأكيد كلمة المرور",
        "login_button": "تسجيل الدخول",
        "register_button": "إنشاء الحساب",
        "guest": "وضع الضيف",
        "history": "السجل",
        "new_chat": "محادثة جديدة",
        "no_history": "لا توجد محادثات محفوظة.",
        "theme": "المظهر",
        "light": "فاتح",
        "dark": "داكن",
        "soft": "ناعم",
        "language": "اللغة",
        "welcome": "مرحباً",
        "login_to_save": "سجّل الدخول لحفظ محادثاتك.",
        "guest_not_saved": "وضع الضيف: لن يتم حفظ هذه المحادثة.",
        "pdf_or_image": "حمّل ملف PDF أو صورة واحدة على الأقل.",
        "empty_question": "يرجى إدخال سؤال.",
        "rag_success": "تم بناء RAG بنجاح.",
        "image_success": "تم تحليل الصورة بنجاح.",
        "login_success": "تم تسجيل الدخول بنجاح.",
        "register_success": "تم إنشاء الحساب بنجاح.",
        "logout_success": "تم تسجيل الخروج.",
        "invalid_login": "اسم المستخدم أو كلمة المرور غير صحيحة.",
        "user_exists": "اسم المستخدم موجود مسبقاً.",
        "password_mismatch": "كلمتا المرور غير متطابقتين.",
        "password_short": "يجب أن تحتوي كلمة المرور على 8 أحرف على الأقل.",
        "username_short": "يجب أن يحتوي اسم المستخدم على 3 أحرف على الأقل.",
        "processing": "جاري المعالجة...",
        "searching": "جاري البحث في الوثائق...",
        "analyzing": "جاري تحليل الصورة...",
        "no_image_analysis": "قم بتحليل الصورة أولاً.",
        "no_rag": "قم ببناء RAG أولاً.",
        "delete_history": "حذف السجل",
        "history_deleted": "تم حذف السجل.",
        "conversation": "المحادثة",
        "guest_label": "ضيف",
    },
}


def tr(key):
    return TEXT[st.session_state.language].get(key, key)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "language": "fr",
    "theme": "light",
    "user_id": None,
    "username": None,
    "auth_view": "login",
    "active_conversation_id": None,
    "messages": [],
    "description_image": None,
    "image_name": None,
    "rag_ready": False,
    "last_sources": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATABASE
# ============================================================

def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        """)

        db.commit()


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310_000,
    )

    return (
        password_hash.hex(),
        salt.hex(),
    )


def verify_password(password, stored_hash, stored_salt):
    salt = bytes.fromhex(stored_salt)

    calculated_hash, _ = hash_password(
        password,
        salt,
    )

    return hmac.compare_digest(
        calculated_hash,
        stored_hash,
    )


def create_user(username, password):
    password_hash, salt = hash_password(password)

    try:
        with get_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    salt,
                    datetime.now().isoformat(),
                ),
            )

            db.commit()
            return cursor.lastrowid

    except sqlite3.IntegrityError:
        return None


def authenticate_user(username, password):
    with get_db() as db:
        user = db.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if user is None:
        return None

    if verify_password(
        password,
        user["password_hash"],
        user["salt"],
    ):
        return user

    return None


def create_conversation(user_id, title):
    now = datetime.now().isoformat()

    with get_db() as db:
        cursor = db.execute(
            """
            INSERT INTO conversations
            (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                title,
                now,
                now,
            ),
        )

        db.commit()
        return cursor.lastrowid


def save_message(conversation_id, role, content):
    now = datetime.now().isoformat()

    with get_db() as db:
        db.execute(
            """
            INSERT INTO messages
            (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                now,
            ),
        )

        db.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                conversation_id,
            ),
        )

        db.commit()


def get_conversations(user_id):
    with get_db() as db:
        return db.execute(
            """
            SELECT *
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()


def get_messages(conversation_id):
    with get_db() as db:
        return db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()


def delete_history(user_id):
    with get_db() as db:
        conversation_ids = db.execute(
            """
            SELECT id
            FROM conversations
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

        for row in conversation_ids:
            db.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                """,
                (row["id"],),
            )

        db.execute(
            """
            DELETE FROM conversations
            WHERE user_id = ?
            """,
            (user_id,),
        )

        db.commit()


init_db()


# ============================================================
# CUSTOM CSS
# ============================================================

def apply_theme():
    """Applique le thème courant avec un background responsive optionnel."""
    if st.session_state.theme == "dark":
        background = "#070B14"
        background_overlay = (
            "linear-gradient("
            "135deg, rgba(7,11,20,0.94), rgba(15,23,42,0.88))"
        )
        surface = "rgba(15,23,42,0.92)"
        surface_2 = "rgba(30,41,59,0.78)"
        text = "#F8FAFC"
        muted = "#94A3B8"
        border = "rgba(148,163,184,0.18)"
        accent = "#8B5CF6"
        accent_2 = "#3B82F6"

    elif st.session_state.theme == "soft":
        # Deux couleurs principales pour le mode Soft.
        background = "#E8F0FF"
        background_overlay = (
            "linear-gradient("
            "135deg, rgba(232,240,255,0.55), rgba(243,232,255,0.55))"
        )
        surface = "rgba(255,255,255,0.78)"
        surface_2 = "rgba(255,255,255,0.62)"
        text = "#1E293B"
        muted = "#64748B"
        border = "rgba(99,102,241,0.18)"
        accent = "#6366F1"
        accent_2 = "#8B5CF6"

    else:
        background = "#F6F8FC"
        background_overlay = (
            "linear-gradient("
            "135deg, rgba(246,248,252,0.92), rgba(255,255,255,0.86))"
        )
        surface = "rgba(255,255,255,0.88)"
        surface_2 = "rgba(241,245,249,0.82)"
        text = "#0F172A"
        muted = "#64748B"
        border = "rgba(148,163,184,0.24)"
        accent = "#4F46E5"
        accent_2 = "#7C3AED"

    if BACKGROUND_IMAGE:
        app_background = (
            f"background-image: {background_overlay}, "
            f"url('{BACKGROUND_IMAGE}');"
        )
    else:
        app_background = f"background: {background_overlay};"

    direction = "rtl" if st.session_state.language == "ar" else "ltr"
    text_align = "right" if st.session_state.language == "ar" else "left"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {background};
            {app_background}
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: {text};
            min-height: 100vh;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: {surface};
            border-right: 1px solid {border};
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }}

        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 14px 18px;
            margin-bottom: 28px;
            background: {surface};
            border: 1px solid {border};
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(15,23,42,0.05);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            direction: ltr;
        }}

        .brand {{
            font-size: 19px;
            font-weight: 750;
            color: {text};
            white-space: nowrap;
        }}

        .brand-sub {{
            font-size: 12px;
            color: {muted};
            margin-top: 2px;
        }}

        .hero {{
            text-align: center;
            margin: 10px auto 30px auto;
            max-width: 760px;
            direction: {direction};
        }}

        .hero-title {{
            font-size: clamp(30px, 5vw, 48px);
            font-weight: 800;
            letter-spacing: -1.5px;
            color: {text};
            margin-bottom: 8px;
        }}

        .hero-subtitle {{
            font-size: 15px;
            color: {muted};
        }}

        .card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 18px;
            direction: {direction};
            text-align: {text_align};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }}

        .bot-message {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 18px;
            margin: 10px 0 18px 0;
            color: {text};
            line-height: 1.75;
            direction: {direction};
            text-align: {text_align};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }}

        .user-message {{
            background: {surface_2};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 14px 16px;
            margin: 10px 0;
            color: {text};
            direction: {direction};
            text-align: {text_align};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}

        .small-muted {{
            color: {muted};
            font-size: 13px;
        }}

        .status-pill {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: {surface_2};
            border: 1px solid {border};
            color: {muted};
            font-size: 12px;
        }}

        .stButton > button {{
            border-radius: 11px;
            border: 1px solid {border};
            font-weight: 600;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            border-color: {accent};
            box-shadow: 0 9px 22px rgba(99,102,241,0.16);
        }}

        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {accent}, {accent_2});
            color: white;
            border: none;
        }}

        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 12px 26px rgba(99,102,241,0.28);
        }}

        .stTextInput input,
        .stTextInput textarea {{
            border-radius: 11px !important;
        }}

        .auth-box {{
            max-width: 470px;
            margin: 45px auto;
            padding: 28px;
            background: {surface};
            border: 1px solid {border};
            border-radius: 20px;
            box-shadow: 0 15px 45px rgba(15,23,42,0.08);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            direction: {direction};
        }}

        @media (max-width: 900px) {{
            .topbar {{
                gap: 10px;
                padding: 12px;
            }}

            .brand {{
                font-size: 16px;
            }}
        }}

        @media (max-width: 640px) {{
            .hero {{
                margin-top: 4px;
                margin-bottom: 20px;
            }}

            .hero-title {{
                font-size: 32px;
            }}

            .card,
            .bot-message,
            .user-message {{
                border-radius: 14px;
                padding: 15px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP BAR
# ============================================================

def render_topbar():
    left, lang_col, theme_col, account_col = st.columns(
        [4.2, 1.6, 1.8, 1.8],
        vertical_alignment="center",
    )

    with left:
        st.markdown(
            f"""
            <div class="brand">
                📚 {tr("app_name")}
                <div class="brand-sub">{tr("tagline")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lang_col:
        language = st.selectbox(
            tr("language"),
            options=["fr", "en", "ar"],
            format_func=lambda x: {
                "fr": "Français",
                "en": "English",
                "ar": "العربية",
            }[x],
            key="language_selector",
            label_visibility="collapsed",
        )

        if language != st.session_state.language:
            st.session_state.language = language
            st.rerun()

    with theme_col:
        theme = st.selectbox(
            tr("theme"),
            options=["light", "dark", "soft"],
            format_func=lambda x: {
                "light": f"☀️ {tr('light')}",
                "dark": f"🌙 {tr('dark')}",
                "soft": f"🪶 {tr('soft')}",
            }[x],
            key="theme_selector",
            label_visibility="collapsed",
        )

        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

    with account_col:
        if st.session_state.user_id:
            if st.button(
                f"↪ {tr('logout')}",
                use_container_width=True,
            ):
                st.session_state.user_id = None
                st.session_state.username = None
                st.session_state.active_conversation_id = None
                st.session_state.messages = []
                st.rerun()
        else:
            if st.button(
                f"👤 {tr('login')}",
                use_container_width=True,
            ):
                st.session_state.auth_view = "login"
                st.session_state.show_auth = True
                st.rerun()


# ============================================================
# AUTHENTICATION
# ============================================================

def render_auth():
    st.markdown(
        f"""
        <div class="auth-box">
            <h2 style="margin-bottom:6px;">🔐 {tr(st.session_state.auth_view)}</h2>
            <div class="small-muted">
                {tr("login_to_save")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.session_state.auth_view == "login":
            username = st.text_input(
                tr("username"),
                key="login_username",
            )

            password = st.text_input(
                tr("password"),
                type="password",
                key="login_password",
            )

            if st.button(
                tr("login_button"),
                use_container_width=True,
                type="primary",
            ):
                user = authenticate_user(
                    username.strip(),
                    password,
                )

                if user:
                    st.session_state.user_id = user["id"]
                    st.session_state.username = user["username"]
                    st.session_state.messages = []
                    st.session_state.active_conversation_id = None
                    st.session_state.show_auth = False
                    st.success(tr("login_success"))
                    st.rerun()
                else:
                    st.error(tr("invalid_login"))

            if st.button(
                tr("register"),
                use_container_width=True,
            ):
                st.session_state.auth_view = "register"
                st.rerun()

        else:
            username = st.text_input(
                tr("username"),
                key="register_username",
            )

            password = st.text_input(
                tr("password"),
                type="password",
                key="register_password",
            )

            confirm = st.text_input(
                tr("confirm_password"),
                type="password",
                key="register_confirm",
            )

            if st.button(
                tr("register_button"),
                use_container_width=True,
                type="primary",
            ):
                if len(username.strip()) < 3:
                    st.error(tr("username_short"))

                elif len(password) < 8:
                    st.error(tr("password_short"))

                elif password != confirm:
                    st.error(tr("password_mismatch"))

                else:
                    user_id = create_user(
                        username.strip(),
                        password,
                    )

                    if user_id is None:
                        st.error(tr("user_exists"))
                    else:
                        st.session_state.user_id = user_id
                        st.session_state.username = username.strip()
                        st.session_state.messages = []
                        st.session_state.active_conversation_id = None
                        st.session_state.show_auth = False
                        st.success(tr("register_success"))
                        st.rerun()

            if st.button(
                tr("login"),
                use_container_width=True,
            ):
                st.session_state.auth_view = "login"
                st.rerun()

        if st.button(
            "← " + tr("guest"),
            use_container_width=True,
        ):
            st.session_state.show_auth = False
            st.rerun()


# ============================================================
# SIDEBAR / HISTORY
# ============================================================

def render_sidebar():
    with st.sidebar:
        if st.session_state.user_id:
            st.markdown(
                f"### 👤 {st.session_state.username}"
            )

            st.caption(
                tr("login_to_save")
            )

            if st.button(
                f"＋ {tr('new_chat')}",
                use_container_width=True,
            ):
                st.session_state.active_conversation_id = None
                st.session_state.messages = []
                st.rerun()

            st.divider()

            st.markdown(
                f"### 🕘 {tr('history')}"
            )

            conversations = get_conversations(
                st.session_state.user_id
            )

            if not conversations:
                st.caption(
                    tr("no_history")
                )

            for conversation in conversations:
                title = conversation["title"][:34]

                if st.button(
                    title,
                    key=f"conv_{conversation['id']}",
                    use_container_width=True,
                ):
                    st.session_state.active_conversation_id = conversation["id"]

                    rows = get_messages(
                        conversation["id"]
                    )

                    st.session_state.messages = [
                        {
                            "role": row["role"],
                            "content": row["content"],
                        }
                        for row in rows
                    ]

                    st.rerun()

            st.divider()

            if st.button(
                f"🗑️ {tr('delete_history')}",
                use_container_width=True,
            ):
                delete_history(
                    st.session_state.user_id
                )

                st.session_state.active_conversation_id = None
                st.session_state.messages = []

                st.success(
                    tr("history_deleted")
                )
                st.rerun()

        else:
            st.markdown(
                f"### 👤 {tr('guest_label')}"
            )

            st.caption(
                tr("guest_not_saved")
            )

            if st.button(
                f"🔐 {tr('login')}",
                use_container_width=True,
            ):
                st.session_state.auth_view = "login"
                st.session_state.show_auth = True
                st.rerun()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def charger_modele():
    return SentenceTransformer(
        "BAAI/bge-m3"
    )


modele_embedding = charger_modele()


# ============================================================
# API CLIENTS
# ============================================================

@st.cache_resource
def charger_clients():
    groq_client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    gemini_client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    return groq_client, gemini_client


groq_client, gemini_client = charger_clients()


# ============================================================
# OCR
# ============================================================

def faire_ocr(image, lang="fra+ara+eng"):
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(
            temp_dir,
            "page.png",
        )

        output_base = os.path.join(
            temp_dir,
            "ocr",
        )

        image.save(
            image_path,
            format="PNG",
        )

        subprocess.run(
            [
                TESSERACT_CMD,
                image_path,
                output_base,
                "-l",
                lang,
                "--psm",
                "6",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        with open(
            output_base + ".txt",
            "r",
            encoding="utf-8",
        ) as file:
            return file.read().strip()


def extraire_texte_pdf(pdf_bytes, seuil=100, dpi=300):
    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    pages = []

    for i, page in enumerate(doc):
        texte = page.get_text("text").strip()

        if len(texte) < seuil:
            pix = page.get_pixmap(
                dpi=dpi,
                alpha=False,
            )

            image = Image.open(
                io.BytesIO(
                    pix.tobytes("png")
                )
            )

            texte = faire_ocr(image)

        pages.append(
            {
                "page": i + 1,
                "texte": texte,
            }
        )

    doc.close()
    return pages


# ============================================================
# CLEANING + CHUNKING
# ============================================================

def nettoyer_texte(texte):
    texte = re.sub(
        r"\s+",
        " ",
        texte,
    )

    return texte.strip()


def decouper_texte(
    texte,
    taille=900,
    chevauchement=150,
):
    morceaux = []
    debut = 0

    while debut < len(texte):
        fin = min(
            debut + taille,
            len(texte),
        )

        if fin < len(texte):
            espace = texte.rfind(
                " ",
                debut,
                fin,
            )

            if espace > debut:
                fin = espace

        morceau = texte[
            debut:fin
        ].strip()

        if morceau:
            morceaux.append(morceau)

        if fin == len(texte):
            break

        debut = max(
            fin - chevauchement,
            debut + 1,
        )

    return morceaux


def creer_chunks(uploaded_files):
    chunks = []

    for fichier in uploaded_files:
        pages = extraire_texte_pdf(
            fichier.getvalue()
        )

        for page in pages:
            texte = nettoyer_texte(
                page["texte"]
            )

            if not texte:
                continue

            morceaux = decouper_texte(
                texte
            )

            for numero, morceau in enumerate(
                morceaux,
                start=1,
            ):
                chunks.append(
                    {
                        "source": fichier.name,
                        "page": page["page"],
                        "chunk": numero,
                        "texte": morceau,
                    }
                )

    return chunks


# ============================================================
# FAISS
# ============================================================

def creer_index(chunks):
    textes = [
        chunk["texte"]
        for chunk in chunks
    ]

    vecteurs = modele_embedding.encode(
        textes,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(
        vecteurs.shape[1]
    )

    index.add(vecteurs)

    return index


def rechercher(
    question,
    index,
    chunks,
    k=4,
    seuil_score=0.25,
):
    vecteur_question = modele_embedding.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        vecteur_question,
        k,
    )

    resultats = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):
        if idx == -1:
            continue

        score = float(score)

        if score < seuil_score:
            continue

        item = chunks[int(idx)].copy()
        item["score"] = score
        resultats.append(item)

    return resultats


# ============================================================
# GEMINI VISION
# ============================================================

def analyser_image(image_bytes, mime_type):
    prompt = """
Analyse cette image avec précision.

Identifie :
- les éléments principaux ;
- le texte visible ;
- les objets et concepts ;
- les relations entre les éléments ;
- les informations importantes.

Si l'image contient un tableau, schéma,
diagramme ou illustration pédagogique,
explique sa structure et son contenu.

Produis une description claire et exploitable
pour une recherche dans une base documentaire.

Ne fais aucune supposition qui n'est pas justifiée
par l'image.
"""

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    response = gemini_client.models.generate_content(
        model="gemini-3.7-flash",
        contents=[
            image_part,
            prompt,
        ],
    )

    return response.text.strip()


# ============================================================
# PROMPTS
# ============================================================

def construire_prompt_pdf(question, passages):
    contexte = "\n\n".join(
        f'[Source : {p["source"]}, page {p["page"]}]\n'
        f'{p["texte"]}'
        for p in passages
    )

    if not contexte:
        contexte = (
            "Aucun passage suffisamment pertinent "
            "n'a été trouvé."
        )

    return f"""
Tu es un assistant documentaire intelligent.

Réponds uniquement à partir du CONTEXTE fourni.

RÈGLES :
- Réponds de manière claire et concise.
- N'invente aucune information.
- N'utilise pas tes connaissances générales.
- Utilise uniquement les informations du contexte.
- Si l'information est absente, réponds :
« Information non précisée dans les documents fournis. »
- Cite la source et la page lorsque disponibles.

CONTEXTE :
{contexte}

QUESTION :
{question}

RÉPONSE :
""".strip()


def construire_prompt_image(question, description):
    return f"""
Tu es un assistant capable d'analyser des images.

DESCRIPTION DE L'IMAGE :
{description}

QUESTION :
{question}

Réponds uniquement à partir des informations
identifiées dans l'image.

N'invente aucune information.
""".strip()


def construire_prompt_image_pdf(
    question,
    description,
    passages,
):
    contexte = "\n\n".join(
        f'[Source : {p["source"]}, page {p["page"]}]\n'
        f'{p["texte"]}'
        for p in passages
    )

    if not contexte:
        contexte = (
            "Aucun passage pertinent n'a été trouvé "
            "dans les PDF."
        )

    return f"""
Tu es un assistant documentaire multimodal.

DESCRIPTION DE L'IMAGE :
{description}

CONTEXTE DES PDF :
{contexte}

QUESTION :
{question}

RÈGLES :
1. Analyse les informations de l'image.
2. Utilise les PDF pour compléter ou expliquer
   l'image lorsqu'une relation pertinente existe.
3. Ne crée aucune relation artificielle.
4. N'invente aucune information.
5. Distingue les informations provenant de l'image
   et celles provenant des PDF.
6. Cite les sources et pages des PDF utilisées.
7. Si l'information n'est disponible ni dans l'image
   ni dans les PDF, indique-le clairement.

RÉPONSE :
""".strip()


# ============================================================
# GROQ
# ============================================================

def generer_reponse(prompt):
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    return (
        completion
        .choices[0]
        .message
        .content
        .strip()
    )


def repondre_pdf(question, index, chunks):
    passages = rechercher(
        question,
        index,
        chunks,
        k=4,
    )

    reponse = generer_reponse(
        construire_prompt_pdf(
            question,
            passages,
        )
    )

    sources = sorted(
        {
            (p["source"], p["page"])
            for p in passages
        }
    )

    return reponse, sources


def repondre_image(question, description):
    return generer_reponse(
        construire_prompt_image(
            question,
            description,
        )
    )


def repondre_image_pdf(
    question,
    description,
    index,
    chunks,
):
    recherche = (
        f"{question}\n\n"
        f"Informations identifiées dans l'image :\n"
        f"{description}"
    )

    passages = rechercher(
        recherche,
        index,
        chunks,
        k=4,
    )

    reponse = generer_reponse(
        construire_prompt_image_pdf(
            question,
            description,
            passages,
        )
    )

    sources = sorted(
        {
            (p["source"], p["page"])
            for p in passages
        }
    )

    return reponse, sources


# ============================================================
# CHAT HISTORY
# ============================================================

def add_chat_message(role, content):
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
        }
    )


def persist_chat(user_question, assistant_answer):
    if not st.session_state.user_id:
        return

    if not st.session_state.active_conversation_id:
        title = user_question[:60].strip()

        conversation_id = create_conversation(
            st.session_state.user_id,
            title or tr("conversation"),
        )

        st.session_state.active_conversation_id = conversation_id

    save_message(
        st.session_state.active_conversation_id,
        "user",
        user_question,
    )

    save_message(
        st.session_state.active_conversation_id,
        "assistant",
        assistant_answer,
    )


# ============================================================
# APPLY UI
# ============================================================

apply_theme()
render_topbar()
render_sidebar()


# ============================================================
# AUTH SCREEN
# ============================================================

if st.session_state.get("show_auth", False):
    render_auth()
    st.stop()


# ============================================================
# HERO
# ============================================================

direction = "rtl" if st.session_state.language == "ar" else "ltr"

st.markdown(
    f"""
    <div class="hero" dir="{direction}">
        <div class="hero-title">{tr("app_name")}</div>
        <div class="hero-subtitle">{tr("tagline")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# USER STATUS
# ============================================================

if st.session_state.user_id:
    st.markdown(
        f"""
        <div class="card">
            <span class="status-pill">
                👤 {st.session_state.username}
            </span>
            <span class="small-muted">
                &nbsp; {tr("login_to_save")}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info(
        f"👤 {tr('guest_not_saved')}"
    )


# ============================================================
# FILE UPLOADS
# ============================================================

pdf_col, image_col = st.columns(
    2,
    gap="large",
)

with pdf_col:
    st.markdown(
        f"### 📄 {tr('pdf')}"
    )

    uploaded_files = st.file_uploader(
        tr("upload_pdf"),
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

with image_col:
    st.markdown(
        f"### 🖼️ {tr('image')}"
    )

    uploaded_image = st.file_uploader(
        tr("upload_image"),
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader",
    )


# ============================================================
# BUILD RAG
# ============================================================

if uploaded_files:
    if st.button(
        f"🔨 {tr('build_rag')}",
        use_container_width=True,
    ):
        with st.spinner(tr("processing")):
            chunks = creer_chunks(
                uploaded_files
            )

            if not chunks:
                st.error(
                    "❌ Aucun texte exploitable "
                    "n'a été trouvé dans les PDF."
                )
                st.stop()

            index = creer_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.rag_ready = True
            st.session_state.last_sources = []

        st.success(
            f"✅ {tr('rag_ready')} — "
            f"{len(chunks)} chunks."
        )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_image:
    image_bytes = uploaded_image.getvalue()
    mime_type = uploaded_image.type

    st.image(
        image_bytes,
        caption=uploaded_image.name,
        use_container_width=True,
    )

    if st.button(
        f"🔍 {tr('analyze_image')}",
        use_container_width=True,
    ):
        with st.spinner(tr("analyzing")):
            try:
                description = analyser_image(
                    image_bytes,
                    mime_type,
                )

                st.session_state.description_image = description
                st.session_state.image_name = uploaded_image.name

                st.success(
                    f"✅ {tr('image_success')}"
                )

            except Exception as error:
                st.error(
                    f"❌ Gemini : {error}"
                )


if st.session_state.description_image:
    with st.expander(
        f"👁️ {tr('image_understanding')}",
        expanded=False,
    ):
        st.write(
            st.session_state.description_image
        )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.markdown(
    f"### 💬 {tr('assistant')}"
)

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="user-message">
                <strong>Vous</strong><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f"""
            <div class="bot-message">
                <strong>🤖 {tr("assistant")}</strong><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )


question = st.text_input(
    tr("question"),
    placeholder=tr("question_placeholder"),
    key="chat_question",
)


if st.button(
    f"➤ {tr('send')}",
    use_container_width=True,
    type="primary",
):
    if not question.strip():
        st.warning(
            tr("empty_question")
        )
        st.stop()

    has_pdf = (
        "index" in st.session_state
        and "chunks" in st.session_state
        and st.session_state.rag_ready
    )

    has_image = bool(
        st.session_state.description_image
    )

    if not has_pdf and not has_image:
        st.warning(
            tr("pdf_or_image")
        )
        st.stop()

    try:
        if has_image and has_pdf:
            with st.spinner(
                tr("searching")
            ):
                answer, sources = repondre_image_pdf(
                    question,
                    st.session_state.description_image,
                    st.session_state.index,
                    st.session_state.chunks,
                )

        elif has_image:
            with st.spinner(
                tr("analyzing")
            ):
                answer = repondre_image(
                    question,
                    st.session_state.description_image,
                )

            sources = []

        else:
            with st.spinner(
                tr("searching")
            ):
                answer, sources = repondre_pdf(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                )

        add_chat_message(
            "user",
            question,
        )

        add_chat_message(
            "assistant",
            answer,
        )

        st.session_state.last_sources = sources

        # Only authenticated users get permanent storage.
        persist_chat(
            question,
            answer,
        )

        st.rerun()

    except Exception as error:
        st.error(
            f"❌ {error}"
        )


# ============================================================
# SOURCES OF LAST RESPONSE
# ============================================================

if st.session_state.last_sources:
    st.markdown(
        f"### 📚 {tr('sources')}"
    )

    for source, page in st.session_state.last_sources:
        st.write(
            f"📄 {source} — page {page}"
        )
