# ============================================================
# ESTIM AI
# AI-Based Construction Decision-Support System
# ============================================================
#
# EXTRA PACKAGE NEEDED (for GPS location in BuildConnect):
#     pip install streamlit-geolocation
#
# NOTE: Browser camera + GPS only work over HTTPS (or localhost).
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import json
import os
import re
import hashlib
import secrets

from datetime import datetime
from io import BytesIO
from html import escape

# Pillow is used to stamp GPS details onto captured work photos
from PIL import Image, ImageDraw, ImageFont

# GPS location component (optional import so the app never crashes)
try:
    from streamlit_geolocation import streamlit_geolocation
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

# ReportLab is used to generate PDF estimation reports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ESTIM AI",
    page_icon="🏗️",
    layout="wide"
)


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

DB_FILE = "estimai_history.db"

# Folder used for profile and work images
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MENU_OPTIONS = [
    "🏠 Home",
    "📐 Estimation",
    "🔄 What-If Analysis",
    "📁 Project History",
    "📊 Model Results",
    "🤝 BuildConnect",
    "🏛️ Tenders"
]


# ============================================================
# 3. DATABASE CONNECTION
# ============================================================

def db():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(DB_FILE, check_same_thread=False)


# ============================================================
# 4. DATABASE INITIALIZATION
# ============================================================

def init_db():

    con = db()

    # --------------------------------------------------------
    # User accounts (authentication)
    # --------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # Project history table
    # --------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS projects(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            project_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            results_json TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # Construction professional profiles
    # --------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS professionals(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            location TEXT,
            services TEXT,
            contact TEXT,
            description TEXT,
            email TEXT,
            phone TEXT,
            profile_image TEXT,
            work_images TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # Private / unofficial work opportunities
    # --------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS work_posts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT,
            description TEXT,
            location TEXT,
            budget TEXT,
            posted_by TEXT,
            contact TEXT,
            image TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # Messaging table (between registered users)
    # --------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER,
            receiver_id INTEGER,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # Government tender links / records
    # --------------------------------------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS tenders(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            department TEXT,
            location TEXT,
            estimated_value TEXT,
            opening_date TEXT,
            closing_date TEXT,
            status TEXT,
            official_link TEXT,
            created_at TEXT NOT NULL
        )
    """)

    con.commit()
    con.close()


# Run database initialization
init_db()


# ============================================================
# 5. DATABASE MIGRATION
# ============================================================
# This makes the application safer if an older version of
# estimai_history.db already exists.

def add_missing_columns():

    con = db()

    # --------------------------------------------------------
    # Users table migration (in case an older/partial users
    # table exists without the columns this version needs)
    # --------------------------------------------------------
    user_columns = [
        row[1] for row in con.execute("PRAGMA table_info(users)").fetchall()
    ]

    user_new_columns = {
        "username": "TEXT",
        "password_hash": "TEXT",
        "salt": "TEXT",
        "is_admin": "INTEGER DEFAULT 0",
        "created_at": "TEXT"
    }

    for column, data_type in user_new_columns.items():
        if column not in user_columns:
            con.execute(f"ALTER TABLE users ADD COLUMN {column} {data_type}")

    # Check existing professional table columns
    columns = [
        row[1]
        for row in con.execute("PRAGMA table_info(professionals)").fetchall()
    ]

    new_columns = {
        "email": "TEXT",
        "phone": "TEXT",
        "whatsapp_no": "TEXT",
        "whatsapp_enabled": "INTEGER DEFAULT 0",
        "profile_image": "TEXT",
        "work_images": "TEXT",
        # NEW: GPS details (latitude / longitude / time) for each
        # camera-captured work photo, stored as JSON {path: {...}}
        "work_geo": "TEXT",
        "certificate_files": "TEXT",
        "licenses_certificates": "TEXT",
        "max_work_budget": "TEXT",
        "owner_user_id": "INTEGER"
    }

    work_columns = [
        row[1] for row in con.execute("PRAGMA table_info(work_posts)").fetchall()
    ]
    work_new_columns = {
        "whatsapp": "TEXT",
        "email": "TEXT",
        "duration": "TEXT"
    }
    for column, data_type in work_new_columns.items():
        if column not in work_columns:
            con.execute(f"ALTER TABLE work_posts ADD COLUMN {column} {data_type}")

    for column, data_type in new_columns.items():

        if column not in columns:

            con.execute(
                f"ALTER TABLE professionals ADD COLUMN {column} {data_type}"
            )

    con.commit()
    con.close()


add_missing_columns()


# ============================================================
# 6. LOAD MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def load_models():

    return (
        # Building models
        joblib.load("construction_rf_model.pkl"),
        joblib.load("construction_dt_model.pkl"),
        joblib.load("construction_feature_columns.pkl"),

        # Road models
        joblib.load("road_rf_model.pkl"),
        joblib.load("road_dt_model.pkl"),
        joblib.load("road_feature_columns.pkl"),

        # Road target scaler
        joblib.load("road_y_scaler.pkl")
    )


try:

    (
        building_rf,
        building_dt,
        building_cols,
        road_rf,
        road_dt,
        road_cols,
        road_scaler
    ) = load_models()

except FileNotFoundError as e:

    st.error(
        f"Required model file not found: {e.filename}. "
        "Keep all .pkl files beside app.py."
    )

    st.stop()


# ============================================================
# 7. MODEL PERFORMANCE VALUES
# ============================================================

BUILDING_RF_R2 = 0.8385
BUILDING_DT_R2 = 0.70

ROAD_RF_R2 = 0.9632
ROAD_DT_R2 = 0.8623

# Accuracy figures supplied for the presentation
BUILDING_RF_ACCURACY = 85
BUILDING_DT_ACCURACY = 70

ROAD_RF_ACCURACY = 96
ROAD_DT_ACCURACY = 86


# ============================================================
# 8. CONSTRUCTION PROFESSIONAL ROLES
# ============================================================

ROLES = [
    "Builder",
    "Contractor",
    "Crusher Owner",
    "Material Supplier",
    "Government / Civil Professional",
    "Other Construction Professional"
]


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file, prefix="file"):

    if uploaded_file is None:
        return ""

    extension = os.path.splitext(uploaded_file.name)[1]

    filename = (
        f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        f"{extension}"
    )

    path = os.path.join(UPLOAD_FOLDER, filename)

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return path


def delete_file(path):

    if path and os.path.exists(path):

        try:
            os.remove(path)
        except:
            pass


# ============================================================
# 10. UI / CONTACT HELPERS
# ============================================================

def set_flash(message, kind="success"):
    st.session_state["flash_message"] = message
    st.session_state["flash_kind"] = kind

def show_flash():
    message = st.session_state.pop("flash_message", None)
    kind = st.session_state.pop("flash_kind", "success")
    if message:
        if kind == "error": st.error(message)
        elif kind == "warning": st.warning(message)
        elif kind == "info": st.info(message)
        else: st.success(message)

def whatsapp_link(number):
    if not number: return ""
    digits = "".join(ch for ch in str(number) if ch.isdigit())
    if len(digits) == 10: digits = "91" + digits
    return f"https://wa.me/{digits}" if digits else ""

def parse_work_images(value):
    if not value: return []
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else []
    except Exception: return []

def parse_work_geo(value):
    """Return the {image_path: {lat, lon, ...}} dictionary stored in work_geo."""
    if not value: return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception: return {}

def maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def render_work_gallery(work_images, work_geo=None, columns_count=4):
    valid_images = [p for p in parse_work_images(work_images) if p and os.path.exists(p)]
    if not valid_images: return
    geo_map = parse_work_geo(work_geo)
    st.markdown("**🖼️ Work Gallery**")
    cols = st.columns(min(columns_count, len(valid_images)))
    for i, image_path in enumerate(valid_images):
        with cols[i % len(cols)]:
            st.image(image_path, use_container_width=True)
            geo = geo_map.get(image_path)
            if geo:
                st.caption(f"📍 {geo['lat']:.5f}, {geo['lon']:.5f}")
                st.markdown(f"[🗺️ View on map]({maps_link(geo['lat'], geo['lon'])})")


def inject_global_css():
    st.markdown(
        """
        <style>
        /* Center every image (profile photos, work galleries, etc.) */
        div[data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
        div[data-testid="stImage"] img {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 10A. GPS CAMERA CAPTURE FOR WORK PHOTOS
# ============================================================
# Work photos can no longer be picked from the gallery. The user
# opens the camera, the browser GPS location is read, and the photo
# is stamped with latitude / longitude / time before it is saved.

def stamp_photo(image_bytes, lat, lon, captured_at):
    """Burn the GPS coordinates and time into the bottom of the photo."""

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    font_size = max(14, width // 40)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    text = f"GPS: {lat:.6f}, {lon:.6f}  |  {captured_at}"
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = max(6, font_size // 2)

    draw.rectangle(
        [0, height - bbox[3] - 2 * pad, width, height],
        fill=(0, 0, 0, 150)
    )
    draw.text(
        (pad, height - pad - bbox[3]),
        text,
        fill=(255, 255, 255, 255),
        font=font
    )

    out = BytesIO()
    img.save(out, format="JPEG", quality=88)
    return out.getvalue()


def geo_work_capture(prefix):
    """
    Show the camera + GPS capture section.
    Captured photos wait in st.session_state[prefix + '_pending_photos']
    until the profile is saved (see commit_pending_photos).
    """

    pending_key = f"{prefix}_pending_photos"
    ver_key = f"{prefix}_cam_ver"

    if pending_key not in st.session_state:
        st.session_state[pending_key] = []
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0

    st.markdown("**📸 Work Photos (live camera + GPS location)**")
    st.caption(
        "Work photos must be taken live with the camera and are tagged with "
        "your GPS location. Gallery uploads are not allowed for work images."
    )

    pending = st.session_state[pending_key]

    # Photos already captured but not yet saved with the profile
    if pending:
        st.write(f"**{len(pending)} photo(s) captured — they will be saved with your profile.**")
        cols = st.columns(min(4, len(pending)))
        for i, item in enumerate(pending):
            with cols[i % len(cols)]:
                st.image(item["bytes"], use_container_width=True)
                st.caption(f"📍 {item['lat']:.5f}, {item['lon']:.5f}")
                if st.button("❌ Remove", key=f"{prefix}_rm_{item['id']}"):
                    st.session_state[pending_key] = [
                        p for p in pending if p["id"] != item["id"]
                    ]
                    st.rerun()

    open_camera = st.checkbox(
        "📷 Open camera to capture a work photo",
        key=f"{prefix}_open_cam"
    )

    if not open_camera:
        return

    if not GEO_AVAILABLE:
        st.error(
            "GPS component not installed. Run `pip install streamlit-geolocation` "
            "and restart the app."
        )
        return

    # ---------------- Step 1: GPS ----------------
    st.markdown("**Step 1 — Get your current GPS location**")
    st.caption("Click the location button below and allow location access in your browser.")

    try:
        location = streamlit_geolocation(key=f"{prefix}_geo")
    except TypeError:
        # Older versions of the package do not accept a key
        location = streamlit_geolocation()

    lat = location.get("latitude") if isinstance(location, dict) else None
    lon = location.get("longitude") if isinstance(location, dict) else None
    accuracy = location.get("accuracy") if isinstance(location, dict) else None
    has_gps = lat is not None and lon is not None

    if has_gps:
        acc_text = f" (accuracy ≈ {accuracy:.0f} m)" if accuracy else ""
        st.success(f"📍 Location captured: {lat:.6f}, {lon:.6f}{acc_text}")
    else:
        st.info("Waiting for GPS location... click the location button above.")

    # ---------------- Step 2: Camera ----------------
    st.markdown("**Step 2 — Take the work photo**")

    shot = st.camera_input(
        "Take a photo of the work site",
        key=f"{prefix}_cam_{st.session_state[ver_key]}"
    )

    if shot is not None:

        if not has_gps:
            st.warning("A GPS location is required. Complete Step 1 before adding this photo.")
        else:
            if st.button(
                "➕ Add this photo with GPS tag",
                key=f"{prefix}_add_{st.session_state[ver_key]}",
                use_container_width=True
            ):
                captured_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                stamped = stamp_photo(shot.getvalue(), lat, lon, captured_at)

                st.session_state[pending_key].append({
                    "id": secrets.token_hex(4),
                    "bytes": stamped,
                    "lat": float(lat),
                    "lon": float(lon),
                    "accuracy": float(accuracy) if accuracy else None,
                    "captured_at": captured_at
                })

                # New camera widget key = camera resets for the next photo
                st.session_state[ver_key] += 1
                st.rerun()


def commit_pending_photos(prefix):
    """Write pending camera photos to disk. Returns (paths, geo_dict)."""

    items = st.session_state.get(f"{prefix}_pending_photos", [])
    paths = []
    geo = {}

    for item in items:
        filename = (
            f"workgps_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{item['id']}.jpg"
        )
        path = os.path.join(UPLOAD_FOLDER, filename)

        with open(path, "wb") as f:
            f.write(item["bytes"])

        paths.append(path)
        geo[path] = {
            "lat": item["lat"],
            "lon": item["lon"],
            "accuracy": item["accuracy"],
            "captured_at": item["captured_at"]
        }

    return paths, geo


# ============================================================
# 10B. AUTHENTICATION
# ============================================================

COMMON_PASSWORDS = {
    "password", "password1", "password@123", "password123", "qwerty123",
    "12345678", "123456789", "admin123", "admin@123", "welcome1",
    "welcome@123", "abc12345", "iloveyou1", "letmein123", "p@ssw0rd"
}


def validate_username(username):
    """Return a list of problems with the username (empty list = valid)."""

    u = username.strip()
    errors = []

    if not u:
        return ["Username cannot be empty."]

    if not u[0].isalpha():
        errors.append(
            "Username must start with a letter (it cannot start with a number "
            "or symbol, and cannot be only numbers)."
        )

    if len(u) < 4 or len(u) > 20:
        errors.append("Username must be between 4 and 20 characters long.")

    if re.search(r"[^A-Za-z0-9_]", u):
        errors.append(
            "Username may contain only letters, numbers and underscore (_). "
            "No spaces or other symbols."
        )

    return errors


def validate_password(password, username=""):
    """Return a list of problems with the password (empty list = strong)."""

    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")

    if len(password) > 64:
        errors.append("Password must be at most 64 characters long.")

    if re.search(r"\s", password):
        errors.append("Password must not contain spaces.")

    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter (A-Z).")

    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter (a-z).")

    if not re.search(r"[0-9]", password):
        errors.append("Password must contain at least one number (0-9).")

    if not re.search(r"[^A-Za-z0-9\s]", password):
        errors.append("Password must contain at least one special character (e.g. @ # $ % ! &).")

    if username and username.strip().lower() in password.lower():
        errors.append("Password must not contain your username.")

    if password.lower() in COMMON_PASSWORDS:
        errors.append("This password is too common. Choose a less predictable one.")

    return errors


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()
    return pwd_hash, salt


def create_user(username, password, is_admin=False):

    username = username.strip()

    con = db()

    # Case-insensitive duplicate check ("Raj" and "raj" are the same user)
    existing = con.execute(
        "SELECT id FROM users WHERE LOWER(username)=LOWER(?)",
        (username,)
    ).fetchone()

    if existing:
        con.close()
        return False, "That username is already taken."

    pwd_hash, salt = hash_password(password)

    con.execute(
        """
        INSERT INTO users(username, password_hash, salt, is_admin, created_at)
        VALUES(?,?,?,?,?)
        """,
        (
            username,
            pwd_hash,
            salt,
            int(is_admin),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    con.commit()
    con.close()

    return True, "Account created successfully."


def verify_user(username, password):

    con = db()

    row = con.execute(
        "SELECT id, username, password_hash, salt, is_admin FROM users WHERE LOWER(username)=LOWER(?)",
        (username.strip(),)
    ).fetchone()

    con.close()

    if not row:
        return None

    user_id, real_username, stored_hash, salt, is_admin = row

    check_hash, _ = hash_password(password, salt)

    if check_hash == stored_hash:
        return {
            "id": user_id,
            "username": real_username,
            "is_admin": bool(is_admin)
        }

    return None


def auth_page():

    st.markdown(
        """
        <div style="text-align:center; padding:30px 0 10px 0;">
            <h1 style="margin-bottom:0;">🏗️ ESTIM AI</h1>
            <p style="color:#64748B;">AI-Based Construction Decision-Support System</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    _, center, _ = st.columns([1, 1.4, 1])

    with center:

        tab_login, tab_signup = st.tabs(["🔑 Login", "🆕 Sign Up"])

        with tab_login:

            with st.form("login_form"):

                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                submitted = st.form_submit_button(
                    "Login", use_container_width=True
                )

            if submitted:

                user = verify_user(username, password)

                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_signup:

            st.info(
                "**Username rules:** 4–20 characters, must start with a letter, "
                "only letters, numbers and underscore (_).\n\n"
                "**Password rules:** at least 8 characters with an uppercase letter, "
                "a lowercase letter, a number and a special character. "
                "No spaces, and it must not contain your username."
            )

            with st.form("signup_form"):

                new_username = st.text_input("Choose a Username")
                new_password = st.text_input("Choose a Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")

                submitted = st.form_submit_button(
                    "Create Account", use_container_width=True
                )

            if submitted:

                username_errors = validate_username(new_username)
                password_errors = validate_password(new_password, new_username)

                if not new_username.strip() or not new_password:
                    st.error("Please fill in all fields.")
                elif username_errors:
                    for err in username_errors:
                        st.error(err)
                elif password_errors:
                    for err in password_errors:
                        st.error(err)
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    is_admin = new_username.strip().lower() == "admin"
                    ok, msg = create_user(new_username, new_password, is_admin)
                    if ok:
                        st.success(msg + " Please log in from the Login tab.")
                    else:
                        st.error(msg)

    st.stop()


def send_message(sender_user_id, receiver_user_id, text):

    con = db()

    con.execute(
        """
        INSERT INTO messages(sender_id, receiver_id, message, created_at)
        VALUES(?,?,?,?)
        """,
        (
            int(sender_user_id),
            int(receiver_user_id),
            text,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    con.commit()
    con.close()


def get_conversation(user_a, user_b):

    con = db()

    df = pd.read_sql_query(
        """
        SELECT * FROM messages
        WHERE (sender_id=? AND receiver_id=?) OR (sender_id=? AND receiver_id=?)
        ORDER BY id ASC
        """,
        con,
        params=[int(user_a), int(user_b), int(user_b), int(user_a)]
    )

    con.close()

    return df


# ============================================================
# 10C. BUILDING PREDICTION
# ============================================================

def predict_building(values):

    df = pd.DataFrame([values])
    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=building_cols, fill_value=0)

    model = (
        building_rf
        if BUILDING_RF_R2 >= BUILDING_DT_R2
        else building_dt
    )

    prediction = model.predict(df)[0]
    prediction = np.where(np.abs(prediction) < 1e-10, 0, prediction)
    prediction = np.maximum(prediction, 0)

    return dict(
        zip(
            [
                "Cement", "Sand", "Aggregate", "Steel", "Bricks",
                "Paint", "Material Cost", "Labour Cost",
                "Total Construction Cost"
            ],
            map(float, prediction)
        )
    )


# ============================================================
# 11. ROAD PREDICTION
# ============================================================

def predict_road(values):

    df = pd.DataFrame([values])

    df = pd.get_dummies(
        df,
        columns=["Road_Type", "Soil_Type", "Construction_Quality"],
        dtype=int
    )

    df = df.reindex(columns=road_cols, fill_value=0)

    model = (
        road_rf
        if ROAD_RF_R2 >= ROAD_DT_R2
        else road_dt
    )

    scaled_prediction = model.predict(df)
    prediction = road_scaler.inverse_transform(scaled_prediction)[0]

    prediction = np.where(np.abs(prediction) < 1e-10, 0, prediction)
    prediction = np.maximum(prediction, 0)

    return dict(
        zip(
            [
                "Aggregate", "Sand", "Cement", "Bitumen", "Steel",
                "Material Cost", "Labour Cost", "Total Construction Cost"
            ],
            map(float, prediction)
        )
    )


# ============================================================
# 12. PDF REPORT GENERATION
# ============================================================

def create_pdf(project_type, project_name, inputs, results, selected_model="Random Forest"):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=19,
        alignment=TA_CENTER, textColor=colors.HexColor("#11233E"), spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=10,
        alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#2563EB"), spaceBefore=8, spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle", parent=styles["Normal"], fontSize=9, leading=13
    )

    story = [
        Paragraph("ESTIM AI - AI-Based Construction Decision-Support System", title_style),
        Paragraph(f"{escape(str(project_name))} | {project_type} Construction", subtitle_style),
        Paragraph("AI-Based Material Requirement and Cost Estimation", subtitle_style),
        Paragraph("Project Parameters", heading_style)
    ]

    def make_table(data, header_color):

        table = Table(data, colWidths=[230, 270])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2ED")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FC")]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ]))

        return table

    story.append(
        make_table(
            [["Parameter", "Value"]] + [[str(k), str(v)] for k, v in inputs.items()],
            "#11233E"
        )
    )

    story += [Spacer(1, 18), Paragraph("Estimation Results", heading_style)]

    story.append(
        make_table(
            [["Item", "Estimated Value"]] + [[str(k), str(v)] for k, v in results.items()],
            "#0D9488"
        )
    )

    story += [
        Spacer(1, 20),
        Paragraph(f"Selected Model: {escape(str(selected_model))}", normal_style),
        Spacer(1, 8),
        Paragraph("Generated using a trained Machine Learning model.", normal_style),
        Spacer(1, 8),
        Paragraph(
            "Presented by: Madhura Dhatrak | "
            "Diploma in Artificial Intelligence & Machine Learning | "
            "RSM Polytechnic",
            normal_style
        )
    ]

    doc.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 13. BUILDING INPUT FORM
# ============================================================

def building_inputs(prefix):

    c1, c2, c3 = st.columns(3)

    with c1:
        area_type = st.selectbox("Area Type", ["Urban", "Semi-Urban", "Rural"], key=prefix + "_area_type")
        built_up_area = st.number_input("Built-Up Area (sq.ft)", min_value=100.0, value=2000.0, step=100.0, key=prefix + "_area")
        floors = st.number_input("Number of Floors", min_value=1, max_value=10, value=2, step=1, key=prefix + "_floors")

    with c2:
        wall = st.selectbox("Wall Thickness (inches)", [6, 9, 12], key=prefix + "_wall")
        roof = st.selectbox("Roof Type", ["RCC Flat Roof", "Sloped RCC Roof", "Metal Sheet Roof"], key=prefix + "_roof")
        soil = st.selectbox("Soil Type", ["Good", "Medium", "Weak"], key=prefix + "_soil")

    with c3:
        paint = st.selectbox("Paint Type", ["Distemper", "Acrylic Emulsion", "Premium Emulsion"], key=prefix + "_paint")
        flooring = st.selectbox("Flooring", ["Ceramic Tiles", "Vitrified Tiles", "Granite"], key=prefix + "_flooring")
        quality = st.selectbox("Construction Quality", ["Economy", "Standard", "Premium"], key=prefix + "_quality")

    return {
        "Area_Type": area_type,
        "Built_Up_Area_sqft": built_up_area,
        "Floors": floors,
        "Wall_Thickness_inches": wall,
        "Roof_Type": roof,
        "Soil_Type": soil,
        "Paint_Type": paint,
        "Flooring": flooring,
        "Construction_Quality": quality
    }


# ============================================================
# 14. ROAD INPUT FORM
# ============================================================

def road_inputs(prefix):

    c1, c2, c3 = st.columns(3)

    with c1:
        road_type = st.selectbox("Road Type", ["Asphalt", "Concrete"], key=prefix + "_type")
        road_length = st.number_input("Road Length (km)", min_value=0.1, value=5.0, step=0.1, key=prefix + "_length")
        road_width = st.number_input("Road Width (m)", min_value=1.0, value=7.5, step=0.1, key=prefix + "_width")

    with c2:
        road_thickness = st.number_input("Road Thickness (m)", min_value=0.01, value=0.20, step=0.01, key=prefix + "_thickness")
        lanes = st.number_input("Number of Lanes", min_value=1, max_value=8, value=2, step=1, key=prefix + "_lanes")

    with c3:
        soil = st.selectbox("Soil Type", ["Weak", "Medium", "Strong"], key=prefix + "_soil")
        quality = st.selectbox("Construction Quality", ["Standard", "Premium"], key=prefix + "_quality")

    return {
        "Road_Type": road_type,
        "Road_Length_km": road_length,
        "Road_Width_m": road_width,
        "Road_Thickness_m": road_thickness,
        "Number_of_Lanes": lanes,
        "Soil_Type": soil,
        "Construction_Quality": quality
    }


# ============================================================
# 15. INPUT DISPLAY FUNCTIONS
# ============================================================

def building_input_view(v):
    return {
        "Area Type": v["Area_Type"],
        "Built-Up Area": f'{v["Built_Up_Area_sqft"]:,.0f} sq.ft',
        "Number of Floors": v["Floors"],
        "Wall Thickness": f'{v["Wall_Thickness_inches"]} inches',
        "Roof Type": v["Roof_Type"],
        "Soil Type": v["Soil_Type"],
        "Paint Type": v["Paint_Type"],
        "Flooring": v["Flooring"],
        "Construction Quality": v["Construction_Quality"]
    }


def road_input_view(v):
    return {
        "Road Type": v["Road_Type"],
        "Road Length": f'{v["Road_Length_km"]:,.2f} km',
        "Road Width": f'{v["Road_Width_m"]:,.2f} m',
        "Road Thickness": f'{v["Road_Thickness_m"]:,.2f} m',
        "Number of Lanes": v["Number_of_Lanes"],
        "Soil Type": v["Soil_Type"],
        "Construction Quality": v["Construction_Quality"]
    }


# ============================================================
# 16. RESULT DISPLAY FUNCTIONS
# ============================================================

def building_result_view(r):
    return {
        "Cement Required": f'{r["Cement"]:,.2f} bags',
        "Sand Required": f'{r["Sand"]:,.2f} m³',
        "Aggregate Required": f'{r["Aggregate"]:,.2f} m³',
        "Steel Required": f'{r["Steel"]:,.2f} kg',
        "Bricks Required": f'{r["Bricks"]:,.2f} units',
        "Paint Required": f'{r["Paint"]:,.2f} litres',
        "Material Cost": f'₹{r["Material Cost"]:,.2f}',
        "Labour Cost": f'₹{r["Labour Cost"]:,.2f}',
        "Total Construction Cost": f'₹{r["Total Construction Cost"]:,.2f}'
    }


def road_result_view(r):
    return {
        "Aggregate Required": f'{r["Aggregate"]:,.2f} m³',
        "Sand Required": f'{r["Sand"]:,.2f} m³',
        "Cement Required": f'{r["Cement"]:,.2f} tonnes',
        "Bitumen Required": f'{r["Bitumen"]:,.2f} tonnes',
        "Steel Required": f'{r["Steel"]:,.2f} tonnes',
        "Material Cost": f'₹{r["Material Cost"]:,.2f}',
        "Labour Cost": f'₹{r["Labour Cost"]:,.2f}',
        "Total Construction Cost": f'₹{r["Total Construction Cost"]:,.2f}'
    }


# ============================================================
# 17. SAVE PROJECT
# ============================================================

def save_project(name, project_type, inputs, results):

    con = db()

    con.execute(
        """
        INSERT INTO projects(project_name, project_type, created_at, inputs_json, results_json)
        VALUES(?,?,?,?,?)
        """,
        (
            name, project_type,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps(inputs), json.dumps(results)
        )
    )

    con.commit()
    con.close()


# ============================================================
# 18. SHOW BUILDING RESULTS
# ============================================================

def show_building_results(r):

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🧱 Cement", f'{r["Cement"]:,.2f} bags')
        st.metric("🏖️ Sand", f'{r["Sand"]:,.2f} m³')
        st.metric("🪨 Aggregate", f'{r["Aggregate"]:,.2f} m³')

    with c2:
        st.metric("🔩 Steel", f'{r["Steel"]:,.2f} kg')
        st.metric("🧱 Bricks", f'{r["Bricks"]:,.2f} units')
        st.metric("🎨 Paint", f'{r["Paint"]:,.2f} litres')

    with c3:
        st.metric("Material Cost", f'₹{r["Material Cost"]:,.2f}')
        st.metric("Labour Cost", f'₹{r["Labour Cost"]:,.2f}')
        st.metric("Total Cost", f'₹{r["Total Construction Cost"]:,.2f}')


# ============================================================
# 19. SHOW ROAD RESULTS
# ============================================================

def show_road_results(r):

    c1, c2 = st.columns(2)

    with c1:
        st.metric("🪨 Aggregate", f'{r["Aggregate"]:,.2f} m³')
        st.metric("🏖️ Sand", f'{r["Sand"]:,.2f} m³')
        st.metric("🧱 Cement", f'{r["Cement"]:,.2f} tonnes')
        st.metric("🛢️ Bitumen", f'{r["Bitumen"]:,.2f} tonnes')

    with c2:
        st.metric("🔩 Steel", f'{r["Steel"]:,.2f} tonnes')
        st.metric("Material Cost", f'₹{r["Material Cost"]:,.2f}')
        st.metric("Labour Cost", f'₹{r["Labour Cost"]:,.2f}')
        st.metric("Total Cost", f'₹{r["Total Construction Cost"]:,.2f}')


# ============================================================
# 20. HOME PAGE (interactive)
# ============================================================

def get_tender_count():
    con = db()
    count = con.execute("SELECT COUNT(*) FROM tenders").fetchone()[0]
    con.close()
    return count


def go_to(page_name):
    # The sidebar radio is created with key="nav_radio", so Streamlit reads
    # its selection from st.session_state["nav_radio"] on every rerun.
    # Setting that key directly (before the widget is recreated) is what
    # actually moves the selection — a separate "nav_page" variable would
    # be ignored once the widget's own session-state key exists.
    st.session_state["nav_radio"] = page_name
    st.rerun()


def home_page(current_user):

    st.markdown(
        """
        <style>
        .estim-hero { border-radius:22px; padding:28px 30px; color:white;
            background:linear-gradient(135deg,#0f2747,#123b63 58%,#0d9488);
            box-shadow:0 10px 30px rgba(15,39,71,.18); }
        .estim-chip { display:inline-block; padding:6px 11px; margin:7px 5px 0 0;
            border-radius:999px; background:rgba(255,255,255,.13);
            border:1px solid rgba(255,255,255,.18); font-size:13px; }
        .estim-art { transition:transform .25s ease; }
        .estim-art:hover { transform:scale(1.015) translateY(-3px); }
        </style>
        <div class="estim-hero">
          <h1 style="margin:0 0 6px 0">🏗️ ESTIM AI</h1>
          <p style="margin:0;opacity:.92;font-size:16px">AI-Based Construction Decision-Support System</p>
          <div>
            <span class="estim-chip">🤖 Machine Learning</span>
            <span class="estim-chip">🏠 Building</span>
            <span class="estim-chip">🛣️ Road</span>
            <span class="estim-chip">📊 Prediction</span>
            <span class="estim-chip">💰 Cost Estimation</span>
          </div>
          <div class="estim-art" style="margin-top:18px">
            <svg width="100%" height="150" viewBox="0 0 900 150" xmlns="http://www.w3.org/2000/svg">
              <rect width="900" height="150" rx="16" fill="rgba(255,255,255,.06)"/>
              <path d="M25 120H310L395 88H875" stroke="#dbeafe" stroke-width="11" fill="none" stroke-linecap="round"/>
              <path d="M410 88H875" stroke="white" stroke-width="3" stroke-dasharray="18 14"/>
              <rect x="70" y="57" width="170" height="63" rx="4" fill="#e2e8f0"/>
              <path d="M55 58L155 20L255 58Z" fill="#38bdf8"/>
              <rect x="95" y="80" width="34" height="40" fill="#94a3b8"/>
              <rect x="150" y="77" width="55" height="28" fill="#7dd3fc"/>
              <circle cx="535" cy="52" r="35" fill="#0d9488" stroke="#ccfbf1" stroke-width="4"/>
              <text x="535" y="62" text-anchor="middle" font-size="28" fill="white">AI</text>
              <path d="M275 65C350 10 420 10 500 45" stroke="#5eead4" stroke-width="4" fill="none" stroke-dasharray="8 8"/>
              <path d="M570 52C650 35 720 42 800 75" stroke="#93c5fd" stroke-width="4" fill="none" stroke-dasharray="8 8"/>
              <circle cx="800" cy="75" r="24" fill="#2563eb"/>
              <text x="800" y="84" text-anchor="middle" font-size="23" fill="white">₹</text>
            </svg>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"Welcome back, **{current_user['username']}**! Smarter Estimates. Better Planning. Faster Construction.")

    con = db()
    project_count = con.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    professional_count = con.execute("SELECT COUNT(*) FROM professionals").fetchone()[0]
    work_count = con.execute("SELECT COUNT(*) FROM work_posts").fetchone()[0]
    my_messages = con.execute(
        "SELECT COUNT(*) FROM messages WHERE sender_id=? OR receiver_id=?",
        (current_user["id"], current_user["id"])
    ).fetchone()[0]
    recent_projects = pd.read_sql_query(
        "SELECT project_name, project_type, created_at FROM projects ORDER BY id DESC LIMIT 5",
        con
    )
    con.close()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🏠 Building R²", f"{BUILDING_RF_R2:.4f}")
    with c2:
        st.metric("🛣️ Road R²", f"{ROAD_RF_R2:.4f}")
    with c3:
        st.metric("📁 Saved Projects", project_count)
    with c4:
        st.metric("🤝 Professionals", professional_count)

    st.divider()

    st.subheader("🔄 ESTIM AI Workflow")

    st.info(
        "User Input → Data Preprocessing → "
        "Trained ML Model → Prediction → "
        "Material Quantities + Cost → "
        "Result Display → PDF Report"
    )

    st.divider()

    st.subheader("🌐 Platform Activity")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("💼 Work Opportunities", work_count)
    with c2:
        st.metric("💬 Your Messages", my_messages)
    with c3:
        st.metric("🏛️ Tender Records", get_tender_count())

    st.divider()

    st.subheader("⚡ Main Features — jump right in")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown("### 📐 Estimation\nPredict construction materials and cost.")
        if st.button("Open Estimation", key="nav_est", use_container_width=True):
            go_to("📐 Estimation")

    with f2:
        st.markdown("### 🔄 What-If\nChange parameters and compare estimates.")
        if st.button("Open What-If", key="nav_whatif", use_container_width=True):
            go_to("🔄 What-If Analysis")

    with f3:
        st.markdown("### 🤝 BuildConnect\nFind construction professionals and opportunities.")
        if st.button("Open BuildConnect", key="nav_bc", use_container_width=True):
            go_to("🤝 BuildConnect")

    with f4:
        st.markdown("### 🏛️ Tenders\nAccess official government tender portals.")
        if st.button("Open Tenders", key="nav_tenders", use_container_width=True):
            go_to("🏛️ Tenders")

    if not recent_projects.empty:
        st.divider()
        st.subheader("🕓 Recently Saved Projects")
        st.dataframe(recent_projects, use_container_width=True, hide_index=True)


# ============================================================
# 21. ESTIMATION PAGE
# ============================================================

def estimation():

    st.header("📐 Construction Estimation")

    st.write("Enter project parameters to generate an AI-based construction estimate.")

    project_type = st.radio("Select Construction Type", ["Building", "Road"], horizontal=True)

    project_name = st.text_input("Project Name", "My Construction Project")

    if project_type == "Building":

        st.subheader("🏠 Building Construction")

        values = building_inputs("estimate_building")

        st.markdown("### 🤖 Model Selection")
        st.write("The trained models are compared using R² Score before the final prediction.")
        building_comparison = pd.DataFrame({
            "Model": ["Random Forest", "Decision Tree"],
            "R² Score": [BUILDING_RF_R2, BUILDING_DT_R2]
        })
        st.dataframe(building_comparison, use_container_width=True, hide_index=True)
        building_selected = "Random Forest" if BUILDING_RF_R2 >= BUILDING_DT_R2 else "Decision Tree"
        st.info(f"Selected model for final prediction: **{building_selected}**")

        if st.button("🔮 Predict Building Estimate", use_container_width=True):

            results = predict_building(values)
            save_project(project_name, "Building", values, results)

            st.success("Estimate generated and saved successfully!")
            st.subheader("🤖 Selected Model")
            st.success(f"{building_selected} selected for Building estimation.")

            show_building_results(results)

            pdf_data = create_pdf(
                "Building", project_name,
                building_input_view(values), building_result_view(results),
                building_selected
            )

            st.download_button(
                "📄 Download Building PDF", pdf_data,
                f"{project_name}_Building.pdf", "application/pdf",
                use_container_width=True
            )

    else:

        st.subheader("🛣️ Road Construction")

        values = road_inputs("estimate_road")

        st.markdown("### 🤖 Model Selection")
        st.write("The trained models are compared using R² Score before the final prediction.")
        road_comparison = pd.DataFrame({
            "Model": ["Random Forest", "Decision Tree"],
            "R² Score": [ROAD_RF_R2, ROAD_DT_R2]
        })
        st.dataframe(road_comparison, use_container_width=True, hide_index=True)
        road_selected = "Random Forest" if ROAD_RF_R2 >= ROAD_DT_R2 else "Decision Tree"
        st.info(f"Selected model for final prediction: **{road_selected}**")

        if st.button("🔮 Predict Road Estimate", use_container_width=True):

            results = predict_road(values)
            save_project(project_name, "Road", values, results)

            st.success("Estimate generated and saved successfully!")
            st.subheader("🤖 Selected Model")
            st.success(f"{road_selected} selected for Road estimation.")

            show_road_results(results)

            pdf_data = create_pdf(
                "Road", project_name,
                road_input_view(values), road_result_view(results),
                road_selected
            )

            st.download_button(
                "📄 Download Road PDF", pdf_data,
                f"{project_name}_Road.pdf", "application/pdf",
                use_container_width=True
            )


# ============================================================
# 22. WHAT-IF ANALYSIS
# ============================================================

def what_if():

    st.header("🔄 What-If Analysis")

    st.write("Change project parameters and compare the resulting ML estimate with the original estimate.")

    project_type = st.radio("Construction Type", ["Building", "Road"], horizontal=True, key="what_if_type")

    st.markdown("### Original Parameters")
    if project_type == "Building":
        old_values = building_inputs("old_building")
    else:
        old_values = road_inputs("old_road")

    st.markdown("### Changed Parameters")
    if project_type == "Building":
        new_values = building_inputs("new_building")
    else:
        new_values = road_inputs("new_road")

    if st.button("🔄 Compare Original vs Changed", use_container_width=True):

        if project_type == "Building":
            original = predict_building(old_values)
            changed = predict_building(new_values)
        else:
            original = predict_road(old_values)
            changed = predict_road(new_values)

        rows = []
        for item in original.keys():
            rows.append({
                "Item": item,
                "Original": original[item],
                "Changed": changed[item],
                "Difference": changed[item] - original[item]
            })

        result_df = pd.DataFrame(rows)

        st.subheader("📊 Comparison")

        st.dataframe(
            result_df.style.format(
                {"Original": "{:,.2f}", "Changed": "{:,.2f}", "Difference": "{:,.2f}"}
            ),
            use_container_width=True,
            hide_index=True
        )

        difference = changed["Total Construction Cost"] - original["Total Construction Cost"]

        if difference > 0:
            st.warning(f"Estimated total cost increases by ₹{difference:,.2f}.")
        elif difference < 0:
            st.info(f"Estimated total cost decreases by ₹{abs(difference):,.2f}.")
        else:
            st.info("No change in estimated total cost.")


# ============================================================
# 23. PROJECT HISTORY
# ============================================================

def history():

    st.header("📁 Project History")

    search = st.text_input("🔎 Search by project name")
    project_type = st.selectbox("Filter by Type", ["All", "Building", "Road"])

    con = db()
    query = "SELECT * FROM projects WHERE project_name LIKE ?"
    params = [f"%{search}%"]

    if project_type != "All":
        query += " AND project_type = ?"
        params.append(project_type)

    query += " ORDER BY id DESC"

    df = pd.read_sql_query(query, con, params=params)
    con.close()

    if df.empty:
        st.info("No saved projects found.")
        return

    st.dataframe(
        df[["id", "project_name", "project_type", "created_at"]].rename(
            columns={
                "id": "ID", "project_name": "Project Name",
                "project_type": "Type", "created_at": "Created At"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    selected_id = st.selectbox(
        "Select Project",
        df.id.tolist(),
        format_func=lambda x: f'{x} - {df.loc[df.id == x, "project_name"].iloc[0]}'
    )

    row = df[df.id == selected_id].iloc[0]
    inputs = json.loads(row.inputs_json)
    results = json.loads(row.results_json)

    if row.project_type == "Building":
        input_view = building_input_view(inputs)
        result_view = building_result_view(results)
    else:
        input_view = road_input_view(inputs)
        result_view = road_result_view(results)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📝 Inputs")
        st.json(input_view)

    with c2:
        st.subheader("📊 Results")
        st.json(result_view)

    pdf_data = create_pdf(row.project_type, row.project_name, input_view, result_view)

    st.download_button(
        "📄 Download Previous PDF", pdf_data,
        f"{row.project_name}_History.pdf", "application/pdf",
        use_container_width=True
    )

    if st.button("🗑️ Delete Selected Project", use_container_width=True):
        con = db()
        con.execute("DELETE FROM projects WHERE id=?", (int(selected_id),))
        con.commit()
        con.close()
        st.success("Project deleted successfully.")
        st.rerun()


# ============================================================
# 24. BUILDCONNECT - PROFESSIONAL NETWORK
# ============================================================

def buildconnect(current_user):

    st.header("🤝 BuildConnect")

    st.write(
        "Connect builders, contractors, crusher owners, "
        "material suppliers and other construction professionals. "
        "Profiles can include licenses, certificates, project-budget "
        "capacity, profile photos and GPS-tagged work galleries. You can "
        "message any professional directly from their profile card below."
    )

    tabs = st.tabs([
        "🔎 Find Professionals",
        "👤 Create Profile",
        "✏️ Edit My Profile",
        "🗑️ Delete My Profile"
    ])

    # ========================================================
    # TAB 1 - FIND PROFESSIONALS (+ integrated messaging)
    # ========================================================

    with tabs[0]:

        st.subheader("🔎 Find Construction Professionals")

        c1, c2 = st.columns(2)

        with c1:
            role_filter = st.selectbox("Professional Type", ["All"] + ROLES, key="find_role")

        with c2:
            location_filter = st.text_input("📍 Search Location", key="find_location")

        con = db()
        query = "SELECT * FROM professionals WHERE 1=1"
        params = []

        if role_filter != "All":
            query += " AND role = ?"
            params.append(role_filter)

        if location_filter:
            query += " AND location LIKE ?"
            params.append(f"%{location_filter}%")

        query += " ORDER BY id DESC"

        df = pd.read_sql_query(query, con, params=params)
        con.close()

        if df.empty:
            st.info("No professional profiles found.")
        else:
            for _, profile in df.iterrows():

                with st.container(border=True):

                    c1, c2 = st.columns([1, 4])

                    with c1:
                        if profile["profile_image"] and os.path.exists(profile["profile_image"]):
                            st.image(profile["profile_image"], width=130)
                        else:
                            st.markdown("<div style='text-align:center;font-size:48px;'>👤</div>", unsafe_allow_html=True)

                    with c2:

                        st.subheader(profile["name"])
                        st.write(f"**Role:** {profile['role']}")
                        st.write(f"**📍 Location:** {profile['location']}")
                        st.write(f"**Services / Materials:** {profile['services']}")
                        st.write(f"**Description:** {profile['description']}")

                        if "licenses_certificates" in profile.index and profile["licenses_certificates"]:
                            st.write(f"**📜 Licenses / Certificates:** {profile['licenses_certificates']}")

                        certificate_files_display = parse_work_images(
                            profile["certificate_files"] if "certificate_files" in profile.index else "[]"
                        )
                        valid_certificates = [p for p in certificate_files_display if p and os.path.exists(p)]
                        if valid_certificates:
                            with st.expander("📎 View Uploaded Licenses / Certificates"):
                                for cert_path in valid_certificates:
                                    with open(cert_path, "rb") as cert_file:
                                        st.download_button(
                                            "⬇️ Download " + os.path.basename(cert_path),
                                            data=cert_file.read(),
                                            file_name=os.path.basename(cert_path),
                                            key=f"cert_{profile['id']}_{os.path.basename(cert_path)}"
                                        )

                        if "max_work_budget" in profile.index and profile["max_work_budget"]:
                            st.write(f"**💰 Maximum Work / Project Budget:** {profile['max_work_budget']}")

                        if profile["email"]:
                            st.write(f"📧 {profile['email']}")

                        if profile["phone"]:
                            st.write(f"📞 {profile['phone']}")

                        if (
                            "whatsapp_no" in profile.index
                            and "whatsapp_enabled" in profile.index
                            and int(profile["whatsapp_enabled"] or 0) == 1
                            and profile["whatsapp_no"]
                        ):
                            wa_url = whatsapp_link(profile["whatsapp_no"])
                            if wa_url:
                                st.link_button("💬 Contact on WhatsApp", wa_url, use_container_width=True)

                    render_work_gallery(
                        profile["work_images"],
                        profile["work_geo"] if "work_geo" in profile.index else None
                    )

                    # ------------------------------------------------
                    # Integrated in-app messaging (per profile)
                    # ------------------------------------------------
                    owner_id = profile["owner_user_id"] if "owner_user_id" in profile.index else None

                    if owner_id and not pd.isna(owner_id) and int(owner_id) == current_user["id"]:
                        st.caption("This is your own profile.")
                    elif owner_id and not pd.isna(owner_id):
                        with st.expander(f"💬 Message {profile['name']}"):
                            conv = get_conversation(current_user["id"], int(owner_id))
                            if conv.empty:
                                st.caption("No messages yet. Start the conversation below.")
                            else:
                                for _, m in conv.iterrows():
                                    who = "You" if m["sender_id"] == current_user["id"] else profile["name"]
                                    st.markdown(f"**{who}:** {m['message']}  \n*{m['created_at']}*")
                            msg_text = st.text_area("Write a message", key=f"msg_text_{profile['id']}")
                            if st.button("📨 Send", key=f"msg_send_{profile['id']}", use_container_width=True):
                                if msg_text.strip():
                                    send_message(current_user["id"], int(owner_id), msg_text.strip())
                                    st.rerun()
                                else:
                                    st.error("Please write a message before sending.")
                    else:
                        st.caption("This professional hasn't linked an account yet — use the contact details above.")

    # ========================================================
    # TAB 2 - CREATE PROFILE
    # ========================================================
    # This tab no longer uses st.form: the live camera and GPS
    # components must be able to interact with the page immediately,
    # which a form would block. A version number in every widget key
    # is bumped after saving so the whole form resets cleanly.

    with tabs[1]:

        st.subheader("👤 Create Your Professional Profile")
        st.info("Create your profile once. You can edit or delete it later from the other tabs.")

        cv = st.session_state.get("create_profile_ver", 0)

        name = st.text_input("Name / Business Name", key=f"cp_name_{cv}")
        role = st.selectbox("Professional Type", ROLES, key=f"cp_role_{cv}")
        location = st.text_input("📍 Location / City", key=f"cp_location_{cv}")
        email = st.text_input("📧 Email", key=f"cp_email_{cv}")
        phone = st.text_input("📞 Phone / Direct Contact", key=f"cp_phone_{cv}")
        whatsapp_no = st.text_input(
            "💬 WhatsApp Number (Optional)",
            placeholder="Example: 9876543210",
            key=f"cp_wa_{cv}"
        )
        whatsapp_enabled = st.checkbox(
            "Show WhatsApp contact button on my profile",
            value=False,
            key=f"cp_wa_enabled_{cv}"
        )
        services = st.text_input("Services / Materials", key=f"cp_services_{cv}")
        description = st.text_area("Short Description", key=f"cp_desc_{cv}")

        licenses_certificates = st.text_area(
            "📜 Licenses / Certificates (Optional)",
            placeholder="Example: Contractor License, Civil Engineering Certificate, GST/Registration, Safety Certificate...",
            key=f"cp_lic_{cv}"
        )

        certificate_files = st.file_uploader(
            "📎 Upload License / Certificate Documents (Optional)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"cp_cert_files_{cv}"
        )

        max_work_budget = st.text_input(
            "💰 Maximum Work / Project Budget You Can Handle (Optional)",
            placeholder="Example: Up to ₹1.5 Crore",
            key=f"cp_budget_{cv}"
        )

        profile_image = st.file_uploader(
            "📷 Profile Image", type=["png", "jpg", "jpeg"], key=f"cp_profile_img_{cv}"
        )

        st.divider()

        # Work photos: live camera + GPS only (no gallery upload)
        geo_work_capture(f"create_{cv}")

        st.divider()

        submit = st.button("➕ Create Profile", use_container_width=True, key=f"cp_submit_{cv}")

        if submit:

            if not name.strip():
                st.error("Please enter your name or business name.")
            else:

                profile_path = save_uploaded_file(profile_image, "profile")

                # GPS-tagged camera photos
                work_paths, work_geo = commit_pending_photos(f"create_{cv}")

                certificate_paths = []
                for certificate in certificate_files:
                    path = save_uploaded_file(certificate, "certificate")
                    if path:
                        certificate_paths.append(path)

                con = db()

                con.execute(
                    """
                    INSERT INTO professionals(
                        name, role, location, services, contact, description,
                        email, phone, whatsapp_no, whatsapp_enabled,
                        profile_image, work_images, work_geo, certificate_files,
                        licenses_certificates, max_work_budget, owner_user_id, created_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        name.strip(), role, location.strip(), services.strip(),
                        phone.strip(), description.strip(), email.strip(), phone.strip(),
                        whatsapp_no.strip(), int(whatsapp_enabled and bool(whatsapp_no.strip())),
                        profile_path, json.dumps(work_paths), json.dumps(work_geo),
                        json.dumps(certificate_paths),
                        licenses_certificates.strip(), max_work_budget.strip(),
                        current_user["id"],
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )

                con.commit()
                con.close()

                # Reset the whole form by moving to a fresh set of widget keys
                st.session_state["create_profile_ver"] = cv + 1

                set_flash("✅ Profile created successfully! You can now find, edit or delete it from BuildConnect.")
                st.rerun()

    # ========================================================
    # TAB 3 - EDIT MY PROFILE
    # ========================================================

    with tabs[2]:

        st.subheader("✏️ Edit Your Profile")

        con = db()
        profiles = pd.read_sql_query(
            "SELECT * FROM professionals WHERE owner_user_id=? ORDER BY id DESC",
            con, params=[current_user["id"]]
        )
        con.close()

        if profiles.empty:
            st.info("You don't have a profile yet. Create one from the 'Create Profile' tab.")
        else:

            selected_id = st.selectbox(
                "Select Your Profile",
                profiles.id.tolist(),
                format_func=lambda x: f'{x} - {profiles.loc[profiles.id == x, "name"].iloc[0]}',
                key="edit_profile_select"
            )

            profile = profiles[profiles.id == selected_id].iloc[0]

            existing_images_preview = parse_work_images(profile["work_images"])
            existing_geo_preview = parse_work_geo(
                profile["work_geo"] if "work_geo" in profile.index else None
            )
            existing_certificate_files = parse_work_images(
                profile["certificate_files"] if "certificate_files" in profile.index else "[]"
            )
            valid_existing_certificates = [p for p in existing_certificate_files if p and os.path.exists(p)]

            if valid_existing_certificates:
                st.markdown("### 📜 Current Licenses / Certificates")
                for cert_path in valid_existing_certificates:
                    with open(cert_path, "rb") as cert_file:
                        st.download_button(
                            "⬇️ Download " + os.path.basename(cert_path),
                            data=cert_file.read(),
                            file_name=os.path.basename(cert_path),
                            key=f"edit_cert_{selected_id}_{os.path.basename(cert_path)}"
                        )

            valid_preview_images = [p for p in existing_images_preview if p and os.path.exists(p)]
            if valid_preview_images:
                st.markdown("### 🖼️ Current Work Images")
                preview_cols = st.columns(min(4, len(valid_preview_images)))
                for i, image_path in enumerate(valid_preview_images):
                    with preview_cols[i % len(preview_cols)]:
                        st.image(image_path, use_container_width=True)
                        geo = existing_geo_preview.get(image_path)
                        if geo:
                            st.caption(f"📍 {geo['lat']:.5f}, {geo['lon']:.5f}")

            # New work photos: live camera + GPS (outside the form,
            # because camera/GPS components cannot work inside st.form)
            st.divider()
            edit_prefix = f"edit_{selected_id}"
            geo_work_capture(edit_prefix)
            st.divider()

            with st.form("edit_profile_form"):

                new_name = st.text_input("Name / Business Name", value=profile["name"])

                current_role_index = ROLES.index(profile["role"]) if profile["role"] in ROLES else 0
                new_role = st.selectbox("Professional Type", ROLES, index=current_role_index)

                new_location = st.text_input("📍 Location / City", value=profile["location"] or "")
                new_email = st.text_input("📧 Email", value=profile["email"] or "")
                new_phone = st.text_input("📞 Phone / Direct Contact", value=profile["phone"] or "")

                new_whatsapp = st.text_input(
                    "💬 WhatsApp Number (Optional)",
                    value=(profile["whatsapp_no"] if "whatsapp_no" in profile.index else "") or "",
                    placeholder="Example: 9876543210"
                )

                new_whatsapp_enabled = st.checkbox(
                    "Show WhatsApp contact button on my profile",
                    value=bool(int(profile["whatsapp_enabled"] or 0)) if "whatsapp_enabled" in profile.index else False
                )

                new_services = st.text_input("Services / Materials", value=profile["services"] or "")
                new_description = st.text_area("Short Description", value=profile["description"] or "")

                new_licenses_certificates = st.text_area(
                    "📜 Licenses / Certificates (Optional)",
                    value=(profile["licenses_certificates"] if "licenses_certificates" in profile.index else "") or "",
                    placeholder="List your licenses, registrations and certificates."
                )

                new_certificate_files = st.file_uploader(
                    "📎 Add New License / Certificate Documents (Optional)",
                    type=["pdf", "png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    key="edit_certificate_files"
                )

                new_max_work_budget = st.text_input(
                    "💰 Maximum Work / Project Budget You Can Handle (Optional)",
                    value=(profile["max_work_budget"] if "max_work_budget" in profile.index else "") or "",
                    placeholder="Example: Up to ₹1.5 Crore"
                )

                new_profile_image = st.file_uploader(
                    "Replace Profile Image", type=["png", "jpg", "jpeg"], key="edit_profile_image"
                )

                remove_existing_images = st.checkbox(
                    "🗑️ Remove all existing work images when updating",
                    value=False,
                    key="remove_existing_work_images"
                )

                update = st.form_submit_button("💾 Update Profile", use_container_width=True)

            if update:

                profile_image_path = profile["profile_image"]

                if new_profile_image:
                    if profile_image_path:
                        delete_file(profile_image_path)
                    profile_image_path = save_uploaded_file(new_profile_image, "profile")

                existing_work_images = parse_work_images(profile["work_images"])
                existing_work_geo = parse_work_geo(
                    profile["work_geo"] if "work_geo" in profile.index else None
                )

                if remove_existing_images:
                    for old_image in existing_work_images:
                        delete_file(old_image)
                    existing_work_images = []
                    existing_work_geo = {}

                # Add newly captured GPS-tagged camera photos
                new_paths, new_geo = commit_pending_photos(edit_prefix)
                existing_work_images.extend(new_paths)
                existing_work_geo.update(new_geo)
                st.session_state[f"{edit_prefix}_pending_photos"] = []

                existing_certificate_files = parse_work_images(
                    profile["certificate_files"] if "certificate_files" in profile.index else "[]"
                )

                for certificate in new_certificate_files:
                    path = save_uploaded_file(certificate, "certificate")
                    if path:
                        existing_certificate_files.append(path)

                con = db()

                con.execute(
                    """
                    UPDATE professionals
                    SET name=?, role=?, location=?, services=?, contact=?, description=?,
                        email=?, phone=?, whatsapp_no=?, whatsapp_enabled=?,
                        profile_image=?, work_images=?, work_geo=?, certificate_files=?,
                        licenses_certificates=?, max_work_budget=?
                    WHERE id=? AND owner_user_id=?
                    """,
                    (
                        new_name.strip(), new_role, new_location.strip(), new_services.strip(),
                        new_phone.strip(), new_description.strip(), new_email.strip(), new_phone.strip(),
                        new_whatsapp.strip(), int(new_whatsapp_enabled and bool(new_whatsapp.strip())),
                        profile_image_path, json.dumps(existing_work_images),
                        json.dumps(existing_work_geo),
                        json.dumps(existing_certificate_files),
                        new_licenses_certificates.strip(), new_max_work_budget.strip(),
                        int(selected_id), current_user["id"]
                    )
                )

                con.commit()
                con.close()

                set_flash("✅ Profile updated successfully! Your changes are now visible in the profile and dashboard.")
                st.rerun()

    # ========================================================
    # TAB 4 - DELETE MY PROFILE
    # ========================================================

    with tabs[3]:

        st.subheader("🗑️ Delete Your Profile")

        con = db()
        profiles = pd.read_sql_query(
            "SELECT * FROM professionals WHERE owner_user_id=? ORDER BY id DESC",
            con, params=[current_user["id"]]
        )
        con.close()

        if profiles.empty:
            st.info("You don't have a profile to delete.")
        else:

            selected_id = st.selectbox(
                "Select Profile to Delete",
                profiles.id.tolist(),
                format_func=lambda x: f'{x} - {profiles.loc[profiles.id == x, "name"].iloc[0]}',
                key="delete_profile_select"
            )

            profile = profiles[profiles.id == selected_id].iloc[0]

            st.warning("Deleting the profile will remove the profile information from the database.")

            confirm = st.checkbox("I understand that I want to delete this profile.")

            if st.button("🗑️ Delete Profile", disabled=not confirm, use_container_width=True):

                delete_file(profile["profile_image"])

                try:
                    work_images = json.loads(profile["work_images"] or "[]")
                    for image in work_images:
                        delete_file(image)
                except:
                    pass

                try:
                    certificate_files = json.loads(
                        profile["certificate_files"] if "certificate_files" in profile.index else "[]"
                    )
                    for certificate in certificate_files:
                        delete_file(certificate)
                except:
                    pass

                con = db()
                con.execute(
                    "DELETE FROM professionals WHERE id=? AND owner_user_id=?",
                    (int(selected_id), current_user["id"])
                )
                con.commit()
                con.close()

                set_flash("✅ Profile deleted successfully!")
                st.rerun()


# ============================================================
# 26. TENDERS (official tender links + work opportunities)
# ============================================================

def tenders_page(current_user):

    st.header("🏛️ Tenders")

    top_tabs = st.tabs(["🏛️ Official Tenders", "💼 Work Opportunities"])

    # ========================================================
    # OFFICIAL TENDERS
    # ========================================================

    with top_tabs[0]:

        st.write(
            "Official government tender opportunities should "
            "be checked through the relevant government portals. "
            "ESTIM AI does not create or claim to publish official "
            "government tenders."
        )

        st.info("Use official government websites for tender details, registration and submission.")

        st.subheader("🌐 Official Government e-Procurement")

        st.link_button(
            "Open Government e-Procurement Portal",
            "https://eprocure.gov.in/eprocure/app",
            use_container_width=True
        )

        st.divider()

        # ----------------------------------------------------
        # Saved tender links — visible to everyone, read-only.
        # Adding/managing links is admin-only (see below).
        # ----------------------------------------------------

        st.subheader("📋 Available Tender Links")

        con = db()
        tenders_df = pd.read_sql_query(
            "SELECT * FROM tenders ORDER BY id DESC", con
        )
        con.close()

        if tenders_df.empty:
            st.info("No official tender links have been added yet.")
        else:
            for _, tender in tenders_df.iterrows():
                with st.container(border=True):
                    st.subheader(tender["title"])
                    if tender["department"]:
                        st.write(f"**Department:** {tender['department']}")
                    if tender["location"]:
                        st.write(f"**📍 Location:** {tender['location']}")
                    if tender["estimated_value"]:
                        st.write(f"**Estimated Value:** {tender['estimated_value']}")
                    if tender["opening_date"]:
                        st.write(f"**Opening Date:** {tender['opening_date']}")
                    if tender["closing_date"]:
                        st.write(f"**Closing Date:** {tender['closing_date']}")
                    if tender["official_link"]:
                        st.link_button("🌐 Open Official Tender Link", tender["official_link"])

        # ----------------------------------------------------
        # Admin-only: add an official tender link
        # ----------------------------------------------------

        if current_user.get("is_admin"):

            st.divider()

            with st.expander("➕ Add Official Tender Link (Admin)"):

                st.write(
                    "This stores a link to an official tender source (for "
                    "example, a direct link from an official government "
                    "e-procurement website). It does not automatically "
                    "verify or scrape tender data. Only administrators can "
                    "add or manage these links."
                )

                with st.form("official_tender_form"):

                    title = st.text_input("Tender / Portal Name")
                    department = st.text_input("Department / Authority")
                    location = st.text_input("Location")
                    value = st.text_input("Estimated Value")
                    opening = st.date_input("Opening Date")
                    closing = st.date_input("Closing Date")
                    official_link = st.text_input("Official Tender Link (from official website)")

                    submit = st.form_submit_button("💾 Save Tender Link", use_container_width=True)

                if submit:

                    if not title.strip():
                        st.error("Please enter the tender title.")
                    elif not official_link.strip():
                        st.error("Please enter the official tender link.")
                    else:

                        con = db()
                        con.execute(
                            """
                            INSERT INTO tenders(
                                title, department, location, estimated_value,
                                opening_date, closing_date, status, official_link, created_at
                            )
                            VALUES(?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                title.strip(), department.strip(), location.strip(), value.strip(),
                                str(opening), str(closing), "Official Link", official_link.strip(),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        con.commit()
                        con.close()

                        st.success("✅ Official tender link saved successfully!")
                        st.rerun()

    # ========================================================
    # WORK OPPORTUNITIES (moved here from BuildConnect)
    # ========================================================

    with top_tabs[1]:

        st.subheader("💼 Unofficial Work Opportunities")
        st.write("Post private construction work so suitable builders, contractors and professionals can contact you.")
        work_tabs = st.tabs(["🔎 Find Work", "➕ Upload Work"])

        with work_tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                work_search = st.text_input("🔎 Search Work", key="work_search")
            with c2:
                work_location = st.text_input("📍 Place", key="work_location")

            con = db()
            query = "SELECT * FROM work_posts WHERE 1=1"
            params = []
            if work_search:
                query += " AND (title LIKE ? OR description LIKE ? OR category LIKE ?)"
                v = f"%{work_search}%"
                params += [v, v, v]
            if work_location:
                query += " AND location LIKE ?"
                params.append(f"%{work_location}%")
            query += " ORDER BY id DESC"
            work_df = pd.read_sql_query(query, con, params=params)
            con.close()

            if work_df.empty:
                st.info("No unofficial work opportunities found.")
            else:
                for _, work in work_df.iterrows():
                    with st.container(border=True):
                        st.subheader(f"💼 {work['title']}")
                        if work.get('budget'): st.write(f"**💰 Budget:** {work['budget']}")
                        if work.get('location'): st.write(f"**📍 Place:** {work['location']}")
                        if work.get('duration'): st.write(f"**⏱️ Duration:** {work['duration']}")
                        if work.get('contact'): st.write(f"**📞 Contact:** {work['contact']}")
                        if work.get('email'): st.write(f"**✉️ Email:** {work['email']}")
                        if work.get('description'): st.write(f"**Description:** {work['description']}")
                        buttons = st.columns(2)
                        if work.get('whatsapp'):
                            link = whatsapp_link(work['whatsapp'])
                            if link:
                                buttons[0].link_button("💬 WhatsApp", link, use_container_width=True)
                        if work.get('email'):
                            buttons[1].link_button("✉️ Email", f"mailto:{work['email']}", use_container_width=True)
                        if work.get('image') and os.path.exists(work['image']):
                            st.image(work['image'], width=300)

        with work_tabs[1]:
            with st.form("post_work_form"):
                work_name = st.text_input("Work Name")
                budget = st.text_input("💰 Budget")
                place = st.text_input("📍 Place")
                contact = st.text_input("📞 Contact No.")
                whatsapp = st.text_input("💬 WhatsApp No. (Optional)")
                email = st.text_input("✉️ Email (Optional)")
                duration = st.text_input("⏱️ Work Duration", placeholder="e.g. 6 months")
                description = st.text_area("Work Details (Optional)")
                work_image = st.file_uploader("🖼️ Work Image (Optional)", type=["png", "jpg", "jpeg"], key="post_work_image")
                post_work = st.form_submit_button("➕ Upload Work", use_container_width=True)

            if post_work:
                if not work_name.strip():
                    st.error("Please enter the work name.")
                elif not place.strip():
                    st.error("Please enter the work place.")
                else:
                    image_path = save_uploaded_file(work_image, "work_post")
                    con = db()
                    con.execute(
                        """INSERT INTO work_posts(title, category, description, location, budget, posted_by, contact, image, created_at, whatsapp, email, duration) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (work_name.strip(), "Unofficial Work", description.strip(), place.strip(), budget.strip(), current_user["username"], contact.strip(), image_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), whatsapp.strip(), email.strip(), duration.strip())
                    )
                    con.commit()
                    con.close()
                    st.success("✅ Unofficial work uploaded successfully!")
                    st.rerun()


# ============================================================
# 27. MODEL RESULTS PAGE
# ============================================================

def model_results():

    st.header("📊 Model Evaluation")

    st.markdown(
        "**About ESTIM AI:** ESTIM AI is an AI-based construction decision-support system. "
        "It uses trained Random Forest and Decision Tree regression models to predict material "
        "quantities and cost estimates for Building and Road construction projects, and pairs that "
        "with What-If analysis, project history, a BuildConnect professional network with in-app "
        "messaging, and access to official government tender resources."
    )

    st.divider()

    st.write("Comparison of the Machine Learning models used for Building and Road Construction estimation.")

    st.subheader("🛣️ Road Construction")

    road_df = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree"],
        "R² Score": [ROAD_RF_R2, ROAD_DT_R2]
    })

    st.dataframe(road_df, use_container_width=True, hide_index=True)

    st.success("Random Forest has the higher R² score among the supplied road model results.")

    st.subheader("🏠 Building Construction")

    building_df = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree"],
        "R² Score": [BUILDING_RF_R2, BUILDING_DT_R2]
    })

    st.dataframe(building_df, use_container_width=True, hide_index=True)

    st.success("Random Forest has the higher supplied R² score for Building Construction.")

    st.divider()

    st.subheader("🎯 Model Accuracy")

    accuracy_df = pd.DataFrame({
        "Construction Type": ["Road", "Road", "Building", "Building"],
        "Model": ["Random Forest", "Decision Tree", "Random Forest", "Decision Tree"],
        "Accuracy": [
            f"{ROAD_RF_ACCURACY}%", f"{ROAD_DT_ACCURACY}%",
            f"{BUILDING_RF_ACCURACY}%", f"{BUILDING_DT_ACCURACY}%"
        ]
    })

    st.dataframe(accuracy_df, use_container_width=True, hide_index=True)

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric("🛣️ Road — Random Forest", f"{ROAD_RF_ACCURACY}%")
    with a2:
        st.metric("🛣️ Road — Decision Tree", f"{ROAD_DT_ACCURACY}%")
    with a3:
        st.metric("🏠 Building — Random Forest", f"{BUILDING_RF_ACCURACY}%")
    with a4:
        st.metric("🏠 Building — Decision Tree", f"{BUILDING_DT_ACCURACY}%")


# ============================================================
# GLOBAL CSS
# ============================================================

inject_global_css()


# ============================================================
# AUTHENTICATION GATE
# ============================================================

if "user" not in st.session_state:
    auth_page()

current_user = st.session_state["user"]


# ============================================================
# SIDEBAR MENU
# ============================================================

st.sidebar.title("🏗️ ESTIM AI")
st.sidebar.caption("AI-Based Construction Decision-Support System")
st.sidebar.divider()

# The radio's own key ("nav_radio") is its single source of truth. Setting
# it here only seeds the very first run; after that, Streamlit keeps using
# whatever is in st.session_state["nav_radio"] (updated either by the user
# clicking the radio, or by go_to() when a Home-page button is clicked).
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "🏠 Home"

st.sidebar.markdown("### ☰ Menu")

page = st.sidebar.radio(
    "Select Section",
    MENU_OPTIONS,
    key="nav_radio",
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown(f"👤 **{current_user['username']}**" + (" 🛡️ (Admin)" if current_user["is_admin"] else ""))

if st.sidebar.button("🚪 Logout", use_container_width=True):
    del st.session_state["user"]
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Madhura Dhatrak")
st.sidebar.caption("Diploma in Artificial Intelligence & Machine Learning")
st.sidebar.caption("RSM Polytechnic")

# ============================================================
# GLOBAL FLASH MESSAGES
# ============================================================

show_flash()


# ============================================================
# 29. PAGE ROUTING
# ============================================================

if page == "🏠 Home":
    home_page(current_user)

elif page == "📐 Estimation":
    estimation()

elif page == "🔄 What-If Analysis":
    what_if()

elif page == "📁 Project History":
    history()

elif page == "📊 Model Results":
    model_results()

elif page == "🤝 BuildConnect":
    buildconnect(current_user)

elif page == "🏛️ Tenders":
    tenders_page(current_user)


# ============================================================
# 30. FOOTER
# ============================================================

st.markdown("---")

st.caption("ESTIM AI | AI-Based Construction Decision-Support System | RSM Polytechnic")
