import os
import sqlite3
import numpy as np
import tensorflow as tf
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Response
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.applications.densenet import preprocess_input
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

app = FastAPI()

# Ensure these directories exist
for folder in ["uploads", "reports", "static", "templates"]:
    os.makedirs(folder, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB_PATH = "patients.db"
# MODEL_PATH = "exported_model"
# MODEL_PATH = "densenet_osteoporosis_model_v2"
MODEL_PATH = "exported_model"

IMG_SIZE = 224

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY, user_id INTEGER, result TEXT, confidence REAL, image_path TEXT, date TEXT)")
init_db()

try:
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]
except Exception as e:
    print(f"⚠️ Model not found: {e}")

def get_user_id(request: Request):
    uid = request.cookies.get("user_id")
    return int(uid) if uid else None

# --- AUTH ROUTES ---
@app.get("/")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/login")
async def do_login(response: Response, username: str = Form(...), password: str = Form(...)):
    with sqlite3.connect(DB_PATH) as db:
        user = db.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password)).fetchone()
    if not user:
        return RedirectResponse("/", status_code=303)
    res = RedirectResponse("/dashboard", status_code=303)
    res.set_cookie(key="user_id", value=str(user[0]), httponly=True)
    return res

@app.post("/register")
async def do_register(username: str = Form(...), password: str = Form(...)):
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        return RedirectResponse("/", status_code=303)
    except:
        return RedirectResponse("/register", status_code=303)

@app.get("/logout")
def logout():
    res = RedirectResponse("/", status_code=302)
    res.delete_cookie("user_id")
    return res

# --- APP ROUTES ---
@app.get("/dashboard")
def dashboard_page(request: Request):
    uid = get_user_id(request)
    if not uid: return RedirectResponse("/")
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/profile")
def profile_page(request: Request):
    uid = get_user_id(request)
    if not uid: return RedirectResponse("/")
    
    rows = []
    with sqlite3.connect(DB_PATH) as db:
        raw_data = db.execute("SELECT id, result, confidence, date FROM scans WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
        rows = [list(r) for r in raw_data]
    return templates.TemplateResponse("profile.html", {"request": request, "rows": rows})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    uid = get_user_id(request)
    if not uid: raise HTTPException(status_code=401)
    
    path = f"uploads/{datetime.now().timestamp()}_{file.filename}"
    with open(path, "wb") as f: f.write(await file.read())

    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    x = preprocess_input(np.expand_dims(tf.keras.utils.img_to_array(img), axis=0))
    
    input_key = list(infer.structured_input_signature[1].keys())[0]
    output = infer(**{input_key: tf.convert_to_tensor(x)})
    preds = list(output.values())[0].numpy()

    prob = float(preds[0][0] if preds.shape[-1] == 1 else preds[0][1])
    result = "Osteoporosis Detected" if prob >= 0.5 else "Normal Bone Density"
    conf = prob if prob >= 0.5 else 1 - prob

    with sqlite3.connect(DB_PATH) as db:
        db.execute("INSERT INTO scans (user_id, result, confidence, image_path, date) VALUES (?,?,?,?,?)",
                   (uid, result, conf, path, datetime.now().strftime("%Y-%m-%d %H:%M")))
    return {"result": result, "confidence": f"{conf*100:.1f}%"}

@app.delete("/delete-scan/{scan_id}")
async def delete_scan(request: Request, scan_id: int):
    uid = get_user_id(request)
    if not uid: raise HTTPException(status_code=401)
    with sqlite3.connect(DB_PATH) as db:
        img_path = db.execute("SELECT image_path FROM scans WHERE id=? AND user_id=?", (scan_id, uid)).fetchone()
        if img_path:
            db.execute("DELETE FROM scans WHERE id=? AND user_id=?", (scan_id, uid))
            if os.path.exists(img_path[0]): os.remove(img_path[0])
            return {"status": "success"}
    raise HTTPException(status_code=404)

@app.get("/report/{scan_id}")
def generate_report(scan_id: int):
    with sqlite3.connect(DB_PATH) as db:
        r = db.execute("SELECT result, confidence, image_path, date FROM scans WHERE id=?", (scan_id,)).fetchone()
    if not r: 
        raise HTTPException(status_code=404)

    result, conf, img_path, date = r
    pdf_path = f"reports/report_{scan_id}.pdf"
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    is_normal = "Normal" in result

    # --- Header Section ---
    c.setFillColor(colors.indigo)
    c.rect(0, height-80, width, 80, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height-50, "BoneAI Medical Report")
    
    # Report Meta
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height-105, f"Report ID: #BA-{scan_id}")
    c.drawString(50, height-120, f"Date: {date}")

    # --- X-RAY IMAGE DISPLAY ---
    if os.path.exists(img_path):
        try:
            img = ImageReader(img_path)
            c.setStrokeColor(colors.indigo)
            c.roundRect(330, height-330, 230, 200, 10, stroke=1, fill=0) 
            c.drawImage(img, 340, height-320, width=210, height=180, preserveAspectRatio=True)
        except Exception:
            c.drawString(340, height-250, "[Error loading X-ray image]")
    
    # --- Diagnosis Box ---
    c.setStrokeColor(colors.lightgrey)
    c.roundRect(50, height-230, 260, 100, 10, stroke=1, fill=0)
    c.setFont("Helvetica", 12)
    c.drawString(70, height-155, "AI Diagnosis Result:")
    
    c.setFont("Helvetica-Bold", 16)
    if is_normal:
        c.setFillColor(colors.green)
        msg = "Normal Bone Density"
    else:
        c.setFillColor(colors.red)
        msg = "Osteoporosis Detected"
    c.drawString(70, height-185, msg)
    
    # Confidence Score
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(70, height-210, f"AI Confidence Score: {float(conf)*100:.1f}%")
    c.setStrokeColor(colors.lightgrey)
    c.roundRect(70, height-225, 180, 10, 5, stroke=1, fill=0)
    c.setFillColor(colors.indigo)
    c.roundRect(70, height-225, 180 * float(conf), 10, 5, stroke=0, fill=1)

    # --- Dynamic Recommendations Section ---
    y_pos = height - 360
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Personalized Recommendations")
    c.line(50, y_pos-5, 550, y_pos-5)
    
    y_pos -= 30
    c.setFont("Helvetica-Bold", 11)
    
    if is_normal:
        c.drawString(50, y_pos, "Status: No Medication Required")
        c.setFont("Helvetica", 10)
        advice = [
            "• Maintain high intake of Calcium-rich fruits like Oranges and Kiwis.",
            "• Increase consumption of leafy greens (Spinach, Kale) and nuts.",
            "• Regular weight-bearing exercises (walking, jogging) to maintain density.",
            "• Ensure adequate Vitamin D exposure through sunlight or supplements.",
            "• Schedule a follow-up screening in 12-24 months."
        ]
    else:
        c.drawString(50, y_pos, "Action Required: Consult a Specialist")
        c.setFont("Helvetica", 10)
        advice = [
            "• Curing Methods: Discuss Bisphosphonates with a doctor.",
            "• Medication: Common prescriptions include Alendronate or Zoledronic acid.",
            "• Safety: Implement fall-prevention strategies at home.",
            "• Nutrition: High-dose Calcium (1200mg+) and Vitamin D (800IU+) daily.",
            "• Avoid: Limit caffeine and alcohol intake."
        ]

    for line in advice:
        y_pos -= 20
        c.drawString(60, y_pos, line)

    # --- Footer ---
    c.setFillColor(colors.grey)
    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(width/2, 40, "* This is an automated AI screening. Please consult a radiologist for official diagnosis.")
    
    c.save()
    return FileResponse(pdf_path, filename=f"BoneAI_Report_{scan_id}.pdf")