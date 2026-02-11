import sqlite3

conn = sqlite3.connect("patients.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    result TEXT,
    confidence REAL,
    image_path TEXT,
    date TEXT
);
""")

conn.commit()
conn.close()

print("✅ Database initialized successfully")
