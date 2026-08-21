import sqlite3
import os
import uuid
import secrets
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if we need to migrate/recreate (if old db exists without email, password_changed, is_active, or api_key columns)
    recreate = False
    try:
        cursor.execute("SELECT id FROM users WHERE username = 'admin' AND role = 'admin'")
        if not cursor.fetchone():
            recreate = True
        
        # Check if is_active column exists in users
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'")
        sql_row_users = cursor.fetchone()
        if sql_row_users and "is_active" not in sql_row_users["sql"]:
            recreate = True

        # Check if api_key column exists in clients
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='clients'")
        sql_row_clients = cursor.fetchone()
        if sql_row_clients and "api_key" not in sql_row_clients["sql"]:
            recreate = True
            
        # Force recreate if chat_sessions still enforces client_id NOT NULL
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_sessions'")
        sql_row = cursor.fetchone()
        if sql_row and "client_id INTEGER NOT NULL" in sql_row["sql"]:
            recreate = True
    except sqlite3.OperationalError:
        recreate = True
        
    if recreate:
        print("Recreating database to support inactivity checks and API keys...")
        cursor.execute("DROP TABLE IF EXISTS documents")
        cursor.execute("DROP TABLE IF EXISTS messages")
        cursor.execute("DROP TABLE IF EXISTS chat_sessions")
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("DROP TABLE IF EXISTS clients")
        conn.commit()
    
    # Create clients table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        type TEXT NOT NULL, -- 'Campus', 'Bank', 'General'
        api_key TEXT UNIQUE
    )
    """)

    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        token TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL, -- 'superadmin', 'admin', 'admin_client', 'user'
        client_id INTEGER,
        email TEXT,
        password_changed INTEGER DEFAULT 0,
        last_login TEXT,
        is_active INTEGER DEFAULT 1,
        FOREIGN KEY (client_id) REFERENCES clients (id) ON DELETE SET NULL
    )
    """)
    
    # Create chat sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        client_id INTEGER, -- Nullable for Global/General AI Chat
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (client_id) REFERENCES clients (id) ON DELETE SET NULL
    )
    """)
    
    # Create messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
    )
    """)

    # Create documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        doc_type TEXT NOT NULL, -- 'PDF', 'GAMBAR', 'VIDEO'
        upload_date TEXT NOT NULL,
        FOREIGN KEY (client_id) REFERENCES clients (id) ON DELETE CASCADE
    )
    """)

    # Create prompt_cache_entries table (persist semantic cache lintas restart)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompt_cache_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cache_key TEXT NOT NULL,
        query TEXT NOT NULL,
        embedding TEXT NOT NULL,  -- JSON list of float
        response TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_prompt_cache_key ON prompt_cache_entries (cache_key)
    """)
    conn.commit()

    
    # Seed default clients & users if tables are empty
    cursor.execute("SELECT COUNT(*) as count FROM clients")
    if cursor.fetchone()["count"] == 0:
        clients = [
            ("Bank DKI", "Bank", "rc_live_bank_dki_8a9b2c"),
            ("Universitas Gunadarma", "Campus", "rc_live_gunadarma_89327f"),
            ("Universitas Pamulang", "Campus", "rc_live_unpam_3d4e5f"),
            ("Universitas Budi Luhur", "Campus", "rc_live_budi_luhur_6a7b8c"),
            ("warung makan", "General", "rc_live_warung_makan_1e2f3d")
        ]
        cursor.executemany(
            "INSERT INTO clients (name, type, api_key) VALUES (?, ?, ?)",
            clients
        )
        conn.commit()

        # Get client IDs
        cursor.execute("SELECT id, name FROM clients")
        client_map = {row["name"]: row["id"] for row in cursor.fetchall()}

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        default_users = [
            ("admin", "admin123", "token-admin", "admin", None, "admin@chatbot.com", 1, now_str, 1), # admin acts as global admin
            ("Budi Santoso", "client123", "token-budi", "admin_client", client_map["Universitas Gunadarma"], "budi@gunadarma.go.id", 0, now_str, 1),
            ("Siti Rahma", "client123", "token-siti", "admin_client", client_map["Universitas Pamulang"], "siti@unpam.go.id", 0, now_str, 1),
            ("Andi Wijaya", "client123", "token-andi", "admin_client", client_map["Universitas Budi Luhur"], "andi@budiluhur.go.id", 0, now_str, 1),
            ("Andi Rijani", "client123", "token-rijani", "admin_client", client_map["Bank DKI"], "Rijani16@instansi.go.id", 0, now_str, 1),
            ("Wijaya", "client123", "token-wijaya", "admin_client", client_map["warung makan"], "Wijaya45@gmail.com", 0, now_str, 1),
            ("user", "user123", "token-user", "user", None, "user@gmail.com", 1, now_str, 1)
        ]
        cursor.executemany(
            "INSERT INTO users (username, password, token, role, client_id, email, password_changed, last_login, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            default_users
        )
        conn.commit()
        print("Default clients and users seeded successfully.")
        
    conn.close()

# Initialize DB on import
init_db()

# --- CLIENT HELPER FUNCTIONS ---
def get_all_clients():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clients ORDER BY name ASC")
    clients = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return clients

def add_client(name: str, type: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Generate API key
    slug = "".join([c if c.isalnum() else "_" for c in name.lower()])
    random_hex = secrets.token_hex(6)
    api_key = f"rc_live_{slug}_{random_hex}"
    try:
        cursor.execute("INSERT INTO clients (name, type, api_key) VALUES (?, ?, ?)", (name, type, api_key))
        conn.commit()
        client_id = cursor.lastrowid
        conn.close()
        return {"id": client_id, "name": name, "type": type, "api_key": api_key}
    except sqlite3.IntegrityError as e:
        conn.close()
        raise Exception(f"Client name already exists. {str(e)}")

def delete_client(client_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM clients WHERE id = ?", (client_id,))
    conn.commit()
    conn.close()
    return True

def get_client_by_api_key(api_key: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clients WHERE api_key = ?", (api_key,))
    client = cursor.fetchone()
    conn.close()
    if client:
        return dict(client)
    return None

def generate_client_api_key(client_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Check if client exists
    cursor.execute("SELECT name FROM clients WHERE id = ?", (client_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise Exception("Client tidak ditemukan.")
    
    client_name = row["name"]
    # Create a nice slug
    slug = "".join([c if c.isalnum() else "_" for c in client_name.lower()])
    random_hex = secrets.token_hex(6)
    new_key = f"rc_live_{slug}_{random_hex}"
    
    cursor.execute("UPDATE clients SET api_key = ? WHERE id = ?", (new_key, client_id))
    conn.commit()
    conn.close()
    return new_key

# --- USER HELPER FUNCTIONS ---
def get_user_by_token(token: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.*, c.name as client_name 
        FROM users u 
        LEFT JOIN clients c ON u.client_id = c.id 
        WHERE u.token = ?
    """, (token,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

def check_and_deactivate_inactive_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Deactivate users who haven't logged in for > 7 days
    # (We ignore admin/superadmin to prevent locking the main admin out)
    one_week_ago = datetime.now() - timedelta(days=7)
    one_week_ago_str = one_week_ago.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE users 
        SET is_active = 0 
        WHERE last_login IS NOT NULL 
          AND last_login < ? 
          AND role != 'admin'
    """, (one_week_ago_str,))
    conn.commit()
    conn.close()

def get_user_by_credentials(username: str, password: str):
    check_and_deactivate_inactive_users()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.*, c.name as client_name 
        FROM users u 
        LEFT JOIN clients c ON u.client_id = c.id 
        WHERE u.username = ? AND u.password = ?
    """, (username, password))
    user = cursor.fetchone()
    if user:
        user_dict = dict(user)
        if user_dict.get("is_active", 1) == 0:
            conn.close()
            return user_dict
        
        # Update last_login
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_str, user_dict["id"]))
        conn.commit()
        user_dict["last_login"] = now_str
        conn.close()
        return user_dict
    conn.close()
    return None

def get_all_users():
    check_and_deactivate_inactive_users()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.username, u.password, u.token, u.role, u.client_id, u.email, u.password_changed, u.last_login, u.is_active, c.name as client_name, c.type as client_type, c.api_key as client_api_key
        FROM users u
        LEFT JOIN clients c ON u.client_id = c.id
    """)
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

def add_user(username: str, password: str, role: str, client_id: int = None, email: str = None, password_changed: int = 0):
    conn = get_db_connection()
    cursor = conn.cursor()
    token = uuid.uuid4().hex
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("INSERT INTO users (username, password, token, role, client_id, email, password_changed, last_login, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)", 
                       (username, password, token, role, client_id, email, password_changed, now_str))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {"id": user_id, "username": username, "token": token, "role": role, "client_id": client_id, "email": email, "password_changed": password_changed, "last_login": now_str, "is_active": 1}
    except sqlite3.IntegrityError as e:
        conn.close()
        raise Exception(f"Username already exists. {str(e)}")

def update_client_instansi(user_id: int, username: str, instansi_name: str, client_type: str, password: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get user client_id
        cursor.execute("SELECT client_id FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise Exception("User tidak ditemukan.")
        client_id = row["client_id"]
        
        # Update client if exists
        if client_id is not None:
            cursor.execute("UPDATE clients SET name = ?, type = ? WHERE id = ?", (instansi_name, client_type, client_id))
            
        # Update user
        if password:
            cursor.execute("UPDATE users SET username = ?, password = ?, password_changed = 0 WHERE id = ?", (username, password, user_id))
        else:
            cursor.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        raise e

def update_user_password(user_id: int, new_password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password = ?, password_changed = 1 WHERE id = ?", (new_password, user_id))
    conn.commit()
    conn.close()
    return True

def delete_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True

# --- DOCUMENT HELPER FUNCTIONS ---
def get_documents_by_client(client_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE client_id = ? ORDER BY upload_date DESC", (client_id,))
    docs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return docs

def add_document(client_id: int, filename: str, doc_type: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO documents (client_id, filename, doc_type, upload_date) VALUES (?, ?, ?, ?)",
        (client_id, filename, doc_type, upload_date)
    )
    conn.commit()
    doc_id = cursor.lastrowid
    conn.close()
    return {"id": doc_id, "client_id": client_id, "filename": filename, "doc_type": doc_type, "upload_date": upload_date}

def delete_document(doc_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
    doc = cursor.fetchone()
    if doc:
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        return doc["filename"]
    conn.close()
    return None

# --- CHAT SESSION HELPER FUNCTIONS ---
def create_chat_session(user_id: int, client_id: int = None, title: str = ""):
    conn = get_db_connection()
    cursor = conn.cursor()
    session_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO chat_sessions (id, user_id, client_id, title, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, user_id, client_id, title, created_at)
    )
    conn.commit()
    conn.close()
    return {"id": session_id, "user_id": user_id, "client_id": client_id, "title": title, "created_at": created_at}

def get_chat_sessions(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, c.name as client_name 
        FROM chat_sessions s
        LEFT JOIN clients c ON s.client_id = c.id
        WHERE s.user_id = ? 
        ORDER BY s.created_at DESC
    """, (user_id,))
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

def get_chat_messages(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    )
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages

def save_message(session_id: str, role: str, content: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    msg_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        (msg_id, session_id, role, content, timestamp)
    )
    conn.commit()
    conn.close()
    return {"id": msg_id, "session_id": session_id, "role": role, "content": content, "timestamp": timestamp}

def delete_chat_session(session_id: str, user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
    if not cursor.fetchone():
        conn.close()
        return False
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return True

# --- PROMPT CACHE PERSISTENCE ---
 
def replace_cache_entries(cache_key: str, entries: list):
    """Timpa semua baris untuk 1 cache_key dengan snapshot bucket in-memory
    saat ini. Dipanggil tiap kali store_cache() -- jadi urutan (FIFO
    eviction) & isi selalu sinkron 1:1 dengan state.prompt_cache[cache_key].
    """
    import json
    from datetime import datetime
 
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prompt_cache_entries WHERE cache_key = ?", (cache_key,))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.executemany(
        "INSERT INTO prompt_cache_entries (cache_key, query, embedding, response, created_at) VALUES (?, ?, ?, ?, ?)",
        [(cache_key, e["query"], json.dumps(e["embedding"]), e["response"], now_str) for e in entries],
    )
    conn.commit()
    conn.close()
 
 
def load_all_cache_entries() -> dict:
    """Baca semua entri cache tersimpan, dikelompokkan per cache_key.
    Dipanggil sekali saat startup (generation.initialize()) untuk hydrate
    state.prompt_cache -- inilah yang bikin cache SURVIVE restart proses.
    """
    import json
 
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT cache_key, query, embedding, response FROM prompt_cache_entries ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
 
    result: dict = {}
    for row in rows:
        result.setdefault(row["cache_key"], []).append({
            "query": row["query"],
            "embedding": json.loads(row["embedding"]),
            "response": row["response"],
        })
    return result
 
 
def delete_cache_entries_by_prefix(prefix: str):
    """Hapus semua entri cache yang cache_key-nya diawali `prefix`.
    Dipakai invalidate_document_cache() saat dokumen client diganti/dihapus.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prompt_cache_entries WHERE cache_key LIKE ?", (f"{prefix}%",))
    conn.commit()
    conn.close()

 
if __name__ == "__main__":
    print(__doc__)
    print("\n--- SNIPPET_INIT_DB ---")
    print(SNIPPET_INIT_DB)
    print("\n--- SNIPPET_NEW_FUNCTIONS ---")
    print(SNIPPET_NEW_FUNCTIONS)