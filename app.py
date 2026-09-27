from datetime import datetime
import os
import re
import sqlite3
import time
import warnings
from contextlib import contextmanager

from groq import Groq
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------- Voice deps (optional) ----------
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except Exception:
    MIC_RECORDER_AVAILABLE = False

try:
    import streamlit.components.v1 as components
except Exception:
    components = None


# ==================================================================
# 1. CONFIGURATION
# ==================================================================
st.set_page_config(
    page_title="StockSmart AI — Auto-Parts Voice & Text",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_GROQ_KEY = ""


def get_secret(key, default=None):
    try:
        if key in st.secrets and st.secrets[key]:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY", DEFAULT_GROQ_KEY)
MODEL_NAME = get_secret("MODEL_NAME", "openai/gpt-oss-120b")
WHISPER_MODEL = get_secret("WHISPER_MODEL", "whisper-large-v3")
DB_FILE = get_secret("DB_FILE", "test_stock.db")

QUERY_TIMEOUT_SEC = float(get_secret("QUERY_TIMEOUT_SEC", "0.8"))
MAX_ROWS_RETURNED = int(get_secret("MAX_ROWS_RETURNED", "200"))
FUZZY_MIN_SCORE = float(get_secret("FUZZY_MIN_SCORE", "0.45"))

if not GROQ_API_KEY:
    st.error(
        "Missing GROQ_API_KEY. Add it to `.streamlit/secrets.toml` "
        "or set it as an environment variable."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ==================================================================
# 2. STYLING
# ==================================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root {
            --bg: #0a0e14; --panel: #12181f; --panel-light: #18222c;
            --border: #223040; --text: #eef3f8; --muted: #8fa3b8;
            --teal: #2dd4bf; --amber: #fbbf24; --coral: #fb7185; --violet: #a78bfa;
        }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 85% -5%, rgba(45,212,191,0.07), transparent 30%),
                radial-gradient(circle at 5% 100%, rgba(167,139,250,0.05), transparent 30%),
                var(--bg);
            color: var(--text);
        }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.03em; }
        .block-container { max-width: 1450px; padding: 2rem 3rem 4rem; }
        section[data-testid="stSidebar"] { background: #0d1219; border-right: 1px solid var(--border); }
        section[data-testid="stSidebar"] > div { min-width: 380px; }

        .brand { display: flex; align-items: center; gap: 14px; margin-bottom: 2rem; }
        .brand-icon {
            width: 48px; height: 48px; border-radius: 14px;
            display: grid; place-items: center; font-size: 25px;
            background: linear-gradient(135deg, #0d9488, #2dd4bf);
            box-shadow: 0 10px 30px rgba(45,212,191,0.25);
        }
        .brand-title { font-size: 18px; font-weight: 700; }
        .brand-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }

        .hero { margin-bottom: 1rem; animation: fadeUp 0.5s ease both; }
        .hero h1 { color: var(--text); font-size: 2.6rem; margin: 0 0 0.35rem 0; }
        .hero p { color: var(--muted); font-size: 1.02rem; margin: 0; }
        .eyebrow {
            color: var(--teal); text-transform: uppercase; font-size: 12px;
            font-weight: 700; letter-spacing: 0.16em; margin-bottom: 0.5rem;
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .live-tag {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 11px; color: var(--teal);
            background: rgba(45,212,191,0.12);
            padding: 4px 8px; border-radius: 6px; margin-bottom: 12px;
        }
        .pulse-dot {
            width: 7px; height: 7px; background-color: var(--teal);
            border-radius: 50%; animation: pulse 1.6s ease infinite;
        }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

        .filter-chip {
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(251,191,36,0.14); color: var(--amber);
            border: 1px solid rgba(251,191,36,0.4);
            padding: 5px 12px; border-radius: 999px; font-size: 12px;
            margin: 6px 0 10px;
        }

        .answer-card {
            border-left: 4px solid var(--teal); border-radius: 12px;
            padding: 16px 20px; background: rgba(45,212,191,0.08);
            margin-top: 1.25rem; animation: fadeUp 0.4s ease both;
        }
        .answer-title { color: var(--teal); font-weight: 700; margin-bottom: 6px; }
        .recovery-card {
            border-left: 4px solid var(--amber); border-radius: 12px;
            padding: 16px 20px; background: rgba(251,191,36,0.08);
            margin-top: 1rem; animation: fadeUp 0.4s ease both;
        }
        .recovery-title { color: var(--amber); font-weight: 700; margin-bottom: 6px; }

        .beta-chip {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 11px; color: var(--violet);
            background: rgba(167,139,250,0.12);
            padding: 4px 8px; border-radius: 6px;
        }
        .latency-badge { font-size: 11px; color: var(--muted); margin-top: 4px; }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: linear-gradient(145deg, var(--panel-light), var(--panel));
            margin-bottom: 8px;
            overflow: hidden;
        }
        div[data-testid="stExpander"] summary { font-weight: 600; color: var(--text); }
        div[data-testid="stExpander"] summary:hover { color: var(--teal); }

        div[data-baseweb="slider"] div[role="slider"] {
            background-color: var(--teal) !important;
            border-color: var(--teal) !important;
        }
        div[data-baseweb="slider"] div[style*="rgb(255, 75, 75)"] {
            background: var(--teal) !important;
        }
        div[data-testid="stSlider"] label p { color: var(--text); font-size: 13px; }

        div[data-testid="stCheckbox"] { padding: 6px 10px; border-radius: 10px;
            background: rgba(45,212,191,0.06); border: 1px solid var(--border); }
        div[data-testid="stCheckbox"] label p { color: var(--text); font-size: 14px; }
        div[data-testid="stCheckbox"] input { accent-color: var(--teal); width: 16px; height: 16px; }

        /* ================== ANIMATIONS ================== */
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero h1 {
            background: linear-gradient(120deg, #2dd4bf, #45b8ff, #a78bfa, #2dd4bf);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 6s ease infinite;
        }

        @keyframes brandGlow {
            0%, 100% { box-shadow: 0 10px 30px rgba(45,212,191,0.25); }
            50%      { box-shadow: 0 10px 45px rgba(45,212,191,0.60); }
        }
        .brand-icon { animation: brandGlow 3s ease-in-out infinite; }

        @keyframes pageFade {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .block-container { animation: pageFade 0.6s ease both; }

        div[data-testid="stDataFrame"] {
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            border-radius: 14px;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 16px 35px rgba(45,212,191,0.18);
        }

        div[data-testid="stButton"] > button {
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(45,212,191,0.25);
        }

        @keyframes answerFade {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .answer-card, .recovery-card { animation: answerFade 0.35s ease both; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# 3. SQL SANITIZATION
# ==================================================================
ALLOWED_TABLES = {"products"}
FORBIDDEN_TOKENS = re.compile(
    r"\b(insert|update|delete|drop|alter|pragma|attach|detach|replace|create|"
    r"vacuum|reindex|analyze|truncate|grant|revoke|begin|commit|rollback|"
    r"savepoint|exec|execute|sp_|xp_)\b",
    re.IGNORECASE,
)
COMMENT_PATTERN = re.compile(r"(--|/\*|\*/)", re.IGNORECASE)
STACKED_STMT_PATTERN = re.compile(r";\s*\S", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    pass


def sanitize_sql(sql: str) -> str:
    if not sql or not isinstance(sql, str):
        raise UnsafeSQLError("Empty or non-string SQL.")
    s = sql.strip().replace("```sql", "").replace("```", "").strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQLError("SQL is empty after cleaning.")
    lowered = s.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise UnsafeSQLError("Only SELECT / WITH queries are permitted.")
    if COMMENT_PATTERN.search(s):
        raise UnsafeSQLError("SQL comments are not allowed.")
    if STACKED_STMT_PATTERN.search(s):
        raise UnsafeSQLError("Multiple SQL statements are not allowed.")
    if FORBIDDEN_TOKENS.search(s):
        raise UnsafeSQLError("Forbidden keyword detected.")
    for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered):
        if t not in ALLOWED_TABLES:
            raise UnsafeSQLError(f"Table '{t}' is not allowed.")
    if re.search(r"\b(load_extension|readfile|writefile)\b", lowered):
        raise UnsafeSQLError("Dangerous SQLite function blocked.")
    return s


@contextmanager
def readonly_connection(db_path: str):
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=3.0)
    try:
        conn.execute("PRAGMA query_only = ON;")
        yield conn
    finally:
        conn.close()


@contextmanager
def write_connection(db_path: str):
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        yield conn
    finally:
        conn.close()


# ==================================================================
# 4. SCHEMA CACHE
# ==================================================================
@st.cache_data(show_spinner=False, ttl=300)
def get_schema_info(db_path: str) -> dict:
    with readonly_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]
        schema = {}
        for t in tables:
            cur.execute(f"PRAGMA table_info('{t}')")
            schema[t] = [
                {"name": r[1], "type": r[2], "notnull": bool(r[3]), "pk": bool(r[5])}
                for r in cur.fetchall()
            ]
        return schema


def schema_to_prompt(schema: dict) -> str:
    lines = []
    for table, cols in schema.items():
        lines.append(f"Table: {table}")
        lines.append("Columns:")
        for c in cols:
            lines.append(f"- {c['name']} {c['type']}")
        lines.append("")
    return "\n".join(lines)


# ==================================================================
# 5. FUZZY MATCHING
# ==================================================================
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    q, c = query.lower(), candidate.lower()
    if q in c:
        return 1.0
    dist = levenshtein(q, c)
    return max(0.0, 1.0 - dist / max(len(q), len(c)))


def suggest_similar_products(user_query: str, df: pd.DataFrame, top_n: int = 3):
    if df.empty:
        return []
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]{3,}", user_query.lower())
    if not tokens:
        return []
    scored = []
    for _, row in df.iterrows():
        name = str(row["name"])
        best = max((fuzzy_score(tok, name) for tok in tokens), default=0.0)
        if best >= FUZZY_MIN_SCORE:
            scored.append((best, name, row["stock"], row["price"]))
    scored.sort(reverse=True)
    return scored[:top_n]


# ==================================================================
# 6. DATA LAYER
# ==================================================================
@st.cache_data(show_spinner=False, ttl=15)
def fetch_data_cached(low_threshold: int, db_mtime: float):
    with readonly_connection(DB_FILE) as conn:
        df = pd.read_sql_query(
            "SELECT id, name, stock, price, stock * price AS total_val FROM products",
            conn,
        )
    df = df.sort_values("stock", ascending=True).reset_index(drop=True)

    def status(stock):
        if stock == 0:
            return "🔴 Out of Stock"
        if stock < low_threshold:
            return "🟡 Low Stock"
        return "🟢 In Stock"

    df["status"] = df["stock"].apply(status)
    return df


def fetch_data(low_threshold: int) -> pd.DataFrame:
    try:
        mtime = os.path.getmtime(DB_FILE)
    except OSError:
        mtime = 0.0
    return fetch_data_cached(low_threshold, mtime)


def db_last_updated() -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(DB_FILE)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def add_product(name: str, stock: int, price: float):
    name = (name or "").strip()
    if not name:
        return False, "Product name cannot be empty."
    if stock < 0:
        return False, "Stock cannot be negative."
    if price < 0:
        return False, "Price cannot be negative."

    try:
        with write_connection(DB_FILE) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM products WHERE LOWER(name) = LOWER(?)", (name,))
            if cur.fetchone():
                return False, f"'{name}' already exists in inventory."
            cur.execute(
                "INSERT INTO products (name, stock, price) VALUES (?, ?, ?)",
                (name, int(stock), float(price)),
            )
            conn.commit()
        return True, f"Added '{name}' ({stock} pcs, {price:.2f} MAD)."
    except Exception as e:
        return False, f"Database error: {e}"


def delete_product(product_id: int):
    try:
        with write_connection(DB_FILE) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM products WHERE id = ?", (product_id,))
            row = cur.fetchone()
            if not row:
                return False, f"Product #{product_id} not found."
            name = row[0]
            cur.execute("DELETE FROM products WHERE id = ?", (product_id,))
            conn.commit()
        return True, f"Deleted '{name}'."
    except Exception as e:
        return False, f"Database error: {e}"


def update_product(product_id: int, new_stock: int, new_price: float):
    if new_stock < 0:
        return False, "Stock cannot be negative."
    if new_price < 0:
        return False, "Price cannot be negative."
    try:
        with write_connection(DB_FILE) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM products WHERE id = ?", (product_id,))
            row = cur.fetchone()
            if not row:
                return False, f"Product #{product_id} not found."
            name = row[0]
            cur.execute(
                "UPDATE products SET stock = ?, price = ? WHERE id = ?",
                (int(new_stock), float(new_price), product_id),
            )
            conn.commit()
        return True, f"Updated '{name}' → {new_stock} pcs at {new_price:.2f} MAD."
    except Exception as e:
        return False, f"Database error: {e}"


def run_safe_query(sql: str) -> pd.DataFrame:
    clean = sanitize_sql(sql)
    with readonly_connection(DB_FILE) as conn:
        df = pd.read_sql_query(clean, conn)
    return df.head(MAX_ROWS_RETURNED)


# ==================================================================
# 6b. VISUALS — Three.js cube + Plotly charts
# ==================================================================
def render_threejs_cube():
    """A rotating 3D cube rendered with Three.js inside an iframe."""
    if components is None:
        return
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        html, body {
          margin: 0; padding: 0; background: transparent; overflow: hidden;
        }
        #cube-wrap {
          width: 100%; height: 220px;
          display: flex; align-items: center; justify-content: center;
        }
        canvas { display: block; }
      </style>
    </head>
    <body>
      <div id="cube-wrap"></div>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <script>
        const wrap = document.getElementById('cube-wrap');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.1, 1000);
        camera.position.z = 4.2;

        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(wrap.clientWidth, wrap.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        wrap.appendChild(renderer.domElement);

        // Main cube
        const geo = new THREE.BoxGeometry(1.6, 1.6, 1.6);
        const mats = [
          new THREE.MeshStandardMaterial({ color: 0x2dd4bf, metalness: 0.4, roughness: 0.3 }),
          new THREE.MeshStandardMaterial({ color: 0x45b8ff, metalness: 0.4, roughness: 0.3 }),
          new THREE.MeshStandardMaterial({ color: 0xa78bfa, metalness: 0.4, roughness: 0.3 }),
          new THREE.MeshStandardMaterial({ color: 0xfbbf24, metalness: 0.4, roughness: 0.3 }),
          new THREE.MeshStandardMaterial({ color: 0xfb7185, metalness: 0.4, roughness: 0.3 }),
          new THREE.MeshStandardMaterial({ color: 0x2dd4bf, metalness: 0.4, roughness: 0.3 })
        ];
        const cube = new THREE.Mesh(geo, mats);
        scene.add(cube);

        // Wireframe outline
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(geo),
          new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 })
        );
        cube.add(edges);

        // Lights
        const light1 = new THREE.PointLight(0xffffff, 1.4);
        light1.position.set(3, 3, 4);
        scene.add(light1);

        const light2 = new THREE.PointLight(0x2dd4bf, 1.2);
        light2.position.set(-4, -2, 3);
        scene.add(light2);

        const ambient = new THREE.AmbientLight(0x404060, 0.8);
        scene.add(ambient);

        // Animate
        function animate() {
          requestAnimationFrame(animate);
          cube.rotation.x += 0.005;
          cube.rotation.y += 0.008;
          renderer.render(scene, camera);
        }
        animate();

        // Resize
        window.addEventListener('resize', () => {
          const w = wrap.clientWidth, h = wrap.clientHeight;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        });
      </script>
    </body>
    </html>
    """
    components.html(html, height=230)


def render_pie_chart(df: pd.DataFrame):
    """Pie chart of products by stock status."""
    if df.empty:
        return None
    try:
        import plotly.express as px
    except ImportError:
        return None

    counts = df["status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]

    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map={
            "🔴 Out of Stock": "#fb7185",
            "🟡 Low Stock":    "#fbbf24",
            "🟢 In Stock":     "#2dd4bf",
        },
        hole=0.55,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont=dict(color="#eef3f8", size=12),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eef3f8"),
        showlegend=False,
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def render_bar_chart(df: pd.DataFrame):
    """Bar chart of top products by stock value."""
    if df.empty:
        return None
    try:
        import plotly.express as px
    except ImportError:
        return None

    top = df.sort_values("total_val", ascending=False).head(10).copy()
    top["total_val"] = top["total_val"].round(2)

    fig = px.bar(
        top,
        x="name",
        y="total_val",
        color="status",
        color_discrete_map={
            "🔴 Out of Stock": "#fb7185",
            "🟡 Low Stock":    "#fbbf24",
            "🟢 In Stock":     "#2dd4bf",
        },
        text="total_val",
    )
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        textfont=dict(color="#eef3f8", size=11),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eef3f8", size=11),
        xaxis=dict(title="", tickangle=-25),
        yaxis=dict(title="Value (MAD)", gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
        height=360,
        margin=dict(l=10, r=10, t=20, b=80),
    )
    return fig


def render_3d_inventory(df: pd.DataFrame):
    """Return a Plotly 3D scatter chart of the inventory (or None if empty)."""
    if df.empty:
        return None
    try:
        import plotly.express as px
    except ImportError:
        return None

    fig = px.scatter_3d(
        df,
        x="stock",
        y="price",
        z="total_val",
        color="status",
        size="stock",
        size_max=40,
        hover_name="name",
        hover_data={"stock": True, "price": True, "total_val": True},
        color_discrete_map={
            "🔴 Out of Stock": "#fb7185",
            "🟡 Low Stock":    "#fbbf24",
            "🟢 In Stock":     "#2dd4bf",
        },
    )
    fig.update_traces(
        marker=dict(opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.3)"))
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Stock (units)",
            yaxis_title="Price (MAD)",
            zaxis_title="Total value (MAD)",
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=False,
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=False,
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                showbackground=False,
            ),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eef3f8", size=11),
        height=520,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
    )
    return fig
# ==================================================================
# 7. PROMPTS
# ==================================================================
SYSTEM_PROMPT_TEMPLATE = """
You are an expert bilingual (English / French) inventory database assistant
for a Moroccan auto-parts store.

DATABASE SCHEMA (live, cached):
{schema_block}

The merchant speaks English and French, sometimes mixing both in one sentence.

STRICT RULES:
1. Generate EXACTLY ONE valid SQLite SELECT (or WITH ... SELECT) query.
2. Return ONLY raw SQL. No quotes, no backticks, no markdown, no explanation.
3. Use LIKE '%...%' for product names to be typo-resilient.
4. NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, ATTACH, CREATE.
5. NEVER reference tables other than the schema above.
6. NEVER add SQL comments. Do not add a trailing semicolon.
7. ALWAYS SELECT name, stock, price for product lookups.
   Never SELECT stock alone — the summary needs name and price to be accurate.
   Only exception: aggregate queries like SUM/COUNT.

EXAMPLES (all product lookups return name, stock, price):

Q: "How many spark plugs do I have?"
A: SELECT name, stock, price FROM products WHERE name LIKE '%Bougies%'

Q: "Do I have any batteries left, and what's the price?"
A: SELECT name, stock, price FROM products WHERE name LIKE '%Batterie%'

Q: "Do I have any oil in stock?"
A: SELECT name, stock, price FROM products WHERE name LIKE '%Huile%'

Q: "How many oil filters are left?"
A: SELECT name, stock, price FROM products WHERE name LIKE '%Filtre%'

Q: "What is out of stock?"
A: SELECT name, stock, price FROM products WHERE stock = 0

Q: "What is the total value of the stock?"
A: SELECT SUM(stock * price) AS total_value FROM products

Q: "Combien de filtres à huile reste-t-il?"
A: SELECT name, stock, price FROM products WHERE name LIKE '%Filtre%huile%'
"""


def build_system_prompt() -> str:
    schema = get_schema_info(DB_FILE)
    return SYSTEM_PROMPT_TEMPLATE.format(schema_block=schema_to_prompt(schema))


# ==================================================================
# 8. RECOVERY
# ==================================================================
def build_recovery_answer(question: str, reason: str, df_all: pd.DataFrame) -> str:
    suggestions = suggest_similar_products(question, df_all)
    base = {
        "empty": "No product matches your question exactly.",
        "unsafe": "I couldn't process this query for security reasons.",
        "sql_error": "I didn't quite understand the question. Could you rephrase it?",
    }.get(reason, "I couldn't process the request.")

    if not suggestions:
        critical = df_all[df_all["stock"] > 0].sort_values("stock").head(3)
        if critical.empty:
            return base
        alts = ", ".join(f"{r['name']} ({int(r['stock'])} pcs)" for _, r in critical.iterrows())
        return f"{base} Products to watch (still in stock): {alts}."

    alts = ", ".join(
        f"{name} ({int(stock)} pcs at {price:.2f} MAD)"
        for _, name, stock, price in suggestions
    )
    return f"{base} Close suggestions: {alts}."


# ==================================================================
# 9. VOICE
# ==================================================================
def transcribe_audio(audio_bytes) -> str:
    transcription = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("audio.wav", audio_bytes),
        language="fr",
        prompt="English and French auto-parts inventory questions.",
    )
    return transcription.text.strip()


# ==================================================================
# 10. AI PIPELINE
# ==================================================================
def ask_inventory_ai(question: str, df_all: pd.DataFrame):
    t_start = time.perf_counter()

    try:
        sql_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        raw_sql = sql_response.choices[0].message.content
    except Exception as e:
        return question, "", pd.DataFrame(), f"Service unavailable: {e}", 0, "sql_error"

    try:
        clean = sanitize_sql(raw_sql)
    except UnsafeSQLError:
        answer = build_recovery_answer(question, "unsafe", df_all)
        return question, raw_sql, pd.DataFrame(), answer, 0, "unsafe"

    try:
        result_df = run_safe_query(clean)
    except Exception:
        answer = build_recovery_answer(question, "sql_error", df_all)
        return question, clean, pd.DataFrame(), answer, 0, "sql_error"

    if result_df.empty:
        answer = build_recovery_answer(question, "empty", df_all)
        latency = (time.perf_counter() - t_start) * 1000
        return question, clean, result_df, answer, latency, "empty"

    if len(result_df) == 1 and len(result_df.columns) == 1:
        val = result_df.iloc[0, 0]
        if isinstance(val, (int, float)):
            answer = f"The result is: {val}."
            latency = (time.perf_counter() - t_start) * 1000
            return question, clean, result_df, answer, latency, None

    summary_prompt = f"""
    The merchant asked: "{question}" (English or French).

    DATABASE RESULT (this is the ONLY truth — do not invent anything):
    {result_df.to_dict(orient="records")}

    STRICT RULES FOR YOUR ANSWER:
    1. Use ONLY product names, stock counts, and prices from the data above.
    2. NEVER invent a price, product name, or number not present in the data.
    3. If a field is missing from the data, do not mention it.
    4. Keep the answer brief and practical, as on a quick phone call.
    5. Reply in the same language the merchant used (English or French).
    """
    try:
        final = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.1,
        )
        answer = final.choices[0].message.content.strip()
    except Exception as e:
        answer = f"(Summary unavailable: {e})"

    latency = (time.perf_counter() - t_start) * 1000
    return question, clean, result_df, answer, latency, None


# ==================================================================
# 11. TABLE STYLING
# ==================================================================
def style_inventory_table(df: pd.DataFrame, low_threshold: int):
    def row_style(row):
        if row["Stock"] == 0:
            return ["background-color: rgba(251,113,133,0.14); color: #ffd7dd;"] * len(row)
        if row["Stock"] < low_threshold:
            return ["background-color: rgba(251,191,36,0.10); color: #ffedc2;"] * len(row)
        return [""] * len(row)

    view = df[["name", "stock", "price", "status"]].copy()
    view["price"] = view["price"].map(lambda v: f"{v:,.2f}")
    view.columns = ["Product", "Stock", "Price (MAD)", "Status"]
    return view.style.apply(row_style, axis=1)


# ==================================================================
# 12. SESSION STATE
# ==================================================================
defaults = {
    "filter_status": "All",
    "low_threshold": 5,
    "user_query": "How many oil filters are left in stock?",
    "speak_answer": False,
    "last_answer": "",
    "last_latency_ms": 0.0,
    "history": [],
    "last_heard": "",
    "add_product_msg": None,
    "delete_msg": None,
    "edit_msg": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ==================================================================
# 13. DATA LOAD
# ==================================================================
df = fetch_data(st.session_state["low_threshold"])
total_products = len(df)
inventory_value = df["total_val"].sum()
low_stock_count = len(df[(df["stock"] > 0) & (df["stock"] < st.session_state["low_threshold"])])
out_of_stock_count = len(df[df["stock"] == 0])
last_updated = db_last_updated()


# ==================================================================
# 14. SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">📦</div>
            <div>
                <div class="brand-title">StockSmart AI</div>
                <div class="brand-subtitle">Live SQLite database feed</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f'<div class="live-tag"><div class="pulse-dot"></div> Live DB Synced ({now_str})</div>'
        f'<div style="font-size:11px;color:var(--muted);margin-bottom:10px;">'
        f'Last updated: <strong style="color:var(--text);">{last_updated}</strong></div>',
        unsafe_allow_html=True,
    )

    new_low = st.slider(
        "Low-stock threshold", 1, 20,
        value=st.session_state["low_threshold"], key="low_slider",
    )
    if new_low != st.session_state["low_threshold"]:
        st.session_state["low_threshold"] = new_low
        st.rerun()

    st.subheader("Inventory")
    f1, f2, f3 = st.columns(3)
    if f1.button("All", use_container_width=True):
        st.session_state["filter_status"] = "All"
    if f2.button(f"Low ({low_stock_count})", use_container_width=True):
        st.session_state["filter_status"] = "Low"
    if f3.button(f"Out ({out_of_stock_count})", use_container_width=True):
        st.session_state["filter_status"] = "Out"

    filtered_df = df.copy()
    if st.session_state["filter_status"] == "Low":
        filtered_df = filtered_df[
            (filtered_df["stock"] > 0)
            & (filtered_df["stock"] < st.session_state["low_threshold"])
        ]
    elif st.session_state["filter_status"] == "Out":
        filtered_df = filtered_df[filtered_df["stock"] == 0]

    if st.session_state["filter_status"] != "All":
        chip_label = "🟡 Low Stock" if st.session_state["filter_status"] == "Low" else "🔴 Out of Stock"
        st.markdown(
            f'<div class="filter-chip">Filter active: {chip_label} '
            f'({len(filtered_df)} rows) — clear with "All"</div>',
            unsafe_allow_html=True,
        )

    # ---------- Inventory table ----------
    st.dataframe(
        style_inventory_table(filtered_df, st.session_state["low_threshold"]),
        use_container_width=True,
        hide_index=True,
        height=340,
    )

    # ---------- CSV Export ----------
    csv_data = filtered_df[["name", "stock", "price", "status"]].copy()
    csv_data.columns = ["Product", "Stock", "Price (MAD)", "Status"]
    csv_bytes = csv_data.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Export inventory as CSV",
        data=csv_bytes,
        file_name=f"inventory_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="csv_export_btn",
    )

    # ---------- Delete feedback ----------
    if st.session_state.get("delete_msg"):
        ok, msg = st.session_state["delete_msg"]
        (st.success if ok else st.error)(msg)
        st.session_state["delete_msg"] = None

    # ---------- Delete section ----------
    with st.expander("🗑️ Delete a product"):
        if filtered_df.empty:
            st.caption("Nothing to delete in the current filter.")
        else:
            product_options = {
                f"{r['name']} ({int(r['stock'])} pcs)": int(r["id"])
                for _, r in filtered_df.iterrows()
            }
            selected_label = st.selectbox(
                "Choose a product",
                options=list(product_options.keys()),
                key="delete_select",
            )
            selected_id = product_options.get(selected_label)

            confirm = st.checkbox(
                "Yes, I'm sure I want to delete it",
                key="delete_confirm",
            )

            if st.button(
                "🗑️ Delete selected product",
                use_container_width=True,
                disabled=not confirm,
                key="delete_btn",
            ):
                ok, msg = delete_product(selected_id)
                st.session_state["delete_msg"] = (ok, msg)
                if ok:
                    st.cache_data.clear()
                st.rerun()

    # ---------- Edit section ----------
    if st.session_state.get("edit_msg"):
        ok, msg = st.session_state["edit_msg"]
        (st.success if ok else st.error)(msg)
        st.session_state["edit_msg"] = None

    with st.expander("✏️ Edit a product"):
        if df.empty:
            st.caption("No products to edit.")
        else:
            edit_options = {
                f"{r['name']} ({int(r['stock'])} pcs · {r['price']:,.2f} MAD)": int(r["id"])
                for _, r in df.iterrows()
            }
            edit_label = st.selectbox(
                "Choose a product",
                options=list(edit_options.keys()),
                key="edit_select",
            )
            edit_id = edit_options.get(edit_label)

            if edit_id is not None:
                current = df[df["id"] == edit_id].iloc[0]

                st.caption(
                    f"Current: **{int(current['stock'])} pcs** "
                    f"at **{current['price']:,.2f} MAD**"
                )

                new_stock_val = st.number_input(
                    "New stock",
                    min_value=0,
                    step=1,
                    value=int(current["stock"]),
                    format="%d",
                    key=f"edit_stock_{edit_id}",
                )
                new_price_val = st.number_input(
                    "New price (MAD)",
                    min_value=0.0,
                    step=1.0,
                    value=float(current["price"]),
                    format="%.2f",
                    key=f"edit_price_{edit_id}",
                )

                if st.button(
                    "💾 Save changes",
                    use_container_width=True,
                    key="edit_save_btn",
                ):
                    ok, msg = update_product(edit_id, new_stock_val, new_price_val)
                    st.session_state["edit_msg"] = (ok, msg)
                    if ok:
                        st.cache_data.clear()
                    st.rerun()

    # ---------- Example prompts ----------
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 13px; color: #94a3b8;">
            <strong>🇬🇧🇫🇷 Example Voice / Text Prompts:</strong><br>
            • <em>How many oil filters are left?</em><br>
            • <em>Do I have any batteries in stock?</em><br>
            • <em>What is out of stock?</em><br>
            • <em>Combien vaut le stock total?</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Add product form ----------
    st.markdown("---")
    st.markdown("### ➕ Add New Product")

    if st.session_state.get("add_product_msg"):
        ok, msg = st.session_state["add_product_msg"]
        (st.success if ok else st.error)(msg)
        st.session_state["add_product_msg"] = None

    with st.form("add_product_form", clear_on_submit=True):
        new_name = st.text_input(
            "Product name",
            placeholder="e.g. Filtre à gasoil",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            new_stock = st.number_input(
                "Stock", min_value=0, step=1, value=0, format="%d"
            )
        with col_b:
            new_price = st.number_input(
                "Price (MAD)", min_value=0.0, step=1.0, value=0.0, format="%.2f"
            )
        submitted = st.form_submit_button("➕ Add to inventory", use_container_width=True)

    if submitted:
        ok, msg = add_product(new_name, new_stock, new_price)
        st.session_state["add_product_msg"] = (ok, msg)
        if ok:
            st.cache_data.clear()
        st.rerun()


# ==================================================================
# 15. MAIN DASHBOARD
# ==================================================================
hero_col1, hero_col2 = st.columns([3, 1])

with hero_col1:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Local Retail Intelligence • Auto-Parts Morocco</div>
            <h1>StockSmart AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_col2:
    render_threejs_cube()

st.markdown("<br>", unsafe_allow_html=True)


# ==================================================================
# 15b. ANALYTICS — 2D Charts + 3D
# ==================================================================
with st.expander("📊 Inventory Analytics — Pie, Bar & 3D", expanded=False):
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 🥧 Distribution by status")
        pie_fig = render_pie_chart(df)
        if pie_fig is not None:
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.caption("No data to display.")

    with chart_col2:
        st.markdown("#### 📊 Top 10 by stock value")
        bar_fig = render_bar_chart(df)
        if bar_fig is not None:
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.caption("No data to display.")

    st.markdown("#### 🧊 3D Inventory — drag to rotate, scroll to zoom")
    fig3d = render_3d_inventory(df)
    if fig3d is not None:
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.caption("No data to display.")


# ==================================================================
# 16. VOICE + TEXT QUERY
# ==================================================================
st.subheader("💬 Ask Assistant (Voice / Quick Call Simulation)")
st.markdown(
    "<span style='font-size: 13px; color: #94a3b8;'>Try one of these:</span>",
    unsafe_allow_html=True,
)

v_col1, v_col2, v_col3 = st.columns(3)
if v_col1.button("🎙️ 'How many oil filters are left?'", use_container_width=True):
    st.session_state["user_query"] = "How many oil filters are left in stock, and what is the price?"
if v_col2.button("🎙️ 'Do I have any batteries?'", use_container_width=True):
    st.session_state["user_query"] = "Do I have any batteries in stock?"
if v_col3.button("🎙️ 'What is out of stock?'", use_container_width=True):
    st.session_state["user_query"] = "What is out of stock?"

if MIC_RECORDER_AVAILABLE:
    st.markdown('<span class="beta-chip">🎤 Voice input — beta (Groq Whisper)</span>', unsafe_allow_html=True)
    audio = mic_recorder(
        start_prompt="● Start recording",
        stop_prompt="■ Stop & transcribe",
        just_once=True,
        use_container_width=True,
        key="mic",
    )
    if audio and audio.get("bytes"):
        with st.spinner("Transcribing (Whisper)..."):
            try:
                voice_text = transcribe_audio(audio["bytes"])
                st.session_state["user_query"] = voice_text
                st.session_state["last_heard"] = voice_text
                if "query_box" in st.session_state:
                    del st.session_state["query_box"]
                if "mic" in st.session_state:
                    del st.session_state["mic"]
                st.rerun()
            except Exception as e:
                st.warning(f"Voice transcription failed ({e}). Falling back to text input.")

    if st.session_state.get("last_heard"):
        st.success(f"🎤 Heard: {st.session_state['last_heard']}")
        st.session_state["last_heard"] = ""
else:
    st.markdown(
        '<span class="beta-chip">🎤 Voice input disabled — '
        'install <code>streamlit-mic-recorder</code> to enable</span>',
        unsafe_allow_html=True,
    )

st.session_state["speak_answer"] = st.checkbox(
    "🔊 Speak answer back (browser TTS)", value=st.session_state["speak_answer"]
)

user_query = st.text_input(
    "Query input:",
    value=st.session_state["user_query"],
    label_visibility="collapsed",
    key="query_box",
)

run_clicked = st.button("⚡ Fast Response (< 0.8s)", type="primary", use_container_width=True)

if run_clicked and user_query.strip():
    with st.spinner("Processing local query..."):
        try:
            normalized, sql, result_df, answer, latency_ms, reason = ask_inventory_ai(user_query, df)
            st.session_state["last_answer"] = answer
            st.session_state["last_latency_ms"] = latency_ms

            st.session_state["history"].append({
                "q": user_query,
                "a": answer,
                "sql": sql,
                "latency_ms": latency_ms,
                "reason": reason,
                "time": datetime.now().strftime("%H:%M:%S"),
                "rows": len(result_df),
            })
            if len(st.session_state["history"]) > 25:
                st.session_state["history"] = st.session_state["history"][-25:]

            is_recovery = reason in ("empty", "unsafe", "sql_error")
            card_class = "recovery-card" if is_recovery else "answer-card"
            title = "💡 Suggestion" if is_recovery else "📞 Instant Assistant Answer:"
            title_class = "recovery-title" if is_recovery else "answer-title"
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="{title_class}">{title}</div>
                    <div style="font-size: 16px; font-weight: 500;">{answer}</div>
                    <div class="latency-badge">⏱ {latency_ms:.0f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not result_df.empty:
                st.dataframe(result_df, use_container_width=True, hide_index=True)

            with st.expander("🛠️ View Generated SQLite Query"):
                st.code(sql or "(no SQL generated)", language="sql")

            if st.session_state["speak_answer"] and components is not None:
                safe_answer = (
                    answer.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
                )
                components.html(
                    f"""
                    <script>
                        window.speechSynthesis.cancel();
                        const u = new SpeechSynthesisUtterance(`{safe_answer}`);
                        u.lang = 'fr-FR';
                        u.rate = 1.0;
                        window.speechSynthesis.speak(u);
                    </script>
                    """,
                    height=0,
                )

        except Exception as error:
            st.error(f"Error: {error}")
elif run_clicked:
    st.warning("Please enter a question.")


# ==================================================================
# 17. CHAT HISTORY
# ==================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

h_col1, h_col2 = st.columns([6, 1])
with h_col1:
    count = len(st.session_state["history"])
    st.subheader(f"🕘 Conversation History ({count})")
with h_col2:
    if st.session_state["history"]:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_history"):
            st.session_state["history"] = []
            st.rerun()

if not st.session_state["history"]:
    st.info(
        "No conversation yet. Ask a question above — your history will appear here."
    )
else:
    for i, item in enumerate(reversed(st.session_state["history"])):
        is_recovery = item["reason"] in ("empty", "unsafe", "sql_error")
        badge = "💡" if is_recovery else "✅"

        with st.expander(
            f"{badge} [{item['time']}] {item['q'][:80]}"
            f"{'…' if len(item['q']) > 80 else ''}",
            expanded=(i == 0),
        ):
            st.markdown(f"**❓ Question:** {item['q']}")
            st.markdown(f"**💬 Answer:** {item['a']}")

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.caption(f"⏱ {item['latency_ms']:.0f} ms")
            meta_col2.caption(f"📊 {item['rows']} row(s)")
            meta_col3.caption(f"🏷️ {item['reason'] or 'ok'}")

            if item["sql"]:
                with st.expander("🛠️ SQL used"):
                    st.code(item["sql"], language="sql")