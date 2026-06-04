import os, json, time, asyncio, sqlite3, httpx, re
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Config ────────────────────────────────────────────────────────────────────
app          = FastAPI()
LMSTUDIO_URL = "http://127.0.0.1:1234" # Puerto por defecto de LM Studio
DIR_PROMPTS  = "prompts"
DB_FILE      = "experimentos.db"

os.makedirs(DIR_PROMPTS,  exist_ok=True)

if not os.listdir(DIR_PROMPTS):
    with open(os.path.join(DIR_PROMPTS, "experto_ciber.txt"), "w", encoding="utf-8") as f:
        f.write("Eres un experto senior en ciberseguridad. Respondes con precision tecnica, sin saludos y directo al punto.")

_abort_events: Dict[str, asyncio.Event] = {}

# ── DB (Migración a SQLite 100%) ──────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sesiones (
        id_sesion        TEXT PRIMARY KEY,
        titulo_ia        TEXT,
        titulo_humano    TEXT,
        subtitulo_humano TEXT,
        reacciones       TEXT,
        fecha            TEXT,
        duracion_total_s REAL    DEFAULT 0,
        es_resesion      INTEGER DEFAULT 0,
        sesion_origen    TEXT    DEFAULT NULL,
        system_prompt    TEXT    DEFAULT '',
        parametros_iniciales TEXT DEFAULT '{}',
        modelo           TEXT    DEFAULT ''
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS interacciones (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        id_sesion            TEXT,
        turno                INTEGER,
        modelo               TEXT,
        prompt_usuario       TEXT,
        respuesta_modelo     TEXT,
        tiempo_escritura_s   REAL    DEFAULT 0,
        tiempo_inferencia_s  REAL    DEFAULT 0,
        tps                  REAL    DEFAULT 0,
        prompt_tokens        INTEGER DEFAULT 0,
        output_tokens        INTEGER DEFAULT 0,
        delta_tokens         INTEGER DEFAULT 0,
        ctx_acumulado        INTEGER DEFAULT 0,
        ratio_proc_gen       REAL    DEFAULT 0,
        editado              INTEGER DEFAULT 0,
        abortado             INTEGER DEFAULT 0,
        FOREIGN KEY(id_sesion) REFERENCES sesiones(id_sesion) ON DELETE CASCADE
    )""")
    
    migraciones = [
        ("sesiones", "titulo_humano", "TEXT"),
        ("sesiones", "system_prompt", "TEXT DEFAULT ''"),
        ("sesiones", "parametros_iniciales", "TEXT DEFAULT '{}'"),
        ("sesiones", "modelo", "TEXT DEFAULT ''"),
        ("interacciones", "delta_tokens", "INTEGER DEFAULT 0"),
        ("interacciones", "ctx_acumulado", "INTEGER DEFAULT 0"),
        ("interacciones", "ratio_proc_gen", "REAL DEFAULT 0"),
        ("interacciones", "abortado", "INTEGER DEFAULT 0")
    ]
    for tabla, col, defn in migraciones:
        try: c.execute(f"ALTER TABLE {tabla} ADD COLUMN {col} {defn}")
        except Exception: pass
        
    conn.commit()
    conn.close()

init_db()

# ── Pydantic ──────────────────────────────────────────────────────────────────
class IniciarSesionPayload(BaseModel):
    id_sesion:     str; modelo: str; system_prompt: str; parametros: Dict[str, Any]
    es_resesion:   bool = False; sesion_origen: Optional[str] = None

class InferenciaPayload(BaseModel):
    id_sesion: str; modelo: str; system_prompt: str; origen_prompt: str
    mensajes_historial: List[Dict[str, str]]; prompt_actual: str
    temperatura: float; top_p: float; top_k: int; max_tokens: int
    timeout_s: int = 180; tiempo_escritura_s: float = 0.0; prompt_tokens_prev: int = 0

class GuardarTurnoPayload(BaseModel):
    id_sesion: str; turno: int; modelo: str; prompt_usuario: str; respuesta_modelo: str
    tiempo_escritura_s: float; tiempo_inferencia_s: float; tps: float
    prompt_tokens: int; output_tokens: int; delta_tokens: int = 0
    ctx_acumulado: int = 0; ratio_proc_gen: float = 0.0; abortado: bool = False

class EditarTurnoPayload(BaseModel):
    id_sesion: str; turno: int; campo: str; contenido: str

class RenombrarSesionPayload(BaseModel):
    id_sesion: str; titulo_humano: str

class EliminarTurnosPayload(BaseModel):
    id_sesion: str; turnos: List[int]

class FinalizarPayload(BaseModel):
    id_sesion: str; subtitulo_humano: str; reacciones_viscerales: str

class NuevoPromptPayload(BaseModel):
    nombre: str; contenido: str

# ── Utils ─────────────────────────────────────────────────────────────────────
def safe_prompt_path(nombre: str) -> str:
    limpio = os.path.basename(nombre)
    ruta   = os.path.realpath(os.path.join(DIR_PROMPTS, limpio))
    base   = os.path.realpath(DIR_PROMPTS)
    if not ruta.startswith(base + os.sep): raise HTTPException(status_code=400, detail="Invalido")
    return ruta

def extraer_meta(model_id: str) -> Dict[str, Any]:
    meta = {"cuantizacion":"—","tamano":"—","familia":"—","contexto_max":8192,"size_gb":0}
    q = re.search(r'(q\d[_a-z0-9]*)', model_id, re.IGNORECASE)
    if q: meta["cuantizacion"] = q.group(1).upper()
    b = re.search(r'(\d+(?:\.\d+)?b)', model_id, re.IGNORECASE)
    if b: meta["tamano"] = b.group(1).lower()
    meta["familia"] = model_id.split("-")[0]
    return meta

# ── Rutas estáticas & Telemetría ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(ruta, "r", encoding="utf-8") as f: return f.read()

@app.get("/api/telemetria")
async def get_telemetria():
    if HAS_PSUTIL: return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}
    return {"cpu": 0, "ram": 0, "mock": True}

# ── Modelos & Estado (API de LM Studio / OpenAI) ──────────────────────────────
@app.get("/api/modelos")
async def get_modelos():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            # LM Studio usa el estandar de OpenAI
            resp = await client.get(f"{LMSTUDIO_URL}/v1/models")
            modelos = []
            for m in resp.json().get("data", []):
                mid = m.get("id", "")
                base = extraer_meta(mid)
                modelos.append({"id": mid, "metadatos": {
                    "cuantizacion": base["cuantizacion"],
                    "tamano":       base["tamano"],
                    "familia":      base["familia"],
                    "contexto_max": base["contexto_max"],
                    "size_gb":      base["size_gb"],
                }})
            return {"data": modelos}
        except Exception as e:
            return {"data": [], "error": str(e)}

@app.get("/api/ollama/estado")
async def lmstudio_estado():
    # Mantenemos la ruta /api/ollama/estado para no romper el frontend, 
    # pero consultamos a LM Studio.
    async with httpx.AsyncClient(timeout=3) as client:
        try:
            resp = await client.get(f"{LMSTUDIO_URL}/v1/models")
            modelos_activos = resp.json().get("data", [])
            # Si hay modelos cargados, asumimos que está listo
            info = [{"modelo": m.get("id","")} for m in modelos_activos]
            return {"ocupado": False, "modelos": info}
        except Exception:
            return {"ocupado": False, "modelos": [], "error": "sin conexion"}

# ── Prompts ───────────────────────────────────────────────────────────────────
@app.get("/api/prompts")
async def listar_prompts():
    return {"prompts": [f for f in sorted(os.listdir(DIR_PROMPTS)) if f.endswith(".txt")]}

@app.get("/api/prompts/{nombre}")
async def cargar_prompt(nombre: str):
    with open(safe_prompt_path(nombre), "r", encoding="utf-8") as f: return {"contenido": f.read()}

@app.post("/api/prompts")
async def crear_prompt(payload: NuevoPromptPayload):
    if not payload.nombre.endswith(".txt"): payload.nombre += ".txt"
    with open(safe_prompt_path(payload.nombre), "w", encoding="utf-8") as f: f.write(payload.contenido)
    return {"status": "ok"}

# ── Sesiones CRUD ─────────────────────────────────────────────────────────────
@app.post("/api/sesiones/iniciar")
async def iniciar_sesion(payload: IniciarSesionPayload):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""INSERT OR IGNORE INTO sesiones
                 (id_sesion,titulo_ia,subtitulo_humano,reacciones,fecha,
                  duracion_total_s,es_resesion,sesion_origen,system_prompt,parametros_iniciales,modelo)
                 VALUES (?,?,'','',?,0,?,?,?,?,?)""",
              (payload.id_sesion, f"Sesión {payload.modelo}", datetime.now().isoformat(),
               int(payload.es_resesion), payload.sesion_origen, payload.system_prompt,
               json.dumps(payload.parametros), payload.modelo))
    conn.commit(); conn.close()
    return {"status": "ok"}

@app.post("/api/sesiones/turno")
async def guardar_turno(payload: GuardarTurnoPayload):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""INSERT INTO interacciones
                 (id_sesion,turno,modelo,prompt_usuario,respuesta_modelo,
                  tiempo_escritura_s,tiempo_inferencia_s,tps,prompt_tokens,output_tokens,
                  delta_tokens,ctx_acumulado,ratio_proc_gen,abortado)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (payload.id_sesion, payload.turno, payload.modelo, payload.prompt_usuario, payload.respuesta_modelo,
               payload.tiempo_escritura_s, payload.tiempo_inferencia_s, payload.tps, payload.prompt_tokens,
               payload.output_tokens, payload.delta_tokens, payload.ctx_acumulado, payload.ratio_proc_gen, int(payload.abortado)))
    c.execute("""UPDATE sesiones SET duracion_total_s=(SELECT COALESCE(SUM(tiempo_escritura_s+tiempo_inferencia_s),0) FROM interacciones WHERE id_sesion=?) WHERE id_sesion=?""", (payload.id_sesion, payload.id_sesion))
    conn.commit(); conn.close()
    return {"status": "ok"}

@app.put("/api/sesiones/turno/editar")
async def editar_turno(payload: EditarTurnoPayload):
    if payload.campo not in ("prompt_usuario", "respuesta_modelo"): raise HTTPException(status_code=400)
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute(f"UPDATE interacciones SET {payload.campo}=?,editado=1 WHERE id_sesion=? AND turno=?", (payload.contenido, payload.id_sesion, payload.turno))
    conn.commit(); conn.close()
    return {"status": "ok"}

@app.put("/api/sesiones/renombrar")
async def renombrar_sesion(payload: RenombrarSesionPayload):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("UPDATE sesiones SET titulo_humano=? WHERE id_sesion=?", (payload.titulo_humano, payload.id_sesion))
    conn.commit(); conn.close()
    return {"status": "ok"}

@app.delete("/api/sesiones/{id_sesion}")
async def eliminar_sesion(id_sesion: str):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("DELETE FROM interacciones WHERE id_sesion=?", (id_sesion,))
    c.execute("DELETE FROM sesiones WHERE id_sesion=?", (id_sesion,))
    conn.commit(); conn.close()
    return {"status": "ok"}

@app.delete("/api/sesiones/{id_sesion}/turnos")
async def eliminar_turnos(id_sesion: str, payload: EliminarTurnosPayload):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    for t in payload.turnos: c.execute("DELETE FROM interacciones WHERE id_sesion=? AND turno=?", (id_sesion, t))
    conn.commit(); conn.close()
    return {"status": "ok"}

# ── Inferencia (Adaptación a LM Studio) ───────────────────────────────────────
@app.post("/api/inferencia/abortar")
async def abortar(payload: BaseModel):
    id_sesion = getattr(payload, "id_sesion", None)
    if id_sesion and id_sesion in _abort_events: _abort_events[id_sesion].set()
    return {"status": "ok"}

@app.post("/api/inferencia")
async def inferencia(payload: InferenciaPayload, request: Request):
    mensajes = [{"role": "system", "content": payload.system_prompt}]
    mensajes.extend(payload.mensajes_historial)
    mensajes.append({"role": "user", "content": payload.prompt_actual})

    # Formato OpenAI compatible
    lmstudio_payload = {
        "model": payload.modelo,
        "messages": mensajes,
        "stream": True,
        "temperature": payload.temperatura,
        "top_p": payload.top_p,
        "max_tokens": payload.max_tokens,
        "stream_options": {"include_usage": True} # Solicita conteo de tokens en stream si el server lo soporta
    }

    abort_event = asyncio.Event()
    _abort_events[payload.id_sesion] = abort_event
    timeout = httpx.Timeout(connect=10.0, read=float(payload.timeout_s), write=10.0, pool=5.0)

    async def stream_gen():
        start = time.time()
        prompt_tokens = 0
        output_tokens = 0
        got_prompt = False

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{LMSTUDIO_URL}/v1/chat/completions", json=lmstudio_payload) as resp:
                    async for line in resp.aiter_lines():
                        if abort_event.is_set(): await resp.aclose(); yield "[ABORTED]||\n"; return
                        if await request.is_disconnected(): await resp.aclose(); return
                        if not line.strip(): continue
                        
                        # Parseo de Server-Sent Events (SSE)
                        if line.startswith("data: "):
                            content = line[6:].strip()
                            if content == "[DONE]": 
                                break
                            try:
                                data = json.loads(content)
                            except Exception:
                                continue

                            # Extracción de uso de tokens
                            if "usage" in data and data["usage"]:
                                prompt_tokens = data["usage"].get("prompt_tokens", prompt_tokens)
                                output_tokens = data["usage"].get("completion_tokens", output_tokens)

                            if not got_prompt:
                                got_prompt = True
                                # Enviamos primer flag al frontend para cambiar estado visual
                                delta_tok = max(0, prompt_tokens - payload.prompt_tokens_prev)
                                yield f"[PROMPT_DONE]||{prompt_tokens}||0||{delta_tok}\n"

                            choices = data.get("choices", [])
                            if choices:
                                token = choices[0].get("delta", {}).get("content", "")
                                if token:
                                    elapsed = time.time() - start
                                    tps_live = round(output_tokens / elapsed, 1) if elapsed > 0 else 0
                                    tok_enc = token.replace("\\","\\\\").replace("\n","\\n")
                                    yield f"[TK]||{tok_enc}||{output_tokens}||{tps_live}\n"

            # Cierre final
            total = round(time.time() - start, 2)
            tps_final = round(output_tokens / total, 2) if total > 0 else 0
            delta_tok = max(0, prompt_tokens - payload.prompt_tokens_prev)
            yield f"[DONE]||{total}||{tps_final}||{prompt_tokens}||{output_tokens}||{delta_tok}||0\n"

        except httpx.TimeoutException: yield "[ERROR]||timeout\n"
        except Exception as e: yield f"[ERROR]||{str(e)[:120]}\n"
        finally: _abort_events.pop(payload.id_sesion, None)

    return StreamingResponse(stream_gen(), media_type="text/plain")

# ── Cargar & Leer ─────────────────────────────────────────────────────────────
@app.get("/api/sesiones")
async def listar_sesiones():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT id_sesion,titulo_ia,titulo_humano,subtitulo_humano,
                        fecha,duracion_total_s,es_resesion,sesion_origen
                 FROM sesiones ORDER BY fecha DESC LIMIT 100""")
    rows = c.fetchall(); conn.close()
    return {"sesiones": [{"id_sesion": r[0], "titulo": r[2] or r[1] or r[0], "titulo_ia": r[1] or r[0], "titulo_humano": r[2], "subtitulo": r[3] or "", "fecha": r[4], "duracion_total_s": r[5] or 0, "es_resesion": bool(r[6]), "sesion_origen": r[7]} for r in rows]}

@app.get("/api/sesiones/{id_sesion}")
async def cargar_sesion(id_sesion: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM sesiones WHERE id_sesion=?", (id_sesion,))
    s = c.fetchone()
    if not s: conn.close(); raise HTTPException(status_code=404)
    
    c.execute("SELECT * FROM interacciones WHERE id_sesion=? ORDER BY turno", (id_sesion,))
    i_rows = c.fetchall()
    conn.close()

    interacciones = []
    for r in i_rows:
        interacciones.append({
            "turno": r[2], "role_user": r[4], "role_assistant": r[5], "abortado": bool(r[15]), "editado": bool(r[14]),
            "metricas": { "tiempo_escritura_s": r[6], "tiempo_inferencia_s": r[7], "tps": r[8], "prompt_tokens": r[9], "output_tokens": r[10], "delta_tokens": r[11], "ctx_acumulado": r[12], "ratio_proc_gen": r[13] }
        })

    return {
        "id_sesion": s[0], "titulo_ia": s[1], "titulo_humano": s[2], "subtitulo_humano": s[3],
        "reacciones_viscerales": s[4], "fecha": s[5], "duracion_total_s": s[6], "es_resesion": bool(s[7]),
        "sesion_origen": s[8], "system_prompt": s[9], "parametros_iniciales": json.loads(s[10] or '{}'),
        "modelo_activo": {"id": s[11]}, "interacciones": interacciones
    }

@app.get("/api/sesiones/{id_sesion}/guion")
async def exportar_guion(id_sesion: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT titulo_humano, titulo_ia, modelo, system_prompt, parametros_iniciales, fecha FROM sesiones WHERE id_sesion=?", (id_sesion,))
    s = c.fetchone()
    if not s: conn.close(); raise HTTPException(status_code=404)
    
    c.execute("SELECT turno, prompt_usuario FROM interacciones WHERE id_sesion=? AND prompt_usuario IS NOT NULL ORDER BY turno", (id_sesion,))
    i_rows = c.fetchall()
    conn.close()

    titulo = s[0] or s[1] or id_sesion
    modelo = s[2]
    sys_p  = s[3]
    params = json.loads(s[4] or '{}')
    fecha  = s[5]
    prompts = [{"turno": r[0], "prompt": r[1]} for r in i_rows]

    txt = f"# GUIÓN — {titulo}\n# Fecha: {fecha}\n# Modelo: {modelo}\n\n"
    for p in prompts: txt += f"[{p['turno']+1}] {p['prompt']}\n\n"

    # Exportación adaptada a la API de LM Studio
    script  = f'"""Guion de replicacion — {titulo}\\nGenerado por LAB21 (Backend LM Studio)"""\n\nimport httpx, json\n\n'
    script += f'LMSTUDIO_URL = "{LMSTUDIO_URL}"\nMODELO = "{modelo}"\nSYSTEM_PROMPT = """{sys_p}"""\n\n'
    script += f'PARAMS = {json.dumps({"temperature": params.get("temperatura",0.8), "top_p": params.get("top_p",0.95), "max_tokens": params.get("max_tokens",1024)}, indent=4)}\n\n'
    script += 'PROMPTS = [\n'
    for p in prompts: script += f'    "{p["prompt"].replace(chr(92),chr(92)*2).replace(chr(34),chr(92)+chr(34))}",\n'
    script += ']\n\n'
    script += '''def run():
    historial = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\\n[{i+1}/{len(PROMPTS)}] Enviando...")
        mensajes = [{"role": "system", "content": SYSTEM_PROMPT}] + historial + [{"role": "user", "content": prompt}]
        resp = httpx.post(f"{LMSTUDIO_URL}/v1/chat/completions", json={"model": MODELO, "messages": mensajes, "stream": False, **PARAMS}, timeout=300)
        contenido = resp.json().get("choices",[{}])[0].get("message",{}).get("content","")
        print(f"Respuesta:\\n{contenido}\\n")
        historial.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": contenido}])
if __name__ == "__main__": run()'''

    return {"titulo": titulo, "prompts": prompts, "texto_plano": txt, "script_python": script}

# ── Finalizar (Titulación con API de LM Studio) ───────────────────────────────
@app.post("/api/sesiones/finalizar")
async def finalizar_sesion(payload: FinalizarPayload):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT modelo, titulo_ia, titulo_humano FROM sesiones WHERE id_sesion=?", (payload.id_sesion,))
    row = c.fetchone()
    if not row: conn.close(); return {"status": "error"}
    
    modelo, titulo_ia, titulo_humano = row
    
    if titulo_ia.startswith("Sesión ") or not titulo_ia:
        c.execute("SELECT prompt_usuario, respuesta_modelo FROM interacciones WHERE id_sesion=? LIMIT 3", (payload.id_sesion,))
        msgs = [{"role":"system","content":"Titulador tecnico. Responde SOLO el titulo, max 5 palabras, sin puntuacion final."}]
        for r in c.fetchall():
            msgs.append({"role": "user", "content": str(r[0])[:400]})
            msgs.append({"role": "assistant", "content": str(r[1])[:400]})
        msgs.append({"role":"user","content":"Titulo tecnico de esta conversacion."})
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(f"{LMSTUDIO_URL}/v1/chat/completions", json={"model": modelo, "messages": msgs, "stream": False, "temperature": 0.3, "max_tokens": 20})
                titulo_ia = r.json().get("choices",[{}])[0].get("message",{}).get("content", titulo_ia).strip().strip('"\'')
        except Exception: pass

    c.execute("UPDATE sesiones SET titulo_ia=?, subtitulo_humano=?, reacciones=? WHERE id_sesion=?",
              (titulo_ia, payload.subtitulo_humano, payload.reacciones_viscerales, payload.id_sesion))
    conn.commit(); conn.close()
    return {"status": "ok", "titulo_ia": titulo_ia}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
