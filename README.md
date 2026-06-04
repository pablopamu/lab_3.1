pip install fastapi uvicorn httpx pydantic psutil

🐧 Linux (Debian / Raspberry Pi)
En entornos basados en Debian, a veces es necesario instalar el paquete para crear entornos virtuales antes de empezar.

1. Instalar dependencias del sistema (si es primera vez):

Bash
sudo apt update
sudo apt install python3-venv python3-pip
2. Crear y activar el entorno virtual:

Bash
python3 -m venv venv
source venv/bin/activate
3. Instalar las librerías de LAB21:

Bash
pip install -r requirements.txt
4. Iniciar el servidor:

Bash
uvicorn servidor:app --host 0.0.0.0 --port 8000
(Nota: Usar 0.0.0.0 es ideal aquí. Te permitirá abrir el navegador en el propio equipo usando http://127.0.0.1:8000, pero también te dejará acceder desde otro dispositivo en tu misma red wifi si ingresas la IP local de tu máquina Linux).

🍏 Mac (macOS)
En Mac, Python 3 ya viene preinstalado en las versiones modernas, por lo que el proceso es muy directo usando la terminal nativa.

1. Crear y activar el entorno virtual:

Bash
python3 -m venv venv
source venv/bin/activate
2. Instalar las librerías:

Bash
pip install -r requirements.txt
3. Iniciar el servidor:

Bash
uvicorn servidor:app --host 127.0.0.1 --port 8000
(Abre tu navegador web y entra a http://127.0.0.1:8000 para ver la interfaz).

🪟 Windows
Abre tu consola (Símbolo del sistema o PowerShell). El único cambio respecto a los demás sistemas es cómo se activa el entorno virtual por la estructura de carpetas de Windows.

1. Crear el entorno virtual:

DOS
python -m venv venv
2. Activar el entorno virtual:

DOS
venv\Scripts\activate
3. Instalar las librerías:

DOS
pip install -r requirements.txt
4. Iniciar el servidor:

DOS
uvicorn servidor:app --host 127.0.0.1 --port 8000
(Abre tu navegador web y entra a http://127.0.0.1:8000).

NOTA:
El servidor_lms.py es para conectarse con LM-studio
