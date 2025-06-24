# wsgi.py

import sys
from a2wsgi import ASGIMiddleware

# Ganti 'main' dengan nama file utama FastAPI Anda, 
# dan 'app' dengan nama variabel FastAPI Anda (misal: app = FastAPI())
from main import app

# === Bagian ini akan diubah nanti di server PythonAnywhere ===
# Ini adalah placeholder untuk path proyek Anda di server
path = '/home/USERNAME_PYTHONANYWHERE/FOLDERNAME_PROJECT'
if path not in sys.path:
    sys.path.insert(0, path)
# =============================================================

# Bungkus aplikasi ASGI (FastAPI) Anda dengan middleware WSGI
application = ASGIMiddleware(app)