# 🚀 Image Embedding Service with CLIP

This service uses a CLIP model from Hugging Face to generate image embeddings via an API.  
Below are the steps to set up and run the service in a development environment.

---

## 📦 Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Service (Development Mode)

Start the service with hot-reload enabled:

```bash
uvicorn main:app --reload
```

---

## ⏳ First-Time Model Download

On the first run, the service will **download the CLIP model** from Hugging Face in the background.
This may take a few moments depending on your internet connection.

---

## ✅ Service Ready

Once you see the following message in the terminal:

```
Application startup complete.
```

...the service is ready to use. Open your browser and navigate to:

```
http://localhost:8000/docs
```

You can now interact with the API via the interactive Swagger UI.

---

## 📘 Notes

- Ensure a stable internet connection for the initial model download.
- The model will be cached, so future runs will be faster.
