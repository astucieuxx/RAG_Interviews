# 🚀 Guía de Despliegue - Compartir la Aplicación

Hay varias formas de compartir esta aplicación con tus compañeros. Aquí están las opciones:

## Opción 1: Streamlit Cloud (Recomendado - Gratis y Fácil)

Streamlit Cloud es la forma más fácil de compartir aplicaciones Streamlit.

### Pasos:

1. **Sube tu código a GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - RAG Interviews app"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/RAG_Interviews.git
   git push -u origin main
   ```

2. **Ve a [share.streamlit.io](https://share.streamlit.io)**

3. **Conecta tu repositorio de GitHub**

4. **Configura la aplicación:**
   - **Main file path:** `app.py`
   - **Python version:** 3.9 o superior
   - **Secrets:** Agrega tu `OPENAI_API_KEY`:
     ```
     OPENAI_API_KEY=tu-api-key-aqui
     ```

5. **¡Listo!** Streamlit Cloud te dará una URL como:
   `https://rag-interviews.streamlit.app`

### ⚠️ Importante:
- Los PDFs NO se suben a GitHub (están en `.gitignore` por seguridad)
- Necesitarás que alguien ejecute `ingest.py` localmente y suba la carpeta `chromadb_data/` al repositorio, O
- Configurar un proceso de ingesta automática en Streamlit Cloud

---

## Opción 2: Render (Alternativa Gratuita)

1. Ve a [render.com](https://render.com)
2. Crea una cuenta y conecta tu repositorio de GitHub
3. Crea un nuevo "Web Service"
4. Configura:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Environment Variables:** Agrega `OPENAI_API_KEY`

---

## Opción 3: ngrok (Para Pruebas Rápidas)

Si solo quieres compartir temporalmente desde tu máquina:

1. **Instala ngrok:**
   ```bash
   brew install ngrok  # macOS
   # O descarga desde https://ngrok.com
   ```

2. **Ejecuta tu app:**
   ```bash
   streamlit run app.py
   ```

3. **En otra terminal, ejecuta ngrok:**
   ```bash
   ngrok http 8501
   ```

4. **Comparte la URL que ngrok te da** (ej: `https://abc123.ngrok.io`)

⚠️ **Nota:** La URL expira cuando cierras ngrok o después de un tiempo.

---

## Opción 4: Subir Base de Datos a GitHub

Si quieres que la app funcione inmediatamente en Streamlit Cloud sin que otros ejecuten `ingest.py`:

1. **Sube la carpeta `chromadb_data/` a GitHub:**
   ```bash
   # Remueve chromadb_data/ del .gitignore temporalmente
   git add chromadb_data/
   git commit -m "Add pre-processed database"
   git push
   ```

2. **Luego vuelve a agregarlo al .gitignore** para futuros commits

⚠️ **Nota:** La carpeta `chromadb_data/` puede ser grande (~20MB). GitHub tiene límites de tamaño de archivo.

---

## Recomendación Final

**Para producción:** Usa **Streamlit Cloud** (Opción 1)
- Es gratuito
- Fácil de configurar
- Maneja actualizaciones automáticas desde GitHub
- URL permanente y profesional

**Para pruebas rápidas:** Usa **ngrok** (Opción 3)
- No requiere configuración de servidor
- Perfecto para demos temporales

---

## Configuración de Seguridad

⚠️ **IMPORTANTE:** Nunca subas tu `OPENAI_API_KEY` al código. Siempre usa:
- Variables de entorno en Streamlit Cloud
- Secrets en Render
- Archivo `.env` local (que está en `.gitignore`)
