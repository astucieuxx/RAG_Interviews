# 📤 Guía para Subir el Proyecto a GitHub

## Paso 1: Crear el Repositorio en GitHub

1. Ve a [github.com](https://github.com) e inicia sesión
2. Haz clic en el botón **"+"** (arriba a la derecha) → **"New repository"**
3. Configura:
   - **Repository name:** `RAG_Interviews` (o el nombre que prefieras)
   - **Description:** "RAG-powered analysis of competitive intelligence interviews"
   - **Visibility:** Private (recomendado) o Public
   - **NO marques** "Add a README file" (ya tenemos uno)
   - **NO marques** "Add .gitignore" (ya tenemos uno)
4. Haz clic en **"Create repository"**

## Paso 2: Inicializar Git en tu Proyecto

Abre tu terminal en la carpeta del proyecto y ejecuta:

```bash
cd /Users/benjaminmiranda/Desktop/RAG_Interviews

# Inicializar git
git init

# Agregar todos los archivos (excepto los que están en .gitignore)
git add .

# Hacer el primer commit
git commit -m "Initial commit - RAG Interviews app"
```

## Paso 3: Decidir sobre chromadb_data/

**IMPORTANTE:** La carpeta `chromadb_data/` está en `.gitignore` por defecto. Tienes dos opciones:

### Opción A: Subir chromadb_data/ (Recomendado)

Si quieres que la app funcione inmediatamente en Streamlit Cloud:

```bash
# Remover chromadb_data/ del .gitignore temporalmente
# Edita .gitignore y comenta o elimina la línea: chromadb_data/

# Luego agrega la carpeta
git add chromadb_data/
git commit -m "Add pre-processed database"
```

⚠️ **Nota:** La carpeta puede ser grande (~20MB). GitHub permite archivos hasta 100MB.

### Opción B: NO subir chromadb_data/

Si prefieres mantener el repo pequeño:
- Los usuarios necesitarán ejecutar `ingest.py` localmente
- O configurar un proceso de ingesta en Streamlit Cloud

## Paso 4: Conectar con GitHub y Subir

```bash
# Agregar el repositorio remoto (reemplaza TU_USUARIO con tu usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/RAG_Interviews.git

# Cambiar a la rama main
git branch -M main

# Subir el código
git push -u origin main
```

Si GitHub te pide autenticación:
- Puede pedirte usuario y contraseña
- O puedes usar un Personal Access Token (más seguro)
- Para crear un token: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

## Paso 5: Verificar

Ve a tu repositorio en GitHub y verifica que todos los archivos estén ahí:
- ✅ app.py
- ✅ ingest.py
- ✅ requirements.txt
- ✅ README.md
- ✅ DEPLOY.md
- ✅ .gitignore
- ✅ chromadb_data/ (si decidiste subirlo)

## Siguiente Paso: Conectar con Streamlit Cloud

Una vez que el código esté en GitHub:
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con GitHub
3. Haz clic en "New app"
4. Selecciona tu repositorio `RAG_Interviews`
5. Configura:
   - **Main file path:** `app.py`
   - **Python version:** 3.9
6. En "Secrets", agrega:
   ```
   OPENAI_API_KEY=tu-api-key-aqui
   ```
7. Haz clic en "Deploy"

¡Listo! Tu app estará disponible en una URL como: `https://rag-interviews.streamlit.app`
