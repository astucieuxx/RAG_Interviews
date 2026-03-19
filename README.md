# 🔍 RAG Interviews - Análisis de Inteligencia Competitiva

Sistema RAG (Retrieval-Augmented Generation) para analizar entrevistas de inteligencia competitiva sobre proveedores de atención al cliente con IA.

## 📋 Requisitos Previos

1. **Python 3.8+** (ya tienes Python 3.14.0 ✅)
2. **API Key de OpenAI** - Necesitas una clave de API de OpenAI
3. **PDFs de entrevistas** - Archivos PDF con las transcripciones de entrevistas

## 🚀 Instalación y Configuración

### Paso 1: Instalar dependencias

```bash
pip3 install -r requirements.txt
```

O si prefieres usar un entorno virtual (recomendado):

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Paso 2: Configurar la API Key de OpenAI

Tienes dos opciones:

**Opción A: Variable de entorno (recomendado)**
```bash
export OPENAI_API_KEY="tu-api-key-aqui"
```

**Opción B: Crear archivo .env** (si prefieres no usar variables de entorno)
```bash
echo "OPENAI_API_KEY=tu-api-key-aqui" > .env
```

> **Nota:** Si usas `.env`, necesitarás instalar `python-dotenv` y modificar el código para cargarlo.

### Paso 3: Preparar los PDFs

1. Crea la carpeta `pdfs` en el directorio del proyecto:
```bash
mkdir pdfs
```

2. Coloca tus archivos PDF de entrevistas en la carpeta `pdfs/`

**Importante:** Los nombres de los archivos deben contener:
- **Vendor:** `decagon`, `sierra`, `intercom`, o `forethought`
- **Tipo de fuente:** `ex-cliente` o `ex-empleado` (o sus variantes en inglés)

**Ejemplos de nombres válidos:**
- `decagon_ex-cliente_entrevista1.pdf`
- `sierra_exempleado_juan.pdf`
- `intercom_customer_interview.pdf`
- `forethought_ex-cliente_maria.pdf`

### Paso 4: Ingestionar los PDFs

Ejecuta el script de ingesta para procesar los PDFs y crear la base de datos vectorial:

```bash
python3 ingest.py
```

Este script:
- Extrae texto de todos los PDFs en `pdfs/`
- Divide el texto en chunks
- Genera embeddings usando OpenAI
- Almacena todo en `vector_db/`

### Paso 5: Ejecutar la aplicación

Inicia la aplicación Streamlit:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador (normalmente en `http://localhost:8501`)

## 💡 Uso de la Aplicación

Una vez que la app esté corriendo:

1. **Filtros (sidebar):**
   - Selecciona los vendors que quieres incluir
   - Filtra por tipo de fuente (ex-cliente, ex-empleado)
   - Ajusta el número de chunks a recuperar

2. **Haz preguntas:**
   - "¿Cuáles son las quejas comunes sobre el onboarding de Decagon?"
   - "Compara Sierra vs Intercom desde la perspectiva del cliente"
   - "¿Qué dicen los ex-empleados sobre la precisión de IA de Forethought?"
   - "¿Cuáles son los buying factors más mencionados?"

3. **Ver fuentes:**
   - Cada respuesta incluye un expander con las fuentes utilizadas
   - Puedes ver qué vendor, tipo de fuente y archivo se usó

## 📁 Estructura del Proyecto

```
RAG_Interviews/
├── app.py              # Aplicación Streamlit (interfaz de chat)
├── ingest.py           # Script de ingesta de PDFs
├── requirements.txt    # Dependencias Python
├── pdfs/               # Carpeta con PDFs de entrevistas (crear manualmente)
└── vector_db/          # Base de datos vectorial (se crea automáticamente)
```

## 🌐 Compartir la Aplicación

Para compartir esta aplicación con tus compañeros, consulta el archivo **[DEPLOY.md](DEPLOY.md)** que incluye instrucciones detalladas para:

- **Streamlit Cloud** (recomendado - gratis y fácil)
- **Render** (alternativa gratuita)
- **ngrok** (para pruebas rápidas)

## 🔧 Solución de Problemas

### Error: "OPENAI_API_KEY not found"
- Verifica que hayas configurado la variable de entorno o el archivo .env
- Reinicia la terminal después de configurar la variable

### Error: "Database not found"
- Ejecuta `python3 ingest.py` primero para crear la base de datos

### Error: "No PDF files found"
- Asegúrate de que la carpeta `pdfs/` existe y contiene archivos PDF
- Verifica que los nombres de archivo contengan el vendor y tipo de fuente

### La app no se abre en el navegador
- Abre manualmente `http://localhost:8501` en tu navegador
- Verifica que el puerto 8501 no esté en uso

## 📝 Notas

- El modelo de chat usa `gpt-4o-mini` por defecto (más económico). Puedes cambiarlo a `gpt-4o` en `app.py` si necesitas más capacidad.
- Los embeddings usan `text-embedding-3-small` (1536 dimensiones)
- El sistema responde en el mismo idioma que uses (español o inglés)
