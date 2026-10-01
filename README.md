# Portal Pericial AGRO · Sancor Seguros

App de tasadores / peritos (Portal Pericial) que reutiliza los **mismos datos** del MVP AGRO (`../AGRO/data`) sin modificar ese proyecto.

## Flujo

Denuncia ingresada → Análisis zona/cultivo → Asignación a tasador → Visita y muestreo → Cierre de tasación

En el muestreo podés aplicar:

1. **Método original** — promedio simple de muestras GPS  
2. **Método nuevo** — K-Means + promedio ponderado por área + ambientes reducidos por similitud  

## Puertos

- Backend: `http://127.0.0.1:8001`
- Frontend: `http://127.0.0.1:5174`

## Arranque

```bash
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload --reload-dir app

# Frontend
cd frontend
npm install
npm run dev
```

Las sesiones de este portal se guardan en `Portal-Pericial-AGRO/data/sessions` (aisladas del MVP AGRO).
