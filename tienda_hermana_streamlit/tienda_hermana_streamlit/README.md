# Tienda Hermana — Streamlit

Herramienta para cargar Book/JUN, capturar primero las características de un punto nuevo, encontrar una tienda hermana abierta y después completar en paralelo Tipo, Transparencia y Accesibilidad antes de estimar la venta.

## Flujo en la app

1. Cargar Book. La aplicación procesa solo `ESTADO = ABIERTA`.
2. Completar todos los datos disponibles del nuevo punto. Los campos categóricos —departamento, municipio, zona, UPZ/comuna, estrato, tipo de local, parqueadero, 24H, generador, SEG26 y PTH— ofrecen los valores únicos presentes en las tiendas abiertas del Book. Los datos numéricos se ingresan como números. `TIE26` / Generación se excluye.
3. Revisar el ranking. La recomendación muestra nombre, CR/ID, similitud y cobertura. La lista Top 15 permite elegir otro espejo si hace falta.
4. Con la tienda espejo elegida, llenar en paralelo para el punto nuevo y el espejo los tres campos del formato histórico: Tipo, Transparencia y Accesibilidad.
5. Pulsar **Estimar venta**. El resultado se muestra destacado y debajo aparecen los puntos disponibles y el detalle de los comparables que forman la estimación.

## Modelo

- La similitud usa distancia tipo Gower agrupada por familias: Ubicación, Perfil de entorno, Infraestructura, Vivienda y empleo, y Tráfico externo. Las magnitudes de hogares, empleo y flujos usan `log1p` y diferencia acotada por el rango P5–P95; los datos faltantes reducen cobertura en lugar de imputarse.
- El campo `TE` se utiliza internamente como etiqueta histórica para ajustar los pesos por familia del ranking. No se muestra al usuario como variable de entrada ni como columna/validación. `TE` no se usa como predictor del punto nuevo.
- Las ventas se estiman con un promedio ponderado de los `k` comparables con venta disponible, incluyendo la tienda hermana seleccionada. El peso es inverso a la distancia: `1 / (distancia + 0,025)²`. La tabla detalla el CR, similitud, venta, peso relativo y aporte de cada comparable.
- Se puede estimar `VU6M` o `VENTAS OUM`; no se convierte ni se presume la unidad. P10–P90 son percentiles de las ventas de los comparables, no un intervalo de confianza.

## Puntajes del formato histórico

Después de elegir el espejo, la app puntúa Tipo, Transparencia y Accesibilidad con las mismas categorías/lookups exactos de `Base Datos` del formato suministrado. También puntúa Fachada y Área de piso de venta cuando los valores de Book coinciden exactamente con la tabla. Muestra un subtotal parcial de hasta 25 puntos y cuántas de las cinco variables tuvieron lookup.

Este subtotal no es el puntaje total del formato de 12 rubros. Transparencia y Accesibilidad no existen de forma consistente para todas las tiendas del Book; por esa razón se muestran en el resumen, pero no se usan como predictores de venta. Para incorporarlas al modelo de ventas se necesitarían esos atributos históricos para el universo de tiendas o una fórmula de ajuste aprobada.

## Ejecutar localmente

Requiere Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Pruebas unitarias:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Publicar con GitHub y Streamlit

1. Crea un repositorio de GitHub **privado** y sube `app.py`, `core.py`, `requirements.txt`, `requirements-dev.txt`, `README.md`, `.gitignore` y `.streamlit/config.toml`.
2. No subas Book, el formato histórico, otros Excel empresariales ni los CSV descargados. El archivo se carga desde la app.
3. Conecta el repositorio en Streamlit y define `app.py` como archivo de entrada.
4. Revisa quién puede acceder a la aplicación publicada y si el tratamiento de los datos empresariales es aceptable. Un repositorio privado no garantiza por sí solo que la aplicación web sea privada.
5. El Excel se conserva en la memoria de la sesión durante la interacción; la app no escribe el archivo al repositorio ni usa caché global.
