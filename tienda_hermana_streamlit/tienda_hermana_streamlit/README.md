# Tienda Hermana — comparables y ventas de referencia

App de Streamlit para cargar el libro Book, hoja **JUN**, encontrar tiendas abiertas parecidas a un punto propuesto y estimar una venta de referencia a partir de sus análogos.

## Alcance del MVP

- Filtra la base a `ESTADO = ABIERTA`; `CR` se conserva y se presenta como identificador del punto.
- Usa distancia de tipo **Gower agrupada por familias** para combinar categorías y magnitudes numéricas. Variables activas: ubicación, perfil/segmentación, infraestructura, viviendas/empleo y tráfico externo cada 15 minutos.
- El score es interpretable: 0–100%, con cobertura de variables disponibles y contribución/distancia por familia. Los faltantes se omiten y reducen cobertura; no se imputan.
- Para magnitudes de hogares, empleos y flujos aplica `log1p`; después limita la diferencia al rango robusto P5–P95 de las tiendas abiertas. Para categorías compara coincidencia normalizada (0/1).
- No incorpora como predictores `TE`, `CR`, `NAME`, `ESTADO`, ventas, tráfico de tienda, ticket, contribución, margen, renta, `%Batting`, `ME`, `MESOP` ni coordenadas. Así se evita usar el resultado comercial para seleccionar el comparable, y se evita calcular distancia geográfica sin confirmar sistema de coordenadas.
- La venta estimada es el promedio ponderado inversamente por distancia de los `k` vecinos con ventas disponibles. Muestra además mediana, P10 y P90 de las ventas observadas entre esos vecinos. P10–P90 es dispersión del grupo, **no** un intervalo de confianza.
- `TE` se reserva para un backtest retrospectivo de concordancia con referencias históricas que se enlazan de forma única a una tienda abierta. Referencias genéricas, ambiguas, autorreferencias y no enlazadas no se cuentan como aciertos.
- La app crea un CSV con dos filas (proyecto y tienda hermana). Tipo histórico, transparencia y accesibilidad se guardan en paralelo. Transparencia y accesibilidad del espejo deben capturarse manualmente porque no están en JUN; `TIPO DE LOCAL` de Book no se equipara automáticamente con el `Tipo` categórico del formato histórico.
- La ficha muestra un **subtotal histórico parcial** de Tipo, Fachada, Transparencia, Área de piso de venta y Accesibilidad, replicando los lookups exactos de las filas 2–7 de `Base Datos`. Solo puntúa coincidencias exactas; no es el puntaje total, no completa los demás rubros y no inventa umbrales para valores intermedios.
- El Excel cargado se conserva en memoria de la sesión de Streamlit para permitir la interacción; la app no lo guarda en el repositorio ni lo pone en una caché global.

## Limitaciones importantes

1. Los pesos iniciales son iguales por familia y ajustables por el usuario; no se presentan como pesos aprendidos. La validación con `TE` mide reproducción de decisiones previas, no causalidad ni calidad óptima de esas decisiones.
2. El estimador de ventas es un benchmark de análogos. La validación leave-one-out es transversal con la foto actual de las tiendas abiertas; no equivale a validación temporal de aperturas nuevas.
3. Las unidades de `VENTAS OUM` y `VU6M` no se convierten ni se presumen. La etiqueta de unidad del archivo se puede editar en la interfaz.
4. La instrucción de filtrar únicamente tiendas abiertas genera sesgo de supervivencia: los resultados describen el universo abierto, no todos los puntos que alguna vez abrieron.
5. La app no puede estimar por separado el efecto de transparencia o accesibilidad con JUN. Si más adelante se agregan valores consistentes para estas variables, se puede ampliar el modelo y revalidarlo.
6. El subtotal de la ficha histórica es deliberadamente parcial. Para completar el puntaje original se requieren las reglas/datos de los demás rubros y una confirmación de cómo redondear o categorizar Fachada y Área cuando no coinciden exactamente con la tabla.

## Ejecutar localmente

Requiere Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Pruebas unitarias: `pip install -r requirements-dev.txt && python -m pytest -q`.

En la app, sube el Excel de Book y completa las características conocidas del punto nuevo. Se requieren al menos dos variables en dos familias para calcular el ranking. Los campos desconocidos pueden dejarse vacíos; la cobertura del score quedará visible.

## Publicar mediante GitHub y Streamlit

1. Crea un repositorio de GitHub **privado** para el código y sube `app.py`, `core.py`, `requirements.txt`, `README.md` y `.gitignore`.
2. No subas `Book (2).xlsx`, la ficha histórica, otros Excel corporativos ni CSV descargados. El libro se carga desde la interfaz.
3. En Streamlit Community Cloud, conecta el repositorio, elige la rama y define `app.py` como archivo de entrada.
4. Antes de usar datos reales, comprueba quién puede acceder a la aplicación desplegada y el tratamiento aceptable de información corporativa. Un repositorio privado no garantiza por sí solo que la aplicación web sea privada.
5. Para datos sensibles, utiliza controles de acceso aprobados por la organización; no coloques usuarios, contraseñas ni credenciales en el repositorio. El proyecto no contiene autenticación.

## Validaciones disponibles en la aplicación

- **Backtest TE:** `Top 1`, `Top 3`, `Top 5`, rango mediano y MRR entre etiquetas históricas enlazables.
- **Validación de venta leave-one-out:** MAE, comparación con la mediana global del resto de tiendas, mejora porcentual, MAPE y mediana del error porcentual.

La validación usa los pesos y cobertura del análisis vigente. Es recomendable guardar los resultados y revisar los casos de mayor error antes de usar la estimación como insumo de una propuesta.
