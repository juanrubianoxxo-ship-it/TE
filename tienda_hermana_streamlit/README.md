# Tienda Hermana — Streamlit

Aplicación para comparar un punto nuevo con tiendas abiertas de Book/JUN, diligenciar en paralelo la ficha del punto y la tienda hermana y estimar la venta con la lógica de la ficha anterior. La interfaz usa rojo, amarillo y azul inspirados en OXXO, sin incorporar su logotipo.

## Flujo de uso

1. Carga Book manualmente o configura su enlace de OneDrive en Streamlit Secrets. La app lee `JUN` y conserva únicamente registros `ESTADO = ABIERTA`.
2. Diligencia el nombre y las variables disponibles del punto nuevo. Los desplegables categóricos muestran valores existentes en Book. `TIE26` no se usa como atributo; `TE` se reserva para entrenar internamente el ranking de tiendas espejo.
3. Pulsa **Buscar tienda hermana**. La aplicación muestra el Top 10 estadístico; puedes escoger otra opción de la lista. Se muestra el CR/ID para distinguir tiendas con nombres repetidos.
4. En las dos fichas interactivas, completa **Tipo, Transparencia y Accesibilidad**. También puedes ingresar Estudiantes a 100 m, Estudiantes en el entorno/300 m y Celdas de parqueo, tanto para el punto nuevo como para la hermana. Esos datos no están desglosados de esa manera en Book. Si se dejan vacíos, se toman como cero, como en las sumas del Excel.
5. Pulsa **Estimar venta**. La cifra principal aparece en miles de pesos, con los 12 componentes de puntaje, la venta real del espejo, su ticket y las transacciones estimadas.

## Cálculo de ventas

La estimación reproduce las fórmulas de la hoja `Tienda Espejo` del formato compartido:

- `D47 = SUM(D35:D46)`: puntaje total de la tienda hermana.
- `F47 = SUM(F35:F46)`: puntaje total del punto nuevo.
- `G47 = F47 / D47`: factor de ajuste.
- `E52 = C52 × G47`: transacciones estimadas para el punto nuevo.
- `E53 = C53`: se conserva el ticket de la hermana.
- `E54 = E53 × E52 / 1.000`: venta mensual estimada en miles de pesos.

Para los datos actuales de Book, `C52` usa `TRAFICO UM`, `C53` se deriva de `VENTAS OUM × 1.000 / TRAFICO UM`, y la venta real base es `VENTAS OUM`. Cuando falta el par mensual, la app usa el promedio mensual de `TRU6` y `VU6M`. En la aritmética, la proyección equivale a escalar la venta real mensual del espejo por `G47`.

Los puntajes incluyen Tipo, Fachada, Transparencia, Área de piso de venta, Accesibilidad, Celdas de parqueo, NSE, Horas de operación, Proximidad a 100 m, Tamaño de mercado, Tráfico externo y Capacidad de parqueo. La app calcula proximidad con `V100 + E100 + estudiantes_100m`, mercado total con `VT + ET + estudiantes_300m` y tráfico externo con `TR15MIN + TV15MIN + TM15MIN`. Fachada y piso de venta se califican por tramos con los cortes del formato (10/9/8/7/6 ml y 100/90/80/70/60 m²). Los puntajes de Tipo, Transparencia y Accesibilidad se muestran en ambas fichas y sí afectan el factor `G47`. El rubro de capacidad de parqueo se calcula como celdas del punto nuevo/celdas del espejo × 10; si el espejo tiene 0 celdas, el punto nuevo recibe 0 cuando también tiene 0 y 10 si tiene una o más. Este es el criterio usado para convertir el dato de celdas capturado en la app.

## Modelo de similitud

El ranking estadístico usa una distancia por familias sobre los atributos de Book/JUN, con normalización de variables numéricas y ajuste interno de pesos a partir de las parejas históricas de `TE`. Se filtran tiendas cerradas. `TE` sirve como objetivo histórico del ajuste, no como variable predictora. No se muestran métricas de validación en la interfaz.

La ficha histórica es una regla de transferencia por puntos, no una regresión supervisada. Por eso Tipo, Transparencia y Accesibilidad influyen directamente mediante sus puntajes en `F47/D47`; no se inventan coeficientes de venta para esos campos, que no existen de forma completa en Book para entrenarlos. El ranking estadístico aporta la selección de la tienda análoga y la ficha aporta el factor de venta reproducible.

## Conectar Book desde OneDrive

En Streamlit Cloud, abre **Settings / App settings → Secrets** y configura:

```toml
BOOK_ONEDRIVE_URL = "https://1drv.ms/x/s!TU_ENLACE_COMPARTIDO"
```

También se incluye `.streamlit/secrets.toml.example`. No pegues el enlace real en el código, README ni GitHub. El archivo local `.streamlit/secrets.toml` está excluido de `.gitignore`.

La app añade `download=1`, descarga el `.xlsx` durante la sesión y ofrece el botón **Actualizar Book desde OneDrive**. El enlace debe permitir descarga desde el servidor; si OneDrive exige login, se requiere autenticación aprobada por la organización (por ejemplo, Microsoft Graph). El enlace secreto no se muestra en pantalla.

## Ejecutar localmente

Requiere Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Para usar OneDrive localmente, crea `.streamlit/secrets.toml` a partir del ejemplo y agrega el enlace compartido. No subas ese archivo a GitHub. Ejecuta las pruebas con:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
