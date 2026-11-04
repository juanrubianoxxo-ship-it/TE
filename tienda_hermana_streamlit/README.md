# Tienda Hermana — Streamlit

Herramienta para comparar un punto nuevo con tiendas abiertas de Book/JUN, completar una ficha paralela con la tienda hermana y estimar la venta. La presentación utiliza una paleta rojo, amarillo y azul inspirada en OXXO, sin incorporar su logotipo.

## Uso

1. Carga Book manualmente o configura la fuente de OneDrive (ver abajo). La app lee la hoja `JUN` y filtra `ESTADO = ABIERTA`.
2. Escribe el nombre del punto nuevo y completa las variables disponibles. Los desplegables se alimentan de los valores únicos en Book; los datos numéricos se ingresan como números. `TIE26`/Generación no se utiliza.
3. Pulsa **Buscar tienda hermana**. La app muestra en columnas el nombre del punto nuevo y el de su recomendación, junto al CR/ID, similitud y cobertura. Si la cobertura solicitada no devuelve candidatas, muestra la mejor coincidencia disponible y avisa que su cobertura es menor.
4. Debajo de ambos nombres completa **Tipo, Transparencia y Accesibilidad** en paralelo para el punto nuevo y su espejo.
5. Pulsa **Estimar venta** para ver la cifra destacada, el subtotal parcial del formato histórico y los comparables que contribuyen a la estimación.

Puedes desplegar la lista de otras tiendas similares si necesitas cambiar la recomendación. El CR/ID acompaña cada tienda para distinguir nombres repetidos.

## Conectar Book desde OneDrive usando Streamlit Secrets

En Streamlit Cloud abre la app, ve a **Settings / App settings → Secrets** y pega una línea TOML como esta, sustituyendo el ejemplo por el enlace compartido de tu archivo Book:

```toml
BOOK_ONEDRIVE_URL = "https://1drv.ms/x/s!TU_ENLACE_COMPARTIDO"
```

También se incluye `.streamlit/secrets.toml.example` como referencia. **No pegues el enlace real en el código, README ni GitHub**: el enlace de OneDrive puede funcionar como una credencial. El archivo local `.streamlit/secrets.toml` está excluido por `.gitignore`.

La app añade `download=1` al enlace, descarga el `.xlsx` durante la sesión y ofrece el botón **Actualizar Book desde OneDrive** para volver a leerlo cuando cambie. También se puede cargar un Excel manual temporal desde la barra lateral.

El enlace debe permitir la descarga desde el servidor de Streamlit, normalmente con permiso de lectura “Anyone with the link”/“Cualquiera con el vínculo”, si la política de la organización lo autoriza. Si OneDrive exige inicio de sesión, el enlace por sí solo no basta; para ese escenario se requiere una integración Microsoft Graph con autenticación aprobada por la organización. La app no guarda ni muestra el enlace secreto.

## Modelo y límites

- La similitud usa una distancia tipo Gower agrupada por familias: Ubicación, Perfil de entorno, Infraestructura, Vivienda y empleo, y Tráfico externo. Las magnitudes se transforman/normalizan para limitar valores extremos; los faltantes reducen cobertura en vez de imputarse.
- `TE` se usa internamente para ajustar los pesos del ranking con tiendas espejo históricas; no se muestra como predictor o columna en la interfaz.
- La venta es un promedio ponderado de hasta `k` comparables que tienen la venta seleccionada (`VU6M` o `VENTAS OUM`), incluyendo el espejo elegido. El peso es inverso a distancia: `1 / (distancia + 0,025)²`. P10–P90 describe la dispersión observada entre comparables, no un intervalo de confianza.
- Después de elegir el espejo, Tipo, Transparencia y Accesibilidad se puntúan con los lookups exactos del formato histórico. Fachada y Área de piso de venta puntúan cuando su valor de Book coincide exactamente con la tabla; el subtotal es parcial, no el puntaje total de 12 rubros.
- Transparencia y Accesibilidad no están cargadas de forma consistente en JUN para todas las tiendas. Se muestran en el resumen, pero no ajustan numéricamente la venta hasta que esos datos históricos estén completos o se defina una fórmula aprobada.
- El Excel se mantiene en memoria de la sesión de Streamlit y no se escribe al repositorio ni a una caché global.

## Ejecutar localmente

Requiere Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Para usar OneDrive localmente, crea `.streamlit/secrets.toml` a partir del ejemplo y agrega el enlace compartido. No subas ese archivo a GitHub.

Pruebas unitarias:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Publicar con GitHub y Streamlit

1. Sube juntos `app.py`, `core.py`, `book_source.py`, `requirements.txt`, `requirements-dev.txt`, `README.md`, `.gitignore` y `.streamlit/secrets.toml.example`. No mezcles versiones de `app.py` y `core.py`.
2. No subas Book, otros Excel empresariales ni `.streamlit/secrets.toml`.
3. Configura como archivo principal `app.py` si el proyecto está en la raíz del repo; si conservas la carpeta del ZIP, selecciona `tienda_hermana_streamlit/app.py`. `core.py` y `book_source.py` deben quedar junto a `app.py`. Streamlit Cloud busca normalmente `requirements.txt` en la raíz del repo; si dejas el proyecto anidado, copia ese archivo también a la raíz.
4. Pega `BOOK_ONEDRIVE_URL` en Settings → Secrets de Streamlit Cloud. Un repositorio privado no garantiza que la app publicada también lo sea; comprueba el acceso permitido y la política de datos de tu organización.
