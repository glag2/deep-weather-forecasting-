# Immagine CPU per addestramento e inferenza.
#
# La build usa `uv` con il lockfile: le versioni installate sono esattamente quelle
# risolte in locale, quindi l'immagine non puo' divergere dall'ambiente di sviluppo.
# L'indice `pytorch-cpu` dichiarato in pyproject.toml evita i wheel CUDA da circa
# 2,5 GB, inutili senza GPU.

FROM python:3.12-slim-bookworm

# `eccodes` e' una libreria C: cfgrib la carica a runtime e senza di essa la lettura
# dei GRIB fallisce con un errore poco leggibile all'import.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libeccodes0 \
        libeccodes-data \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Il download dei wheel di torch supera spesso il timeout predefinito.
    UV_HTTP_TIMEOUT=600

WORKDIR /app

# Le dipendenze si installano prima del codice: cambiare un sorgente non invalida
# la cache dello strato piu' costoso della build.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --extra notebooks

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
# `pyproject.toml` dichiara readme e licenza: senza questi file l'installazione del
# progetto fallisce nel backend di build, non a runtime, quindi l'errore arriva a
# immagine quasi completata.
COPY README.md LICENSE ./
RUN uv sync --locked --extra notebooks

# I dati stanno su un volume: sono decine di gigabyte e non appartengono all'immagine.
VOLUME ["/app/datasets"]

# Torch sceglie i thread in base alla macchina host; qui si lascia il default e si
# regola con la variabile d'ambiente quando serve limitarlo.
CMD ["python", "-c", "import dwf; print('dwf pronto')"]
