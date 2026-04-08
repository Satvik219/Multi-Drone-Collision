FROM ghcr.io/meta-pytorch/openenv-base:latest

RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/.cache/uv && chown -R user:user /home/user/.cache

USER user
ENV HOME=/home/user
ENV PATH="$HOME/.local/bin:$PATH"
WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        true; \
    fi

RUN if [ -f uv.lock ]; then \
        uv sync --no-editable; \
    else \
        uv sync --no-editable; \
    fi

ENV PATH="$HOME/app/.venv/bin:$PATH"
ENV PYTHONPATH="$HOME/app:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
