FROM python:3.11.14-slim as builder

COPY vehicle_inspection/requirements/service.txt requirements.txt
COPY vehicle_inspection/docker/service/gunicorn_conf.py /gunicorn_conf.py

ENV PATH /venv/bin:$PATH

RUN set -x \
    && python -m venv /venv \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.11.14-slim

COPY --from=builder /venv /venv
COPY --from=builder /gunicorn_conf.py /gunicorn_conf.py

ENV PATH /venv/bin:$PATH

ENV PYTHONUNBUFFERED True
ENV PYTHONDONTWRITEBYTECODE=1

ENV MPLCONFIGDIR /tmp

ENV APP_HOME /app

EXPOSE 8080

WORKDIR $APP_HOME
COPY . $APP_HOME/

# Necessary for opencv library to address:
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN set -x \
    && apt-get update \
    && apt-get install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd -r gunicorn && useradd --no-log-init -r -g gunicorn gunicorn
USER gunicorn

ENTRYPOINT ["gunicorn", "-c", "/gunicorn_conf.py", "service:create_app()"]
