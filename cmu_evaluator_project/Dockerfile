FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir "pearmut==1.1.5"

WORKDIR /app

COPY scripts/pearmut_public_proxy.py scripts/pearmut_user_backup.py /app/scripts/
COPY scripts/annotator_backup.js scripts/start_railway_pearmut.sh /app/scripts/
COPY data/pearmut/external_smoke_01/campaigns/*.json /app/campaigns/
COPY data/pearmut/pool_01/campaigns/*.json /app/campaigns/

RUN chmod 0555 /app/scripts/start_railway_pearmut.sh \
    && chmod 0444 /app/scripts/pearmut_public_proxy.py \
      /app/scripts/pearmut_user_backup.py \
      /app/scripts/annotator_backup.js \
      /app/campaigns/*.json

CMD ["/app/scripts/start_railway_pearmut.sh"]
