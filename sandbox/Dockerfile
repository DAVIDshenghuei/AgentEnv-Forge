FROM python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

RUN groupadd --gid 10001 forge \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin forge \
    && mkdir /workspace \
    && chown 10001:10001 /workspace

USER 10001:10001
WORKDIR /workspace

CMD ["sleep", "infinity"]
