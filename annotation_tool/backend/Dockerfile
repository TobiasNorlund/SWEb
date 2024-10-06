FROM python:3.11

ARG DOCKER_UID
ARG DOCKER_GID
RUN groupadd -r docker-user -g $DOCKER_GID && useradd -r -u $DOCKER_UID -g $DOCKER_GID -m -s /bin/false -g docker-user docker-user

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

USER $DOCKER_UID:$DOCKER_GID