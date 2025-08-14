FROM --platform=linux/amd64 pytorch/pytorch
# Use an appropriate base image for your algorithm.
# As an example, we use the official pytorch image.

# In reality this baseline algorithm only needs numpy so we could use a smaller image:
#FROM --platform=linux/amd64 python:3.11-slim


# Ensure that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# 安装 git
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
    
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# Install requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user XMemModel.py /opt/app/
COPY --chown=user:user train.py /opt/app/
COPY --chown=user:user test.py /opt/app/
COPY --chown=user:user eval.py /opt/app/
COPY --chown=user:user interactive_demo.py /opt/app/
COPY --chown=user:user merge_multi_scale.py /opt/app/
COPY --chown=user:user util /opt/app/util
COPY --chown=user:user dataset /opt/app/dataset
COPY --chown=user:user model /opt/app/model
COPY --chown=user:user scripts /opt/app/scripts
COPY --chown=user:user save /opt/app/save
COPY --chown=user:user inference /opt/app/inference
COPY --chown=user:user docs /opt/app/docs


# Add any other files that are needed for your algorithm
# COPY --chown=user:user <source> <destination>
COPY --chown=user:user *.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
