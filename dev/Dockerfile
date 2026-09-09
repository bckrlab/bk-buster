FROM debian:stable-slim

# install basic utilities and clean up afterward
RUN apt-get update --allow-releaseinfo-change && \
    apt-get -y install --no-install-recommends \
        procps wget curl ca-certificates unzip bzip2 git vim tmux texlive-full \
        build-essential graphviz && \
    apt-get -y autoremove && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install micromamba
ENV MAMBA_ROOT_PREFIX=/opt/mamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && \
    /bin/micromamba shell init -s bash --root-prefix ${MAMBA_ROOT_PREFIX}

# copy and install environment
COPY env.yml /tmp/env.yml
RUN micromamba create --yes -f /tmp/env.yml -n dev && \
    micromamba clean --all --yes
ENV PATH="${MAMBA_ROOT_PREFIX}/envs/dev/bin:$PATH"

# set workspace directory
WORKDIR /workspace

# expose jupyter and tensorboard
EXPOSE 8888 6006

# health check
HEALTHCHECK CMD [ "micromamba", "--version" ]