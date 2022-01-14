# Base image
FROM python:3.7-slim
WORKDIR /root

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt-get install -y wget && \
apt clean && rm -rf /var/lib/apt/lists/*

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
# RUN mkdir /root/mnist

# we copy the folder structure
COPY requirements_gcp.txt requirements.txt
COPY setup.py setup.py
COPY key.json key.json
COPY src/ src/
COPY data/ data/
COPY reports/ reports/

# 
#WORKDIR /
WORKDIR /root
RUN pip install -r requirements.txt --no-cache-dir

# 
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]