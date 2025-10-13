FROM continuumio/miniconda3:latest
COPY env_cuda_latest.yaml .
RUN conda env create -f env_cuda_latest.yaml
ENV PATH /opt/conda/envs/pfllib/bin:$PATH
WORKDIR /app
COPY . .
WORKDIR /app/src/system 
CMD ["python", "server.py"]