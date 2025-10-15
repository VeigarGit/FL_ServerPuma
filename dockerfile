# Define the base image for the container
# Uses the latest version of Miniconda3, a minimal distribution of Anaconda
FROM continuumio/miniconda3:latest

# Copies the Conda environment configuration file to the container's current directory
# This YAML file contains all required dependencies and packages
COPY env_cuda_latest.yaml .

# Executes the command to create the Conda environment based on the YAML file
# The environment will be created with the name defined in the file (likely 'pfllib')
RUN conda env create -f env_cuda_latest.yaml

# Sets the PATH environment variable to include the binary of the created Conda environment
# This ensures commands will use the 'pfllib' environment by default
ENV PATH /opt/conda/envs/pfllib/bin:$PATH

# Sets the default working directory inside the container to /app
WORKDIR /app

# Copies all files from the current build context (local directory) to /app in the container
COPY . .

# Changes the working directory to the specific system folder within src
WORKDIR /app/src/system 

# Defines the default command that will execute when the container starts
# Starts the Python server by running the server.py file
CMD ["python", "server.py"]