# Base our image on slim python version, contains only basic needed libraries
FROM python:3.9.14-slim

# Install common libraries needed for ML development
# --no-cache-dir flag is used to disable saving pip install installation files (.whl) to save space and reduce the image size
RUN pip install --no-cache-dir numpy==1.23.2 pandas==1.4.4
RUN pip install --no-cache-dir matplotlib==3.5.3 scipy==1.9.1
RUN pip install --no-cache-dir scikit-learn==1.1.2 seaborn==0.11.2

# Install tensorflow
RUN pip install --no-cache-dir tensorflow==2.9.2

# Install pytorch and other torch libraries we might need
RUN pip install --no-cache-dir torch==1.12.1
RUN pip install --no-cache-dir torchvision==0.13.1
RUN pip install --no-cache-dir torchaudio==0.12.1

# Install MlFlow for tracking
RUN pip install --no-cache-dir mlflow==1.28.0

# pytorch lightning is needed for mlflow to work with pytorch.
# It also provides some useful functionalities
RUN pip install --no-cache-dir pytorch-lightning==1.7.1

# Install xgboost (used in the US Accidents example notebook)
RUN pip install --no-cache-dir xgboost==0.90

# Commands to copy directories from local file system into the docker image
COPY accidents_usecase ./accidents_usecase/
COPY mlflow_torch_example ./mlflow_torch_example/
COPY mlflow_tensorflow_example ./mlflow_tensorflow_example/

# Start a background operation to keep the container running.
# This enables students to attach vscode to the running container and start developing locally
ENTRYPOINT ["tail", "-f", "/dev/null"]

# To run the container and use mlflow monitoring
## Run container and expose port 5000 (used by mlflow ui): docker container run -p 5000:5000 ml_env:version
## After training a model and want to view tracked data using mlflow ui: mlflow ui --host 0.0.0.0 (after that visit localhost:5000 on your machine)