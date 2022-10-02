# Base our image on slim python version, contains only basic needed libraries
FROM python:3.9.14-slim


# Install common libraries needed for ML development
# --no-cache-dir flag is used to disable saving pip install installation files (.whl) to save space and reduce the image size
RUN pip install --no-cache-dir numpy==1.23.2 pandas==1.4.4
RUN pip install --no-cache-dir matplotlib==3.5.3 scipy==1.9.1
RUN pip install --no-cache-dir scikit-learn==1.1.2 seaborn==0.11.2

# Install tensorflow
RUN pip install --no-cache-dir tensorflow==2.9.2

# Install MlFlow for tracking
RUN pip install --no-cache-dir mlflow==1.28.0

# Install xgboost (used in the US Accidents example notebook)
RUN pip install --no-cache-dir xgboost==0.90

# Install nltk for NLP projects
RUN pip install --no-cache-dir nltk==3.7

# Install ipykernel package to enable jupyter notebook development
RUN pip install --no-cache-dir ipykernel==6.15.3

# # Install libraries needed for the 3rd usecase
# RUN apt install --no-cache-dir graphviz==0.20.1
# RUN pip install --no-cache-dir pydot==1.0.28

# to avoid xgboost import error
RUN apt-get update
RUN apt-get install -y libgomp1

# Commands to copy directories from local file system into the docker image
COPY accidents_usecase ./accidents_usecase/
COPY mlflow_torch_example ./mlflow_torch_example/
COPY mlflow_tensorflow_example ./mlflow_tensorflow_example/
COPY smart_cities_usecase ./smart_cities_usecase/
COPY images_usecase ./images_usecase/

# Start a background operation to keep the container running.
# This enables students to attach vscode to the running container and start developing locally
ENTRYPOINT ["tail", "-f", "/dev/null"]

# To run the container and use mlflow monitoring
## Run container and expose port 5000 (used by mlflow ui): docker container run -p 5000:5000 ml_env:version
## After training a model and want to view tracked data using mlflow ui: mlflow ui --host 0.0.0.0 (after that visit localhost:5000 on your machine)