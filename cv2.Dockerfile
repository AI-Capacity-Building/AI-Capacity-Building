# Base our image on python version, contains only basic needed libraries
FROM python:3.9

# Install common libraries needed for ML development
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir numpy==1.23.2 pandas==1.4.4
RUN pip install --no-cache-dir matplotlib==3.5.3 scipy==1.9.1
RUN pip install --no-cache-dir scikit-learn==1.1.2 seaborn==0.11.2
RUN pip install opencv-python==4.8.1.78
RUN pip install --no-cache-dir parso==0.8.3
RUN pip install --no-cache-dir mlflow==1.28.0
RUN pip install --no-cache-dir ipykernel==6.15.3
RUN pip install --no-cache-dir gym==0.15.4 ipython==7.9.0

# To avoid xgboost import error
RUN apt-get update && apt-get install -y libgomp1 libgl1-mesa-glx libglib2.0-0 libxcb-xinerama0

# Commands to copy directories from local file system into the docker image
COPY . /home/cv
WORKDIR /home/cv

# Start a background operation to keep the container running.
# This enables students to attach vscode to the running container and start developing locally
ENTRYPOINT ["tail", "-f", "/dev/null"]
