# AI Capacity Building Project

The AI Capacity Building project aims to provide a sandbox environment for students to develop their machine learning projects. The project includes various use cases for hands-on learning.

## Use Cases

1. **Accident Severity Classification:**
   - A machine learning project that labels accident severity from 1 to 4 and provides data analysis.
   - Data format: CSV file.

2. **Smart Cities Sentiment Analysis:**
   - An NLP project classifying Arabic tweets discussing smart cities into positive, negative, or neutral.
   - Data format: TSV file.

3. **Reinforcement Learning:**
   - Simulation of a self-driving cab using RL techniques in an OpenAI Gym environment.

4. **Object Detection:**
   - A computer vision project detecting objects in images and videos using YOLO and OpenCV.

## Additional Files

1.	A hands-on example of how to use MLFlow with TensorFlow
2.	A hands-on example of how to use MLFLow with PyTorch

## Tech Stack

Libraries covered:
NumPy, pandas, matplotlib, scipy, scikit-learn, seaborn, TensorFlow, PyTorch, MLFlow, xgboost, nltk, OpenCV, Gym.

## Prerequisites

- Install Docker Desktop from [Docker](https://www.docker.com/products/docker-desktop/).

- Choose the relevant Dockerfile:
  - `cv2.Dockerfile` for object detection.
  - `rl.Dockerfile` for reinforcement learning.
  - `torch.Dockerfile` for PyTorch environments.
  - `tensorflow.Dockerfile` for TensorFlow environments.

- Execute the build command for the docker image:

  ```bash
  docker build -f <Relevant_Dockerfile> -t ml_env:2.0 .
  ```
- Running the container by executing the following command: 

  ```bash
  docker container run -p 5000:5000 ml_env:2.0
  ```


# Running the use cases     
## 1- Integrating with VSCode (Optional)
  1. Install VSCode from : https://code.visualstudio.com/Download
  2. Installation of 2 extensions in VSCode: Docker, and Remote – Containers
  
  ![Docker extension](Documentation/Pictures%20used%20in%20documentation/Fig4%20-%20Docker%20extension.png)
  ![remote - containers extension](Documentation/Pictures%20used%20in%20documentation/Fig5%20-%20Dev%20Containers%20extension.png)
   3. If you haven't run the image yet, in your vscode terminal execute: docker container run -p 5000:5000 ml_env:2.0
  
 4. In VSCode, go to the docker icon present in the left
  
  ![Icon](Documentation/Pictures%20used%20in%20documentation/Fig6%20-%20Docker%20Icon.png)
  
  5. Under containers -> choose the running container (ml_env:2.0) -> right click -> choose 'Attach Visual Studio Code'.
  6. A newly opened VSCode will appear, install Jupyter & Python extensions (press install in container)
  7. Press open folder
  8. You can either choose a usecase to run, or open root folder (folder containing all usecases) by just pressing ok
  9. You can now use VSCode to develop inside the container
  10. To run your code, open a terminal inside the example folder and type : python (python file name)

  Example : 
  
  ```bash
  python3 mlflow_torch_example.py
  ``` 

## 2- Using MLFlow 
  To open MLFlow user interface and view the runs output (make sure your code uses mlflow and logs the parameters you want to see - US accidents doesn't use mlflow)
  1. Open a terminal in your project
  2. Execute:  

  ```bash
  mlflow ui --host 0.0.0.0
  ``` 

  3. In your browser open : http://0.0.0.0:5000  (mlflow uses port 5000 on the host)
  4. If your experiment has a name (example : experiment_name = "MNIST_Classification"), it will create a folder with the same name in mlflow ui, open that folder to view the runs