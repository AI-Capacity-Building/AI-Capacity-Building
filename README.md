# AI-Capacity-Building
The AI capacity building project is a project that aims to develop a sandbox for students to use when they are developing their machine learning projects.

# Use cases:
1.	Accident Severity Classification: a machine learning project that labels the accident severity from 1- 4 (inclusive) and provides analysis of the data. Data format is a csv file
2.	Smart Cities Sentiment Analysis: an NLP project in which Arabic tweets discussing smart cities are classified either positive, negative, or neutral. Data format is a tsv file.  
# Additional files:
1.	A hands-on example of how to use MLFlow with TensorFlow
2.	A hands-on example of how to use MLFLow with PyTorch

# Tech Stack:
The covered libraries are: NumPy, pandas, matplotlib, scipy, sklearn, seaborn, TensorFlow, PyTorch, MLFlow, xgboost, nltk.

# Prerequisites:
-	Installation of docker desktop 
-	Installation of VSCode
-	Installation of 2 extensions in VSCode: Docker, and Remote – Containers
![Docker extension](https://media.eos2git.cec.lab.emc.com/user/17974/files/0fc3c7aa-4c92-49a2-b658-d476c7a0dcf3)
![remote - containers extension](https://media.eos2git.cec.lab.emc.com/user/17974/files/20c48ff0-ed77-422e-a551-cf3ddba6628a)
-	Pull the docker image from the following link: 

# Running the image:
-	In your VSCode terminal, execute the following command: docker container run -p 5000:5000 ml_env:2.0

# Developing with VSCode 
-	To integrate docker with VSCode :
  1. In VSCode, go to the docker icon present in the left
  
  ![Icon](https://media.eos2git.cec.lab.emc.com/user/17974/files/2d0d9c0f-12ce-46cf-8525-f5f94b1ba7a8)
  
  2. Under containers -> choose the running container (ml_env:2.0) -> right click -> choose 'Attach Visual Studio Code'.
  3. A newly opened VSCode will appear, install Jupyter & Python extensions (press install in container)
  4. Press open folder
  5. You can either choose a usecase to run, or open root folder (folder containing all usecases) by just pressing ok
  6. You can now use VSCode to develop inside the container
  7. To run your code, open a terminal inside the example folder and type : python (python file name)

    Example : python mlflow_torch_example.py
  8. To open MLFlow user interface and view the runs output
     - Right click on the usecase folder
     - Choose open in integrated terminal 
     - Execute mlflow ui --host 0.0.0.0
     - In your browser open : http://0.0.0.0:5000  (mlflow uses port 5000 on the host)
     - If your experiment has a name (example : experiment_name = "MNIST_Classification"), it will create a folder with the same name in mlflow ui, open that folder to view the        runs




