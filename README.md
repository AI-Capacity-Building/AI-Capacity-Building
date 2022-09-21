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

# Running the project:
-	In your VSCode terminal, execute the following command: docker container run -p 5000:5000 ml_env:2.0
-	To integrate docker with VSCode :
   1.	In vscode, go to remote containers extension
   2.	Under container you will find your running container. Right click, then choose attach visual studio code.
   3.	You are good to go, you can now use vscode to develop inside the container

-	To open the MLFlow user interface and view the runs output, execute the following command in the container terminal
mlflow ui --host 0.0.0.0
MLFlow uses port 5000 on the host



