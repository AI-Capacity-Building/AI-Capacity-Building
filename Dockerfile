FROM python:3.9

ADD docker-test/ann_model.py .
ADD docker-test/one_hot_encoded_dataset.csv .

RUN pip install numpy==1.23.2 pandas==1.4.4
RUN pip install matplotlib==3.5.3 scipy==1.9.1
RUN pip install scikit-learn==1.1.2 seaborn==0.11.2 
RUN pip install tensorflow==2.9.2 torch==1.12.1
RUN pip install mlflow==1.28.0
CMD ["python","./ann_model.py"]
