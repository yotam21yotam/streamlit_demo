import streamlit as st
from transformers import pipeline

# Load the question-answering model
# qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


# Personal context about Yotam
context = """
I am Yotam, a Data Scientist and ML Engineer with a robust background in Statistics, Machine Learning engineering, 
and cybersecurity. I have built ML pipelines and led projects from proof of concept (POC) to 
production across various organizations, enhancing their ability to leverage data effectively. 
My diverse experience has equipped me with a unique blend of skills, making me a quick learner and 
a creative problem solver who brings insightful and innovative solutions to complex challenges.

Key Details:
- Age: I am 36 years old.
- Email: You can contact me at yotam21@gmail.com.
- Residence: I live in Tel Aviv.
- Current Position: I am currently working at the Prime Minister's Office as a Data Scientist 
  specializing in cybersecurity.
- Expertise: My expertise primarily lies in computer networks and operating systems.
- Hobbies: I enjoy extreme sports like motocross and hill climbs.
- Military Service: I served in the army as a fighter for 3 years.

Professional Experience:
From 2021 to Present:
- Position: Data Scientist & ML Engineer
- Organization: Prime Minister's Office, Tel Aviv
- Responsibilities: Conducted R&D for ML-driven Intrusion Detection Systems (IDS), with a deep 
  understanding of the OSI model and operating systems implemented with Python, C, Zeek, and Splunk. 
  Led research and development efforts on TLS fingerprinting tools and built an MLOps infrastructure 
  to support CI/CD/CT for a diverse range of projects, include data collection and preparation, model 
  development and training, ML service deployment, continuous feedback, and monitoring using MLflow, 
  Airflow, Docker, Kubernetes, S3, Jenkins, and Kafka.

From 2020 to 2021:
- Position: Data Scientist
- Organization: Nogamy, Tel Aviv
- Projects:
  BioRad - Developed and deployed ML-driven projects for predictive maintenance and fault prediction 
  on biological equipment, reducing equipment malfunctions by over 80% using Python and SQL.
  InTelos - Developed and deployed ML-driven projects for BI and fraud detection, leveraging network 
  data using Python and SQL.
  Golan Telecom - Conducted R&D on ETL processes to reduce time and space complexities, achieving over 
  a 20% reduction in running time using Python and SQL.

From 2019 to 2020:
- Position: Data Scientist
- Organization: eLoomina, Tel Aviv
- Achievements: Designed and developed statistical models predicting human behavior based on large 
  datasets using R, Python, and SQL. Successfully implemented ML models that increased the accuracy 
  of behavioral predictions. Engineered and refined data architectures that supported complex data 
  operations, significantly reducing latency.

Technical Skills:
Statistics, Deep Learning, Machine Learning, Signal Processing, System Design, Predictive Modeling, 
Spark, Splunk, Docker, Linux, GIT, Kubernetes, Grafana, MongoDB, Kafka, Jenkins, Apache, IDS, Snort, 
Zeek, Suricata, Wireshark, tshark, Nmap, TCP/IP, OSI model, AWS, Azure, Snowflake, Databricks, S3, 
Python, R, C, SQL, MLflow, Airflow, TensorFlow, Keras, Pandas, scikit-learn, Gensim, Matplotlib, 
Seaborn, NumPy, PyTorch, Dask, Scapy, PyCaret, Transformers, PySpark, Boto3, Joblib.
"""

st.title('Yotam\'s Chatbot')

st.write('I am a chatbot that can answer questions about Yotam! Please ask me anything you would like to know about him:')

user_input = st.text_input("Type your question here:")

if user_input:
    try:
        # Generate response using the model
        response = qa_pipeline({'question': user_input, 'context': context})
        answer = response['answer']
        st.text_area("Response", value=answer, height=150, max_chars=None, help="Response from the chatbot.")
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
