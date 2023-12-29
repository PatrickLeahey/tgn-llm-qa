# Databricks notebook source
# MAGIC %md
# MAGIC # Construct RAG QA Chatbot Chain

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 databricks-sdk==0.12.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chat_models import ChatDatabricks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

# COMMAND ----------

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
vector_search_endpoint_name = "tgn_vs_endpoint"
vector_search_index_name=f"leahey_sandbox.tgn_llm_qa.tgn_vs_index"

vsc = VectorSearchClient()
vs_index = vsc.get_index(
    endpoint_name=vector_search_endpoint_name,
    index_name=vector_search_index_name
)

# Create the retriever
retriever = DatabricksVectorSearch(
    vs_index, text_column="content", embedding=embedding_model
).as_retriever()

# COMMAND ----------

llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

# COMMAND ----------

TEMPLATE = """You are an assistant for listeners of The Grey Nato podcast. You are answering questions related to the podcast and its hosts: Jason Heaton and James Stacey. The podcast is about watches, diving, cars, gear, and other topics like these. If the question is not related to the podcast, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

question = {"query": "How did the hosts meet?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

question = {"query": "Do the podcast hosts like Scurfa watches?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

question = {"query": "Who is going to win the Super Bowl?"}
answer = chain.run(question)
print(answer)
