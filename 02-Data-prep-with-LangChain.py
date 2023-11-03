# Databricks notebook source
# MAGIC %md
# MAGIC # Document Ingestion and Preparation
# MAGIC
# MAGIC 1. Download and organize episode transcripts into a directory on DBFS
# MAGIC 2. Use LangChain to ingest those documents and split them into manageable chunks using a text splitter
# MAGIC 3. Use a sentence transformer NLP model to create embeddings of those text chunks and store them in a vectorstore
# MAGIC     * Embeddings are basically creating a high-dimension vector encoding the semantic meaning of a chunk of text

# COMMAND ----------

# MAGIC %run ./util/install-prep-libraries

# COMMAND ----------

import os
import urllib

# COMMAND ----------

# where you want the transcripts to be saved
dbutils.widgets.text("transcripts_path", "/dbfs/tmp/tgn_llm_qa/transcripts")

# which embeddings model from Hugging Face 🤗  you would like to use
dbutils.widgets.text("embeddings_model", "sentence-transformers/multi-qa-distilbert-cos-v1")

# where you want the vectorstore to be persisted across sessions, so that you don't have to regenerate
dbutils.widgets.text("vectorstore_path", "/dbfs/tmp/langchain_hls/db")

# where you want the Hugging Face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

#get widget values
transcripts_path = dbutils.widgets.get("transcripts_path")
embeddings_model = dbutils.widgets.get("embeddings_model")
vectorstore_path = dbutils.widgets.get("vectorstore_path")

#set cache path
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download documents
# MAGIC
# MAGIC Download transcripts from https://www.phfactor.net/tgn and save each in DBFS

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/" + transcripts_path.lstrip("/dbfs"))

# COMMAND ----------

for i in range(1,261):
  script = urllib.request.urlopen(f"https://www.phfactor.net/tgn/{i}.0/episode.txt").read().decode("utf-8")
  with open(f"{transcripts_path}/episode_{i}", "w") as f:
    f.write(script.replace("\'", "'").replace('"',"").replace("\n",""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Document DB
# MAGIC
# MAGIC Now it's time to load the texts that have been generated, and create a searchable database of text for use in the `langchain` pipeline. 
# MAGIC These documents are embedded, so that later queries can be embedded too, and matched to relevant text chunks by embedding.
# MAGIC
# MAGIC - Use `langchain` to reading directly from PDFs, although LangChain also supports txt, HTML, Word docs, GDrive, PDFs, etc.
# MAGIC - Create a simple in-memory Chroma vector DB for storage
# MAGIC - Instantiate an embedding function from `sentence-transformers`
# MAGIC - Populate the database and save it

# COMMAND ----------

# MAGIC %md
# MAGIC Prepare a directory to store the document database. Any path on `/dbfs` will do.

# COMMAND ----------

!(rm -r {vectorstore_path} || true) && mkdir -p {vectorstore_path}

# COMMAND ----------

# MAGIC %md
# MAGIC Create the document database:
# MAGIC - Here we are using the `DirectoryLoader` loader from LangChain ([docs page](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html)) to form `documents`

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader(transcripts_path, show_progress=True)
docs = loader.load()

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are using a text splitter from LangChain to split our transcripts into manageable chunks. This is for a few reasons, primarily:
# MAGIC - LLMs (currently) have a limited context length. MPT-7b-Instruct by default can only accept 2048 tokens (roughly words) in the prompt, although it can accept 4096 with a small settings change. This is rapidly changing, though, so keep an eye on it.
# MAGIC - When we create embeddings for these documents, an NLP model (sentence transformer) creates a numerical representation (a high-dimensional vector) of that chunk of text that captures the semantic meaning of what is being embedded. If we were to embed large documents, the NLP model would need to capture the meaning of the entire document in one vector; by splitting the document, we can capture the meaning of chunks throughout that document and retrieve only what is most relevant.
# MAGIC - In this case, the embeddings model we use can except a very limited number of tokens. The default one we have selected in this notebook, [
# MAGIC S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO), has also been finetuned on a PubMed dataset, so it is particularly good at generating embeddings for medical documents.
# MAGIC - More info on embeddings: [Hugging Face: Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)

# COMMAND ----------

# For documents containing long bits of text we need to split them for embedding:
from langchain.text_splitter import TokenTextSplitter

# this is splitting into chunks based on a fixed number of tokens
# the embeddings model we use below can take a maximum of 512 tokens (and truncates beyond that)
# further, the model specs state that the model was trained on 250 token samples, so we use that
text_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=64)
documents = text_splitter.split_documents(docs)

# COMMAND ----------

display(documents)

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

hf_embed = HuggingFaceEmbeddings(model_name=embeddings_model)
sample_query = "What watch brands do Jason and James like the most?"
db = Chroma.from_documents(collection_name="tgn_docs", documents=documents, embedding=hf_embed, persist_directory=vectorstore_path)
db.persist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a `langchain` Chain
# MAGIC
# MAGIC Now we can compose the database with a language model and prompting strategy to make a `langchain` chain that answers questions.
# MAGIC
# MAGIC - Load the Chroma DB
# MAGIC - Instantiate an LLM, like Dolly here, but could be other models or even OpenAI models
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt

# COMMAND ----------

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

hf_embed = HuggingFaceEmbeddings(model_name=embeddings_model)
db = Chroma(collection_name="tgn_docs", embedding_function=hf_embed, persist_directory=vectorstore_path)
