# tgn-llm-qa

### Overview
tgn-llm-qa is a Retrieval Augmented Generation (RAG) application developed specifically for fans and listeners of 'The Grey Nato' podcast. This project demonstrates how advanced language model chains can be used to create an interactive question-answering system. Listeners can ask questions about the podcast's content, and the system will provide responses based on the information extracted from the show's transcripts.

### Key Features
Podcast Transcript Processing: Utilizes notebooks to download, chunk, and embed transcripts from 'The Grey Nato' podcast.
Embedding and Storage: Leverages Databricks Foundations Models to create embeddings, which are then stored as a Delta table.
Embedding Retrieval Endpoint: Implements Databricks Vector Store to create an efficient retrieval system for accessing the stored embeddings.
RAG LangChain Application: Develops a LangChain application for the Retrieval Augmented Generation model, facilitating accurate and contextually relevant answers.


### Steps
##### Transcript Processing Notebook:
- Downloads podcast transcripts.
- Chunks the transcripts for embedding.
- Embeds the chunks using Databricks Foundations Models.
- Stores embeddings in a Delta table.
- Uses Databricks Vector Search to index the embeddings.
- Creates an endpoint for querying and accessing these embeddings.

##### RAG LangChain Notebook:
- Sets up the LangChain application for Retrieval Augmented Generation.
- Integrates the embeddings retrieval system for enhanced question answering.

Currently, this project is in a demonstration phase and not fully deployed. Upon completion, users will be able to query the system with questions related to 'The Grey Nato' podcast, receiving answers generated based on the podcast's transcripts.
