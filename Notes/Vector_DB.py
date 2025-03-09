# Using various vector databases.
# 1. Using ChromaDB : It is local vector database.

# Importing dependencies
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Setting up the api key
os.environ['OPEN_API_KEY'] = ""


# Loading the data
loader = DirectoryLoader('File_Path',
                glob = './*.txt',
                loader_cls = TextLoader)   # This glob parameter specifies which type of files should be accessed.

document = loader.load()

# Now the object document contains all the text present in our files but when converting the these into embeddings we need to take into account that the llm model can convert only a fixed amount of text into embeddings. So, we need to split this large corpus of data into small chunks of data.
# Chunk : It is the maximum number of characters our corpus can contain after splitting.
# Chunk Overlap : It is the number of characters that should overlap between two adjacent chunks.

# Splitting the large corpus into chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
text = text_splitter.split_documents(documents = document)
print("The no. of chunks it has formed from the given amount of text :: ", len(text))

# Creating DB object
persist_directory = 'db'

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents = text,
                                 embedding = embedding,
                                 persist_directory = persist_directory)

# Saving the converted embeddings into disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal embeddigns
vectordb = Chroma(persist_directory = persist_directory,
                  embedding_function = embedding)

# Making a retriever, to retrieve answers regarding the documents provided.
retriever = vectordb.as_retriever(search_kwargs = {'k' : 2})   # search_kwargs parameter tells that how many answers should be generated for a given query.
docs = retriever.get_relevant_documents(" Enter your query regarding the ")   # This will fetch 2 chunks of data similar to the query asked. The model is using cosine similarity for generating answers.
print(f"The no. of answers generated for the given query :: {len(docs)}")

# Creating a chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm = OpenAI(),
                                       chain_type = 'stuff',
                                       retriever = retriever,
                                       return_source_documents = True)

query = 'Your query regarding the document.'
llm_response = qa_chain(query)   # Stores the answer related to the query.

# Deleting the db
vectordb.delete_collection()
vectordb.persist()