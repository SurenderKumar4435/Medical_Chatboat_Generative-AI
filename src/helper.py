from langchain_community.document_loaders import PyPDFLoader

file_path = "D:\Generative.AI\Medical_Chatboat_Generative-AI\Data\Medical_book.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()



from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

texts_chunk = text_splitter.split_documents(docs)


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")