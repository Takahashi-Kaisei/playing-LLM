import os
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import nltk
# from langchain_experimental.text_splitter import SemanticChunker
nltk.download('punkt')
from dotenv import load_dotenv



class Gemini_RAG:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.model_name = os.getenv('MODEL')

    def _loader(self, path="data/rag.txt"):
        loader = TextLoader(path)
        self.documents = loader.load()

    def _text_splitter(
        self,
        document_list,
        chunk_size: int=100,
        chunk_overlap: int=70
    ):
        ### RecursiveCharacterTextSplitter ###
        document_list = [self.documents[0].page_content]
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents(document_list)
        self.list_ = []
        for text in texts:
            self.list_.append(text.page_content)

    def _vector_store(self):
        # retrieverの作成
        self.vectorstore = FAISS.from_texts(
            self.list_,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )

    def _retriever(self, k: int=8):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def _chain(self):
        # promptの作成
        template = """contextに従って回答してください:
        {context}

        質問: {question}
        """
        prompt = PromptTemplate.from_template(template)

        # modelの作成
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key
        )

        # chainを作成

        chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def save_text(self, path: str, chunk_size: int=100, chunk_overlap: int=70):
        self._loader(path)
        self._text_splitter(self.documents, chunk_size, chunk_overlap)
        self._vector_store()

    def run(self,prompt: str, k: int=8) -> str:
        self._retriever(k)
        return self._chain().invoke(prompt)
