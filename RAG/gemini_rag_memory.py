import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import nltk
nltk.download('punkt')
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables import chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages, filter_messages
from operator import itemgetter


class Gemini_RAG_Memory:
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
        chunk_size: int = 100,
        chunk_overlap: int = 70
    ):
        document_list = [self.documents[0].page_content]
        text_splitter = RecursiveCharacterTextSplitter(
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

    def _retriever(self, k: int = 8):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def _get_msg_content(self, msg):
        """メッセージの内容のみを抽出する関数"""
        return msg.content

    def save_text(self, path: str, chunk_size: int = 100, chunk_overlap: int = 70, k: int = 8):
        self._loader(path)
        self._text_splitter(self.documents, chunk_size, chunk_overlap)
        self._vector_store()
        self._retriever(k)

    def _preparation_prompt(self):
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key
        )
        contextualize_system_prompt = """
        チャット履歴と最新のユーザーの質問を与えられた場合、
        チャット履歴の文脈を参照する必要があるため、
        理解できるような質問に再構成してください。
        チャット履歴がなければ質問に答えないでください。
        ただ必要に応じて再構成し、そのまま返してください。
        """

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        self.contextualize_chain = (
            contextualize_prompt
            | llm
            | self._get_msg_content
        )

        qa_system_prompt = """
        あなたは優秀なAIアシスタントです。
        以下の取得された文脈を使用して質問に答えてください。
        もし答えがわからない場合は、その旨を伝えてください。
        3つの文を上限にして、簡潔に答えてください。
        \n\n
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        self.qa_chain = (
            qa_prompt
            | llm
            | self._get_msg_content
        )

# ここにデコレータを入れると，sekf, input_の二つが必要になり，エラーになる．
# よって，wrapper_for_chainでラップし，それにだけデコレータをつけるとよし．
    def _history_aware_qa(self, input_: dict):
        """
        実際の処理ロジック:
        RunnableWithMessageHistoryから渡される辞書を受け取り、
        チャット履歴を考慮した再構成を行ってからリトリーバ検索+QA実行
        """
        print("型:", type(input_))
        print("input_:", input_)
        print("-----------------")

        if input_.get('chat_history'):
            question = self.contextualize_chain.invoke(input_)
        else:
            question = input_["input"]

        context = self.retriever.invoke(question)
        print("context:", context)

        return self.qa_chain.invoke({
            **input_,
            "context": context,
        })

    def _preparation_run(self) -> None:
        chat_history_for_chain = InMemoryChatMessageHistory()

        # ─── ラッパを定義し、@chainを付けて「引数1つだけの関数」にする ───
        @chain
        def wrapper_for_chain(input_: dict):
            """ self._history_aware_qa を呼び出すためのラッパ関数 """
            return self._history_aware_qa(input_)
        # ────────────────────────────────────────────────

        self.qa_with_history = RunnableWithMessageHistory(
            wrapper_for_chain,
            lambda _: chat_history_for_chain,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def run(self):
        self._preparation_prompt()
        self._preparation_run()

    def ask(self, prompt: str, session_id: str) -> str:
        return self.qa_with_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}}
        )
