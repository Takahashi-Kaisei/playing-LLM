import os
import nltk
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables import chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages, filter_messages
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

# 必要なnltkデータをダウンロード
# nltk.download('punkt')


class Gemini_RAG_Trimmed_Memory:
    """
    Gemini_RAG_Memory は、
    ・ ドキュメントのロードと分割
    ・ ベクトルストアへの格納とリトリーバー作成
    ・ チャット履歴を考慮した質問の再構成と回答生成
    を担当するクラスです。

    Attributes:
        api_key (str): APIキー（.env から読み込み）
        model_name (str): モデル名（.env から読み込み）
        documents (list): 読み込んだドキュメント
        list_ (list): chunk split後のテキストのリスト
        vectorstore: FAISSなどのベクトルストアインスタンス
        retriever: ベクトルストアから生成したリトリーバ
        contextualize_chain: チャット履歴に基づく質問再構成チェーン
        qa_chain: 質問に対して最終回答を行うチェーン
        qa_with_history (RunnableWithMessageHistory): チャット履歴管理付きの Runnable
    """

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.model_name = os.getenv('MODEL')

        self.documents = None
        self.list_ = []
        self.vectorstore = None
        self.retriever = None
        self.contextualize_chain = None
        self.qa_chain = None
        self.qa_with_history = None


    def _loader(self, path: str = "data/rag.txt"):
        """
        指定したファイルパスからドキュメントを読み込む
        """
        loader = TextLoader(path)
        self.documents = loader.load()

    def _text_splitter(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 70
    ):
        """
        ドキュメントを指定したchunkサイズ・重なり量で分割し、テキストのリストを生成する
        """
        # 実際には document_list引数を使わず、このクラスの self.documents を参照している
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([self.documents[0].page_content])
        self.list_ = [text.page_content for text in texts]

    def _vector_store(self):
        """
        テキストをベクトル化してFAISSに格納する
        """
        self.vectorstore = FAISS.from_texts(
            self.list_,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )

    def _retriever(self, k: int = 8):
        """
        ベクトルストアからリトリーバーを作成する
        """
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def _get_msg_content(self, msg):
        """
        メッセージオブジェクトから実際のテキスト（content）だけを抽出する
        """
        return msg.content

    def save_text(self, path: str, chunk_size: int = 100, chunk_overlap: int = 70, k: int = 8):
        """
        指定パスのファイルをロード→分割→ベクトルストアへ格納→リトリーバー作成 までを行う
        """
        self._loader(path)
        self._text_splitter(chunk_size, chunk_overlap)
        self._vector_store()
        self._retriever(k)

    def _trimmer(self, input_messages):
        trimmer = trim_messages(
        max_tokens=3, # system message含め，5つのメッセージまで保持する．
        strategy="last", # ここがlastだと最後のメッセージからtrimしてくれる
        token_counter=len, #ここがlenだとメッセージ数でtrimしてくれる
        include_system=True,
        allow_partial=True,
        start_on="human",
        )

        print("トリマーに入力されたメッセージ", type(input_messages))
        print("トリマーに入力されたメッセージ", input_messages)
        self.chat_history_for_chain.messages = trimmer.invoke(input_messages)
        print("トリマーによってトリミングされた後のメッセージ", self.chat_history_for_chain.messages)
        print("トリマーによってトリミングされた後のメッセージ", type(self.chat_history_for_chain.messages))

    # def _print_trimmed(self, messages):
    #     print("Trimmed messages:")
    #     for msg in messages:
    #         print(f"{msg.type}: {msg.content}")
    #     self.chat_history_for_chain.messages = messages
    #     return messages

    def _preparation_prompt(self):
        """
        ユーザーの質問を再構成するContextualize Chainと、
        文脈を取り込んだ最終回答を生成するQA Chainを定義する
        """
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key
        )

        # 再構成用システムプロンプト
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

        # QA用システムプロンプト
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

        @chain
        def wrapper_for_trim(input_messages):
            return self._trimmer(input_messages)


        self.qa_chain = (
        RunnablePassthrough().assign(
            messages=itemgetter("chat_history")
        | wrapper_for_trim)
        | qa_prompt
        | llm
        | self._get_msg_content
    )

    def _history_aware_qa(self, input_: dict):
        """
        実際の質問応答処理:
        RunnableWithMessageHistoryから渡される辞書を受け取り、
        チャット履歴を考慮した再構成を行ってからリトリーバ検索 + QA実行
        """
        print("型:", type(input_))
        print("input_:", input_)
        print("-----------------")

        # チャット履歴が存在する場合、入力質問を再構成
        if input_.get("chat_history"):
            question = self.contextualize_chain.invoke(input_)
        else:
            question = input_["input"]

        # リトリーバを用いて文脈検索
        context = self.retriever.invoke(question)
        print("context:", context)

        # QAチェーンへ
        return self.qa_chain.invoke({
            **input_,
            "context": context,
        })

    def _preparation_run(self):
        """
        RunnableWithMessageHistoryを生成し、qa_with_history属性へ格納する
        """
        self.chat_history_for_chain = InMemoryChatMessageHistory()

        @chain
        def wrapper_for_chain(input_: dict):
            """
            _history_aware_qa を呼び出すためのラッパ関数
            （@chain では引数が1つの関数を想定するため）
            """
            return self._history_aware_qa(input_)

        self.qa_with_history = RunnableWithMessageHistory(
            wrapper_for_chain,
            lambda _: self.chat_history_for_chain,  # session_idによって分けたい場合にはここを変える
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def run(self):
        """
        QA実行に最低限必要なチェーンなどの準備を行う
        """
        self._preparation_prompt()
        self._preparation_run()

    def ask(self, prompt: str, session_id: str) -> str:
        """
        チャット履歴管理（RunnableWithMessageHistory）付きのQAを実行し、回答を返す
        """
        return self.qa_with_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}}
        )
