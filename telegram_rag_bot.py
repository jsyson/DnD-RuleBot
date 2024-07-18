import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes, CommandHandler
# pip install python-telegram-bot

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# # # # # 기본 파라미터 # # # # #
import auth

BOT_TOKEN = auth.BOT_TOKEN

# ADMIN
ADMIN_CHAT_ID = auth.ADMIN_CHAT_ID


# 로깅 설정.
logging.basicConfig(
    # format='%(asctime)s - %(message)s',
    level=logging.INFO,
    # filename='telegram_rag_bot.log'
)


# # # # # 함수들 # # # # #


# 기본 로깅 함수.
def log_info(chat_id, msg=''):
    log_msg = str(chat_id) + ' : ' + msg + '\n'
    logging.info(log_msg)
    # print(log_msg)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    vectorstore = FAISS.load_local('./faiss_db', embeddings=OpenAIEmbeddings(),
                                   allow_dangerous_deserialization=True)

    # 검색기 생성.
    retriever = vectorstore.as_retriever(k=10)

    # 기본 프롬프트 생성
    rag_default_prompt = ("You are an assistant for question-answering tasks. "
                          "Use the following pieces of retrieved context to answer the question. "
                          "If you don't know the answer, just say that '잘 모르겠어요.' "
                          "Use three sentences maximum and keep the answer concise."
                          "\n"
                          "Question: {question} "
                          "\n"
                          "Context: {context} "
                          "\n"
                          "Answer:")

    my_prompt = ChatPromptTemplate(input_variables=['context', 'question'],
                                   messages=[HumanMessagePromptTemplate(
                                       prompt=PromptTemplate(input_variables=['context', 'question'],
                                                             template=rag_default_prompt))])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # rag 체인을 생성합니다.
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | my_prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


# /start
async def welcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    log_info(chat_id, update.message.text)

    text = f'안녕하세요? RM 도우미 챗봇입니다. 도움말은 /help'
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


# 채팅이 들어오면.
async def process_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RAG

    chat_id = update.message.chat_id
    log_info(chat_id, update.message.text)

    user_chat = update.message.text
    CHAT_HISTORY.append({"role": "user", "content": user_chat})  # 히스토리 추가.

    answer = RAG.invoke(user_chat)
    CHAT_HISTORY.append({"role": "assistant", "content": answer})

    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


# /help
async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    log_info(chat_id, update.message.text)

    text = f'''RM 도우미 챗봇 (Ver 0.1) 사용 안내
 - 만든이: 정승용
이것은 도움말입니다.
ㅁㅁㅁ
ㅂㅂㅂ
'''

    await update.message.reply_text(text)


# # # # # MAIN 프로그램 시작 # # # # #


RAG = get_rag_chain()
logging.info('DB 로딩, RAG Chain 생성 완료.')
CHAT_HISTORY = []

application = ApplicationBuilder().token(BOT_TOKEN).build()

application.add_handler(CommandHandler('start', welcome))
application.add_handler(CommandHandler('help', show_help))
application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), process_chat))

application.run_polling()


