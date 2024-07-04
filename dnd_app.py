import os
import streamlit as st
# from lanchain.llms import OpenAI
from openai import OpenAI
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm


# 코사인 유사도 계산 함수
def cos_sim(a, b):
    return dot(a, b)/(norm(a) * norm(b))


# 유저 질문과 비슷한 컨텍스트를 모아주는 함수
def create_context(question, embedding_df, max_len=5000):  # , size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

    # Get the similarity from the embeddings
    if 'similarity' in embedding_df:
        del embedding_df['similarity']
    embedding_df['similarity'] = embedding_df.apply(lambda x: cos_sim(q_embeddings, x['embeddings']), axis=1)

    returns = []
    cur_len = 0

    # Sort by similarity and add the text to the context until the context is too long
    for i, row in embedding_df.sort_values('similarity', ascending=False).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


# 답변 함수
def answer_question(
        embedding_df=None,  # default parameters
        model="gpt-3.5-turbo",
        question=None,
        max_len=3000,
        # size="ada",
        debug=True,
        max_tokens=500,
        stop_sequence=None,
        container=None,
):

    if question is None or container is None:
        return

    # Answer a question based on the most similar context from the dataframe texts
    if embedding_df is not None:
        context = create_context(
            question,
            embedding_df,
            max_len=max_len,
            # size=size,
        )
    else:
        context = '없음'

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print()

    try:
        temp_msg = st.session_state.messages.copy()
        temp_msg.append({"role": "user",
                         "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"})

        with container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in client.chat.completions.create(
                    messages=[
                        {"role": m["role"], "content": m["content"]} for m in temp_msg
                    ],
                    model=model,
                    # messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=stop_sequence,
                    stream=True,
                ):
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        if debug:
            print(response.choices)
            print('\n')

    except Exception as e:
        print(e)


def get_openai_client():
    global client

    if client is None:
        if openai_api_key.startswith('sk-'):
            # st.warning('Please enter your OpenAI API key!', icon='⚠')
            client = OpenAI(api_key=openai_api_key)
        else:
            client = OpenAI()

    return client


def add_chat(role, msg):
    # st.session_state.messages.append({"role": role, "content": msg})
    with st.chat_message(role):
        st.markdown(msg)


st.set_page_config(layout="wide")
st.title('D&D 룰 도우미')


# os.getenv('OPENAI_API_KEY')
openai_api_key = st.sidebar.text_input('OpenAI API Key', value='Input your ChatGPT API key.')

client = None  # OpenAI()
df = pd.read_csv('scraped3000.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)  # 문자열로 읽혔을 경우 apply(eval) 필요

# st.write('# 룰북 벡터 DF')
# df

# message key 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 목록을 출력해줄 컨테이너 생성
text_container = st.container(border=True)
# text_container.title('AI와의 오붓한 채팅방')

for message in st.session_state.messages:
    with text_container:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# st.divider()

prompt = st.chat_input("Say something")

if prompt:
    with text_container:
        with st.chat_message('user'):
            st.markdown(prompt)

    # response
    client = get_openai_client()  # OpenAI()
    answer_question(embedding_df=df, model='gpt-4o', question=prompt,
                    max_len=5000, max_tokens=2000, debug=True, container=text_container)


