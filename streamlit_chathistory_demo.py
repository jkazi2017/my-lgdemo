
#Required packages installed
import os
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#API Key, URL and Project_ID created as environment variables in windows
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

#All required Parameters
parameters = {
    GenTextParamsMetaNames.DECODING_METHOD: "sample",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
    GenTextParamsMetaNames.TEMPERATURE: 0.5,
    GenTextParamsMetaNames.TOP_K: 50,
    GenTextParamsMetaNames.TOP_P: 1,
}

#ChatWatsonx class to connect with IBM Granite LLM model to answer the queries for the questoins asked
llm=chat = ChatWatsonx(
    model_id="ibm/granite-20b-code-instruct",
    #model_id="ibm/granite-guardian-3-8b",
    url=WATSONX_URL,
    apikey=WATSONX_APIKEY,
    project_id=WATSONX_PROJECT_ID,
    params=parameters,
)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","You are a RPA Coach.Answer any questions"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
]
)

#Used Chain
chain = prompt_template | llm

#To maintain the ChatHistory
history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id : history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

st.title("RPA Guide")

#Input Text to type the question
input = st.text_input("Enter the question:")

if input:
    response = chain_with_history.invoke({"input":input},
                                         {"configurable":{"session_id":"abc123"}})
    st.write(response.content)

#Write chathistory for all the questions asked
st.write("HISTORY")
st.write(history_for_chain)