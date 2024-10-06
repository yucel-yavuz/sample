import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import yaml

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentType, initialize_agent, create_react_agent
from langchain.tools import tool, Tool
from langchain_community.chat_models import ChatOllama
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from langchain.callbacks import StreamlitCallbackHandler

# Load credentials from YAML file
with open('/Users/yucelyavuz/07_GenAI/50_credentials/01_credentials.yml') as file:
    credentials = yaml.safe_load(file)
    OPENAI_API_KEY = credentials['openai']
    SERPER_API_KEY = credentials['serper']
    HUGGINGFACE_API_KEY = credentials['huggingface']

# Load sample data
df = pd.read_csv("/Users/yucelyavuz/07_GenAI/21_Data_Copilot/data/sample_data.csv", encoding='windows-1252')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Postal Code'] = df['Postal Code'].astype(str).str.zfill(5)

@tool
def plotChart(data: str) -> str:
    """
    Plots JSON data using a Plotly Figure and also displays a Plotly pie chart.

    Args:
        data (str): JSON string representing the figure.

    Returns:
        str: Confirmation message after plotting.
    """
    import json
    from plotly.io import from_json
    import plotly.graph_objects as go

    figure_dict = json.loads(data)
    fig = from_json(json.dumps(figure_dict))

    st.plotly_chart(fig)
    fig.show()

    pie_labels = figure_dict.get('pie_labels', [])
    pie_values = figure_dict.get('pie_values', [])

    if pie_labels and pie_values:
        pie_chart = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values)])
        st.plotly_chart(pie_chart)
        pie_chart.show()

    return "Charts plotted successfully."

extra_tools = [plotChart]

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
    api_key=OPENAI_API_KEY
)

pandas_agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    max_iterations=30,
    max_execution_time=45,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    extra_tools=extra_tools
)

llm_search = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.7,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
)

search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

search_tool = Tool(
    name="Intermediate Answer",
    func=search.run,
    description="Useful for when you need to answer questions using search",
)

google_search_agent = initialize_agent(
    tools=[search_tool],
    llm=llm_search,
    agent_type=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True,
)

# 1.0 Streamlit App

copilot_image = Image.open("/Users/yucelyavuz/07_GenAI/21_Data_Copilot/copilot.jpeg")

col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(copilot_image, width=50)

with col2:
    st.markdown("<h2 style='text-align:center; color:white; padding:0em;\
                border-radius:6px; '\
                >Sample AI Assistant </h2>", unsafe_allow_html=True)

with st.expander("🔎 Dataframe Preview"):
    st.write(df.head(3))


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input("🗣️ Chat with Dataframe"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Generating output..."):
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_agent.run(st.session_state.messages, callbacks=[st_cb])
            
            # answer_search = ""  # Initialize answer_search
            # # Check for "google search" keyword in the query
            # if "google" in prompt.lower():
            #     answer_search = google_search_agent.run({"input": prompt}, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

