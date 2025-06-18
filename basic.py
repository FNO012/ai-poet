import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

OEPNAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OEPNAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key = OEPNAI_API_KEY
)

question = "화창한 날은 뭐하면 좋을까?"

messages = [
    SystemMessage( content="너는 심리 상담가 입니다."),
    HumanMessage( content=question )
]

try:
    response = chat.invoke( messages)

    print( response)
except Exception as e:
    print("Error: {e}")