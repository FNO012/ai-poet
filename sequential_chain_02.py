import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 환경변수 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7

#llm 모델 초기화
llm = ChatOpenAI( api_key=OPENAI_API_KEY , model=MODEL_NAME , temperature=TEMPERATURE )
output_parser = StrOutputParser()

# 프롬프트 템플릿 정의
prompt_translation = PromptTemplate(
    input_variables=['review'],
    template="다음 숙박 시설 리뷰를 한글로 번역하세요.\n\n{review}"
)

prompt_summary = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한 문장으로 요약하세요.\n\n{translation}"
)

prompt_score = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 읽고 0점 부터 10점 사이에서 긍정/부정 점수를 매기세요. 숫자만 대답하세요.\n\n{translation}"
)

prompt_language = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰에 사용된 언어가 무엇인가요? 언어 이름만 답하세요.\n\n{review}"
)

prompt_reply_native = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰 요약에 대해 공손한 답볍을 작성하세요.\n답변 언어: {language}\n리뷰 요약: {summary}"
)

prompt_replay_korean = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한국어로 번역해 주세요.\리뷰 번역{reply_native}"
)

# 각 체인 구성
chain_translation = prompt_translation | llm | output_parser
chain_summary = prompt_summary | llm | output_parser
chain_score = prompt_score | llm | output_parser
chain_language = prompt_language | llm | output_parser
chain_reply_native = prompt_reply_native | llm | output_parser
chain_reply_korean = prompt_replay_korean | llm | output_parser

# RunnablePassthrough.assign 사용하여 각 단계의 출력을 다음 단계의 입력으로 전달하고,
# 중간 결과들을 딕셔너리에 누적 시킴.
combined_lcel_chain = (
    RunnablePassthrough.assign(
        translation=lambda x: chain_translation.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        summary=lambda x:chain_summary.invoke({"translation": x["translation"]})
    )
    | RunnablePassthrough.assign(
        score=lambda x: chain_score.invoke({"translation": x["translation"]})
    )
    | RunnablePassthrough.assign(
        language=lambda x: chain_language.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        reply_native=lambda x: chain_reply_native.invoke({"language": x["language"], "summary": x["summary"]})
    )
    | RunnablePassthrough.assign(
        reply_korean=lambda x: chain_reply_korean.invoke({"reply_native": x["reply_native"]})
    )
)

review_text = """
The hotel was clean and the staff were very helpful.
The location was convenient, close to many attractions.
However, the room was a bit small and the breakfast options were limited.
Overall, a decent stay but there is room for improvement.
"""

# 체인 실행 및 결과 출력
try:
    result = combined_lcel_chain.invoke(input={'review': review_text})

    print(f'translation 결과: {result.get("translation", "N/A")}')
    print(f'summary 결과: {result.get("'summary'", "N/A")}')
    print(f'score 결과: {result.get("score", "N/A")}')
    print(f'language 결과: {result.get("language", "N/A")}')
    print(f'reply_native 결과: {result.get("reply_native", "N/A")}')
    print(f'reply_korean 결과: {result.get("reply_korean", "N/A")}')
except Exception as e:
    print(f"Error: {e}")