"""레스토랑 리뷰 분석을 위한 LangChain 파이프라인 모듈"""

import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7

# LLM 모델 초기화
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL_NAME, temperature=TEMPERATURE)
output_parser = StrOutputParser()

# 프롬프트 템플릿 정의
summary_prompt = PromptTemplate.from_template(
    "다음 식당 리뷰를 한 문장으로 요약하세요.\n\n{review}"
)

sentiment_prompt = PromptTemplate.from_template(
    "다음 식당 리뷰를 읽고 0점부터 10점 사이에서 부정/긍정 점수를 매기세요. 숫자만 대답하세요.\n\n{review}"
)

# 응답 템플릿은 {summary}를 직접 입력받는 대신 입력 변수로 선언
reply_prompt = ChatPromptTemplate.from_template(
    "다음 식당 리뷰 요약에 대해 공손한 답변을 작성하세요.\n리뷰 요약: {summary}"
)

# 각 체인 구성
summary_chain = summary_prompt | llm | output_parser
sentiment_chain = sentiment_prompt | llm | output_parser
reply_chain = reply_prompt | llm | output_parser

# 요약 및 감정 분석 체인 (병렬 실행)
first_chain = RunnableParallel(
    {
        "summary": summary_chain,
        "sentiment_score": sentiment_chain,
        "review": RunnablePassthrough()
    }
)

# 완전히 새로운 접근법: 중간 단계에서 데이터 처리를 위한 함수
def prepare_for_reply(inputs):
    # 입력값에서 필요한 정보만 reply_chain으로 전달
    return {"summary": inputs["summary"]}

# 최종 결과를 조합하는 함수
def combine_results(inputs):
    # 수정된 구조에 맞게 접근 방식 변경
    first_result = inputs["first"]
    reply = inputs["second"]

    return {
        "summary": first_result["summary"],
        "sentiment_score": first_result["sentiment_score"],
        "review": first_result["review"],
        "reply": reply
    }

# 체인 재구성 - 더 명확한 단계별 접근
# 1. 요약 및 감정 분석 수행
# 2. 답변을 위한 데이터 준비
# 3. 답변 생성
# 4. 결과 조합
reply_generation = (
        first_chain |
        prepare_for_reply |
        reply_chain
)

# 최종 체인 구성 - 원본 데이터와 답변 결과를 조합
all_chain = RunnableParallel({"first": first_chain, "second": reply_generation}) | combine_results

# 리뷰 샘플
review = """
이 식당은 맛도 좋고 분위기도 좋았습니다. 가격 대비 만족도가 높아요.
하지만, 서비스 속도가 너무 느려서 조금 실망스러워습니다.
전반적으로는 다시 방문할 의사가 있습니다.
"""

try:
    # 체인 실행
    result = all_chain.invoke({"review": review})
    print(f"summary 결과 \n {result['summary']} \n")
    print(f"sentiment score 결과 \n {result['sentiment_score']} \n")
    print(f"reply 결과 \n {result['reply']} \n")
except Exception as e:
    print(f"Error: {e}")