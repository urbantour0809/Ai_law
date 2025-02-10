from search import get_relevant_docs
from answer import generate_answer

def main():
    while True:
        user_query = input("\n💬 법률 관련 질문을 입력하세요 (종료하려면 'exit' 입력): ").strip()
        
        if not user_query:
            print("⚠️ 질문을 입력해야 합니다!")
            continue

        if user_query.lower() == "exit":
            print("🔚 프로그램을 종료합니다.")
            break

        try:
            # ✅ ChromaDB에서 관련 법률 검색
            relevant_docs, sources, scores, law_numbers = get_relevant_docs(user_query)
            
            if not relevant_docs:
                print("📌 참고할 법률 데이터를 찾을 수 없습니다.")
                continue

            # ✅ EXAONE 모델을 이용한 답변 생성
            response = generate_answer(user_query, relevant_docs, sources, scores)
            
            # ✅ 답변 먼저 출력
            print(f"\n📝 Lawyer.ai 법률 답변:\n{response}")

            # ✅ 검색된 법률 정보 & 출처 출력 (답변 아래에 위치)
            print("\n📌 참고한 법률 출처 및 근거:")
            for i, (src, score) in enumerate(zip(sources, scores)):
                print(f"  {i+1}. {src} (유사도 점수: {score:.2f})")

            print("\n📌 관련 법률 및 판례:")
            for i, (law_num, doc) in enumerate(zip(law_numbers, relevant_docs)):
                print(f"  {i+1}. [{law_num}] {doc[:100]}...")  # ✅ 법률 번호 포함하여 출력

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
