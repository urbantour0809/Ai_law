import chromadb
import numpy as np

# ✅ ChromaDB 불러오기
CHROMA_DB_PATH = "../dataset/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name="legal_data")

# ✅ 샘플 데이터 가져오기 (최대 10개)
sample_data = collection.get(include=["embeddings", "metadatas", "documents"], limit=10)

# ✅ 필수 데이터 가져오기
embeddings = sample_data.get("embeddings", [])
metadata_list = sample_data.get("metadatas", [])
documents = sample_data.get("documents", [])

# ✅ 임베딩 데이터 차원 확인
if isinstance(embeddings, list) and len(embeddings) > 0:  # 🔹 리스트이고, 데이터가 있는지 확인
    emb_dim = len(embeddings[0])
    print(f"📌 현재 ChromaDB에 저장된 임베딩 차원: {emb_dim} (정상: 768)")

    # ✅ 모든 임베딩이 올바른 리스트인지 확인
    if not all(isinstance(emb, list) and len(emb) == emb_dim for emb in embeddings):
        print("❌ 오류: 일부 임베딩 데이터가 잘못된 형식입니다!")
else:
    print("⚠ ChromaDB에 저장된 임베딩 데이터가 없습니다.")

# ✅ 샘플 데이터 출력 (최대 5개)
if metadata_list:
    print("\n📌 저장된 임베딩 데이터 샘플 (최대 5개):")
    for i, metadata in enumerate(metadata_list[:5]):
        case_no = metadata.get("caseNo", "⚠ 사건번호 없음")
        text = metadata.get("text", "⚠ 텍스트 데이터 없음")
        source = metadata.get("source", "⚠ 출처 없음")
        keywords = metadata.get("keywords", "⚠ 키워드 없음")

        print(f"\n🔹 샘플 {i+1}:")
        print(f"📖 사건번호: {case_no}")
        print(f"📖 출처 정보: {source}")
        print(f"📖 임베딩된 텍스트 미리보기: {text[:300]}...")  # 🔹 300자까지만 출력
        print(f"🔑 주요 키워드: {keywords}")

        # ✅ None 값이 있는지 확인
        if None in [case_no, text, source, keywords]:
            print("⚠ 경고: 일부 필드에 None 값이 포함되어 있습니다!")

        # ✅ 빈 문자열 체크
        if not case_no.strip() or not text.strip():
            print("⚠ 경고: 사건번호 또는 텍스트 필드가 비어 있습니다!")
else:
    print("⚠ 저장된 메타데이터가 없습니다.")

# ✅ 전체 데이터 개수 확인
total_count = collection.count()
print(f"\n📌 현재 저장된 총 데이터 개수: {total_count}개")
