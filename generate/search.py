import os
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np  # ✅ numpy 추가
import re

# ✅ ChromaDB 설정
CHROMA_DB_PATH = "../dataset/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
legal_collection = chroma_client.get_or_create_collection(name="legal_data")

# ✅ Fine-tuned `legal-bert-base` 모델 로드
MODEL_PATH = "../ft_legal_bert/checkpoint-1185"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ VRAM 최적화를 위해 `.half()` 사용
embedding_model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)

def embed_text(text):
    """ ✅ Fine-tuned 모델을 사용하여 문장을 벡터화 (GPU 활용) """

    # ✅ None 값 체크
    if text is None:
        raise ValueError("❌ 오류: `text` 값이 None 입니다. 올바른 문자열을 입력하세요.")

    # ✅ 문자열 변환 및 공백 정리
    if isinstance(text, list):
        text = " ".join(map(str, text))  # 리스트 → 문자열 변환

    if isinstance(text, (int, float, np.ndarray)):
        text = str(text)  # 숫자나 numpy 배열 → 문자열 변환

    if not isinstance(text, str):
        raise ValueError(f"❌ 오류: `text`의 타입이 문자열이 아닙니다. (현재 타입: {type(text)})")

    # ✅ 불필요한 공백 제거 및 특수문자 제거
    text = re.sub(r"\s+", " ", text.strip())  # 여러 개 공백 → 하나로 변환

    if not text:
        raise ValueError("❌ 오류: 빈 문자열이 입력되었습니다.")

    # ✅ 토크나이저 실행 전에 테스트
    try:
        _ = tokenizer.encode(text, add_special_tokens=True)
    except Exception as e:
        raise ValueError(f"❌ 토크나이저 실행 중 오류 발생: {e}")

    # ✅ 텍스트 임베딩 처리
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

    return embedding.cpu().numpy().flatten().tolist()  # ✅ 1차원 리스트 변환

def get_relevant_docs(query, top_k=5):
    """ ✅ 사용자의 질문을 벡터화하고, 관련 법률 데이터를 검색 """

    # ✅ 입력값 검증
    if not isinstance(query, str):
        raise ValueError("❌ 오류: 입력된 질문이 유효하지 않습니다. (문자열이 아닙니다.)")

    query = query.strip()  # 앞뒤 공백 제거
    query = re.sub(r"\s+", " ", query)  # 여러 개 공백 → 하나로 변환

    if not query:
        raise ValueError("❌ 오류: 빈 문자열이 입력되었습니다.")

    query_embedding = embed_text(query)

    # ✅ `query_embedding`이 올바른 리스트인지 확인
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()  # numpy 배열 → 리스트 변환

    if isinstance(query_embedding[0], list):  # 🔥 중첩 리스트 방지
        query_embedding = query_embedding[0]

    # ✅ ChromaDB에서 가장 관련성이 높은 법률 데이터 검색
    results = legal_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    relevant_texts = []
    sources = []
    scores = []
    law_numbers = []  # ✅ 법률 번호/판례 번호 추가 저장

    # ✅ 검색된 결과 가공
    for i, meta_list in enumerate(results.get("metadatas", [])):
        for meta in meta_list:
            text = meta.get("text", results["documents"][i] if results.get("documents") else None)
            score = results.get("distances", [])[i] if results.get("distances") else 0.0
            law_number = meta.get("law_number", meta.get("case_number", "법률 번호 없음"))  # ✅ 사건번호 or 법률 번호 추가

            if isinstance(text, list):
                text = "\n".join([t if t is not None else "" for t in text])

            if text and text.strip():
                relevant_texts.append(text)
                sources.append(meta.get("source", "출처 정보 없음"))
                scores.append(score)
                law_numbers.append(law_number)  # ✅ 법률 번호 리스트에 추가

    # ✅ 2차원 리스트(`list[list[float]]`)인 경우 1차원 리스트로 변환
    scores = [s for sublist in scores for s in (sublist if isinstance(sublist, list) else [sublist])]

    return relevant_texts, sources, scores, law_numbers
