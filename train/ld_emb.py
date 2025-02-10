import os
import json
import chromadb
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ 데이터 경로 설정
JSON_PATH = os.path.abspath("../dataset/판례목록.json")

# ✅ GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ ChromaDB 설정
CHROMA_DB_PATH = "../dataset/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="legal_data")

# ✅ Fine-tuned `legal-bert-base` 모델 로드
MODEL_PATH = "../ft_legal_bert/checkpoint-1185"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
embedding_model = AutoModel.from_pretrained(MODEL_PATH).to(device)

# ✅ 불용어 리스트
STOPWORDS = set([
    "있다", "하는", "되어", "때문", "위해", "으로", "그리고", "따라서", "하지만", "대하여", 
    "같은", "그러한", "경우", "그런데", "뿐만", "이러한", "그것", "이것", "저것",
    "이다", "였다", "되지", "없이", "하는", "하며", "라고", "까지", "하면서", "등등",
    "및", "이다", "에서", "고", "가", "의", "를", "을", "에", "는", "도", "과", "와", "은"
])

def extract_keywords(text, top_n=5):
    """ ✅ TF-IDF를 활용하여 주요 키워드 추출 (불용어 제거 포함) """
    if not text.strip():  # ✅ 빈 문자열 체크
        return "키워드 없음"
    
    words = text.split()
    if len(words) < 3:  # ✅ 단어가 너무 적을 경우 TF-IDF 적용 불가
        return "키워드 없음"

    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=top_n)

    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        return ", ".join(feature_names) if feature_names.size > 0 else "키워드 없음"
    except ValueError:
        return "키워드 없음"

def embed_text(text):
    """ ✅ Fine-tuned 모델을 사용하여 문장을 벡터화 (GPU 활용) """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

# ✅ 실행
if __name__ == "__main__":
    print("📌 판례 목록 데이터 벡터화 시작...")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for data in data_list:
        case_number = data.get("사건번호", "사건번호 없음").strip()
        title = data.get("제목", "").strip()
        ruling = data.get("판시사항", "").strip()

        if not title and not ruling:  # ✅ 둘 다 비어 있으면 스킵
            print(f"⚠ 데이터 부족으로 건너뜀: {case_number}")
            continue

        text_data = f"{title} {ruling}".strip()

        keywords = extract_keywords(text_data)
        embedding = embed_text(text_data)

        # ✅ 중복 확인 후 데이터 추가
        existing_data = collection.get(ids=[case_number])
        if existing_data and existing_data["ids"]:
            print(f"🔄 기존 데이터 스킵: {case_number}")
            continue

        print(f"✅ 새로운 데이터 추가: {case_number}")
        collection.add(
            ids=[case_number],
            embeddings=[embedding],
            metadatas=[{"text": text_data, "keywords": keywords}]
        )

    print("✅ 모든 판례 목록 데이터 벡터화 완료!")
