from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ EXAONE 모델 로드 설정
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ CUDA 오류 체크
print(f"✅ 현재 PyTorch CUDA 버전: {torch.version.cuda}")
print(f"✅ GPU 사용 가능 여부: {torch.cuda.is_available()}")
print(f"✅ 현재 사용 가능한 GPU 개수: {torch.cuda.device_count()}개")

if device == "cuda":
    try:
        torch.cuda.current_device()
        print("✅ CUDA 디바이스가 정상적으로 사용 가능합니다.")
    except Exception as e:
        print(f"❌ CUDA 디바이스 문제 발생: {e}")
        device = "cpu"  # 오류 발생 시 CPU로 강제 변경

# ✅ 모델 로드 (VRAM 절약을 위해 `torch_dtype=torch.float16` 사용)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
).to(device)

def generate_answer(query, relevant_docs, sources, scores):
    """ ✅ EXAONE 모델을 이용하여 법률 답변 생성 (GPU 활용) """

    # ✅ 검색된 법률 정보가 부족할 경우 기본 메시지 설정
    if not relevant_docs:
        relevant_docs = ["📌 참고할 법률 조항을 찾을 수 없습니다. 일반적인 법률 원칙을 적용하세요."]

    # ✅ `scores`가 리스트 안에 리스트(`list[list]`)인지 확인 후 변환
    if isinstance(scores[0], list):
        scores = [s for sublist in scores for s in sublist]

    # ✅ 프롬프트 최적화
    prompt = f"""
    [사용자 질문]
    당신은 대한민국 변호사입니다.
    그리고 모든 답변은 한국어로 해주세요.
    고객의 질문을 세밀하게 분석하여, 전문적인 답변을 사용자에게 제공해주세요.
    주로 아래에 관련 법률 및 판례를 위주로 답변을 생상하세요.
    {query}

    [관련 법률 및 판례]
    {relevant_docs}

    [변호사 답변]
    """

    # ✅ tokenizer 실행 전에 `query` 타입 체크
    if not isinstance(query, str) or not query.strip():
        raise ValueError("❌ 오류: `query` 값이 올바른 문자열이 아닙니다.")

    # ✅ 모델 입력 처리 (GPU로 이동)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ✅ 모델 실행 (답변 생성)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=4096,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            stopping_criteria=None,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True).split("[변호사 답변]")[-1].strip()

    return f""":{answer}"""
