"""
영어 AI 텍스트 판별 모델 — 추론 코드
모델: DeBERTa-v3-large + LoRA
사용법: 아래 예시 참고

필수 라이브러리:
    pip install torch transformers peft

필수 파일 (같은 폴더에 배치):
    adapter_model.safetensors
    adapter_config.json
    tokenizer.json
    tokenizer_config.json
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


class TextDetector:
    def __init__(self, model_dir, device=None):
        """
        Args:
            model_dir: 모델 파일 4개가 있는 폴더 경로
            device: 'cuda' 또는 'cpu' (None이면 자동 감지)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # base 모델 + LoRA 어댑터 로드
        base = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large", num_labels=2, local_files_only=True
        )
        self.model = PeftModel.from_pretrained(base, model_dir)
        self.model.to(self.device)
        self.model.eval()

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-large",
            local_files_only=True,
            use_fast=True,
        )

    def predict(self, text):
        """
        텍스트를 분석하여 AI 생성 여부를 판별합니다.

        Args:
            text: 판별할 영어 텍스트 (str)

        Returns:
            dict: {
                "label": "AI 생성" 또는 "사람 작성",
                "ai_probability": 87.3,  (0~100%)
                "reason": "판단 근거 설명"
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        ai_prob = probs[1].item() * 100

        # 확률 구간별 판단 근거
        if ai_prob >= 90:
            reason = "문장 구조와 어휘 선택이 LLM의 전형적인 생성 패턴과 매우 유사합니다."
        elif ai_prob >= 70:
            reason = "일부 문장에서 AI 생성 텍스트의 특징이 관찰되나, 인간 작성 가능성도 있습니다."
        elif ai_prob >= 50:
            reason = "AI와 인간 작성의 특징이 혼재되어 있어 판단이 불확실합니다."
        elif ai_prob >= 30:
            reason = "대체로 인간이 작성한 것으로 보이나, 일부 AI 특징이 관찰됩니다."
        else:
            reason = "어휘 다양성과 문체 변동이 자연스러워 인간이 작성한 것으로 판단됩니다."

        label = "AI 생성" if ai_prob >= 50 else "사람 작성"

        return {
            "label": label,
            "ai_probability": round(ai_prob, 1),
            "reason": reason,
        }

    def predict_batch(self, texts):
        """
        여러 텍스트를 한 번에 판별합니다.

        Args:
            texts: 텍스트 리스트 (list[str])

        Returns:
            list[dict]: 각 텍스트에 대한 판별 결과
        """
        return [self.predict(text) for text in texts]


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    # 모델 로드 (최초 1회, 이후 재사용)
    detector = TextDetector(model_dir="./DeBERTa_v3_large_English_Detector")

    # 단일 텍스트 판별
    result = detector.predict(
        "The rapid advancement of artificial intelligence has fundamentally "
        "transformed the landscape of modern technology, enabling unprecedented "
        "capabilities in natural language processing and computer vision."
    )
    print(f"판정: {result['label']}")
    print(f"AI 생성 확률: {result['ai_probability']}%")
    print(f"판단 근거: {result['reason']}")

    print()

    # 여러 텍스트 한 번에 판별
    texts = [
        "I went to the store yesterday and bought some groceries. The weather was nice.",
        "Furthermore, the implementation of machine learning algorithms has demonstrated "
        "significant improvements in predictive accuracy across various domains.",
    ]
    results = detector.predict_batch(texts)
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['label']} ({r['ai_probability']}%) - {r['reason']}")
