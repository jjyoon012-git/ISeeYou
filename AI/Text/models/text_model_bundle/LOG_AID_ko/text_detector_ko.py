"""
한국어 AI 텍스트 판별 모델 — 추론 코드
모델: LOG-AID (Qwen2.5-7B base/instruct) + Logistic Regression
사용법: 아래 예시 참고

필수 라이브러리:
    pip install torch transformers scipy scikit-learn joblib

필수 파일 (같은 폴더에 배치):
    logistic_regression.pkl   ← Logistic Regression 분류기
    standard_scaler.pkl       ← Z-score 정규화기

주의사항:
    - Qwen2.5-7B 모델 2개를 로드하므로 GPU 최소 32GB VRAM 필요 (A100 권장)
    - 8-bit 양자화 사용 시 16GB GPU에서도 실행 가능 (pip install bitsandbytes)
    - 최초 실행 시 Hugging Face에서 모델 다운로드 (~14GB × 2)
"""

import torch
import numpy as np
import joblib
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextDetectorKO:
    FEATURE_NAMES = [
        'mean_surprisal_base', 'mean_surprisal_instruct', 'mean_jsd',
        'mean_entropy_diff', 'mean_entropy_base', 'mean_log_rank',
    ]

    FEATURE_REASONS = {
        'mean_surprisal_instruct': {
            'ai': "Instruct 모델이 텍스트를 쉽게 예측하여 AI 생성 패턴이 감지됨",
            'human': "Instruct 모델이 텍스트를 예측하기 어려워 인간 작성 특성이 관찰됨",
        },
        'mean_entropy_base': {
            'ai': "Base 모델에서 높은 불확실성이 관찰되어 AI 생성 가능성이 높음",
            'human': "Base 모델에서 낮은 불확실성이 관찰되어 인간 작성 가능성이 높음",
        },
        'mean_log_rank': {
            'ai': "토큰 예측 순위가 낮아 비전형적인 단어 선택이 감지됨",
            'human': "토큰 예측 순위가 높아 자연스러운 단어 선택이 관찰됨",
        },
        'mean_entropy_diff': {
            'ai': "두 모델 간 엔트로피 차이가 작아 AI 생성 패턴이 감지됨",
            'human': "두 모델 간 엔트로피 차이가 커 인간 작성 특성이 관찰됨",
        },
        'mean_jsd': {
            'ai': "두 모델 간 분포 차이가 커 AI 생성 가능성이 높음",
            'human': "두 모델 간 분포 차이가 작아 인간 작성 가능성이 높음",
        },
        'mean_surprisal_base': {
            'ai': "Base 모델의 예측 어려움이 높아 AI 생성 특성이 관찰됨",
            'human': "Base 모델의 예측이 자연스러워 인간 작성 특성이 관찰됨",
        },
    }

    def __init__(self, classifier_dir, quantization='none', max_length=1024, device=None):
        """
        Args:
            classifier_dir: logistic_regression.pkl, standard_scaler.pkl이 있는 폴더
            quantization: 'none'(fp16, 32GB+), '8bit'(16GB), '4bit'(10GB)
            max_length: 최대 토큰 길이
            device: 'cuda' 또는 'cpu'
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.max_length = max_length
        self.quantization = quantization

        # 분류기 로드
        self.clf = joblib.load(f"{classifier_dir}/logistic_regression.pkl")
        self.scaler = joblib.load(f"{classifier_dir}/standard_scaler.pkl")

        # Qwen2.5-7B 모델 2개 로드
        self.base_model, self.base_tokenizer = self._load_model("Qwen/Qwen2.5-7B")
        self.inst_model, self.inst_tokenizer = self._load_model("Qwen/Qwen2.5-7B-Instruct")

        print(f"GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    def _load_model(self, model_name):
        print(f"  모델 로드: {model_name} ({self.quantization})")
        kwargs = {'trust_remote_code': True, 'device_map': 'auto', 'local_files_only': True}

        if self.quantization == '8bit':
            kwargs['load_in_8bit'] = True
        elif self.quantization == '4bit':
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4',
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    @torch.no_grad()
    def _extract_features(self, text):
        """텍스트 → 6차원 feature 벡터"""
        try:
            # Base 모델
            inp_b = self.base_tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
            ids_b = inp_b['input_ids'].to(self.base_model.device)
            if ids_b.shape[1] < 2:
                return None
            logits_b = self.base_model(ids_b).logits
            sl_b = logits_b[0, :-1, :]
            lb_b = ids_b[0, 1:]
            probs_b = torch.softmax(sl_b.float(), dim=-1)

            # base surprisal
            tp_b = probs_b.gather(1, lb_b.unsqueeze(1)).squeeze(1)
            surp_b = -torch.log(tp_b + 1e-10)

            # base entropy
            log_p_b = torch.log(probs_b + 1e-10)
            ent_b = -(probs_b * log_p_b).sum(dim=-1)

            # base log-rank
            sorted_b = torch.argsort(probs_b, dim=-1, descending=True)
            n_tok = lb_b.shape[0]
            ranks_b = torch.zeros(n_tok, device=logits_b.device)
            for t in range(n_tok):
                pos = (sorted_b[t] == lb_b[t]).nonzero(as_tuple=True)[0]
                ranks_b[t] = (pos[0].float() + 1) if len(pos) > 0 else probs_b.shape[-1]
            lr_b = torch.log(ranks_b + 1e-10)

            del logits_b, sl_b, sorted_b, log_p_b
            torch.cuda.empty_cache()

            # Instruct 모델
            inp_i = self.inst_tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
            ids_i = inp_i['input_ids'].to(self.inst_model.device)
            if ids_i.shape[1] < 2:
                return None
            logits_i = self.inst_model(ids_i).logits
            sl_i = logits_i[0, :-1, :]
            lb_i = ids_i[0, 1:]
            probs_i = torch.softmax(sl_i.float(), dim=-1)

            tp_i = probs_i.gather(1, lb_i.unsqueeze(1)).squeeze(1)
            surp_i = -torch.log(tp_i + 1e-10)
            log_p_i = torch.log(probs_i + 1e-10)
            ent_i = -(probs_i * log_p_i).sum(dim=-1)

            del logits_i, sl_i, log_p_i
            torch.cuda.empty_cache()

            # 6개 feature 계산
            n = min(probs_b.shape[0], probs_i.shape[0])
            p = probs_b[:n]
            q = probs_i[:n]
            m = 0.5 * (p + q)
            log_m = torch.log(m + 1e-10)
            kl_pm = (p * (torch.log(p + 1e-10) - log_m)).sum(dim=-1)
            kl_qm = (q * (torch.log(q + 1e-10) - log_m)).sum(dim=-1)
            jsd = 0.5 * (kl_pm + kl_qm)
            ent_diff = torch.abs(ent_b[:n] - ent_i[:n])

            feature = np.array([
                surp_b.mean().cpu().item(),
                surp_i.mean().cpu().item(),
                jsd.mean().cpu().item(),
                ent_diff.mean().cpu().item(),
                ent_b.mean().cpu().item(),
                lr_b.mean().cpu().item(),
            ])

            del probs_b, probs_i, p, q, m, log_m
            torch.cuda.empty_cache()

            return feature

        except Exception as e:
            print(f"  [오류] {str(e)[:80]}")
            torch.cuda.empty_cache()
            return None

    def predict(self, text):
        """
        텍스트를 분석하여 AI 생성 여부를 판별합니다.

        Args:
            text: 판별할 한국어 텍스트 (str)

        Returns:
            dict: {
                "label": "AI 생성" 또는 "사람 작성",
                "ai_probability": 92.1,  (0~100%)
                "reason": "판단 근거 설명"
            }
        """
        feature = self._extract_features(text)
        if feature is None:
            return {
                "label": "판단 불가",
                "ai_probability": 0.0,
                "reason": "텍스트가 너무 짧거나 처리 중 오류가 발생했습니다.",
            }

        X = self.scaler.transform(feature.reshape(1, -1))
        prob = self.clf.predict_proba(X)[0]
        ai_prob = prob[1] * 100

        # feature별 기여도 → 상위 2개로 이유 생성
        contributions = self.clf.coef_[0] * X[0]
        top_indices = np.argsort(np.abs(contributions))[::-1][:2]

        reason_parts = []
        for idx in top_indices:
            name = self.FEATURE_NAMES[idx]
            direction = 'ai' if contributions[idx] > 0 else 'human'
            reason_parts.append(self.FEATURE_REASONS.get(name, {}).get(direction, ""))

        label = "AI 생성" if ai_prob >= 50 else "사람 작성"

        return {
            "label": label,
            "ai_probability": round(ai_prob, 1),
            "reason": " / ".join(reason_parts),
        }


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    # 모델 로드 (최초 1회, GPU 32GB 이상 필요)
    detector = TextDetectorKO(
        classifier_dir="./LOG_AID_results",
        quantization='none',  # A100: 'none', T4: '8bit'
    )

    # 단일 텍스트 판별
    result = detector.predict(
        "인공지능 기술의 발전은 자연어 처리 분야에 혁신적인 변화를 가져왔으며, "
        "다양한 산업 분야에서 활용 가능성이 크게 확대되고 있다."
    )
    print(f"판정: {result['label']}")
    print(f"AI 생성 확률: {result['ai_probability']}%")
    print(f"판단 근거: {result['reason']}")
