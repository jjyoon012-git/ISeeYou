from text_detector_en import TextDetector

# 모델 로드 (서버 시작 시 1회)
detector_en = TextDetector(model_dir="./DeBERTa_v3_large_English_Detector")

# 판별 (요청마다)
result = detector_en.predict("판별할 영어 텍스트")
print(result)

# 여러 건 한 번에
results = detector_en.predict_batch(["텍스트1", "텍스트2"])