from text_detector_ko import TextDetectorKO

# 모델 로드 (서버 시작 시 1회, GPU 32GB 이상 필요)
detector_ko = TextDetectorKO(
    classifier_dir="./LOG_AID_results",
    quantization="none",  # A100: 'none', T4: '8bit'
)

# 판별 (요청마다)
result = detector_ko.predict("판별할 한국어 텍스트")
print(result)
