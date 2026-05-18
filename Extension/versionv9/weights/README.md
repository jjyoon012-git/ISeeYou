# 이미지 모델 가중치

이 폴더에 이미지 판별 모델의 가중치 파일을 넣어주세요.

```
versionv9/weights/
└── best.pt   ← 이 파일이 필요합니다
```

파일은 **[Hugging Face Hub](https://huggingface.co/ryujiho/isy-weights)** 에서 다운로드할 수 있습니다.  
HF Hub의 `image/best.pt` 를 이 폴더에 `best.pt` 로 저장한 뒤 `python server.py` 로 서버를 실행하면 됩니다.
