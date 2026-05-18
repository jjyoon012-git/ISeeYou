# Chrome Extension 연동 메모

이 폴더는 `ryujihos0105/isy-extention` 저장소를 ISeeYou 프로젝트 내부로 가져온 것입니다. 원본 저장소의 `.git` 폴더는 제외했고, ISeeYou 메인 저장소에서 함께 버전 관리합니다.

## 포함된 구성

- `extension/`: Chrome Manifest V3 확장 프로그램 본체
- `server.py`: 실제 모델 추론용 FastAPI 서버
- `mock_server.py`: UI/흐름 확인용 mock 서버
- `demo_platform/`: YouTube 스타일 데모 플랫폼
- `versionv9/`: 이미지 모델 코드와 weight 배치 위치
- `video/`, `video_inference.py`: 비디오 모델 코드와 7개 앙상블 weight 배치 위치

## 현재 API 연결 상태

확장 프로그램은 현재 아래 로컬 서버를 호출합니다.

```text
http://localhost:8000
```

주요 호출 경로:

- `POST /api/analyze/image`
- `POST /api/analyze/video`
- `POST /api/analyze/video-file`
- `POST /api/analyze/text`

반면 현재 ISeeYou 웹 UI가 사용하는 통합 백엔드는 `http://127.0.0.1:8001`이며, 엔드포인트 이름이 다릅니다. 따라서 확장 프로그램을 메인 백엔드와 완전히 통합하려면 다음 중 하나가 필요합니다.

1. 확장 프로그램 `background.js`의 API 호출 경로를 `8001` 백엔드 형식에 맞게 수정
2. `8001` 백엔드에 확장 프로그램 호환용 `/api/analyze/*` 라우트 추가
3. `8000` 확장 서버를 유지하고 내부에서 ISeeYou 모델/백엔드를 호출하는 adapter 서버로 사용

현재는 원본 확장 구조를 보존해 가져왔고, GitHub 업로드 대상에는 모델 weight 파일을 포함하지 않습니다.

## 로컬 실행

### 서버 실행

```powershell
cd C:\Users\jjeong\Desktop\ISeeYou\Extension
python server.py
```

모델 없이 UI 흐름만 확인하려면:

```powershell
python mock_server.py
```

### Chrome에 확장 로드

1. Chrome에서 `chrome://extensions`를 엽니다.
2. 개발자 모드를 켭니다.
3. `압축해제된 확장 프로그램을 로드`를 선택합니다.
4. `C:\Users\jjeong\Desktop\ISeeYou\Extension\extension` 폴더를 선택합니다.

## GitHub 업로드 제외

`.pt`, `.pth`, `.pkl`, `.onnx`, `.safetensors`, `.env`, 로그, 업로드 데이터는 Git에 포함하지 않습니다. 모델 weight는 별도 로컬 경로 또는 외부 스토리지에서 관리해야 합니다.


## 현재 로컬 실행 상태

로컬 테스트는 다음 두 단계로 실행합니다.

```powershell
cd C:\Users\jjeong\Desktop\ISeeYou\Extension
powershell -ExecutionPolicy Bypass -File .\start_extension_mock_server.ps1
```

다른 터미널에서:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch_chrome_extension.ps1
```

`mock_server.py`는 모델 가중치 없이 확장 UI와 페이지 배지 흐름을 확인하기 위한 서버입니다. 실제 추론으로 전환하려면 `server.py`를 실행하고 `versionv9/weights`, `video/checkpoints_*` 아래에 필요한 weight를 배치해야 합니다.
