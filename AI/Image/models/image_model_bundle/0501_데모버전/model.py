# 이 파일은 Late Fusion 모델 구조 정의를 담당합니다

import torch
import torch.nn as nn
import timm


# EfficientNet-B4 백본을 생성하는 함수
# num_classes=0 이면 classifier를 제거하고 feature vector만 반환 (Late Fusion 브랜치용)
def make_efficientnet_b4(in_channels: int = 3, num_classes: int = 2) -> nn.Module:
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,      # 추론 데모이므로 pretrained 불필요 (가중치를 직접 로드)
        num_classes=num_classes,
    )

    # 입력 채널이 3이 아닐 경우 conv_stem을 확장
    if in_channels != 3:
        old_conv = model.conv_stem
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight  # 기존 RGB 가중치 복사
            avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for c in range(3, in_channels):
                new_conv.weight[:, c:c+1, :, :] = avg_weight  # 추가 채널은 평균으로 초기화
        model.conv_stem = new_conv

    return model


# Late Fusion 모델 클래스
# RGB 브랜치와 FFT 브랜치가 각각 따로 특징을 추출한 뒤, 합쳐서 최종 판단
class LateFusionModel(nn.Module):
    """
    구조:
      rgb_branch : EfficientNet-B4 (num_classes=0) → (B, 1792)
      fft_branch : EfficientNet-B4 (num_classes=0) → (B, 1792)
      concat     : (B, 3584)
      fc         : Linear(3584, 512) → ReLU → Dropout(0.3) → Linear(512, 2)
    """

    def __init__(self):
        super().__init__()
        # 각 브랜치는 3채널 입력, classifier 제거 (feature extractor 전용)
        self.rgb_branch = make_efficientnet_b4(in_channels=3, num_classes=0)
        self.fft_branch = make_efficientnet_b4(in_channels=3, num_classes=0)

        feat_dim = self.rgb_branch.num_features  # EfficientNet-B4: 1792
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),  # RGB(1792) + FFT(1792) = 3584 → 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),             # 최종 출력: [Fake 확률, Real 확률]
        )

    def forward(self, x_rgb: torch.Tensor, x_fft: torch.Tensor) -> torch.Tensor:
        feat_rgb = self.rgb_branch(x_rgb)              # (B, 1792)
        feat_fft = self.fft_branch(x_fft)              # (B, 1792)
        feat_cat = torch.cat([feat_rgb, feat_fft], dim=1)  # (B, 3584)
        return self.fc(feat_cat)                        # (B, 2)


# 저장된 가중치를 불러와 추론 준비 상태의 모델을 반환하는 함수
def load_model(model_path: str, device: str) -> LateFusionModel:
    model = LateFusionModel()

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise FileNotFoundError(f"[에러] 가중치 파일을 찾을 수 없습니다: {model_path}")
    except RuntimeError as e:
        raise RuntimeError(f"[에러] 가중치 로드 실패 (모델 구조가 맞지 않을 수 있습니다): {e}")

    model.to(device)
    model.eval()  # 추론 모드로 전환 (Dropout, BatchNorm 비활성화)
    return model
