⸻

📦 SSD300 Custom Transform Object Detection

SSD300 + VGG16 Backbone 기반의 객체 탐지(Object Detection) 프로젝트입니다.
본 프로젝트는 페이스 중심 Random Crop, LetterBox Resize, Bounding Box 좌표 보정 등
직접 구현한 Custom Transform Pipeline을 적용해
소규모 데이터셋에서도 안정적인 학습을 수행하는 것을 목표로 합니다.

⸻

🚀 Features

✅ 1. SSD300(VGG16) 모델 사용
	•	torchvision.models.detection.ssd300_vgg16
	•	Backbone만 ImageNet pretrain 사용 (weights_backbone=VGG16_Weights.IMAGENET1K_V1)
	•	Detection head는 랜덤 초기화 후 새로 학습

✅ 2. Custom Data Preprocessing

🔹 RandomFaceCrop
	•	GT bbox 주변을 확장하여 face 중심 crop
	•	bbox 좌표 보정(clamp, offset 적용)

🔹 LetterBoxResize
	•	비율 유지(resize with aspect ratio)
	•	새 canvas(300×300)에 중앙 배치
	•	bbox scale + offset 적용

✅ 3. Balanced Loss 학습
	•	Classification & BBox regression loss 개별 로깅
	•	overfitting 방지 lr scheduler 적용

✅ 4. 평가 지표(Evaluation)

✔ Simple Metrics
	•	Precision
	•	Recall
	•	Mean IoU (GT–Pred 매칭 기반)

✔ Custom mAP(0.5 IoU)
	•	클래스별 AP 계산(cat/dog)
	•	IoU matrix 기반 matching
	•	trapezoidal integration 사용

🧪 Evaluation Example

===== SSD Evaluation (thr=0.5) =====
Precision  : 0.1927
Recall     : 0.2016
mIoU       : 0.6507
TP         : 148
Detections : 768
GT Boxes   : 734
===================================

===== mAP Evaluation (IoU=0.5) =====
cat        AP : 0.1682
dog        AP : 0.1109
mAP              : 0.1396
=====================================


⸻

🛠 How to Train

from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights

model = ssd300_vgg16(
    weights_backbone=VGG16_Weights.IMAGENET1K_V1,
    num_classes=3  # background 포함
).to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4
)

for epoch in range(num_epochs):
    train_one_epoch(...)
    lr_scheduler.step()


⸻

📊 Evaluation

Simple evaluation

from eval_simple import eval_ssd_simple
m = eval_ssd_simple(model, val_dataset, device, iou_thresh=0.5)

mAP evaluation

from eval_map import eval_map_ssd

aps, mAP = eval_map_ssd(
    model, val_dataset, device,
    iou_thresh=0.5,
    num_classes=3
)


⸻

📌 TODO
	•	SSD 전용 augmentation(photometric distortion, expand, crop) 적용
	•	Focal Loss 기반 imbalance 개선
	•	RetinaNet 또는 Faster R-CNN baseline 비교
	•	ONNX 변환 + 추론 속도 측정

⸻

📝 License

MIT License.

⸻
