import os
import cv2
import json
import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as F

# データ保存用ディレクトリの作成
data_dir = os.path.join(os.path.dirname(__file__), '../data')
os.makedirs(data_dir, exist_ok=True)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# セグメンテーションモデルの初期化
weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
segmentation_model = maskrcnn_resnet50_fpn(weights=weights).to(device)
segmentation_model.eval()

# 姿勢推定モデルの初期化（ここではDeepLabV3を使用）
pose_model = deeplabv3_resnet50(pretrained=True).to(device)
pose_model.eval()

# Webカメラの読み込み
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラが開けません")
    exit()

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("フレームが読み込めませんでした")
        break

    # フレームの取得確認
    print(f"フレーム {frame_id} 読み込み成功")

    # フレームをテンソルに変換してGPUに転送
    frame_tensor = F.to_tensor(frame).to(device).unsqueeze(0)

    # セグメンテーション
    with torch.no_grad():
        predictions = segmentation_model(frame_tensor)

    # セグメンテーションマスクの取得（最初の人のマスクのみ使用）
    masks = predictions[0]['masks']
    if len(masks) > 0:
        mask = masks[0, 0].mul(255).byte().cpu().numpy()
    else:
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # 姿勢推定
    with torch.no_grad():
        pose_predictions = pose_model(frame_tensor)['out']
        pose_predictions = torch.argmax(pose_predictions.squeeze(), dim=0).detach().cpu().numpy()

    # セグメンテーションマスクの保存
    seg_mask_path = os.path.join(data_dir, f'segmentation_mask_{frame_id}.png')
    cv2.imwrite(seg_mask_path, mask)

    # 姿勢推定データの保存（仮の例としてフレームごとの簡単な情報を保存）
    joint_data = {
        "frame_id": frame_id,
        "mask_sum": mask.sum().item(),  # 例としてマスクのピクセル数を保存
        "pose_predictions": pose_predictions.tolist()  # 姿勢推定結果を保存
    }
    joint_data_path = os.path.join(data_dir, f'joint_data_{frame_id}.json')
    with open(joint_data_path, 'w') as f:
        json.dump(joint_data, f)

    # フレームにマスクを重ねて表示
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined_frame = cv2.addWeighted(frame, 0.5, mask_colored, 0.5, 0)
    cv2.imshow('Segmentation', combined_frame)

    frame_id += 1

    # 一定数のフレームをキャプチャしたら終了（例えば、1000フレーム）
    if frame_id >= 1000:
        print("1000フレームをキャプチャしました。終了します。")
        break

    # 'q'キーで途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("カメラを解放しました")
