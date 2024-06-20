import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, segmentation
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np
from torch import nn
import mediapipe as mp

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前訓練済みモデルのロード
resnet_model = resnet50(weights="IMAGENET1K_V1").to(device)
resnet_model.eval()
segmentation_model = segmentation.deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1").to(device)
segmentation_model.eval()
face_detection_model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1").to(device)
face_detection_model.eval()

# 学習済み3Dモデルのロード
class Simple3DRenderingModel(nn.Module):
    def __init__(self):
        super(Simple3DRenderingModel, self).__init__()
        self.fc = nn.Linear(1, 3)  # 1次元の入力を3次元に変換

    def forward(self, x):
        return self.fc(x)

    def apply_3d_model(self, frame_tensor):
        middle_pixel = frame_tensor[:, :, frame_tensor.shape[2] // 2, frame_tensor.shape[3] // 2]
        middle_pixel = middle_pixel.mean().unsqueeze(0).unsqueeze(0)  # 中央ピクセルの値を1次元に縮小
        outputs = self.forward(middle_pixel).squeeze(0).detach().cpu().numpy()
        return outputs

# 3Dモデルの初期化
model_path = "/home/kanengi/notebook/cgmodel/models/3d_model.pth"
three_d_model = Simple3DRenderingModel().to(device)
three_d_model.load_state_dict(torch.load(model_path))
three_d_model.eval()

# 中間層の特徴量を抽出するフック
def get_intermediate_layer(layer_name):
    def hook(module, input, output):
        global features
        features = output
    return hook

layer_name = 'layer3'  # ResNetの任意の中間層名
layer = dict([*resnet_model.named_modules()])[layer_name]
layer.register_forward_hook(get_intermediate_layer(layer_name))

# 画像前処理のためのトランスフォーム
preprocess_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_segmentation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mediapipeの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def apply_pseudo_3d_effect(frame):
    global features  # グローバル変数として特徴量を宣言
    features = None  # 初期化
    
    # フレームの前処理
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess_resnet(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 特徴量抽出
        resnet_model(input_tensor)

    if features is None:
        raise ValueError("特徴量が取得できませんでした。")

    # 特徴量を取得し、CPUに転送
    features_np = features.squeeze().cpu().numpy()
    
    # 特徴量をチャネル方向に平均化して2Dグレースケール画像に変換
    gray = np.mean(features_np, axis=0)
    
    # 特徴量を正規化
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    
    # 8ビットのグレースケール画像に変換
    gray = (gray * 255).astype(np.uint8)

    # グレースケール画像を元のフレームサイズにリサイズ
    gray_resized = cv2.resize(gray, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    return gray_resized

def segment_frame(frame):
    # フレームの前処理
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess_segmentation(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # セグメンテーション実行
        output = segmentation_model(input_tensor)['out']
    output_predictions = output.argmax(1).cpu().numpy()[0]

    return output_predictions

def detect_eyes(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_x = int(face_landmarks.landmark[362].x * frame.shape[1])
            left_eye_y = int(face_landmarks.landmark[362].y * frame.shape[0])
            right_eye_x = int(face_landmarks.landmark[133].x * frame.shape[1])
            right_eye_y = int(face_landmarks.landmark[133].y * frame.shape[0])
            return (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)
    return None, None

def draw_3d_segmentation(frame, seg_map, gray):
    # 人間部分のマスク
    mask = seg_map == 15  # クラスID 15は人間

    # 人間部分を黒塗りにする
    black_texture = np.zeros_like(frame)
    textured_frame = frame.copy()
    textured_frame[mask] = black_texture[mask]

    # 陰影を追加（擬似的な方法としてグレースケール画像を使用）
    shaded_frame = cv2.addWeighted(textured_frame, 0.7, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    # 目の検出
    left_eye, right_eye = detect_eyes(frame)

    if left_eye and right_eye:
        # 目の位置に円を描画
        eye_radius = 5
        cv2.circle(shaded_frame, left_eye, eye_radius, (255, 255, 255), -1)
        cv2.circle(shaded_frame, right_eye, eye_radius, (255, 255, 255), -1)

    return shaded_frame

def main():
    cap = cv2.VideoCapture(0)  # デバイスIDを0に設定

    if not cap.isOpened():
        print("カメラを開くことができません。デバイスIDを確認してください。")
        return

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("フレームを取得できません。")
            break
        
        # 擬似3D効果の適用
        gray_resized = apply_pseudo_3d_effect(frame)
        
        # セグメンテーション実行
        seg_map = segment_frame(frame)
        
        # 3Dモデルの適用
        frame_tensor = torch.from_numpy(frame).float().to(device).permute(2, 0, 1).unsqueeze(0)
        three_d_effect = three_d_model.apply_3d_model(frame_tensor)
        
        # 擬似3D効果とセグメンテーション結果を描画
        result_frame = draw_3d_segmentation(frame, seg_map, gray_resized)
        
        # フレームを表示
        cv2.imshow('Pseudo 3D Effect with Segmentation', result_frame)
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
