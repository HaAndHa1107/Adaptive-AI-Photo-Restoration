
import os
import sys
import types

# ============================================================
# 1. 相容性補丁
# ============================================================

# 保留目前已經測試成功的 BasicSR 版本相容性處理
if "basicsr.version" not in sys.modules:
    v_mod = types.ModuleType("basicsr.version")
    v_mod.__version__ = "1.4.2"
    v_mod.__gitsha__ = "unknown"
    sys.modules["basicsr.version"] = v_mod

import torch
import cv2

from quality_assessment import (
    calculate_sharpness,
    test_blur_sensitivity,
    select_fidelity_weight,
)

from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer


def main():

    print("[*] 正在啟動 4K 修復流程...")

    device = get_device()
    print(f"[*] 使用裝置: {device}")

    # ========================================================
    # 2. 讀取輸入影像
    # ========================================================

    input_path = "inputs/test.jpg"

    if not os.path.exists(input_path):
        print(f"[!] 找不到輸入圖片：{input_path}")
        print(f"[*] 當前工作目錄：{os.getcwd()}")
        return

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[!] 無法讀取圖片：{input_path}")
        return

    print(f"[*] 成功讀取圖片：{input_path}")
    print(f"[*] 原始解析度：{img.shape[1]} x {img.shape[0]}")

    os.makedirs("results", exist_ok=True)

    # ========================================================
    # 3. 載入 Real-ESRGAN
    # ========================================================

    print("[1/3] 載入 Real-ESRGAN (4x 升頻)...")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    bg_upsampler = RealESRGANer(
        scale=4,
        model_path="weights/realesrgan/RealESRGAN_x4plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=False,
    )

    # ========================================================
    # 4. 載入 CodeFormer
    # ========================================================

    print("[2/3] 載入 CodeFormer 人臉修復模型...")

    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    ckpt_path = "weights/CodeFormer/codeformer.pth"

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )["params_ema"]

    net.load_state_dict(checkpoint)
    net.eval()

    # ========================================================
    # 5. 人臉偵測與對齊
    # ========================================================

    print("[3/3] 開始偵測與對齊人臉...")

    face_helper = FaceRestoreHelper(
        upscale_factor=4,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )

    face_helper.clean_all()
    face_helper.read_image(img)

    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False,
        resize=640,
        eye_dist_threshold=5,
    )

    print(f"[*] 偵測到人臉數量：{num_det_faces}")

    face_helper.align_warp_face()

    # ========================================================
    # 6. 品質評估與自適應參數選擇
    # ========================================================

    print("\n[*] 開始分析原始人臉品質...")

    face_sharpness_scores = []
    adaptive_weights = []

    for idx, cropped_face in enumerate(face_helper.cropped_faces):

        sharpness = calculate_sharpness(cropped_face)
        adaptive_w = select_fidelity_weight(sharpness)

        face_sharpness_scores.append(sharpness)
        adaptive_weights.append(adaptive_w)

        print(f"\n[品質分析] 人臉 {idx + 1}")
        print(f"    Laplacian Variance：{sharpness:.4f}")
        print(f"    自適應選擇 w = {adaptive_w}")

        blur_results = test_blur_sensitivity(cropped_face)

        print("    不同模糊程度測試：")

        for kernel_size, score in blur_results.items():
            print(
                f"    Kernel {kernel_size:2d}: {score:.4f}"
            )

    # ========================================================
    # 7. Real-ESRGAN 背景放大（只執行一次）
    # ========================================================

    print("\n[*] 正在進行背景 x4 放大...")

    with torch.inference_mode():
        bg_img = bg_upsampler.enhance(
            img,
            outscale=4,
        )[0]

    print(
        f"[*] 放大後解析度："
        f"{bg_img.shape[1]} x {bg_img.shape[0]}"
    )

    # ========================================================
    # 8. 預先轉換人臉 Tensor
    # ========================================================

    face_tensors = []

    for cropped_face in face_helper.cropped_faces:

        cropped_face_t = img2tensor(
            cropped_face / 255.0,
            bgr2rgb=True,
            float32=True,
        )

        normalize(
            cropped_face_t,
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            inplace=True,
        )

        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        face_tensors.append(cropped_face_t)

    # ========================================================
    # 9. 初始化反向仿射矩陣
    # ========================================================

    face_helper.inverse_affine_matrices = []

    # 只計算一次
    face_helper.get_inverse_affine(None)

    # 保存原始矩陣，避免貼回人臉時產生累積位移
    original_inverse_matrices = [
        matrix.copy()
        for matrix in face_helper.inverse_affine_matrices
    ]

    print(
        "[DEBUG] 初始化反向矩陣數量：",
        len(original_inverse_matrices),
    )

    assert len(face_tensors) == len(
        original_inverse_matrices
    ), "人臉數量與反向仿射矩陣數量不一致"

    # ========================================================
    # 10. 固定 fidelity_weight 對照實驗
    # ========================================================

    fidelity_weights = [0.3, 0.6, 0.9]

    for w in fidelity_weights:

        print(f"\n[*] 正在測試 fidelity_weight = {w}")

        # 清除上一輪結果
        face_helper.restored_faces = []

        for cropped_face, cropped_face_t in zip(
            face_helper.cropped_faces,
            face_tensors,
        ):

            with torch.inference_mode():
                output = net(
                    cropped_face_t,
                    w=w,
                    adain=True,
                )[0]

            restored_face = tensor2img(
                output,
                rgb2bgr=True,
                min_max=(-1, 1),
            )

            face_helper.add_restored_face(
                restored_face.astype("uint8"),
                cropped_face,
            )

        # 每一輪都還原原始矩陣
        face_helper.inverse_affine_matrices = [
            matrix.copy()
            for matrix in original_inverse_matrices
        ]

        print(
            f"[DEBUG] cropped_faces = "
            f"{len(face_helper.cropped_faces)}"
        )

        print(
            f"[DEBUG] restored_faces = "
            f"{len(face_helper.restored_faces)}"
        )

        print(
            f"[DEBUG] inverse_affine_matrices = "
            f"{len(face_helper.inverse_affine_matrices)}"
        )

        assert len(face_helper.restored_faces) == len(
            face_helper.inverse_affine_matrices
        ), "人臉與矩陣數量不一致"

        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img.copy()
        )

        out_path = f"results/output_w{w:.1f}.png"

        imwrite(restored_img, out_path)

        print(
            f"[+] 已儲存：{os.path.abspath(out_path)}"
        )

    print("\n[+] 三組 fidelity_weight 對照實驗完成！")

    # ========================================================
    # 11. Adaptive Fidelity Weight 自適應修復
    # ========================================================

    print("\n[*] 開始自適應人臉修復...")

    # 清空固定參數實驗的修復結果
    face_helper.restored_faces = []

    for idx, (cropped_face, cropped_face_t) in enumerate(
        zip(
            face_helper.cropped_faces,
            face_tensors,
        )
    ):

        # 每張人臉使用自己的品質評估結果
        w = adaptive_weights[idx]

        print(
            f"[*] 人臉 {idx + 1} 使用自適應 w = {w}"
        )

        with torch.inference_mode():
            output = net(
                cropped_face_t,
                w=w,
                adain=True,
            )[0]

        restored_face = tensor2img(
            output,
            rgb2bgr=True,
            min_max=(-1, 1),
        )

        face_helper.add_restored_face(
            restored_face.astype("uint8"),
            cropped_face,
        )

    # 還原原始矩陣，避免位移累積
    face_helper.inverse_affine_matrices = [
        matrix.copy()
        for matrix in original_inverse_matrices
    ]

    assert len(face_helper.restored_faces) == len(
        face_helper.inverse_affine_matrices
    ), "自適應修復的人臉與矩陣數量不一致"

    adaptive_img = face_helper.paste_faces_to_input_image(
        upsample_img=bg_img.copy()
    )

    adaptive_path = "results/output_adaptive.png"

    imwrite(adaptive_img, adaptive_path)

    print(
        f"[+] 自適應修復完成："
        f"{os.path.abspath(adaptive_path)}"
    )

    # ========================================================
    # 12. 驗證自適應結果
    # ========================================================

    print("\n[*] 開始驗證自適應修復結果...")

    # 只有單張人臉且自適應選擇 w=0.6 時，
    # 才與固定 w=0.6 的結果做像素比較
    if len(adaptive_weights) == 1 and adaptive_weights[0] == 0.6:

        baseline = cv2.imread("results/output_w0.6.png")
        adaptive = cv2.imread(adaptive_path)

        if baseline is not None and adaptive is not None:

            if baseline.shape == adaptive.shape:

                difference = cv2.absdiff(
                    baseline,
                    adaptive,
                )

                max_difference = int(difference.max())

                print(
                    f"[驗證] 自適應與 w=0.6 "
                    f"最大像素差異：{max_difference}"
                )

                if max_difference == 0:
                    print("[驗證] PASS：兩張圖片完全一致")
                else:
                    print("[驗證] WARNING：兩張圖片存在差異")

            else:
                print("[驗證] WARNING：兩張圖片尺寸不同")

        else:
            print("[驗證] WARNING：無法讀取比較圖片")

    else:
        print(
            "[驗證] 本次自適應參數不全為 w=0.6，"
            "略過單張人臉的像素一致性測試"
        )

    print("\n[+] 所有影像修復流程完成！")


if __name__ == "__main__":
    main()
