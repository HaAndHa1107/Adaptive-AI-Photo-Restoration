
import cv2


def calculate_sharpness(image):
    """
    計算影像的拉普拉斯變異數。

    Parameters:
        image: OpenCV BGR 影像

    Returns:
        float: 銳利度指標
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return float(laplacian.var())


def test_blur_sensitivity(image):
    """
    測試 Laplacian Variance 對不同模糊程度的反應。
    """
    blur_levels = [0, 3, 5, 9, 15]

    results = {}

    for kernel_size in blur_levels:
        if kernel_size == 0:
            test_image = image.copy()
        else:
            test_image = cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                0
            )

        score = calculate_sharpness(test_image)
        results[kernel_size] = score

    return results
def select_fidelity_weight(sharpness):
    """
    根據 Laplacian Variance 選擇 CodeFormer fidelity_weight。

    注意：目前門檻為實驗性設定，尚未經過資料集驗證。
    """
    if sharpness < 5:
        return 0.3
    elif sharpness < 15:
        return 0.6
    else:
        return 0.9