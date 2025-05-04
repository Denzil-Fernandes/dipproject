import os
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from rembg import remove
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score

# Create directory
os.makedirs("original", exist_ok=True)

st.title("🧰 Image Restoration & Background Removal App")

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    original_path = os.path.join("original", uploaded_file.name)

    with open(original_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and normalize image
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_np = np.array(image).astype(np.float32) / 255.0

    st.sidebar.title("🔘 Choose an Operation")
    choice = st.sidebar.radio(
        "Select processing step:",
        (
            "Show Original",
            "Remove Background",
            "Denoise",
            "Remove Scratches",
            "Enhance Contrast",
            "Apply Gaussian Filter",
            "Apply Median Filter",
            "Apply Sharpening",
            "Compute Metrics"
        )
    )

    def denoise_image(img):
        img_8bit = (img * 255).astype(np.uint8)
        return cv2.fastNlMeansDenoisingColored(img_8bit, None, 10, 10, 7, 21).astype(np.float32) / 255.0

    def create_mask(img_shape, vertical=True):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if vertical:
            mask[:, 100:105] = 255
        else:
            mask[50:100, 30:40] = 255
        cv2.imwrite("damage_mask.png", mask)
        return mask

    def remove_scratches(img, mask_path):
        mask = cv2.imread(mask_path, 0)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        img_8bit = (img * 255).astype(np.uint8)
        return cv2.inpaint(img_8bit, mask, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0

    def enhance_contrast(img):
        img_8bit = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    def apply_gaussian_filter(img, k=5):
        return cv2.GaussianBlur((img * 255).astype(np.uint8), (k, k), 0).astype(np.float32) / 255.0

    def apply_median_filter(img, k=3):
        return cv2.medianBlur((img * 255).astype(np.uint8), k).astype(np.float32) / 255.0

    def apply_sharpening(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D((img * 255).astype(np.uint8), -1, kernel).astype(np.float32) / 255.0

    def compute_metrics(original, processed, mask=None):
        original_gray = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        psnr_val = psnr(original_gray, processed_gray, data_range=255)
        ssim_val = ssim(original_gray, processed_gray, data_range=255)
        miou_val = None
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8).flatten()
            diff_mask = (original_gray != processed_gray).astype(np.uint8).flatten()
            miou_val = jaccard_score(mask_bin, diff_mask)
        return psnr_val, ssim_val, miou_val

    # Processing Steps
    denoised = denoise_image(img_np)
    mask = create_mask(img_np.shape, vertical=True)
    restored = remove_scratches(denoised, "damage_mask.png")
    contrast_enhanced = enhance_contrast(restored)

    if choice == "Show Original":
        st.image(image, caption="Original Image", use_column_width=True)

    elif choice == "Remove Background":
        with st.spinner("Removing background..."):
            output = remove(file_bytes)
        bg_removed = Image.open(io.BytesIO(output))
        st.image(bg_removed, caption="No Background", use_column_width=True)
        st.download_button("Download Transparent Image", output, file_name="no_bg.png")

    elif choice == "Denoise":
        st.image(denoised, caption="Denoised Image", use_column_width=True)

    elif choice == "Remove Scratches":
        st.image(restored, caption="Scratch Removed", use_column_width=True)

    elif choice == "Enhance Contrast":
        st.image(contrast_enhanced, caption="Contrast Enhanced", use_column_width=True)

    elif choice == "Apply Gaussian Filter":
        gaussian_filtered = apply_gaussian_filter(contrast_enhanced)
        st.image(gaussian_filtered, caption="Gaussian Filter Applied", use_column_width=True)

    elif choice == "Apply Median Filter":
        median_filtered = apply_median_filter(contrast_enhanced)
        st.image(median_filtered, caption="Median Filter Applied", use_column_width=True)

    elif choice == "Apply Sharpening":
        sharpened = apply_sharpening(contrast_enhanced)
        st.image(sharpened, caption="Sharpened Image", use_column_width=True)

    elif choice == "Compute Metrics":
        psnr_val, ssim_val, miou_val = compute_metrics(img_np, contrast_enhanced, mask)
        st.markdown("### 📊 Image Quality Metrics")
        st.write(f"**PSNR**: {psnr_val:.2f} dB")
        st.write(f"**SSIM**: {ssim_val:.4f}")
        if miou_val is not None:
            st.write(f"**mIoU**: {miou_val:.4f}")

