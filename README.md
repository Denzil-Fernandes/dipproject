🔍 Overview
The Image Restoration & Background Removal App is a comprehensive image enhancement tool built using Python and Streamlit that enables users to upload images and apply various image restoration and analysis techniques. It is designed for both casual users and professionals who want to clean, enhance, and analyze visual quality of images—especially useful for old photos, scanned documents, or any degraded visual content.

This tool combines several classic and AI-powered image processing methods into a single, easy-to-use web interface.

🧠 Core Features & Functionalities
🖼️ Show Original
Displays the uploaded image without any processing.

🪄 Background Removal
Utilizes the rembg library (which employs a pre-trained deep learning model) to separate the foreground (main object/person) from the background.

Output: A transparent PNG with background removed.

Download option included.

🔕 Denoise
Applies Non-local Means Denoising using OpenCV’s fastNlMeansDenoisingColored method.

Suitable for reducing color and luminance noise in photographs, especially scanned or old images.

🩹 Remove Scratches
Uses a custom-generated binary damage mask and OpenCV’s inpainting technique (cv2.INPAINT_TELEA) to restore damaged parts.

A vertical scratch mask is simulated and used to demonstrate the process.

Ideal for historical photo restoration.

🌓 Enhance Contrast
Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the contrast in the L channel of the LAB color space.

Effective for improving low-contrast images while avoiding over-amplification of noise.

🌫️ Apply Gaussian Filter
Smooths the image by applying a Gaussian Blur with a configurable kernel size (default is 5x5).

Useful for reducing high-frequency noise and minor artifacts.

🔳 Apply Median Filter
Applies a Median Blur, which is highly effective in removing salt-and-pepper noise.

🔪 Apply Sharpening
Uses a convolution kernel to enhance edges and make the image appear more crisp and clear.

📈 Compute Metrics
Evaluates the quality of the restored image compared to the original using:

PSNR (Peak Signal-to-Noise Ratio) – Indicates reconstruction quality; higher is better.

SSIM (Structural Similarity Index) – Measures visual similarity; ranges from 0 to 1.

mIoU (Mean Intersection over Union) – Optional metric used when a damage mask is applied; compares the restoration accuracy in damaged regions.
________________________________________
🔧 Technologies Used
•	Python – Main programming language used for backend processing.
•	Streamlit – For building the interactive web application UI.
•	OpenCV – For core image processing operations (e.g., denoising, filtering, scratch removal).
•	NumPy – For numerical and array operations.
•	Pillow (PIL) – For image handling and conversion.
•	rembg – For background removal using deep learning-based segmentation.
•	scikit-image – For computing image quality metrics (PSNR, SSIM).
•	scikit-learn – For calculating mIoU (Jaccard index) using classification metrics.

TEAM MEMBERS:
DENZIL FERNANDES-4SO22CD017
ANWYL RYAN SOANS-4S022CD008
ADEN RYAN DSOUZA-4SO22CD001

