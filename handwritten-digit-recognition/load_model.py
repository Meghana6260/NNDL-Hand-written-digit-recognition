
import cv2
import numpy as np
import argparse
from tensorflow.keras import models

MODEL_PATH = "tf-cnn-model.h5"  # Path 

def center_and_resize(img, size=28, box_size=20):
    """Crop to digit, pad to square, resize, and center using centroid alignment."""
    ys, xs = np.where(img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    roi = img[y_min:y_max+1, x_min:x_max+1]

    # Pad to square
    h, w = roi.shape
    side = max(h, w)
    pad_vert = (side - h) // 2
    pad_horiz = (side - w) // 2
    roi_padded = np.pad(
        roi,
        ((pad_vert, side - h - pad_vert), (pad_horiz, side - w - pad_horiz)),
        mode='constant',
        constant_values=0
    )

    # Resize to box_size
    roi_small = cv2.resize(roi_padded, (box_size, box_size), interpolation=cv2.INTER_AREA)

    # Place in center
    final = np.zeros((size, size), dtype=np.uint8)
    start = (size - box_size) // 2
    final[start:start+box_size, start:start+box_size] = roi_small

    # Align centroid
    cy, cx = centroid(final)
    shiftx = int(np.round(size/2.0 - cx))
    shifty = int(np.round(size/2.0 - cy))
    final = shift_image(final, shiftx, shifty)
    return final

def centroid(img):
    """Compute centroid of non-zero pixels."""
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return img.shape[0]//2, img.shape[1]//2
    return ys.mean(), xs.mean()

def shift_image(img, sx, sy):
    """Shift image by sx (cols) and sy (rows)."""
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted.astype(np.uint8)

def preprocess_for_model(gray):
    """Preprocess grayscale image to 1x28x28x1 for model input."""
    img = gray.copy().astype(np.uint8)

    # Invert if background is light
    h, w = img.shape
    corners = np.concatenate([
        img[0:10,0:10].ravel(),
        img[0:10,w-10:w].ravel(),
        img[h-10:h,0:10].ravel(),
        img[h-10:h,w-10:w].ravel()
    ])
    if np.mean(corners) > 127:
        img = cv2.bitwise_not(img)

    # Denoise
    img = cv2.GaussianBlur(img, (3,3), 0)

    
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    final = center_and_resize(img, size=28, box_size=20)

    # Normalize and reshape
    final_norm = final.astype('float32') / 255.0
    final_norm = final_norm.reshape(1,28,28,1)
    return final_norm, final  # also return 2D for display

def predict_digit(image_path):
    # Load model
    model = models.load_model(MODEL_PATH, compile=False)
    print("[INFO] Loaded model.")

    # Load image
    orig = cv2.imread(image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Invalid image path or cannot read image.")

    img_input, img_vis = preprocess_for_model(gray)

    # Predict
    preds = model.predict(img_input)
    label = int(np.argmax(preds))
    conf = float(np.max(preds) * 100)

    # Prediction window
    pred_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
    text = f"Prediction: {label} ({conf:.2f}%)"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    x = (pred_window.shape[1] - text_w) // 2
    y = (pred_window.shape[0] + text_h) // 2
    cv2.putText(pred_window, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)

    # Display original and processed side by side
    proc_display = cv2.resize(img_vis, (200,200), interpolation=cv2.INTER_NEAREST)
    orig_disp = cv2.resize(orig, (200,200)) if orig is not None else np.zeros((200,200,3), dtype=np.uint8)
    combined = np.hstack([orig_disp, cv2.cvtColor(proc_display, cv2.COLOR_GRAY2BGR)])
    cv2.imshow("Original (left) | Processed (right)", combined)
    cv2.imshow("Prediction", pred_window)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a handwritten digit image using trained CNN.")
    parser.add_argument("image_path", help="Path to input image")
    args = parser.parse_args()
    result = predict_digit(args.image_path)
    print("Predicted:", result)
