# detector.py
import cv2
import os

# import your model loading and detection helper(s)
# e.g. from your_module import load_svm_model, detect_boxes_in_frame

# -- model init (load once) --
# model = load_svm_model("svm_model.joblib")
# hog = init_hog()   # if you have an init function

def detect_vehicles_in_video(input_path: str, output_path: str = "outputs/out.mp4",
                             resize_to: tuple = None, frame_step: int = 1) -> dict:
    """
    Processes input_path, writes annotated video to output_path.
    Returns stats dict: {"total_vehicles": int}
    - resize_to: optional (w, h) to resize frames before detection (faster)
    - frame_step: process every nth frame (skip frames to speed up)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_to:
        out_size = resize_to
    else:
        out_size = (width, height)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    total_vehicles = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_step > 1 and (frame_idx % frame_step != 0):
            # still write original frame (or skip writing if you prefer)
            writer.write(cv2.resize(frame, out_size))
            continue

        frame_proc = (cv2.resize(frame, resize_to) if resize_to else frame.copy())

        # === CALL YOUR DETECTION LOGIC HERE ===
        # Example contract: detect_boxes_in_frame returns list of (x,y,w,h)
        # boxes = detect_boxes_in_frame(frame_proc, model, hog)
        # For now put a placeholder:
        # boxes = []  # REPLACE with actual detection
        # =======================================

        # draw boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_proc, (x, y), (x + w, y + h), (0, 255, 0), 2)

        total_vehicles += len(boxes)

        # ensure output frame size matches writer
        out_frame = cv2.resize(frame_proc, out_size)
        writer.write(out_frame)

    cap.release()
    writer.release()
    return {"total_vehicles": total_vehicles}
