import cv2
import numpy as np
import os
import glob

POSITIVE_TRAINING_VIDEO = "C:/Users/HP/PycharmProjects/SVM_Vehicle_Detection/vehicles.mp4"
NEGATIVE_TRAINING_VIDEO = "C:/Users/HP/PycharmProjects/SVM_Vehicle_Detection/empty_road.mp4"
POSITIVE_TRAINING_SET_PATH = "C:/Users/HP/PycharmProjects/SVM_Vehicle_Detection/DATASET/POSITIVE"
NEGATIVE_TRAINING_SET_PATH = "C:/Users/HP/PycharmProjects/SVM_Vehicle_Detection/DATASET/NEGATIVE"
TRAFFIC_VIDEO_FILE = r"C:\Users\HP\PycharmProjects\SVM_Vehicle_Detection\video.mp4"
TRAINED_SVM = r"C:\Users\HP\PycharmProjects\SVM_Vehicle_Detection\vehicle_detector.yml"
WINDOW_NAME = "WINDOW"
IMAGE_SIZE = (40, 40)

def model_exists(filepath):
    return os.path.exists(filepath)
    # return False
def file_paths(directory):
    return glob.glob(os.path.join(directory, "*"))

def load_images(directory):
    image_list = []
    for file in file_paths(directory):
        img = cv2.imread(file)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            image_list.append(img)
    return image_list

def hog_calculator(image_list):
    hog = cv2.HOGDescriptor(_winSize=IMAGE_SIZE,
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    gradients = []
    for img in image_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        descriptors = hog.compute(gray)
        gradients.append(descriptors.flatten())
    return gradients

def hog_convertor(gradient_list):
    return np.array(gradient_list, dtype=np.float32)

def train_svm_model(gradient_list, labels):
    print("Training SVM...")
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3))

    train_data = hog_convertor(gradient_list)
    labels = np.array(labels, dtype=np.int32)

    svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)
    svm.save(TRAINED_SVM)
    print("SVM training completed and saved.")
    return svm

def svm_detector(svm):
    support_vectors = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    detector = np.append(support_vectors[0], -rho)
    return detector

def visualise(img, locations, color):
    for loc in locations:
        cv2.rectangle(img, loc, color, 2)

def extract_frames_from_video(video_path, output_folder, max_frames=100, resize_dim=(40, 40)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    extracted = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize_dim)
        save_path = os.path.join(output_folder, f"frame_{extracted}.jpg")
        cv2.imwrite(save_path, frame)
        extracted += 1
        frame_count += 1

    cap.release()
    print(f"{extracted} frames saved to {output_folder}")

def test():
    svm = cv2.ml.SVM_load(TRAINED_SVM)

    hog = cv2.HOGDescriptor(_winSize=IMAGE_SIZE,
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    detector = svm_detector(svm)
    hog.setSVMDetector(detector)

    cap = cv2.VideoCapture(TRAFFIC_VIDEO_FILE)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    num_of_vehicles = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw = frame.copy()
        frame[:, frame.shape[1]//2:] = 0

        rects, _ = hog.detectMultiScale(frame)
        visualise(draw, rects, (0, 255, 0))

        for r in rects:
            center = (r[0] + r[2] // 2, r[1] + r[3] // 2)
            if abs(center[1] - frame.shape[0] * 2 // 3) < 2:
                num_of_vehicles += 1
                cv2.line(draw, (0, frame.shape[0]*2//3), (frame.shape[1]//2, frame.shape[0]*2//3), (0, 255, 0), 3)
            else:
                cv2.line(draw, (0, frame.shape[0]*2//3), (frame.shape[1]//2, frame.shape[0]*2//3), (0, 0, 255), 3)

        cv2.imshow(WINDOW_NAME, draw)
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    if not model_exists(TRAINED_SVM):
        print("Model not found. Extracting frames from videos for training...")

        extract_frames_from_video(POSITIVE_TRAINING_VIDEO, POSITIVE_TRAINING_SET_PATH, max_frames=200)
        extract_frames_from_video(NEGATIVE_TRAINING_VIDEO, NEGATIVE_TRAINING_SET_PATH, max_frames=200)

        pos_images = load_images(POSITIVE_TRAINING_SET_PATH)
        neg_images = load_images(NEGATIVE_TRAINING_SET_PATH)

        pos_gradients = hog_calculator(pos_images)
        neg_gradients = hog_calculator(neg_images)

        gradients = pos_gradients + neg_gradients
        labels = [1]*len(pos_gradients) + [-1]*len(neg_gradients)

        train_svm_model(gradients, labels)

    test()

if __name__ == "__main__":
    main()
