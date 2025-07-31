import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, # True for images, False for video streams
    model_complexity=1,      # 0, 1, or 2. Higher complexity means more accurate but slower.
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def capture_and_preprocess_frame(camera_index=0, target_width=640, target_height=480):
    """
    Captures a frame from the camera and applies basic preprocessing.

    Args:
        camera_index (int): The index of the camera to use (e.g., 0 for default webcam).
        target_width (int): Desired width of the preprocessed frame.
        target_height (int): Desired height of the preprocessed frame.

    Returns:
        numpy.ndarray: The preprocessed frame, or None if frame capture fails.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame from camera.")
        return None

    frame = cv2.resize(frame, (target_width, target_height))
    return frame

def estimate_pose(image):
    """
    Estimates human pose from an image using MediaPipe Pose.

    Args:
        image (numpy.ndarray): The input image (BGR format).

    Returns:
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None: Detected pose landmarks,
        or None if no pose is detected.
    """
    # Convert the BGR image to RGB before processing.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        return results.pose_landmarks
    return None

def draw_landmarks(image, landmarks):
    """
    Draws MediaPipe pose landmarks on the image.

    Args:
        image (numpy.ndarray): The image to draw on.
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The pose landmarks.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
    )

if __name__ == '__main__':

    cap = cv2.VideoCapture(0) #0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            display_frame = cv2.resize(frame, (640, 480))
            display_frame = cv2.flip(display_frame, 1)
            
            landmarks = estimate_pose(display_frame)

            if landmarks:
                draw_landmarks(display_frame, landmarks)

            cv2.imshow('MediaPipe Pose', display_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    pose.close()


