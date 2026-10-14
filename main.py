import cv2
import mediapipe as mp
import pyautogui
from screeninfo import get_monitors
import time

# Get screen resolution
monitors = get_monitors()
virtual_screen_width = sum([monitor.width for monitor in monitors])
virtual_screen_height = max([monitor.height for monitor in monitors])

# Initialize Mediapipe and OpenCV
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hand_detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

prev_index_x, prev_index_y = 0, 0
smooth_factor = 0.5
dead_zone = 10
last_click_time = 0  # For debounce

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)

    if output.multi_hand_landmarks:
        for hand in output.multi_hand_landmarks:
            landmarks = hand.landmark

            # Index finger tip
            index_finger_tip = landmarks[8]
            index_x = int(index_finger_tip.x * frame.shape[1])
            index_y = int(index_finger_tip.y * frame.shape[0])

            # Middle finger tip
            middle_finger_tip = landmarks[12]
            middle_x = int(middle_finger_tip.x * frame.shape[1])
            middle_y = int(middle_finger_tip.y * frame.shape[0])

            # Map to screen space
            screen_x = int(index_x * (virtual_screen_width / frame.shape[1]))
            screen_y = int(index_y * (virtual_screen_height / frame.shape[0]))

            # Smooth mouse movement
            smoothed_x = smooth_factor * prev_index_x + (1 - smooth_factor) * screen_x
            smoothed_y = smooth_factor * prev_index_y + (1 - smooth_factor) * screen_y

            # Move mouse
            if abs(smoothed_x - prev_index_x) > dead_zone or abs(smoothed_y - prev_index_y) > dead_zone:
                pyautogui.moveTo(smoothed_x, smoothed_y)
                prev_index_x, prev_index_y = smoothed_x, smoothed_y

            # Click detection using index and middle fingers
            distance = ((index_x - middle_x)**2 + (index_y - middle_y)**2)**0.5
            current_time = time.time()

            if distance < 30 and (current_time - last_click_time > 0.5):  # Threshold and debounce
                pyautogui.click()
                last_click_time = current_time  # Update last click time

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
