import cv2
import pyautogui
pyautogui.FAILSAFE = False
# Constants for gesture recognition
gesture_open_palm_area = 4000
gesture_close_fist_area = 1000

# Initialize variables for hand gesture
hand_detected = False

# Start capturing video from your webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a hand detection model or a simple thresholding technique
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort the contours by area, and choose the largest
        max_contour = max(contours, key=cv2.contourArea)

        # Check if the area of the contour is greater than the gesture threshold
        if cv2.contourArea(max_contour) > gesture_open_palm_area:
            hand_detected = True
        else:
            hand_detected = False

        # Perform actions based on the detected hand gesture
        if hand_detected:
            # Move the mouse cursor with hand movement
            x, y = max_contour[0][0]
            screen_width, screen_height = pyautogui.size()
            x = x * screen_width // 640  # Adjust for screen resolution
            y = y * screen_height // 480  # Adjust for screen resolution
            pyautogui.moveTo(x, y)

            # Detect a fist gesture and perform a left mouse click
            if cv2.contourArea(max_contour) < gesture_close_fist_area:
                pyautogui.click()

    # Display the video feed
    cv2.imshow("Hand Gesture Control", frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
