import cv2
import numpy as np
import os
import time
from datetime import datetime

# Create alert folder
os.makedirs("alerts", exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Fire and Smoke Detection Started")
print("Press Q to quit.")

last_alert_time = 0
alert_cooldown = 5


def detect_fire(frame):
    """
    Detect possible fire using color and brightness rules.
    This is NOT an AI model.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire-like color ranges
    lower_fire1 = np.array([0, 120, 150])
    upper_fire1 = np.array([25, 255, 255])

    lower_fire2 = np.array([160, 100, 150])
    upper_fire2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)

    fire_mask = mask1 | mask2

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(
        fire_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    fire_mask = cv2.morphologyEx(
        fire_mask,
        cv2.MORPH_DILATE,
        kernel
    )

    fire_pixels = cv2.countNonZero(fire_mask)
    total_pixels = frame.shape[0] * frame.shape[1]

    fire_percentage = (fire_pixels / total_pixels) * 100

    return fire_percentage, fire_mask


def detect_smoke(frame):
    """
    Detect possible smoke using low-saturation gray regions.
    This is NOT an AI model.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Smoke tends to have relatively low saturation
    lower_smoke = np.array([0, 0, 40])
    upper_smoke = np.array([180, 90, 220])

    smoke_mask = cv2.inRange(
        hsv,
        lower_smoke,
        upper_smoke
    )

    # Remove noise
    kernel = np.ones((7, 7), np.uint8)

    smoke_mask = cv2.morphologyEx(
        smoke_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    smoke_mask = cv2.morphologyEx(
        smoke_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    smoke_pixels = cv2.countNonZero(smoke_mask)
    total_pixels = frame.shape[0] * frame.shape[1]

    smoke_percentage = (smoke_pixels / total_pixels) * 100

    return smoke_percentage, smoke_mask


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (960, 540))

    # Detect fire
    fire_percentage, fire_mask = detect_fire(frame)

    # Detect smoke
    smoke_percentage, smoke_mask = detect_smoke(frame)

    # Default status
    status = "NORMAL"
    alert_type = ""

    # Decision rules
    if fire_percentage > 2.0:
        status = "FIRE DETECTED"
        alert_type = "Possible Fire"

    elif smoke_percentage > 25.0:
        status = "SMOKE DETECTED"
        alert_type = "Possible Smoke"

    # Display status
    if status == "NORMAL":

        cv2.putText(
            frame,
            "STATUS: NORMAL",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "WARNING: " + status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        # Alert cooldown
        current_time = time.time()

        if current_time - last_alert_time > alert_cooldown:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = f"alerts/alert_{timestamp}.jpg"

            cv2.imwrite(filename, frame)

            print(
                f"ALERT: {alert_type} | "
                f"Saved: {filename}"
            )

            last_alert_time = current_time

    # Display detection percentages
    cv2.putText(
        frame,
        f"Fire: {fire_percentage:.2f}%",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Smoke: {smoke_percentage:.2f}%",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show output
    cv2.imshow(
        "Fire and Smoke Detection System",
        frame
    )

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
