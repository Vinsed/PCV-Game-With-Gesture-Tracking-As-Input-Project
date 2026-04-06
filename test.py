import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    # Increased frame processing area (Full Frame)
    h, w, _ = frame.shape

    # 1. Improved Color Masking (YCrCb for stability)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # 2. Aggressive Cleanup for Full-Frame noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter for hand-sized objects to avoid background noise
        large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
        
        if large_contours:
            cnt = max(large_contours, key=cv2.contourArea)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            # --- PALM CENTER LOGIC (Distance Transform) ---
            # This calculates the distance from every pixel inside the contour to the nearest edge
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # The point with the maximum distance to the edge is the center of the palm
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
            
            # max_loc gives us the (x, y) coordinates of the palm center
            palm_center = max_loc
            radius = int(max_val) # This represents the "thickness" of the palm

            # Draw the palm center
            cv2.circle(frame, palm_center, 10, (0, 0, 255), -1)
            # Draw the inscribed circle (optional, shows the palm area)
            cv2.circle(frame, palm_center, radius, (255, 255, 0), 2)

            cv2.putText(frame, f"Palm: {palm_center}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Full Frame Palm Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()