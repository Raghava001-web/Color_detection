import cv2
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import webcolors

def get_dominant_color(image, mask, k=1):
    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    if len(pixels) == 0:
        return None
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    dominant = kmeans.cluster_centers_[0].astype(int)
    return tuple(dominant)

def closest_color_name(requested_color):
    min_distance = float("inf")
    closest_name = "Unknown"
    try:
        return webcolors.rgb_to_name(requested_color)
    except ValueError:
        for name in webcolors.names():
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            distance = (r_c - requested_color[0])**2 + (g_c - requested_color[1])**2 + (b_c - requested_color[2])**2
            if distance < min_distance:
                min_distance = distance
                closest_name = name
    return closest_name

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)
report_file = open("color_report.txt", "w")
report_file.write("Color Detection Report\n")
report_file.write("======================\n\n")

print("Show an object in your hand. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    dom_color = get_dominant_color(frame, fgmask, k=1)
    if dom_color is not None:
        rgb_color = (dom_color[2], dom_color[1], dom_color[0])
        color_name = closest_color_name(rgb_color)
        color_bgr = tuple(map(int, dom_color))
        cv2.rectangle(frame, (50, 50), (150, 150), color_bgr, -1)
        cv2.putText(frame, f"Color: {color_name}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected: {color_name}")
        report_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {color_name}\n")
    else:
        cv2.putText(frame, "You are not holding anything", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] You are not holding anything")
        report_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] Nothing detected\n")
    cv2.imshow("Webcam (with background)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
report_file.close()
print("\nReport saved as 'color_report.txt'")
