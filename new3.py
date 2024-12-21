import cv2
import numpy as np
from collections import deque

# Parameters
MAX_TRACK_LENGTH = 20  # Maximum number of trajectory points to store
SPEED_THRESHOLD = 2.0  # Speed threshold to classify behavior

# Object Tracking Class
class ObjectTracker:
    def __init__(self):  # Correct constructor definition
        self.tracks = {}
        self.track_id = 0

    def update_tracks(self, frame, detections):
        updated_tracks = {}

        for detection in detections:
            x, y, w, h = detection
            center = (int(x + w / 2), int(y + h / 2))

            # Match detection to existing track
            matched = False
            for track_id, track in self.tracks.items():
                if cv2.norm(center, track[-1]) < 50:  # Distance threshold
                    track.append(center)
                    if len(track) > MAX_TRACK_LENGTH:
                        track.pop(0)
                    updated_tracks[track_id] = track
                    matched = True
                    break

            # If no match, create a new track
            if not matched:
                updated_tracks[self.track_id] = deque([center], maxlen=MAX_TRACK_LENGTH)
                self.track_id += 1

        self.tracks = updated_tracks

    def draw_tracks(self, frame):
        for track_id, track in self.tracks.items():
            for i in range(1, len(track)):
                cv2.line(frame, track[i - 1], track[i], (0, 255, 0), 2)

            # Label track with behavior classification
            if len(track) >= 2:
                speed = cv2.norm(np.array(track[-1]) - np.array(track[-2]))
                behavior = "Walking" if speed < SPEED_THRESHOLD else "Running"
                cv2.putText(frame, behavior, track[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# Object Detection
def detect_objects(frame, background_subtractor):
    mask = background_subtractor.apply(frame)
    mask = cv2.medianBlur(mask, 5)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            detections.append((x, y, w, h))
    return detections

# Main Function
def main(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Initialize tracker and background subtractor
    tracker = ObjectTracker()
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect moving objects
        detections = detect_objects(frame, background_subtractor)

        # Update and draw tracks
        tracker.update_tracks(frame, detections)
        tracker.draw_tracks(frame)

        # Draw detections
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Write frame to output video
        out.write(frame)

        # Display the frame
        cv2.imshow("Dynamic Object Tracking and Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_video}")

# Run the Program
if __name__ == "__main__":
    main("dynamic_ob_tracking.mp4", "output_video.mp4")
