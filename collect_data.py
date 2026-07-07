import cv2
import os
import uuid

def main():
    # Create directories if they don't exist
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/screen', exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Data Collection Started.")
    print("Press 'r' to save as REAL photo.")
    print("Press 's' to save as SCREEN photo.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display instructions on the frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 'r': REAL | 's': SCREEN | 'q': QUIT", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Data Collection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            filename = f"dataset/real/real_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('s'):
            filename = f"dataset/screen/screen_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
