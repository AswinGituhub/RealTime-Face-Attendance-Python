import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# --- Configuration ---
DATASET_PATH = "data/students"
ATTENDANCE_FILE = "data/attendance/attendance.csv"
MODEL = "VGG-Face"

# --- Utility Function ---
def mark_attendance_csv(name, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # --- DIAGNOSTIC ADDITION START ---
    # This line prints the absolute path of the file being written to.
    full_path = os.path.abspath(file_path)
    print(f"File is being saved at: {full_path}")
    # --- DIAGNOSTIC ADDITION END ---
    
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)

    if name not in df['Name'].values:
        now = datetime.now().strftime("%H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(file_path, index=False)
        print(f"Marked attendance for {name}")

# --- Main Attendance Function ---
def start_attendance():
    
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"Error: Face database not found or empty. Please run encode_faces.py first!")
        return
        
    cap = cv2.VideoCapture(0)
    print("-" * 30)
    print("Starting camera... Look directly at the camera. Press Q to quit.")
    print("-" * 30)

    marked_in_session = set() 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        try:
            results = DeepFace.find(
                img_path=frame, 
                db_path=DATASET_PATH, 
                model_name=MODEL, 
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            for result in results:
                if not result.empty:
                    best_match = result.iloc[0]
                    identity_path = best_match['identity']
                    name = os.path.splitext(os.path.basename(identity_path))[0]
                    distance = best_match['distance']

                    if distance < 0.70: 
                        if name not in marked_in_session:
                            mark_attendance_csv(name, ATTENDANCE_FILE)
                            marked_in_session.add(name)

                        x = int(best_match['source_x'])
                        y = int(best_match['source_y'])
                        w = int(best_match['source_w'])
                        h = int(best_match['source_h'])
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        x = int(best_match['source_x'])
                        y = int(best_match['source_y'])
                        w = int(best_match['source_w'])
                        h = int(best_match['source_h'])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception:
            pass 

        cv2.imshow("Attendance System - DeepFace", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_attendance()