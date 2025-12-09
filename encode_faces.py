import os
from deepface import DeepFace

DATASET_PATH = "data/students"

def generate_encodings():
    print("Starting face encoding process...")
    
    # Check if student directory exists and has files
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"Error: The directory '{DATASET_PATH}' is empty or does not exist.")
        print("Please place student images inside this folder.")
        return

    try:
        # Check for CUDA library warnings (expected if you don't have a GPU)
        print("DeepFace is building the face database. This may take a minute...")
        
        # We search one of the images against the entire folder to trigger database creation.
        # This will download VGG-Face weights and create the representations file.
        first_image = os.path.join(DATASET_PATH, os.listdir(DATASET_PATH)[0])
        
        DeepFace.find(
            img_path=first_image, 
            db_path=DATASET_PATH, 
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        print("-" * 30)
        print(f"✅ Encoding complete. Face database built in: {DATASET_PATH}")
        print("Ready to start attendance.")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred during encoding: {e}")

if __name__ == "__main__":
    # --- The two original os.makedirs lines are now removed/commented out ---
    # os.makedirs(DATASET_PATH, exist_ok=True)
    # os.makedirs("data/encodings", exist_ok=True)
    generate_encodings()