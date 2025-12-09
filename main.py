from src.encode_faces import generate_encodings
from src.mark_attendance import start_attendance

if __name__ == "__main__":
    print("1. Generate face encodings")
    print("2. Start attendance")
    ch = int(input("Select: "))

    if ch == 1:
        generate_encodings()
    elif ch == 2:
        start_attendance()
