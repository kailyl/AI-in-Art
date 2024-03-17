import cv2
from deepface import DeepFace
import os
import time

# Delay between each face analysis (in seconds)
ANALYSIS_DELAY = 15

def analyze_face(img_path):
    # Analyze the face in the given image
    # result = DeepFace.analyze(img_path=img_path, actions=["gender", "race"])
    # return [result[0]['race'], result[0]['dominant_gender']]
    try:
        result = DeepFace.analyze(img_path=img_path, actions=["race"])
        return [result[0]['race'], -1]
    except ValueError as e: 
        print(f"Error analyzing face: {e}")
        return None
    
def face_analysis_worker(face_img):
    # Save the cropped face image temporarily
    cv2.imwrite('temp_face.jpg', face_img)

    # Analyze the cropped face image
    result = analyze_face('temp_face.jpg')

    # Remove the temporary face image file
    os.remove('temp_face.jpg')

    return result

def calculate_similarity(occupation_result, face_result):
    min_difference = float('inf')

    for occupation_race, occupation_percentage in occupation_result['race_distribution'].items():
        difference = abs(occupation_percentage - face_result[occupation_race])
        if difference < min_difference:
            min_difference = difference

    return min_difference

def detect_and_analyze_faces():
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    last_analysis_time = time.time()
    first_analysis_done = False

    # Example face analysis results for occupations
    cashier_result = {
        'race_distribution': {'asian': 3.96, 'indian': 2.07, 'black': 0.20, 'white': 56.20, 'middle eastern': 17.17, 'latino hispanic': 20.40}
    }

    ceo_result = {
        'race_distribution': {'asian': 8.83, 'indian': 0, 'black': 0, 'white': 99.36, 'middle eastern': 0.19, 'latino hispanic': 0.45}
    }

    criminal_result = {
        'race_distribution': {'asian': 16.51, 'indian': 21.61, 'black': 5.43, 'white': 11.58, 'middle eastern': 8.39, 'latino hispanic': 36.48}
    }

    doctor_result = {
        'race_distribution': {'asian': 0.50, 'indian': 0.89, 'black': 0.03, 'white': 70.24, 'middle eastern': 18.71, 'latino hispanic': 9.63}
    }

    # List of occupations and their corresponding face analysis results
    occupations = {
        'Cashier': cashier_result,
        'Criminal': criminal_result,
        'Doctor': doctor_result,
        'CEO': ceo_result,
    }

    predicted_occupation = "calculating..."  # Initialize predicted occupation

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300))

        # Only perform first face analysis once
        if not first_analysis_done and len(faces) > 0:
            x, y, w, h = faces[0]
            # Crop the face region from the frame
            face_img = frame[y:y+h, x:x+w]
            first_analysis_result = face_analysis_worker(face_img)
            first_analysis_done = True

        # Analyze the cropped face image if enough time has passed since the last analysis
        current_time = time.time()
        if current_time - last_analysis_time >= ANALYSIS_DELAY:
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Crop the face region from the frame
                face_img = frame[y:y+h, x:x+w]
                face_result = face_analysis_worker(face_img)

                if face_result == None:
                    predicted_occupation = "calculating..."
                else:
                    # Find the most similar-looking occupation
                    min_difference = float('inf')
                    most_similar_occupation = None

                    for occupation, occupation_result in occupations.items():
                        difference = calculate_similarity(occupation_result, face_result[0])
                        if difference < min_difference:
                            min_difference = difference
                            most_similar_occupation = occupation

                    predicted_occupation = most_similar_occupation
                
            last_analysis_time = current_time

        # Draw a rectangle around each face and display the resulting frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)  # Make the box white
            # Draw the predicted occupation next to the face box in white color
            cv2.putText(frame, f'Occupation: {predicted_occupation}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Add padding to the face region
            padding = 200
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding
            
            # Ensure the coordinates are within the frame bounds
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, frame.shape[1])
            h = min(h, frame.shape[0])
            
        # Display the resulting frame
        cv2.imshow('Midterm Project - Kaily Liu', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_analyze_faces()
