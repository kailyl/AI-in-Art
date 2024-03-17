from deepface import DeepFace

result = DeepFace.analyze(img_path = "doctor-test.jpg", actions = ["age", "gender", "emotion", "race"])
print(result)

