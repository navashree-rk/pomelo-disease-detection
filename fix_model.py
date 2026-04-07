from tensorflow.keras.models import load_model

model = load_model("plant_disease_model.h5", compile=False)

model.save("new_model.h5")

print("Model fixed and saved!")