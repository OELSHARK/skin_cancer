from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# تحميل النموذج المدرب
model = load_model("model.h5")

# دالة للتنبؤ بالصورة
def predict_skin_class(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    
    class_names = ["benign", "malignant"]
    return class_names[np.argmax(pred)], float(np.max(pred))

# الراوت الأساسي
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    confidence = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            upload_folder = os.path.join("static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            predicted_class, confidence_val = predict_skin_class(filepath)
            result = predicted_class
            confidence = f"{confidence_val * 100:.2f}%"
            image_path = filepath

    return render_template("index.html", result=result, confidence=confidence, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
