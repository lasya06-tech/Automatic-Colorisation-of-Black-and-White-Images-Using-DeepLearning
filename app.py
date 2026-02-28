import os
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image

# Import official colorizers
from colorizers import *

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ----------------------------
# Load Pretrained Model
# ----------------------------
# You can change to siggraph17 for better quality
colorizer = eccv16(pretrained=True).eval()
# colorizer = siggraph17(pretrained=True).eval()

# ----------------------------
# Route
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files["image"]

        if file:

            # Save uploaded image
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(upload_path)

            # Open image correctly
            img = Image.open(upload_path).convert("RGB")


            img = np.array(img)

            # Preprocess for model
            tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

            # Run model
            with torch.no_grad():
                out_ab = colorizer(tens_l_rs)

            # Postprocess output
            out_img = postprocess_tens(tens_l_orig, out_ab)


            out_img = (out_img * 255).clip(0, 255).astype("uint8")

            # Save result
            result_path = os.path.join(app.config["RESULT_FOLDER"], "result.png")
            Image.fromarray(out_img).save(result_path)

            return render_template(
                "index.html",
                uploaded_image=upload_path,
                result_image=result_path,
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)