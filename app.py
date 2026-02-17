import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dummy demo results
demo_results = [
    {"vehicle_id": 1, "speed": 65, "plate": "KA01AB1234"},
    {"vehicle_id": 2, "speed": 72, "plate": "KA05XY9876"},
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return redirect("/")

    file = request.files["video"]

    if file.filename == "":
        return redirect("/")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    return render_template("result.html", results=demo_results, video=file.filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
