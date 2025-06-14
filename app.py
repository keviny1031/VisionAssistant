from flask import Flask, jsonify, render_template, request, Response
from detect import get_detections, generate_frames

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect")
def detect():
    data = get_detections()
    return jsonify(data)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)