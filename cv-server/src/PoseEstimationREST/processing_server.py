import base64
import threading
from flask import Flask, jsonify, request



app = Flask(__name__)

@app.route('/camera-input', methods=['POST'])
def post_camera_input():
    b64_image = request.data
    try:
        image = base64_to_image(b64_image)
        # processed_image = mp_pose.process(image)

        return jsonify({
            # "data": processed_image.pose_landmarks,
            "msg": "OK",
        })
    except (base64.binascii.Error, UnicodeDecodeError):
        return jsonify({
            "msg": "Error",
            "status": 400
        })
    


# def start_flask_server():
#     app.run(port=5002)

if __name__ == '__main__':
    app.run(port=5002)
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True  
    flask_thread.start()