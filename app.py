from flask import Flask,request,jsonify,send_file
from flask_cors import CORS
import os
from ultralytics import YOLO

app=Flask(__name__)
CORS(app)
UPLOAD_FOLDER="uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
model=YOLO('last.pt')
@app.route('/calculate',methods=['POST'])
def calculate():
    try:
        data=request.get_json()
        number=data.get('number')
        if number is None or not isinstance(number, (int, float)):
            return jsonify({'error': 'Invalid number'}), 400

        result = number * number  # 2乗計算
        return jsonify({'result': result})  # 結果を返す
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "ファイルが見つかりません", 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    results=model(file_path,conf=0.5 ,show_conf=False,show_labels=False, stream=True)
    for result in results:
       boxes = result.boxes 
       masks = result.masks 
       keypoints = result.keypoints  
       probs = result.probs 
       obb = result.obb
       disease_detected=False

    for obj in boxes:
        class_id=int(obj.cls)
        class_name=model.names[class_id]
        if class_name=='disease1' or class_name=='disease2' or class_name=='disease3':
            disease_detected=True
            break

    if disease_detected:
        result_filename=os.path.join(UPLOAD_FOLDER,f"result_{os.path.basename(result.path)}")
        return send_file(result_filename,as_attachment=True, mimetype="image/jpeg")  # ファイルを保存
    

    return jsonify({
        "message":"Not Detected",
        "file_path":file_path
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
