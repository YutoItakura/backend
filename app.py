from flask import Flask,request,jsonify
from flask_cors import CORS

app=Flask(__name__)
CORS(app)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)