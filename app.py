from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import logging

app = Flask(__name__)

# ตั้งค่าการบันทึก
logging.basicConfig(level=logging.INFO)

# กำหนดตัวแปรสำหรับผลลัพธ์ประวัติและการติดตาม
historical_results = []  # เริ่มต้นด้วยผลลัพธ์ว่าง
data = []
labels = []

# เริ่มต้นโมเดล
model = RandomForestClassifier()

# ฟังก์ชันสำหรับแปลงผลลัพธ์เป็นค่าตัวเลข
def result_to_numeric(result):
    if result == "Player":
        return 1
    elif result == "Banker":
        return 0
    return 2  # Tie

# ฟังก์ชันสำหรับฝึกโมเดลการคาดการณ์ด้วยผลลัพธ์ประวัติ
def train_model():
    global data, labels
    data = []
    labels = []
    if len(historical_results) >= 3:  # ต้องมีผลลัพธ์อย่างน้อย 3 รายการเพื่อฝึก
        for i in range(2, len(historical_results)):
            last_two = [result_to_numeric(x) for x in historical_results[i-2:i]]
            data.append(last_two)
            labels.append(result_to_numeric(historical_results[i]))
        model.fit(np.array(data), np.array(labels))  # ฝึกโมเดล
        # บันทึกโมเดล
        dump(model, 'baccarat_model.joblib')

# ฟังก์ชันสำหรับการคาดการณ์
def predict_next(results):
    if len(results) < 6:
        raise ValueError("ต้องการผลลัพธ์อย่างน้อย 6 รายการเพื่อทำการคาดการณ์.")
    last_two = [result_to_numeric(x) for x in results[-2:]]
    probabilities = model.predict_proba([last_two])[0]
    prediction = model.predict([last_two])[0]
    return ["Banker", "Player", "Tie"][prediction], probabilities * 100

@app.route('/')
def index():
    prediction, probabilities = "N/A", [0, 0, 0]  # ค่าปริยาย
    if len(historical_results) >= 6:  # คาดการณ์เฉพาะเมื่อมีข้อมูลเพียงพอ
        try:
            prediction, probabilities = predict_next(historical_results)
        except ValueError:
            prediction = "ต้องการผลลัพธ์อย่างน้อย 6 รายการเพื่อทำการคาดการณ์."
    
    # จัดรูปแบบความน่าจะเป็นเป็นทศนิยม 2 ตำแหน่ง
    probabilities = [round(prob, 2) for prob in probabilities]

    return render_template('index.html', prediction=prediction, probabilities=probabilities, historical_results=historical_results)

@app.route('/submit', methods=['POST'])
def submit_result():
    actual_result = request.form['result'].strip().capitalize()
    if actual_result not in ["Banker", "Player", "Tie"]:
        return "ข้อมูลไม่ถูกต้อง กรุณาใส่ Banker, Player, หรือ Tie.", 400

    historical_results.append(actual_result)
    train_model()  # ฝึกโมเดลใหม่ด้วยผลลัพธ์ประวัติที่อัปเดต

    # บันทึกผลลัพธ์ลงในไฟล์ CSV ในรูปแบบคู่
    if len(historical_results) > 1:
        previous_result = historical_results[-2]  # ผลลัพธ์ก่อนหน้า
        with open('historical_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([previous_result, actual_result])  # บันทึกผลลัพธ์เป็นคู่

    return redirect(url_for('index'))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global historical_results
    historical_results.clear()  # ลบประวัติ
    return redirect(url_for('index'))  # กลับไปยังหน้าหลัก

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

    # ฝึกโมเดลโดยใช้ข้อมูลที่มีอยู่ในไฟล์ CSV
    try:
        model = load('baccarat_model.joblib')
    except Exception as e:
        logging.warning(f"ไม่สามารถโหลดโมเดล: {e}")

    # อ่านผลลัพธ์จากไฟล์ CSV สำหรับการคาดการณ์
    historical_results_from_csv = []
    try:
        with open('historical_results.csv', mode='r') as file:
            reader = csv.reader(file)
            historical_results_from_csv = [row[1] for row in reader if row]  # อ่านผลลัพธ์ทั้งหมด
    except FileNotFoundError:
        logging.warning("ไม่พบไฟล์ historical_results.csv, จะเริ่มต้นด้วยประวัติผลลัพธ์ว่าง")

    # ใช้ข้อมูลจาก historical_results_from_csv สำหรับการคาดการณ์
    historical_results.extend(historical_results_from_csv)  # เริ่มต้นด้วยข้อมูลจาก CSV

    app.run(debug=True)
