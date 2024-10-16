from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import logging
import threading

app = Flask(__name__)

# ตั้งค่าการบันทึก
logging.basicConfig(level=logging.INFO)

# กำหนดตัวแปรสำหรับผลลัพธ์ประวัติและการติดตาม
historical_results = []
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
    if len(historical_results) >= 3:
        for i in range(2, len(historical_results)):
            last_two = [result_to_numeric(x) for x in historical_results[i-2:i]]
            data.append(last_two)
            labels.append(result_to_numeric(historical_results[i]))
        model.fit(np.array(data), np.array(labels))
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
    prediction, probabilities = "N/A", [0, 0, 0]
    if len(historical_results) >= 6:
        try:
            prediction, probabilities = predict_next(historical_results)
        except ValueError:
            prediction = "ต้องการผลลัพธ์อย่างน้อย 6 รายการเพื่อทำการคาดการณ์."
    probabilities = [round(prob, 2) for prob in probabilities]

    return render_template('index.html', prediction=prediction, probabilities=probabilities, historical_results=historical_results)

@app.route('/submit', methods=['POST'])
def submit_result():
    actual_result = request.form['result'].strip().capitalize()
    if actual_result not in ["Banker", "Player", "Tie"]:
        return "ข้อมูลไม่ถูกต้อง กรุณาใส่ Banker, Player, หรือ Tie.", 400

    historical_results.append(actual_result)
    train_model()

    # ใช้ล็อกในการบันทึกข้อมูลในไฟล์ CSV
    threading.Thread(target=write_to_csv, args=(actual_result,)).start()

    return redirect(url_for('index'))

def write_to_csv(actual_result):
    if len(historical_results) > 1:
        previous_result = historical_results[-2]
        try:
            with open('historical_results.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([previous_result, actual_result])
                logging.info(f"บันทึกผลลัพธ์: {previous_result}, {actual_result}")
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์ CSV: {e}")

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global historical_results
    historical_results.clear()
    open('historical_results.csv', 'w').close()
    return redirect(url_for('index'))

if __name__ == "__main__":
    try:
        model = load('baccarat_model.joblib')
    except Exception as e:
        logging.warning(f"ไม่สามารถโหลดโมเดล: {e}")

    try:
        with open('historical_results.csv', mode='r') as file:
            reader = csv.reader(file)
            historical_results = [row[1] for row in reader if row]
    except FileNotFoundError:
        logging.warning("ไม่พบไฟล์ historical_results.csv, จะเริ่มต้นด้วยประวัติผลลัพธ์ว่าง")

    app.run(host="0.0.0.0", port=10000, debug=True)
