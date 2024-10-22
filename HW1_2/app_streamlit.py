from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)

# 設定靜態資料夾，存放生成的圖表
app.config['UPLOAD_FOLDER'] = 'static'

# 路由：主頁，顯示上傳 CSV 的表單
@app.route('/')
def index():
    return render_template('index.html')

# 路由：處理上傳的 CSV，進行預測並生成圖表
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # 讀取上傳的 CSV 檔案
        data = pd.read_csv(file)
        
        # 資料處理
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data['y'] = data['y'].str.replace(',', '').astype(float)
        data = data.rename(columns={'Date': 'ds'})
        
        # Prophet 模型訓練
        model = Prophet(changepoint_prior_scale=0.5, interval_width=0.95)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(data)
        
        # 預測未來 60 天
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        # 繪製預測圖表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['ds'], data['y'], 'k-', label='Actual Data')
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.4)
        historical_avg = data['y'].mean()
        ax.axhline(y=historical_avg, color='gray', linestyle='--', label='Historical Average')

        # 添加紅色虛線標記預測的初始化時間點
        forecast_init_date = data['ds'].max()
        ax.axvline(x=forecast_init_date, color='red', linestyle='--')
        ax.text(forecast_init_date, ax.get_ylim()[1], 'Forecast Initialization', color='red', ha='right')

        # 儲存圖表到靜態資料夾
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'forecast_plot.png')
        plt.savefig(image_path)
        plt.close()

        return render_template('result.html', image_path=image_path)

if __name__ == '__main__':
    # 啟動 Flask 應用
    app.run(debug=True)
