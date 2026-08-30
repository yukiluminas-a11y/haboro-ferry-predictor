import os
import sqlite3
import requests
import pandas as pd
from datetime import datetime

# --- 設定 ---
DB_PATH = "ferry_data.sqlite"
LATITUDE = 44.38   # 羽幌〜焼尻・天売エリアの緯度
LONGITUDE = 141.70 # 経度

def setup_database():
    """データベースとテーブルの初期化"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ferry_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            status TEXT,
            wind_speed REAL,
            wave_height REAL,
            weathercode INTEGER,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def fetch_weather_data():
    """Open-Meteoから気象データを取得（日本時間指定）"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["wind_speed_10m", "wave_height"],
        "timezone": "Asia/Tokyo"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def calculate_risk(row):
    """【フェリー専用判定ロジック】"""
    wave = row["wave_height"]
    wind = row["wind_speed"]
    
    if pd.isna(wave) or pd.isna(wind):
        return "データなし", "gray", 0

    score = (wave * 25) + (wind * 3.5)
    
    if wave >= 2.5 or wind >= 13 or score >= 65:
        return "欠航警戒（赤）", "red", min(100, int(score))
    elif wave >= 1.8 or wind >= 10 or score >= 45:
        return "運航注意（黄）", "yellow", int(score)
    else:
        return "通常運航（緑）", "green", max(0, int(score))

def process_forecast_data(data):
    """公式ダイヤ（第1便: 08:30-12:10 / 第2便: 14:00-17:35）に合わせた気象データ抽出"""
    df_hourly = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "wave_height": data["hourly"]["wave_height"]
    })
    
    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")
    df_hourly["hour"] = df_hourly["datetime"].dt.hour
    
    # 【公式ダイヤ適用】
    # 第1便 (8時〜12時) および 第2便 (14時〜18時) の航行時間帯のみを抽出
    df_operating = df_hourly[
        ((df_hourly["hour"] >= 8) & (df_hourly["hour"] <= 12)) |
        ((df_hourly["hour"] >= 14) & (df_hourly["hour"] <= 18))
    ]
    
    daily_summary = df_operating.groupby("date").agg({
        "wind_speed": "max",
        "wave_height": "max"
    }).reset_index()
    
    daily_summary[["status", "color", "probability"]] = daily_summary.apply(
        calculate_risk, axis=1, result_type="expand"
    )
    
    return daily_summary

def get_past_records_html():
    """データベースから直近の実績を取得してHTML化"""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date AS 日付, status AS 運航状況, 
               wind_speed AS '風速(m/s)', wave_height AS '波高(m)'
        FROM ferry_records 
        ORDER BY date DESC LIMIT 10
    """
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            html = "<p>過去の実績データはまだ蓄積されていません。</p>"
        else:
            html = df.to_html(index=False, classes="past-records", border=1, justify="center")
    except Exception as e:
        html = f"<p>データ取得エラー: {e}</p>"
    conn.close()
    return html

def save_today_record(daily_summary):
    """今日のフェリー運航データをデータベースに記録"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_data = daily_summary[daily_summary["date"] == today_str]
    
    if not today_data.empty:
        row = today_data.iloc[0]
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ferry_records (date, status, wind_speed, wave_height, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (today_str, row["status"], row["wind_speed"], row["wave_height"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

def generate_html(daily_summary):
    """Web公開用の index.html を生成"""
    forecast_rows = ""
    for _, row in daily_summary.iterrows():
        date = row["date"]
        wave = f'{row["wave_height"]:.1f}' if not pd.isna(row["wave_height"]) else "-"
        wind = f'{row["wind_speed"]:.1f}' if not pd.isna(row["wind_speed"]) else "-"
        status = row["status"]
        color = row["color"]
        prob = row["probability"]
        
        bg_color = {"green": "#e6ffe6", "yellow": "#ffffe6", "red": "#ffe6e6", "gray": "#f0f0f0"}.get(color, "#fff")
        font_color = {"green": "green", "yellow": "#b38f00", "red": "red", "gray": "gray"}.get(color, "black")
        
        forecast_rows += f"""
        <tr style="background-color: {bg_color};">
            <td>{date}</td>
            <td>{wave}</td>
            <td>{wind}</td>
            <td>{prob}%</td>
            <td style="color: {font_color}; font-weight: bold;">{status}</td>
        </tr>
        """
        
    past_records_html = get_past_records_html()
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>羽幌沿海フェリー 欠航予測ダッシュボード</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
            h1, h2 {{ color: #2c3e50; }}
            .notice {{ background-color: #eef6ff; padding: 10px 15px; border-left: 5px solid #0066cc; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; max-width: 800px; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ccc; padding: 10px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
            .past-records {{ width: 100%; max-width: 800px; }}
        </style>
    </head>
    <body>
        <h1>羽幌沿海フェリー 欠航予測ダッシュボード</h1>
        <div class="notice">
            <strong>【9月ダイヤ対応】</strong> フェリー（おろろん2）運航ダイヤ（第1便: 08:30〜12:10 / 第2便: 14:00〜17:35）に基づき航行時間帯限定で気象判定を行っています。
        </div>
        <p>最終更新: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (JST)</p>
        
        <h2>フェリー週間欠航予測 (公式ダイヤ航行時間帯基準)</h2>
        <table>
            <tr>
                <th>日付</th>
                <th>予測波高 (m)</th>
                <th>予測風速 (m/s)</th>
                <th>欠航確率</th>
                <th>判定</th>
            </tr>
            {forecast_rows}
        </table>
        
        <h2>過去の運航・欠航実績 (直近10件)</h2>
        {past_records_html}
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    print("データベース初期化中...")
    setup_database()
    
    print("気象データ取得中 (Open-Meteo)...")
    weather_data = fetch_weather_data()
    
    print("公式ダイヤに合わせた精度計算中...")
    daily_summary = process_forecast_data(weather_data)
    
    print("本日の実績データをDBへ保存中...")
    save_today_record(daily_summary)
    
    print("Webページ (index.html) 生成中...")
    generate_html(daily_summary)
    
    print("全処理が完了しました！")

if __name__ == "__main__":
    main()
