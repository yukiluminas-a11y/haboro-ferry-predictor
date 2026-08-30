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

def calculate_flight_risk(wave, wind):
    """便ごとの波高・風速から欠航リスクを判定"""
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
    """1便（8-12時）と2便（14-18時）を分離して個別に計算"""
    df_hourly = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "wave_height": data["hourly"]["wave_height"]
    })
    
    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")
    df_hourly["hour"] = df_hourly["datetime"].dt.hour
    
    results = []
    dates = df_hourly["date"].unique()
    
    for d in dates:
        df_day = df_hourly[df_hourly["date"] == d]
        
        # 第1便 (08:00 〜 12:00)
        df_flight1 = df_day[(df_day["hour"] >= 8) & (df_day["hour"] <= 12)]
        if not df_flight1.empty:
            wave1 = df_flight1["wave_height"].max()
            wind1 = df_flight1["wind_speed_10m"].max() if "wind_speed_10m" in df_flight1 else df_flight1["wind_speed"].max()
            status1, color1, prob1 = calculate_flight_risk(wave1, wind1)
        else:
            wave1, wind1, status1, color1, prob1 = None, None, "データなし", "gray", 0

        # 第2便 (14:00 〜 18:00)
        df_flight2 = df_day[(df_day["hour"] >= 14) & (df_day["hour"] <= 18)]
        if not df_flight2.empty:
            wave2 = df_flight2["wave_height"].max()
            wind2 = df_flight2["wind_speed_10m"].max() if "wind_speed_10m" in df_flight2 else df_flight2["wind_speed"].max()
            status2, color2, prob2 = calculate_flight_risk(wave2, wind2)
        else:
            wave2, wind2, status2, color2, prob2 = None, None, "データなし", "gray", 0
            
        results.append({
            "date": d,
            "wave1": wave1, "wind1": wind1, "status1": status1, "color1": color1, "prob1": prob1,
            "wave2": wave2, "wind2": wind2, "status2": status2, "color2": color2, "prob2": prob2
        })
        
    return pd.DataFrame(results)

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

def save_today_record(df_summary):
    """今日のデータをデータベースに記録"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_data = df_summary[df_summary["date"] == today_str]
    
    if not today_data.empty:
        row = today_data.iloc[0]
        # 日全体の代表値（最大波高・最大風速・厳しい方の判定）
        max_wave = max(filter(None, [row["wave1"], row["wave2"]]), default=0)
        max_wind = max(filter(None, [row["wind1"], row["wind2"]]), default=0)
        status_text = f"1便:{row['status1']} / 2便:{row['status2']}"
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ferry_records (date, status, wind_speed, wave_height, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (today_str, status_text, max_wind, max_wave, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

def generate_html(df_summary):
    """1便・2便を併記したWebダッシュボード（index.html）を出力"""
    forecast_rows = ""
    for _, row in df_summary.iterrows():
        date = row["date"]
        
        # 1便データ整形
        w1 = f'{row["wave1"]:.1f}m / {row["wind1"]:.1f}m/s' if row["wave1"] is not None else "-"
        s1 = f'<span style="color:{row["color1"]}; font-weight:bold;">{row["status1"]} ({row["prob1"]}%)</span>'
        
        # 2便データ整形
        w2 = f'{row["wave2"]:.1f}m / {row["wind2"]:.1f}m/s' if row["wave2"] is not None else "-"
        s2 = f'<span style="color:{row["color2"]}; font-weight:bold;">{row["status2"]} ({row["prob2"]}%)</span>'
        
        forecast_rows += f"""
        <tr>
            <td>{date}</td>
            <td>{w1}</td>
            <td>{s1}</td>
            <td>{w2}</td>
            <td>{s2}</td>
        </tr>
        """
        
    past_records_html = get_past_records_html()
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>羽幌沿海フェリー 便別欠航予測ダッシュボード</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
            h1, h2 {{ color: #2c3e50; }}
            .notice {{ background-color: #eef6ff; padding: 10px 15px; border-left: 5px solid #0066cc; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; max-width: 900px; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ccc; padding: 10px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
            .past-records {{ width: 100%; max-width: 900px; }}
        </style>
    </head>
    <body>
        <h1>羽幌沿海フェリー 便別欠航予測ダッシュボード</h1>
        <div class="notice">
            <strong>【9月ダイヤ対応】</strong> 第1便（羽幌発 08:30 / 返り 12:10着）と 第2便（羽幌発 14:00 / 返り 17:35着）の時間帯別気象データに基づき、便ごとに独立して運航予測を行っています。
        </div>
        <p>最終更新: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (JST)</p>
        
        <h2>フェリー週間欠航予測（便別）</h2>
        <table>
            <tr>
                <th rowspan="2">日付</th>
                <th colspan="2">第1便（午前便）</th>
                <th colspan="2">第2便（午後便）</th>
            </tr>
            <tr>
                <th>気象（波高 / 風速）</th>
                <th>予測判定</th>
                <th>気象（波高 / 風速）</th>
                <th>予測判定</th>
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
    
    print("便別（1便・2便）の予測計算中...")
    df_summary = process_forecast_data(weather_data)
    
    print("本日の実績データをDBへ保存中...")
    save_today_record(df_summary)
    
    print("Webページ (index.html) 生成中...")
    generate_html(df_summary)
    
    print("全処理が完了しました！")

if __name__ == "__main__":
    main()
