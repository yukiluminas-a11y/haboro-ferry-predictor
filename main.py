import os
import sqlite3
import requests
import pandas as pd
from datetime import datetime

# --- 設定 ---
DB_PATH = "ferry_data.sqlite"
LATITUDE = 44.38   # 羽幌〜焼尻・天売エリア（緯度）
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
            visibility REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def fetch_combined_weather_data():
    """
    Open-Meteo から気象（風速・視程）と海洋（波高）データを取得して結合
    """
    # 1. 陸上気象 API（風速・視程）
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["wind_speed_10m", "visibility"],
        "timezone": "Asia/Tokyo"
    }
    res_forecast = requests.get(forecast_url, params=forecast_params)
    res_forecast.raise_for_status()
    data_forecast = res_forecast.json()

    # 2. 海洋 API（波高）
    marine_url = "https://api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["wave_height"],
        "timezone": "Asia/Tokyo"
    }
    res_marine = requests.get(marine_url, params=marine_params)
    res_marine.raise_for_status()
    data_marine = res_marine.json()

    # DataFrame 化
    df_forecast = pd.DataFrame({
        "datetime": pd.to_datetime(data_forecast["hourly"]["time"]),
        "wind_speed": data_forecast["hourly"]["wind_speed_10m"],
        "visibility": data_forecast["hourly"]["visibility"] # メートル単位 (例: 10000m)
    })
    
    df_marine = pd.DataFrame({
        "datetime": pd.to_datetime(data_marine["hourly"]["time"]),
        "wave_height": data_marine["hourly"]["wave_height"]
    })

    # 時間軸でマージ
    return pd.merge(df_forecast, df_marine, on="datetime", how="outer")

def calculate_flight_risk(wave, wind, visibility, vessel_type="ferry"):
    """
    羽幌沿海フェリー安全運航基準（regulations02.pdf）に基づく運航判定
    - フェリーおろろん2: 風速 15m/s, 波高 2.5m, 視界 500m
    - 高速船さんらいなぁ2: 風速 12m/s, 波高 1.5m, 視界 500m
    """
    if pd.isna(wave) or pd.isna(wind):
        return "データなし", "gray", 0

    if vessel_type == "high_speed":
        limit_wind, limit_wave = 12.0, 1.5
        warn_wind, warn_wave = 9.0, 1.0
    else:
        limit_wind, limit_wave = 15.0, 2.5
        warn_wind, warn_wave = 11.0, 1.8

    limit_vis = 500.0   # 限界視界 (m)
    warn_vis = 1000.0   # 注意視界 (m)
    vis_meters = visibility if not pd.isna(visibility) else 10000.0

    # 運航中止限界条件（規約基準超過）
    if wind >= limit_wind or wave >= limit_wave or vis_meters <= limit_vis:
        return "欠航警戒（基準超過）", "red", 95
    # 運航注意条件
    elif wind >= warn_wind or wave >= warn_wave or vis_meters <= warn_vis:
        return "運航注意（出港慎重）", "yellow", 60
    # 通常運航
    else:
        return "通常運航", "green", 10

def process_forecast_data(df_hourly):
    """第1便（08-12時）と第2便（14-18時）それぞれでピーク気象値を判定"""
    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")
    df_hourly["hour"] = df_hourly["datetime"].dt.hour
    
    results = []
    dates = df_hourly["date"].unique()
    
    for d in dates:
        df_day = df_hourly[df_hourly["date"] == d]
        
        # 第1便 (08:00 〜 12:00)
        df_f1 = df_day[(df_day["hour"] >= 8) & (df_day["hour"] <= 12)]
        if not df_f1.empty and df_f1["wave_height"].notna().any():
            wave1 = df_f1["wave_height"].max()
            wind1 = df_f1["wind_speed"].max()
            vis1 = df_f1["visibility"].min()
            status1, color1, prob1 = calculate_flight_risk(wave1, wind1, vis1)
        else:
            wave1, wind1, vis1, status1, color1, prob1 = None, None, None, "データなし", "gray", 0

        # 第2便 (14:00 〜 18:00)
        df_f2 = df_day[(df_day["hour"] >= 14) & (df_day["hour"] <= 18)]
        if not df_f2.empty and df_f2["wave_height"].notna().any():
            wave2 = df_f2["wave_height"].max()
            wind2 = df_f2["wind_speed"].max()
            vis2 = df_f2["visibility"].min()
            status2, color2, prob2 = calculate_flight_risk(wave2, wind2, vis2)
        else:
            wave2, wind2, vis2, status2, color2, prob2 = None, None, None, "データなし", "gray", 0
            
        results.append({
            "date": d,
            "wave1": wave1, "wind1": wind1, "vis1": vis1, "status1": status1, "color1": color1, "prob1": prob1,
            "wave2": wave2, "wind2": wind2, "vis2": vis2, "status2": status2, "color2": color2, "prob2": prob2
        })
        
    return pd.DataFrame(results)

def get_past_records_html():
    """DBから過去実績（直近10件）を取得"""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date AS 日付, status AS 運航状況, 
               wind_speed AS '最大風速(m/s)', wave_height AS '最大波高(m)'
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
    """本日の最大値を DB に永続化"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_data = df_summary[df_summary["date"] == today_str]
    
    if not today_data.empty:
        row = today_data.iloc[0]
        max_wave = max(filter(None, [row["wave1"], row["wave2"]]), default=0)
        max_wind = max(filter(None, [row["wind1"], row["wind2"]]), default=0)
        min_vis = min(filter(None, [row["vis1"], row["vis2"]]), default=10000)
        status_text = f"1便:{row['status1']} / 2便:{row['status2']}"
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ferry_records (date, status, wind_speed, wave_height, visibility, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (today_str, status_text, max_wind, max_wave, min_vis, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

def generate_html(df_summary):
    """ダッシュボード (index.html) の出力"""
    forecast_rows = ""
    for _, row in df_summary.iterrows():
        date = row["date"]
        
        vis1_km = f"{row['vis1']/1000:.1f}km" if row["vis1"] is not None else "-"
        w1 = f'{row["wave1"]:.1f}m / {row["wind1"]:.1f}m/s (視界:{vis1_km})' if row["wave1"] is not None else "-"
        s1 = f'<span style="color:{row["color1"]}; font-weight:bold;">{row["status1"]}</span>'
        
        vis2_km = f"{row['vis2']/1000:.1f}km" if row["vis2"] is not None else "-"
        w2 = f'{row["wave2"]:.1f}m / {row["wind2"]:.1f}m/s (視界:{vis2_km})' if row["wave2"] is not None else "-"
        s2 = f'<span style="color:{row["color2"]}; font-weight:bold;">{row["status2"]}</span>'
        
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
            .notice {{ background-color: #eef6ff; padding: 12px 15px; border-left: 5px solid #0066cc; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; max-width: 950px; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ccc; padding: 10px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
            .past-records {{ width: 100%; max-width: 950px; }}
        </style>
    </head>
    <body>
        <h1>羽幌沿海フェリー 便別欠航予測ダッシュボード</h1>
        <div class="notice">
            <strong>【運航管理規約 準拠判定】</strong><br>
            規約（regulations02.pdf）に定められた限界値（フェリーおろろん2: 風速15m/s、波高2.5m、視界500m）に基づき自動判定を行っています。
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
                <th>気象（波高 / 風速 / 視界）</th>
                <th>予測判定</th>
                <th>気象（波高 / 風速 / 視界）</th>
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
    
    print("気象・海洋データ取得中 (Forecast & Marine API)...")
    df_hourly = fetch_combined_weather_data()
    
    print("安全運航基準に照らし合わせて便別予測計算中...")
    df_summary = process_forecast_data(df_hourly)
    
    print("本日の実績データをDBへ保存中...")
    save_today_record(df_summary)
    
    print("Webページ (index.html) 生成中...")
    generate_html(df_summary)
    
    print("全処理が完了しました！")

if __name__ == "__main__":
    main()
