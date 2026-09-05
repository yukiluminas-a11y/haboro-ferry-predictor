import os
import sqlite3
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 設定値 ---
DB_PATH = "ferry_data.sqlite"
HTML_PATH = "index.html"
MIN_ML_SAMPLES = 20  # AI判定に移行するために必要な最低実績件数

# Open-Meteo API (羽幌付近の緯度・経度)
LATITUDE = 44.36
LONGITUDE = 141.70

# --- 1. 気象データの取得 (Open-Meteo API) ---
def fetch_weather_forecast():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=windspeed_10m,winddirection_10m&daily=windspeed_10m_max,winddirection_10m_dominant&timezone=Asia%2FTokyo"
    marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={LATITUDE}&longitude={LONGITUDE}&daily=wave_height_max&timezone=Asia%2FTokyo"
    
    try:
        res_w = requests.get(url, timeout=10).json()
        res_m = requests.get(marine_url, timeout=10).json()
        
        today_date = res_w['daily']['time'][0]
        max_wind = res_w['daily']['windspeed_10m_max'][0]
        wind_dir = res_w['daily']['winddirection_10m_dominant'][0]
        max_wave = res_m['daily']['wave_height_max'][0]
        
        return {
            "date": today_date,
            "max_wind_speed": float(max_wind),
            "wind_direction_deg": float(wind_dir),
            "max_wave_height": float(max_wave)
        }
    except Exception as e:
        print(f"気象データ取得エラー: {e}")
        return None

# --- 2. 公式実績の取得 (スクレイピング) ---
def fetch_official_status():
    # 羽幌沿海フェリーの運航情報ページURL
    url = "https://www.haboro-enkai.com/"
    try:
        res = requests.get(url, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 運航状況テキストの抽出（サイト構造に合わせて適宜調整）
        text = soup.get_text()
        if "全便欠航" in text or "欠航" in text:
            return "欠航"
        elif "平常運航" in text or "運航" in text:
            return "平常運航"
        else:
            return "未取得"
    except Exception as e:
        print(f"公式実績取得エラー: {e}")
        return "未取得"

# --- 3. データベース操作 ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ferry_records (
        date TEXT PRIMARY KEY,
        max_wind_speed REAL,
        max_wave_height REAL,
        wind_direction_deg REAL,
        prev_day_max_wave REAL,
        predicted_status TEXT,
        actual_status TEXT,
        prediction_mode TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_prev_day_wave(today_date):
    """前日の最大波高を取得"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT max_wave_height FROM ferry_records WHERE date < ? ORDER BY date DESC LIMIT 1", (today_date,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# --- 4. 予測ロジック（ルール判定 vs AI学習判定） ---
def predict_with_rules(weather):
    """従来のルールベース判定（データ蓄積期）"""
    wind = weather['max_wind_speed']
    wave = weather['max_wave_height']
    
    if wind >= 14.0 or wave >= 2.5:
        return "欠航予想", 85.0
    elif wind >= 10.0 or wave >= 1.8:
        return "注意予想", 50.0
    else:
        return "平常予想", 10.0

def predict_with_ml(weather, prev_wave):
    """機械学習（ランダムフォレスト）による判定"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT max_wind_speed, max_wave_height, wind_direction_deg, prev_day_max_wave, actual_status 
    FROM ferry_records 
    WHERE actual_status IS NOT NULL AND actual_status IN ('平常運航', '欠航')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) < MIN_ML_SAMPLES:
        return None, None  # サンプル不足

    # 前処理
    df['target'] = df['actual_status'].apply(lambda x: 1 if '欠航' in x else 0)
    df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction_deg'].fillna(0)))
    df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction_deg'].fillna(0)))
    df['prev_day_max_wave'] = df['prev_day_max_wave'].fillna(df['max_wave_height'])

    feature_cols = ['max_wind_speed', 'max_wave_height', 'wind_dir_cos', 'wind_dir_sin', 'prev_day_max_wave']
    X = df[feature_cols]
    y = df['target']

    # モデル学習
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    # 本日のデータ予測
    wind_dir = weather['wind_direction_deg']
    input_data = pd.DataFrame([{
        'max_wind_speed': weather['max_wind_speed'],
        'max_wave_height': weather['max_wave_height'],
        'wind_dir_cos': np.cos(np.radians(wind_dir)),
        'wind_dir_sin': np.sin(np.radians(wind_dir)),
        'prev_day_max_wave': prev_wave if prev_wave else weather['max_wave_height']
    }])[feature_cols]

    cancel_prob = model.predict_proba(input_data)[0][1] * 100.0

    if cancel_prob >= 65.0:
        status = "欠航予想"
    elif cancel_prob >= 35.0:
        status = "注意予想"
    else:
        status = "平常予想"

    return status, cancel_prob

# --- 5. ダッシュボード HTML 生成 ---
def generate_html():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ferry_records ORDER BY date DESC LIMIT 15", conn)
    conn.close()

    if df.empty:
        return

    today = df.iloc[0]
    
    # テーブル行の構築
    rows_html = ""
    for _, row in df.iterrows():
        actual = row['actual_status'] if row['actual_status'] else "未取得"
        mode_badge = "AI" if row['prediction_mode'] == "機械学習" else "固定ルール"
        rows_html += f"""
        <tr>
            <td>{row['date']}</td>
            <td>{row['max_wind_speed']} m/s</td>
            <td>{row['max_wave_height']} m</td>
            <td>{row['predicted_status']}</td>
            <td><strong>{actual}</strong></td>
            <td><small>{mode_badge}</small></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>羽幌沿海フェリー 欠航予測ダッシュボード</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 20px auto; padding: 0 10px; background: #f9f9f9; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .status {{ font-size: 1.8em; font-weight: bold; color: #0056b3; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #fff; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        th {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>羽幌沿海フェリー 運航予測</h1>
    
    <div class="card">
        <h2>本日の予測 ({today['date']})</h2>
        <div class="status">{today['predicted_status']}</div>
        <p>最大風速: {today['max_wind_speed']} m/s | 最大波高: {today['max_wave_height']} m</p>
        <p>判定モード: <strong>{today['prediction_mode']}</strong></p>
        <p>本日の公式実績: <strong>{today['actual_status']}</strong></p>
    </div>

    <div class="card">
        <h3>直近の記録・実績比較</h3>
        <table>
            <thead>
                <tr>
                    <th>日付</th>
                    <th>風速</th>
                    <th>波高</th>
                    <th>予測</th>
                    <th>公式実績</th>
                    <th>判定方式</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

# --- 6. メイン処理 ---
def main():
    init_db()
    
    weather = fetch_weather_forecast()
    if not weather:
        print("気象データが取得できないため終了します。")
        return
        
    today_date = weather['date']
    prev_wave = get_prev_day_wave(today_date)
    official_status = fetch_official_status()
    
    # 予測判定（AI判定試行 ➔ サンプル不足ならルール判定）
    pred_status, cancel_prob = predict_with_ml(weather, prev_wave)
    
    if pred_status is not None:
        mode = "機械学習"
    else:
        pred_status, cancel_prob = predict_with_rules(weather)
        mode = "固定ルール"
        
    print(f"[{today_date}] 判定モード: {mode} | 予測: {pred_status} (欠航確率: {cancel_prob:.1f}%) | 公式実績: {official_status}")
    
    # DBへの保存・更新
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO ferry_records (
        date, max_wind_speed, max_wave_height, wind_direction_deg, 
        prev_day_max_wave, predicted_status, actual_status, prediction_mode
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(date) DO UPDATE SET
        max_wind_speed = excluded.max_wind_speed,
        max_wave_height = excluded.max_wave_height,
        wind_direction_deg = excluded.wind_direction_deg,
        prev_day_max_wave = excluded.prev_day_max_wave,
        predicted_status = excluded.predicted_status,
        actual_status = CASE 
            WHEN excluded.actual_status != '未取得' THEN excluded.actual_status 
            ELSE ferry_records.actual_status 
        END,
        prediction_mode = excluded.prediction_mode
    """, (
        today_date,
        weather['max_wind_speed'],
        weather['max_wave_height'],
        weather['wind_direction_deg'],
        prev_wave if prev_wave else weather['max_wave_height'],
        pred_status,
        official_status,
        mode
    ))
    conn.commit()
    conn.close()
    
    # HTML生成
    generate_html()

if __name__ == "__main__":
    main()
