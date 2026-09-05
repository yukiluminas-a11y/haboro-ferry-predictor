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

# --- 1. 向こう1週間分の気象データを取得 (Open-Meteo API) ---
def fetch_weekly_weather_forecast():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=windspeed_10m,winddirection_10m&daily=windspeed_10m_max,winddirection_10m_dominant&timezone=Asia%2FTokyo"
    marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={LATITUDE}&longitude={LONGITUDE}&daily=wave_height_max&timezone=Asia%2FTokyo"
    
    try:
        res_w = requests.get(url, timeout=10).json()
        res_m = requests.get(marine_url, timeout=10).json()
        
        forecast_list = []
        days_cnt = len(res_w['daily']['time'])
        
        for i in range(days_cnt):
            forecast_list.append({
                "date": res_w['daily']['time'][i],
                "max_wind_speed": float(res_w['daily']['windspeed_10m_max'][i]),
                "wind_direction_deg": float(res_w['daily']['winddirection_10m_dominant'][i]),
                "max_wave_height": float(res_m['daily']['wave_height_max'][i])
            })
        return forecast_list
    except Exception as e:
        print(f"気象データ取得エラー: {e}")
        return []

# --- 2. 公式実績の取得 (スクレイピング) ---
def fetch_official_status():
    url = "https://www.haboro-enkai.com/"
    try:
        res = requests.get(url, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, 'html.parser')
        
        main_content = soup.find("main") or soup.find("div", id="content") or soup.find("body") or soup
        cleaned_text = main_content.get_text().replace(" ", "").replace("\n", "").replace("\r", "")
        
        if any(k in cleaned_text for k in ["平常運航", "通常運航", "全便運航", "欠航はありません", "全便通常"]):
            return "平常運航"
        elif "全便欠航" in cleaned_text or "終日欠航" in cleaned_text or "欠航" in cleaned_text:
            return "欠航"
        elif "条件付" in cleaned_text:
            return "条件付運航"
        elif "見合わせ" in cleaned_text:
            return "見合わせ"
        else:
            return "平常運航"
    except Exception as e:
        print(f"公式実績取得エラー: {e}")
        return "未取得"

# --- 3. データベース操作（安全なマイグレーション対応版） ---
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
    
    cursor.execute("PRAGMA table_info(ferry_records)")
    columns = [column[1] for column in cursor.fetchall()]
    
    fields_to_add = {
        "max_wind_speed": "REAL",
        "max_wave_height": "REAL",
        "wind_direction_deg": "REAL",
        "prev_day_max_wave": "REAL",
        "predicted_status": "TEXT",
        "actual_status": "TEXT",
        "prediction_mode": "TEXT"
    }
    
    for field, col_type in fields_to_add.items():
        if field not in columns:
            cursor.execute(f"ALTER TABLE ferry_records ADD COLUMN {field} {col_type}")
            
    conn.commit()
    conn.close()

def get_prev_day_wave(target_date):
    """指定日の前日の最大波高を取得"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT max_wave_height FROM ferry_records WHERE date < ? ORDER BY date DESC LIMIT 1", (target_date,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# --- 4. 予測ロジック（固定ルール vs AI学習） ---
def predict_with_rules(weather):
    wind = weather['max_wind_speed']
    wave = weather['max_wave_height']
    
    if wind >= 14.0 or wave >= 2.5:
        return "欠航予想", 85.0
    elif wind >= 10.0 or wave >= 1.8:
        return "注意予想", 50.0
    else:
        return "平常予想", 10.0

def train_ml_model():
    """蓄積データからモデルを学習させて返す"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT max_wind_speed, max_wave_height, wind_direction_deg, prev_day_max_wave, actual_status 
    FROM ferry_records 
    WHERE actual_status IS NOT NULL AND actual_status IN ('平常運航', '欠航')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) < MIN_ML_SAMPLES:
        return None

    df['target'] = df['actual_status'].apply(lambda x: 1 if '欠航' in str(x) else 0)
    df['wind_direction_deg'] = df['wind_direction_deg'].fillna(0.0)
    df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction_deg']))
    df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction_deg']))
    df['prev_day_max_wave'] = df['prev_day_max_wave'].fillna(df['max_wave_height'])

    feature_cols = ['max_wind_speed', 'max_wave_height', 'wind_dir_cos', 'wind_dir_sin', 'prev_day_max_wave']
    X = df[feature_cols]
    y = df['target']

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    return model

def predict_single_day(model, weather, prev_wave):
    if model is None:
        status, prob = predict_with_rules(weather)
        return status, prob, "固定ルール"

    wind_dir = weather['wind_direction_deg']
    feature_cols = ['max_wind_speed', 'max_wave_height', 'wind_dir_cos', 'wind_dir_sin', 'prev_day_max_wave']
    input_data = pd.DataFrame([{
        'max_wind_speed': weather['max_wind_speed'],
        'max_wave_height': weather['max_wave_height'],
        'wind_dir_cos': np.cos(np.radians(wind_dir)),
        'wind_dir_sin': np.sin(np.radians(wind_dir)),
        'prev_day_max_wave': prev_wave if prev_wave is not None else weather['max_wave_height']
    }])[feature_cols]

    cancel_prob = model.predict_proba(input_data)[0][1] * 100.0

    if cancel_prob >= 65.0:
        status = "欠航予想"
    elif cancel_prob >= 35.0:
        status = "注意予想"
    else:
        status = "平常予想"

    return status, cancel_prob, "機械学習"

# --- 5. Webダッシュボード（index.html）生成 ---
def generate_html(weekly_predictions):
    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query("SELECT * FROM ferry_records ORDER BY date DESC LIMIT 15", conn)
    conn.close()

    # 1週間予測のテーブル生成
    weekly_rows_html = ""
    for p in weekly_predictions:
        badge_color = "#e74c3c" if "欠航" in p['status'] else ("#f39c12" if "注意" in p['status'] else "#2ecc71")
        weekly_rows_html += f"""
        <tr>
            <td><strong>{p['date']}</strong></td>
            <td>{p['max_wind_speed']:.1f} m/s</td>
            <td>{p['max_wave_height']:.1f} m</td>
            <td><span style="color: {badge_color}; font-weight: bold;">{p['status']}</span> ({p['prob']:.0f}%)</td>
            <td><small>{p['mode']}</small></td>
        </tr>
        """

    # 過去の実績テーブル生成
    history_rows_html = ""
    for _, row in df_history.iterrows():
        actual = row['actual_status'] if row['actual_status'] else "未取得"
        mode_badge = "AI判定" if row['prediction_mode'] == "機械学習" else "固定ルール"
        wind_val = f"{row['max_wind_speed']:.1f}" if pd.notnull(row['max_wind_speed']) else "-"
        wave_val = f"{row['max_wave_height']:.1f}" if pd.notnull(row['max_wave_height']) else "-"
        
        history_rows_html += f"""
        <tr>
            <td>{row['date']}</td>
            <td>{wind_val} m/s</td>
            <td>{wave_val} m</td>
            <td>{row['predicted_status']}</td>
            <td><strong>{actual}</strong></td>
            <td><small>{mode_badge}</small></td>
        </tr>
        """

    today_p = weekly_predictions[0]

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>羽幌沿海フェリー 欠航予測ダッシュボード</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 850px; margin: 20px auto; padding: 0 15px; background: #f8f9fa; color: #333; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.08); margin-bottom: 20px; }}
        .status {{ font-size: 1.8em; font-weight: bold; color: #1c7ed6; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #fff; }}
        th, td {{ border: 1px solid #dee2e6; padding: 10px; text-align: center; font-size: 0.95em; }}
        th {{ background: #f1f3f5; color: #495057; }}
    </style>
</head>
<body>
    <h1>羽幌沿海フェリー 運航予測</h1>
    
    <div class="card">
        <h2>本日の予測 ({today_p['date']})</h2>
        <div class="status">{today_p['status']} (欠航確率: {today_p['prob']:.0f}%)</div>
        <p>最大風速: <strong>{today_p['max_wind_speed']:.1f} m/s</strong> | 最大波高: <strong>{today_p['max_wave_height']:.1f} m</strong></p>
        <p>判定モード: <strong>{today_p['mode']}</strong></p>
    </div>

    <div class="card">
        <h3>向こう1週間の運航予測</h3>
        <table>
            <thead>
                <tr>
                    <th>日付</th>
                    <th>風速</th>
                    <th>波高</th>
                    <th>予測 (確率)</th>
                    <th>判定方式</th>
                </tr>
            </thead>
            <tbody>
                {weekly_rows_html}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h3>過去の記録・公式実績比較 (直近15件)</h3>
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
                {history_rows_html}
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
    print("1. データベース構造の安全初期化...")
    init_db()
    
    print("2. 向こう1週間分の気象データ取得中...")
    weekly_weather = fetch_weekly_weather_forecast()
    if not weekly_weather:
        print("エラー: 気象データ取得失敗のため終了します。")
        return
        
    print("3. 今日の公式実績を取得中...")
    official_status = fetch_official_status()
    
    print("4. 機械学習モデルの準備...")
    model = train_ml_model()
    
    print("5. 1週間分の予測を実行中...")
    weekly_predictions = []
    
    for idx, w in enumerate(weekly_weather):
        prev_wave = get_prev_day_wave(w['date'])
        status, prob, mode = predict_single_day(model, w, prev_wave)
        
        weekly_predictions.append({
            "date": w['date'],
            "max_wind_speed": w['max_wind_speed'],
            "max_wave_height": w['max_wave_height'],
            "status": status,
            "prob": prob,
            "mode": mode
        })
        
        # 本日(先頭)のデータのみDBへ記録・更新
        if idx == 0:
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
                w['date'],
                w['max_wind_speed'],
                w['max_wave_height'],
                w['wind_direction_deg'],
                prev_wave if prev_wave is not None else w['max_wave_height'],
                status,
                official_status,
                mode
            ))
            conn.commit()
            conn.close()
    
    print("6. Webダッシュボード生成中...")
    generate_html(weekly_predictions)
    print("全処理が正常完了しました。")

if __name__ == "__main__":
    main()
