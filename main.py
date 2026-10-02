import os
import sqlite3
import math
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 設定値
# ---------------------------------------------------------
DB_FILE = 'ferry_data.sqlite'
LATITUDE = 44.3600
LONGITUDE = 141.6900

# ---------------------------------------------------------
# 1. データベースの初期化
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ferry_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            predicted_status TEXT,
            actual_status TEXT,
            max_wind_speed REAL,
            max_wave_height REAL,
            min_visibility REAL,
            raw_official_text TEXT,
            updated_at TEXT,
            wind_direction_deg REAL,
            prev_day_max_wave REAL,
            prediction_mode TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# 2. 公式サイトからの実績取得（新ダイヤ対応スクレイピング）
# ---------------------------------------------------------
def fetch_official_status():
    url = "https://haboro-enkai.com/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text_content = soup.get_text() if soup else ""
        
        # 本日日付（例: 10月1日、10月2日）がテキストに含まれているか確認
        today_jst = pd.Timestamp.now(tz='Asia/Tokyo')
        month_day_str = f"{today_jst.month}月{today_jst.day}日"
        
        if "全便欠航" in text_content or "欠航" in text_content:
            # 「欠航」キーワードが含まれており、かつ本日の案内と思われる場合
            return "欠航", text_content[:300]
        elif "平常運航" in text_content or "通常運航" in text_content or "運航いたします" in text_content:
            return "平常運航", text_content[:300]
        else:
            print(f"公式サイトの本日分（{month_day_str}）更新が未確認のため、実績取得を保留します。")
            return None, text_content[:300]
    except Exception as e:
        print(f"公式サイト取得例外（気象予測のみ先行処理します）: {e}")
        return None, ""

# ---------------------------------------------------------
# 3. 気象データAPIからのデータ取得（1便体制：8:00〜14:00）
# ---------------------------------------------------------
def fetch_weather_data():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=windspeed_10m,winddirection_10m,wave_height,visibility&timezone=Asia%2FTokyo"
    
    try:
        res = requests.get(url, timeout=15).json()
        hourly = res.get('hourly', {})
        if not hourly:
            return None

        df = pd.DataFrame(hourly)
        df['time'] = pd.to_datetime(df['time'])
        
        today = pd.Timestamp.now(tz='Asia/Tokyo').date()
        df_today = df[df['time'].dt.date == today]
        # 9:00発/13:00着の1便体制に対応した時間帯絞り込み
        df_sailing = df_today[(df_today['time'].dt.hour >= 8) & (df_today['time'].dt.hour <= 14)]
        
        if df_sailing.empty:
            df_sailing = df_today

        max_wind = float(df_sailing['windspeed_10m'].max())
        max_wave = float(df_sailing['wave_height'].max())
        min_vis = float(df_sailing['visibility'].min())
        avg_wind_dir = float(df_sailing['winddirection_10m'].mean())
        
        yesterday = today - pd.Timedelta(days=1)
        df_yesterday = df[df['time'].dt.date == yesterday]
        prev_max_wave = float(df_yesterday['wave_height'].max()) if not df_yesterday.empty else 1.0

        return {
            'max_wind_speed': max_wind,
            'max_wave_height': max_wave,
            'min_visibility': min_vis,
            'wind_direction_deg': avg_wind_dir,
            'prev_day_max_wave': prev_max_wave
        }
    except Exception as e:
        print(f"気象データ取得エラー: {e}")
        return None

# ---------------------------------------------------------
# 4. 機械学習（AI）による判定モデル
# ---------------------------------------------------------
def predict_status(weather_info, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT max_wind_speed, max_wave_height, wind_direction_deg, prev_day_max_wave, actual_status FROM ferry_records WHERE actual_status IN ('平常運航', '欠航')")
    rows = cursor.fetchall()
    
    if len(rows) < 20:
        if weather_info['max_wave_height'] >= 2.5 or weather_info['max_wind_speed'] >= 14.0:
            return "欠航予想", "固定ルール"
        elif weather_info['max_wave_height'] >= 1.8 or weather_info['max_wind_speed'] >= 10.0:
            return "注意予想", "固定ルール"
        else:
            return "平常予想", "固定ルール"

    X, y = [], []
    for r in rows:
        wind_deg = r[2] if r[2] is not None else 0.0
        rad = math.radians(wind_deg)
        prev_wave = r[3] if r[3] is not None else 1.0
        X.append([r[0], r[1], math.cos(rad), math.sin(rad), prev_wave])
        y.append(1 if r[4] == "欠航" else 0)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    cur_rad = math.radians(weather_info['wind_direction_deg'])
    cur_X = [[
        weather_info['max_wind_speed'],
        weather_info['max_wave_height'],
        math.cos(cur_rad),
        math.sin(cur_rad),
        weather_info['prev_day_max_wave']
    ]]

    cancel_prob = clf.predict_proba(cur_X)[0][1]

    if cancel_prob >= 0.65:
        return "欠航予想", "機械学習"
    elif cancel_prob >= 0.35:
        return "注意予想", "機械学習"
    else:
        return "平常予想", "機械学習"

# ---------------------------------------------------------
# 5. メイン実行処理
# ---------------------------------------------------------
def main():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    today_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d')
    now_timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')

    weather_info = fetch_weather_data()
    official_status, raw_text = fetch_official_status()

    if weather_info is not None:
        predicted, mode = predict_status(weather_info, conn)

        # すでに確定実績が入っている場合は上書き防止
        cursor.execute("SELECT actual_status FROM ferry_records WHERE date = ?", (today_str,))
        existing_row = cursor.fetchone()
        
        final_actual_status = official_status
        if existing_row and existing_row[0] in ['平常運航', '欠航']:
            final_actual_status = existing_row[0]

        cursor.execute('''
            INSERT INTO ferry_records (
                date, predicted_status, actual_status, max_wind_speed, max_wave_height,
                min_visibility, raw_official_text, updated_at, wind_direction_deg,
                prev_day_max_wave, prediction_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                predicted_status = excluded.predicted_status,
                actual_status = COALESCE(excluded.actual_status, ferry_records.actual_status),
                max_wind_speed = excluded.max_wind_speed,
                max_wave_height = excluded.max_wave_height,
                min_visibility = excluded.min_visibility,
                raw_official_text = excluded.raw_official_text,
                updated_at = excluded.updated_at,
                wind_direction_deg = excluded.wind_direction_deg,
                prev_day_max_wave = excluded.prev_day_max_wave,
                prediction_mode = excluded.prediction_mode
        ''', (
            today_str, predicted, final_actual_status,
            weather_info['max_wind_speed'], weather_info['max_wave_height'],
            weather_info['min_visibility'], raw_text, now_timestamp,
            weather_info['wind_direction_deg'], weather_info['prev_day_max_wave'], mode
        ))
        conn.commit()
        print(f"[{now_timestamp}] 更新完了: 日付={today_str} | 予測={predicted} | モード={mode} | 公式実績={final_actual_status}")

    conn.close()

if __name__ == '__main__':
    main()
