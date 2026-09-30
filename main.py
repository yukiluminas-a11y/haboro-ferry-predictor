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
# 羽幌沖（羽幌〜焼尻・天売）の座標
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
# 2. 公式サイトからの実績取得スクレイピング
# ---------------------------------------------------------
def fetch_official_status():
    url = "https://haboro-enkai.com/"
    try:
        response = requests.get(url, timeout=10)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 運航状況テキストの抽出（サイト構造に合わせて適宜調整）
        text_content = soup.get_text()
        
        if "欠航" in text_content:
            return "欠航", text_content
        else:
            return "平常運航", text_content
    except Exception as e:
        print(f"公式サイト取得エラー: {e}")
        return "不明", ""

# ---------------------------------------------------------
# 3. 気象データAPI（Open-Meteo等）からのデータ取得
# ---------------------------------------------------------
def fetch_weather_data():
    # 本日の気象予測データおよび前日の波高データを取得するAPIリクエスト
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=windspeed_10m,winddirection_10m,wave_height,visibility&timezone=Asia%2FTokyo"
    
    try:
        res = requests.get(url, timeout=10).json()
        hourly = res.get('hourly', {})
        df = pd.DataFrame(hourly)
        df['time'] = pd.to_datetime(df['time'])
        
        # 【修正ポイント】1便体制（9:00発 / 13:00帰港）に合わせ、8:00〜14:00の範囲で集計
        today = pd.Timestamp.now(tz='Asia/Tokyo').date()
        df_today = df[df['time'].dt.date == today]
        df_sailing = df_today[(df_today['time'].dt.hour >= 8) & (df_today['time'].dt.hour <= 14)]
        
        if df_sailing.empty:
            df_sailing = df_today

        max_wind = df_sailing['windspeed_10m'].max()
        max_wave = df_sailing['wave_height'].max()
        min_vis = df_sailing['visibility'].min()
        avg_wind_dir = df_sailing['winddirection_10m'].mean()
        
        # 前日データの波高取得（うねりの影響評価）
        yesterday = today - pd.Timedelta(days=1)
        df_yesterday = df[df['time'].dt.date == yesterday]
        prev_max_wave = df_yesterday['wave_height'].max() if not df_yesterday.empty else 1.0

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
# 4. 機械学習（ランダムフォレスト）による判定モデル
# ---------------------------------------------------------
def predict_status(weather_info, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT max_wind_speed, max_wave_height, wind_direction_deg, prev_day_max_wave, actual_status FROM ferry_records WHERE actual_status IN ('平常運航', '欠航')")
    rows = cursor.fetchall()
    
    # 有効実績データが20件未満の場合は固定ルール判定
    if len(rows) < 20:
        if weather_info['max_wave_height'] >= 2.5 or weather_info['max_wind_speed'] >= 14.0:
            return "欠航予想", "固定ルール"
        elif weather_info['max_wave_height'] >= 1.8 or weather_info['max_wind_speed'] >= 10.0:
            return "注意予想", "固定ルール"
        else:
            return "平常予想", "固定ルール"

    # --- 機械学習（AI）モード ---
    X = []
    y = []
    for r in rows:
        wind_deg = r[2] if r[2] is not None else 0.0
        rad = math.radians(wind_deg)
        prev_wave = r[3] if r[3] is not None else 1.0
        # 特徴量: 風速, 波高, 風向(cos), 風向(sin), 前日波高
        X.append([r[0], r[1], math.cos(rad), math.sin(rad), prev_wave])
        y.append(1 if r[4] == "欠航" else 0)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # 当日データの変換
    cur_rad = math.radians(weather_info['wind_direction_deg'])
    cur_X = [[
        weather_info['max_wind_speed'],
        weather_info['max_wave_height'],
        math.cos(cur_rad),
        math.sin(cur_rad),
        weather_info['prev_day_max_wave']
    ]]

    # 欠航確率（クラス1の確率）を算出
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

    # 気象データおよび実績の取得
    weather_info = fetch_weather_data()
    official_status, raw_text = fetch_official_status()

    if weather_info is not None:
        predicted, mode = predict_status(weather_info, conn)

        # データベースへのUpsert処理
        cursor.execute('''
            INSERT INTO ferry_records (
                date, predicted_status, actual_status, max_wind_speed, max_wave_height,
                min_visibility, raw_official_text, updated_at, wind_direction_deg,
                prev_day_max_wave, prediction_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                predicted_status = excluded.predicted_status,
                actual_status = excluded.actual_status,
                max_wind_speed = excluded.max_wind_speed,
                max_wave_height = excluded.max_wave_height,
                min_visibility = excluded.min_visibility,
                raw_official_text = excluded.raw_official_text,
                updated_at = excluded.updated_at,
                wind_direction_deg = excluded.wind_direction_deg,
                prev_day_max_wave = excluded.prev_day_max_wave,
                prediction_mode = excluded.prediction_mode
        ''', (
            today_str, predicted, official_status,
            weather_info['max_wind_speed'], weather_info['max_wave_height'],
            weather_info['min_visibility'], raw_text, now_timestamp,
            weather_info['wind_direction_deg'], weather_info['prev_day_max_wave'], mode
        ))
        conn.commit()
        print(f"[{now_timestamp}] 判定更新完了: {today_str} | 予測={predicted} | モード={mode} | 公式={official_status}")

    conn.close()

if __name__ == '__main__':
    main()
