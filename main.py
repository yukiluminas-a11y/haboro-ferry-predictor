import os
import sqlite3
import math
import requests
import joblib
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 設定値
# ---------------------------------------------------------
DB_FILE = 'ferry_data.sqlite'
MODEL_FILE = 'ferry_model.pkl'
HTML_FILE = 'index.html'
LATITUDE = 44.3600
LONGITUDE = 141.6900

# ---------------------------------------------------------
# 1. データベースの初期化
# ---------------------------------------------------------
def init_db():
    try:
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
                prediction_mode TEXT,
                consecutive_cancels INTEGER DEFAULT 0
            )
        ''')
        try:
            cursor.execute("ALTER TABLE ferry_records ADD COLUMN consecutive_cancels INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] DB初期化失敗: {e}")

# ---------------------------------------------------------
# 2. 直前連続欠航日数の計算ロジック
# ---------------------------------------------------------
def get_consecutive_cancels(conn, target_date_str):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT actual_status FROM ferry_records 
            WHERE date < ? AND actual_status IS NOT NULL 
            ORDER BY date DESC LIMIT 5
        """, (target_date_str,))
        rows = cursor.fetchall()
        
        count = 0
        for r in rows:
            if r[0] == '欠航':
                count += 1
            else:
                break
        return count
    except Exception as e:
        print(f"[WARN] 連続欠航日数計算エラー: {e}")
        return 0

# ---------------------------------------------------------
# 3. 公式サイトからの実績取得
# ---------------------------------------------------------
def fetch_official_status():
    url = "https://haboro-enkai.com/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text() if soup else ""
        
        if "欠航" in text_content:
            return "欠航", text_content[:300]
        elif "平常" in text_content or "通常" in text_content or "運航" in text_content:
            return "平常運航", text_content[:300]
        else:
            return None, text_content[:300]
    except Exception as e:
        print(f"[WARN] 公式サイト取得スキップ: {e}")
        return None, ""

# ---------------------------------------------------------
# 4. 気象データAPI（一般気象＋海上気象マージ取得）
# ---------------------------------------------------------
def fetch_weather_forecast_10days():
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=windspeed_10m,winddirection_10m,visibility&forecast_days=10&timezone=Asia%2FTokyo"
    marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=wave_height&forecast_days=10&timezone=Asia%2FTokyo"

    try:
        res_w = requests.get(weather_url, timeout=15)
        res_w.raise_for_status()
        df_w = pd.DataFrame(res_w.json().get('hourly', {}))
        df_w['time'] = pd.to_datetime(df_w['time'])

        try:
            res_m = requests.get(marine_url, timeout=15)
            res_m.raise_for_status()
            df_m = pd.DataFrame(res_m.json().get('hourly', {}))
            df_m['time'] = pd.to_datetime(df_m['time'])
            df = pd.merge(df_w, df_m[['time', 'wave_height']], on='time', how='left')
        except Exception as e_m:
            print(f"[WARN] 海上気象API取得失敗のため風速より波高補完: {e_m}")
            df = df_w
            df['wave_height'] = None

        df['date_str'] = df['time'].dt.strftime('%Y-%m-%d')
        daily_forecasts = []
        unique_dates = df['date_str'].unique()

        for idx, date_str in enumerate(unique_dates):
            df_day = df[df['date_str'] == date_str]
            df_sailing = df_day[(df_day['time'].dt.hour >= 8) & (df_day['time'].dt.hour <= 14)]
            if df_sailing.empty:
                df_sailing = df_day

            max_wind = float(df_sailing['windspeed_10m'].max()) if not df_sailing.empty and pd.notna(df_sailing['windspeed_10m'].max()) else 0.0
            raw_wave = df_sailing['wave_height'].max() if 'wave_height' in df_sailing.columns else None
            
            if pd.notna(raw_wave) and raw_wave is not None:
                max_wave = float(raw_wave)
            else:
                max_wave = round(max_wind * 0.12, 2)

            min_vis = float(df_sailing['visibility'].min()) if 'visibility' in df_sailing.columns and not df_sailing.empty and pd.notna(df_sailing['visibility'].min()) else 10000.0
            avg_wind_dir = float(df_sailing['winddirection_10m'].mean()) if not df_sailing.empty and pd.notna(df_sailing['winddirection_10m'].mean()) else 0.0

            if idx > 0:
                prev_date_str = unique_dates[idx - 1]
                df_prev = df[df['date_str'] == prev_date_str]
                prev_wave_val = df_prev['wave_height'].max() if 'wave_height' in df_prev.columns else None
                prev_max_wave = float(prev_wave_val) if pd.notna(prev_wave_val) and prev_wave_val is not None else 1.0
            else:
                prev_max_wave = 1.0

            daily_forecasts.append({
                'date': date_str,
                'max_wind_speed': round(max_wind, 1),
                'max_wave_height': round(max_wave, 2),
                'min_visibility': round(min_vis, 1),
                'wind_direction_deg': round(avg_wind_dir, 1),
                'prev_day_max_wave': round(prev_max_wave, 2)
            })

        return daily_forecasts
    except Exception as e:
        print(f"[ERROR] 10日分気象データ取得失敗: {e}")
        return []

# ---------------------------------------------------------
# 5. 自己学習（モデルの再学習・保存）エンジン
# ---------------------------------------------------------
def train_and_update_model(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT max_wind_speed, max_wave_height, wind_direction_deg, prev_day_max_wave, consecutive_cancels, actual_status 
            FROM ferry_records 
            WHERE actual_status IN ('平常運航', '欠航') 
              AND max_wind_speed IS NOT NULL 
              AND max_wave_height IS NOT NULL
        """)
        rows = cursor.fetchall()

        statuses = set(r[5] for r in rows) if rows else set()
        
        if len(rows) < 20 or len(statuses) < 2:
            return None

        X, y = [], []
        for r in rows:
            wind_deg = r[2] if r[2] is not None else 0.0
            rad = math.radians(wind_deg)
            prev_wave = r[3] if r[3] is not None else 1.0
            cancels = r[4] if r[4] is not None else 0
            
            X.append([r[0], r[1], math.cos(rad), math.sin(rad), prev_wave, cancels])
            y.append(1 if r[5] == "欠航" else 0)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        joblib.dump(clf, MODEL_FILE)
        print(f"[INFO] 自己学習完了: {len(rows)}件のデータ（連続欠航指標含む）でモデル更新・保存")
        return clf
    except Exception as e:
        print(f"[ERROR] 自己学習処理エラー: {e}")
        return None

# ---------------------------------------------------------
# 6. 判定推論関数（連続欠航による出港圧力補正入り）
# ---------------------------------------------------------
def predict_status_for_day(weather_info, cancels, clf):
    def fallback_rule():
        wave_limit = 2.8 if cancels >= 2 else 2.5
        wind_limit = 15.0 if cancels >= 2 else 14.0
        
        if weather_info['max_wave_height'] >= wave_limit or weather_info['max_wind_speed'] >= wind_limit:
            return "欠航予想", "固定ルール"
        elif weather_info['max_wave_height'] >= 1.8 or weather_info['max_wind_speed'] >= 10.0:
            return "注意予想", "固定ルール"
        else:
            return "平常予想", "固定ルール"

    if clf is None:
        return fallback_rule()

    try:
        cur_rad = math.radians(weather_info['wind_direction_deg'])
        cur_X = [[
            weather_info['max_wind_speed'],
            weather_info['max_wave_height'],
            math.cos(cur_rad),
            math.sin(cur_rad),
            weather_info['prev_day_max_wave'],
            cancels
        ]]

        cancel_prob = clf.predict_proba(cur_X)[0][1]

        if cancel_prob >= 0.65:
            return "欠航予想", "自己学習AI"
        elif cancel_prob >= 0.35:
            return "注意予想", "自己学習AI"
        else:
            return "平常予想", "自己学習AI"
    except Exception as e:
        return fallback_rule()

# ---------------------------------------------------------
# 7. index.html 自動生成処理（完全に文字列エスケープ・改行事故を防止）
# ---------------------------------------------------------
def generate_html(conn):
    try:
        today_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d')
        now_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')

        df_forecast = pd.read_sql_query("SELECT * FROM ferry_records WHERE date >= ? ORDER BY date ASC LIMIT 10", conn, params=(today_str,))
        df_history = pd.read_sql_query("SELECT * FROM ferry_records WHERE date <= ? ORDER BY date DESC LIMIT 30", conn, params=(today_str,))

        def build_rows_html(df_data):
            out = []
            row_fmt = '
