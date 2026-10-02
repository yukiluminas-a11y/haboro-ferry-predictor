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
                prediction_mode TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] DB初期化失敗: {e}")

# ---------------------------------------------------------
# 2. 公式サイトからの実績取得
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
# 3. 気象データAPIからのデータ取得
# ---------------------------------------------------------
def fetch_weather_data():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=windspeed_10m,winddirection_10m,wave_height,visibility&timezone=Asia%2FTokyo"
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()
        hourly = data.get('hourly', {})
        if not hourly or 'time' not in hourly:
            print("[WARN] 気象データの hourly パラメータが空です")
            return None

        df = pd.DataFrame(hourly)
        df['time'] = pd.to_datetime(df['time'])
        
        today = pd.Timestamp.now(tz='Asia/Tokyo').date()
        df_today = df[df['time'].dt.date == today]
        df_sailing = df_today[(df_today['time'].dt.hour >= 8) & (df_today['time'].dt.hour <= 14)]
        
        if df_sailing.empty:
            df_sailing = df_today

        max_wind = float(df_sailing['windspeed_10m'].max()) if not df_sailing.empty and pd.notna(df_sailing['windspeed_10m'].max()) else 0.0
        max_wave = float(df_sailing['wave_height'].max()) if not df_sailing.empty and pd.notna(df_sailing['wave_height'].max()) else 0.0
        min_vis = float(df_sailing['visibility'].min()) if not df_sailing.empty and pd.notna(df_sailing['visibility'].min()) else 10000.0
        avg_wind_dir = float(df_sailing['winddirection_10m'].mean()) if not df_sailing.empty and pd.notna(df_sailing['winddirection_10m'].mean()) else 0.0
        
        yesterday = today - pd.Timedelta(days=1)
        df_yesterday = df[df['time'].dt.date == yesterday]
        prev_max_wave = float(df_yesterday['wave_height'].max()) if not df_yesterday.empty and pd.notna(df_yesterday['wave_height'].max()) else 1.0

        return {
            'max_wind_speed': round(max_wind, 1),
            'max_wave_height': round(max_wave, 2),
            'min_visibility': round(min_vis, 1),
            'wind_direction_deg': round(avg_wind_dir, 1),
            'prev_day_max_wave': round(prev_max_wave, 2)
        }
    except Exception as e:
        print(f"[ERROR] 気象データ取得失敗: {e}")
        return None

# ---------------------------------------------------------
# 4. 自己学習（モデルの再学習・保存）エンジン
# ---------------------------------------------------------
def train_and_update_model(conn):
    """DBに蓄積された確定データ（平常運航 / 欠航）からモデルを学習・更新する"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT max_wind_speed, max_wave_height, wind_direction_deg, prev_day_max_wave, actual_status 
            FROM ferry_records 
            WHERE actual_status IN ('平常運航', '欠航') 
              AND max_wind_speed IS NOT NULL 
              AND max_wave_height IS NOT NULL
        """)
        rows = cursor.fetchall()

        statuses = set(r[4] for r in rows) if rows else set()
        
        # 学習に必要な最小条件：20件以上のデータかつ両判定（平常・欠航）が存在すること
        if len(rows) < 20 or len(statuses) < 2:
            print(f"[INFO] 再学習スキップ: データ件数不足 (確定データ数: {len(rows)}件)")
            return None

        X, y = [], []
        for r in rows:
            wind_deg = r[2] if r[2] is not None else 0.0
            rad = math.radians(wind_deg)
            prev_wave = r[3] if r[3] is not None else 1.0
            
            # 特徴量: [最大風速, 最大波高, 風向Cos, 風向Sin, 前日波高]
            X.append([r[0], r[1], math.cos(rad), math.sin(rad), prev_wave])
            y.append(1 if r[4] == "欠航" else 0)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        # モデルを保存
        joblib.dump(clf, MODEL_FILE)
        print(f"[INFO] 自己学習完了: {len(rows)}件のデータでモデル更新・保存 ({MODEL_FILE})")
        return clf
    except Exception as e:
        print(f"[ERROR] 自己学習処理エラー: {e}")
        return None

# ---------------------------------------------------------
# 5. 判定推論（学習済みモデル または 固定ルール）
# ---------------------------------------------------------
def predict_status(weather_info, conn):
    def fallback_rule():
        if weather_info['max_wave_height'] >= 2.5 or weather_info['max_wind_speed'] >= 14.0:
            return "欠航予想", "固定ルール"
        elif weather_info['max_wave_height'] >= 1.8 or weather_info['max_wind_speed'] >= 10.0:
            return "注意予想", "固定ルール"
        else:
            return "平常予想", "固定ルール"

    try:
        # 1. 最新の蓄積データから再学習を試みる
        clf = train_and_update_model(conn)

        # 2. 再学習がスキップされた場合、既存の保存済みモデル読み込みを試みる
        if clf is None and os.path.exists(MODEL_FILE):
            try:
                clf = joblib.load(MODEL_FILE)
                print("[INFO] 保存済みモデルを読み込みました。")
            except Exception as e:
                print(f"[WARN] 保存済みモデル読み込み失敗: {e}")
                clf = None

        # 3. モデルが準備できていない場合は固定ルールにフォールバック
        if clf is None:
            return fallback_rule()

        # 4. モデルによる予測確率の計算
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
            return "欠航予想", "自己学習AI"
        elif cancel_prob >= 0.35:
            return "注意予想", "自己学習AI"
        else:
            return "平常予想", "自己学習AI"
    except Exception as e:
        print(f"[WARN] AI予測例外のため固定ルールにフォールバック: {e}")
        return fallback_rule()

# ---------------------------------------------------------
# 6. index.html 自動生成処理
# ---------------------------------------------------------
def generate_html(conn):
    try:
        df = pd.read_sql_query("SELECT * FROM ferry_records ORDER BY date DESC LIMIT 30", conn)
        now_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')

        rows_html = ""
        for _, r in df.iterrows():
            pred = r['predicted_status'] if pd.notna(r['predicted_status']) else "-"
            actual = r['actual_status'] if pd.notna(r['actual_status']) and r['actual_status'] else "確認中"
            mode = r['prediction_mode'] if pd.notna(r['prediction_mode']) else "-"
            
            color = "#e74c3c" if pred == "欠航予想" else ("#f39c12" if pred == "注意予想" else "#2ecc71")
            
            rows_html += f"""
