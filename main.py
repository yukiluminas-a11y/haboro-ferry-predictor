import os
import sqlite3
import math
import requests
import joblib
import pandas as pd
import xml.etree.ElementTree as ET
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
# 7. index.html 自動生成処理 (ElementTreeによるDOMツリー生成)
# ---------------------------------------------------------
def generate_html(conn):
    try:
        today_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d')
        now_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')

        df_forecast = pd.read_sql_query("SELECT * FROM ferry_records WHERE date >= ? ORDER BY date ASC LIMIT 10", conn, params=(today_str,))
        df_history = pd.read_sql_query("SELECT * FROM ferry_records WHERE date <= ? ORDER BY date DESC LIMIT 30", conn, params=(today_str,))

        html = ET.Element('html', lang='ja')
        head = ET.SubElement(html, 'head')
        
        meta1 = ET.SubElement(head, 'meta', charset='UTF-8')
        meta2 = ET.SubElement(head, 'meta', name='viewport', content='width=device-width, initial-scale=1.0')
        
        title = ET.SubElement(head, 'title')
        title.text = '羽幌沿海フェリー 運航予測'

        style = ET.SubElement(head, 'style')
        style.text = (
            'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:20px;background:#f4f6f8;color:#333;}'
            'h1{font-size:1.5rem;}'
            'h2{font-size:1.2rem;margin-top:30px;color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:5px;}'
            '.meta{font-size:0.85rem;color:#666;margin-bottom:15px;}'
            'table{width:100%;border-collapse:collapse;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.1);border-radius:4px;overflow:hidden;margin-bottom:20px;}'
            'th,td{padding:10px 12px;text-align:center;border-bottom:1px solid #eee;font-size:0.9rem;}'
            'th{background:#2c3e50;color:#fff;font-weight:normal;}'
            'tr:hover{background:#f8f9fa;}'
        )

        body = ET.SubElement(html, 'body')
        
        h1 = ET.SubElement(body, 'h1')
        h1.text = '羽幌沿海フェリー 運航予測・実績'

        meta_div = ET.SubElement(body, 'div', {'class': 'meta'})
        meta_div.text = f'最終更新時刻: {now_str} (JST)'

        def append_section(parent_elem, section_title, df_data):
            h2 = ET.SubElement(parent_elem, 'h2')
            h2.text = section_title
            
            table = ET.SubElement(parent_elem, 'table')
            thead = ET.SubElement(table, 'thead')
            tr_head = ET.SubElement(thead, 'tr')
            
            headers = ['日付', 'AI予測', '公式実績', '最大風速', '最大波高', '判定モード', '更新時刻']
            for h in headers:
                th = ET.SubElement(tr_head, 'th')
                th.text = h

            tbody = ET.SubElement(table, 'tbody')
            for _, r in df_data.iterrows():
                pred = str(r['predicted_status']) if pd.notna(r['predicted_status']) else '-'
                actual = str(r['actual_status']) if pd.notna(r['actual_status']) and r['actual_status'] else '確認中'
                mode = str(r['prediction_mode']) if pd.notna(r['prediction_mode']) else '-'

                if pred == '欠航予想':
                    color = '#e74c3c'
                elif pred == '注意予想':
                    color = '#f39c12'
                else:
                    color = '#2ecc71'

                tr = ET.SubElement(tbody, 'tr')
                
                td_date = ET.SubElement(tr, 'td')
                td_date.text = str(r['date'])

                td_pred = ET.SubElement(tr, 'td', style=f'color:{color};font-weight:bold;')
                td_pred.text = pred

                td_actual = ET.SubElement(tr, 'td')
                td_actual.text = actual

                td_wind = ET.SubElement(tr, 'td')
                td_wind.text = f"{r['max_wind_speed']} m/s"

                td_wave = ET.SubElement(tr, 'td')
                td_wave.text = f"{r['max_wave_height']} m"

                td_mode = ET.SubElement(tr, 'td')
                td_mode.text = mode

                td_upd = ET.SubElement(tr, 'td')
                td_upd.text = str(r['updated_at'])

        append_section(body, '📅 向こう10日間の運航予測', df_forecast)
        append_section(body, '📜 過去の運航実績・予測履歴', df_history)

        doc_type = '\n'
        raw_xml = ET.tostring(html, encoding='utf-8').decode('utf-8')
        
        with open(HTML_FILE, 'w', encoding='utf-8') as f:
            f.write(doc_type + raw_xml)

        print("10日分対応 index.html の生成が正常完了しました。")
    except Exception as e:
        print(f"[ERROR] HTML生成失敗: {e}")

# ---------------------------------------------------------
# 8. メイン実行処理
# ---------------------------------------------------------
def main():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    
    today_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d')
    now_timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')

    clf = train_and_update_model(conn)
    if clf is None and os.path.exists(MODEL_FILE):
        try:
            clf = joblib.load(MODEL_FILE)
        except Exception:
            clf = None

    forecasts_10days = fetch_weather_forecast_10days()
    official_status, raw_text = fetch_official_status()

    cursor = conn.cursor()

    simulated_cancels = get_consecutive_cancels(conn, today_str)

    for w_info in forecasts_10days:
        date_str = w_info['date']
        predicted, mode = predict_status_for_day(w_info, simulated_cancels, clf)

        final_actual_status = None
        if date_str == today_str:
            cursor.execute("SELECT actual_status FROM ferry_records WHERE date = ?", (today_str,))
            existing_row = cursor.fetchone()
            if existing_row and existing_row[0] in ['平常運航', '欠航']:
                final_actual_status = existing_row[0]
            else:
                final_actual_status = official_status

        cursor.execute('''
            INSERT INTO ferry_records (
                date, predicted_status, actual_status, max_wind_speed, max_wave_height,
                min_visibility, raw_official_text, updated_at, wind_direction_deg,
                prev_day_max_wave, prediction_mode, consecutive_cancels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                predicted_status = excluded.predicted_status,
                actual_status = COALESCE(excluded.actual_status, ferry_records.actual_status),
                max_wind_speed = excluded.max_wind_speed,
                max_wave_height = excluded.max_wave_height,
                min_visibility = excluded.min_visibility,
                raw_official_text = COALESCE(excluded.raw_official_text, ferry_records.raw_official_text),
                updated_at = excluded.updated_at,
                wind_direction_deg = excluded.wind_direction_deg,
                prev_day_max_wave = excluded.prev_day_max_wave,
                prediction_mode = excluded.prediction_mode,
                consecutive_cancels = excluded.consecutive_cancels
        ''', (
            date_str, predicted, final_actual_status,
            w_info['max_wind_speed'], w_info['max_wave_height'],
            w_info['min_visibility'], raw_text if date_str == today_str else "", now_timestamp,
            w_info['wind_direction_deg'], w_info['prev_day_max_wave'], mode, simulated_cancels
        ))

        if predicted == "欠航予想":
            simulated_cancels += 1
        else:
            simulated_cancels = 0

    conn.commit()
    print(f"[{now_timestamp}] 連続欠航指標組み込み済み10日分予報DB更新が完了しました。")

    generate_html(conn)
    conn.close()

if __name__ == '__main__':
    main()
