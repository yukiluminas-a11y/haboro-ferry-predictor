import os
import re
import sqlite3
import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

# --- 基本設定 ---
DB_PATH = "ferry_data.sqlite"
TARGET_URL = "https://haboro-enkai.com/"

# 取得を試みる座標リスト
LOCATION_CANDIDATES = [
    {"lat": 44.38, "lon": 141.30, "name": "日本海沖合1（沿岸寄り）"},
    {"lat": 44.38, "lon": 141.10, "name": "日本海沖合2（中央部）"},
    {"lat": 44.40, "lon": 140.80, "name": "日本海沖合3（広域）"}
]

WEATHER_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_msm"]
MARINE_MODELS = ["ecmwf_wam", "gfs_wave", "best_match"]

# ==========================================
# 1. データベース管理・初期化
# ==========================================
def setup_database():
    """データベースおよびテーブルの初期化"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 予測データ・気象・公式実績の一括管理テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ferry_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                predicted_status TEXT,   -- システム予測 (例: 欠航警戒 / 通常運航)
                actual_status TEXT,      -- 公式実績 (例: 平常運航 / 欠航)
                max_wind_speed REAL,     -- 日中最大風速 (m/s)
                max_wave_height REAL,    -- 日中最大波高 (m)
                min_visibility REAL,     -- 日中最悪視界 (m)
                raw_official_text TEXT,  -- 公式サイト原文
                updated_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"データベース初期化エラー: {e}")

# ==========================================
# 2. 羽幌沿海フェリー公式サイトからの実績スクレイピング
# ==========================================
def parse_status_text(text):
    """テキストから状態を判別"""
    if "欠航" in text:
        return "欠航"
    elif "平常運航" in text or "通常運航" in text:
        return "平常運航"
    elif "条件付" in text:
        return "条件付運航"
    elif "見合わせ" in text:
        return "見合わせ"
    else:
        return "不明"

def fetch_actual_ferry_status():
    """公式サイトから本日の実際の運航結果を取得"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        res = requests.get(TARGET_URL, headers=headers, timeout=10)
        res.encoding = res.apparent_encoding
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, "html.parser")
        page_text = soup.get_text()
        
        actual_status = parse_status_text(page_text)
        print(f"公式サイト実績取得成功: [{actual_status}]")
        return actual_status, page_text[:150].replace("\n", " ").strip()
    except Exception as e:
        print(f"公式サイト実績取得失敗: {e}")
        return "未取得", ""

# ==========================================
# 3. 気象データ取得 (多重フォールバック & 風速推計)
# ==========================================
def estimate_wave_from_wind(wind_speed):
    """波高API失敗時の風速物理推計式"""
    if wind_speed is None or pd.isna(wind_speed):
        return 0.5
    estimated_wave = 0.025 * (wind_speed ** 1.5)
    return round(max(0.2, min(estimated_wave, 6.0)), 2)

def fetch_weather_data():
    """気象・海洋データの多重バックアップ取得"""
    headers = {'User-Agent': 'FerryForecastApp/1.0'}
    
    # 1. 風速・視程
    df_forecast = pd.DataFrame()
    for model in WEATHER_MODELS:
        try:
            res = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": LOCATION_CANDIDATES[0]["lat"],
                    "longitude": LOCATION_CANDIDATES[0]["lon"],
                    "hourly": ["wind_speed_10m", "visibility"],
                    "models": model,
                    "wind_speed_unit": "ms",
                    "timezone": "Asia/Tokyo"
                },
                headers=headers, timeout=8
            )
            res.raise_for_status()
            data = res.json().get("hourly", {})
            times, winds, vis = data.get("time", []), data.get("wind_speed_10m", []), data.get("visibility", [])
            
            if times and any(w is not None for w in winds):
                # 視界のNaN対策
                vis_clean = [v if (v is not None and not pd.isna(v)) else 10000.0 for v in vis]
                df_forecast = pd.DataFrame({"datetime": pd.to_datetime(times), "wind_speed": winds, "visibility": vis_clean})
                print(f"風速データ取得成功 (モデル: {model})")
                break
        except Exception:
            continue

    if df_forecast.empty:
        return pd.DataFrame()

    # 2. 波高
    df_marine = pd.DataFrame()
    for loc in LOCATION_CANDIDATES:
        for model in MARINE_MODELS:
            try:
                res = requests.get(
                    "https://api.open-meteo.com/v1/marine",
                    params={
                        "latitude": loc["lat"],
                        "longitude": loc["lon"],
                        "hourly": ["wave_height"],
                        "models": model,
                        "timezone": "Asia/Tokyo"
                    },
                    headers=headers, timeout=8
                )
                res.raise_for_status()
                m_data = res.json().get("hourly", {})
                m_times, m_waves = m_data.get("time", []), m_data.get("wave_height", [])
                
                if m_times and any(w is not None for w in m_waves):
                    df_marine = pd.DataFrame({"datetime": pd.to_datetime(m_times), "wave_height": m_waves})
                    print(f"波高データ取得成功 (座標: {loc['name']} / モデル: {model})")
                    break
            except Exception:
                continue
        if not df_marine.empty:
            break

    # マージと補完処理
    if not df_marine.empty:
        df_merged = pd.merge(df_forecast, df_marine, on="datetime", how="left")
    else:
        df_merged = df_forecast
        df_merged["wave_height"] = None

    df_merged["wave_height"] = df_merged.apply(
        lambda r: estimate_wave_from_wind(r["wind_speed"]) if pd.isna(r["wave_height"]) or r["wave_height"] is None else r["wave_height"],
        axis=1
    )
    return df_merged

# ==========================================
# 4. 運航リスク判定ロジック
# ==========================================
def calculate_flight_risk(wave, wind, visibility):
    """安全運航基準（風速15m/s、波高2.5m、視界500m）判定"""
    if pd.isna(wind) or wind is None:
        return "データなし", "#6c757d", "#ffffff"

    wave_val = wave if (wave is not None and not pd.isna(wave)) else estimate_wave_from_wind(wind)
    vis_meters = visibility if (not pd.isna(visibility) and visibility is not None) else 10000.0

    exceeded = []
    warned = []

    if wind >= 15.0: exceeded.append("風速")
    elif wind >= 11.0: warned.append("風速")

    if wave_val >= 2.5: exceeded.append("波高")
    elif wave_val >= 1.8: warned.append("波高")

    if vis_meters <= 500.0: exceeded.append("視界")
    elif vis_meters <= 1000.0: warned.append("視界")

    if exceeded:
        return f"欠航警戒（{'・'.join(exceeded)}超過）", "#721c24", "#f8d7da"
    elif warned:
        return f"運航注意（{'・'.join(warned)}注意）", "#856404", "#fff3cd"
    else:
        return "通常運航", "#155724", "#d4edda"

def process_forecast_data(df_hourly):
    """便別（午前/午後）の気象ピーク値およびリスク判定"""
    if df_hourly.empty: return pd.DataFrame()

    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")
    df_hourly["hour"] = df_hourly["datetime"].dt.hour
    
    results = []
    for d in df_hourly["date"].unique():
        df_day = df_hourly[df_hourly["date"] == d]
        
        # 午前便 (08-12時)
        df_f1 = df_day[(df_day["hour"] >= 8) & (df_day["hour"] <= 12)]
        if not df_f1.empty:
            w1, wave1, vis1 = df_f1["wind_speed"].max(), df_f1["wave_height"].max(), df_f1["visibility"].min()
            st1, tc1, bg1 = calculate_flight_risk(wave1, w1, vis1)
        else:
            w1, wave1, vis1, st1, tc1, bg1 = None, None, None, "データなし", "#6c757d", "#e2e3e5"

        # 午後便 (14-18時)
        df_f2 = df_day[(df_day["hour"] >= 14) & (df_day["hour"] <= 18)]
        if not df_f2.empty:
            w2, wave2, vis2 = df_f2["wind_speed"].max(), df_f2["wave_height"].max(), df_f2["visibility"].min()
            st2, tc2, bg2 = calculate_flight_risk(wave2, w2, vis2)
        else:
            w2, wave2, vis2, st2, tc2, bg2 = None, None, None, "データなし", "#6c757d", "#e2e3e5"

        results.append({
            "date": d,
            "wave1": wave1, "wind1": w1, "vis1": vis1, "status1": st1, "tc1": tc1, "bg1": bg1,
            "wave2": wave2, "wind2": w2, "vis2": vis2, "status2": st2, "tc2": tc2, "bg2": bg2
        })
    return pd.DataFrame(results)

# ==========================================
# 5. 精度分析・学習・ログ保存処理
# ==========================================
def save_and_evaluate_today(df_summary, actual_status, raw_text):
    """本日分の「気象データ + 予測判定 + 公式実績」をDBへ自動更新保存"""
    if df_summary.empty: return

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    today_data = df_summary[df_summary["date"] == today_str]

    if not today_data.empty:
        row = today_data.iloc[0]
        max_wind = max([w for w in [row["wind1"], row["wind2"]] if w is not None], default=0.0)
        max_wave = max([w for w in [row["wave1"], row["wave2"]] if w is not None], default=0.0)
        min_vis = min([v for v in [row["vis1"], row["vis2"]] if v is not None], default=10000.0)
        
        pred_status = f"1便:{row['status1']} / 2便:{row['status2']}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ferry_records 
                (date, predicted_status, actual_status, max_wind_speed, max_wave_height, min_visibility, raw_official_text, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    predicted_status = excluded.predicted_status,
                    actual_status = CASE WHEN excluded.actual_status != '未取得' THEN excluded.actual_status ELSE ferry_records.actual_status END,
                    max_wind_speed = excluded.max_wind_speed,
                    max_wave_height = excluded.max_wave_height,
                    min_visibility = excluded.min_visibility,
                    raw_official_text = CASE WHEN excluded.raw_official_text != '' THEN excluded.raw_official_text ELSE ferry_records.raw_official_text END,
                    updated_at = excluded.updated_at
            ''', (today_str, pred_status, actual_status, max_wind, max_wave, min_vis, raw_text, now_str))
            conn.commit()
            conn.close()
            print("本日の予測および実績データをDBに蓄積しました。")
        except Exception as e:
            print(f"DB更新失敗: {e}")

def generate_accuracy_report_html():
    """蓄積データから予測精度と傾向を自動算出・表示"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM ferry_records WHERE actual_status != '未取得' AND actual_status != '不明'", conn)
        conn.close()

        if df.empty or len(df) < 2:
            return "<p style='color:#6c757d;'>蓄積された実績データが十分でないため、精度分析は準備中です（データ数: 2件以上で自動表示）。</p>"

        # 簡易精度判定ロジック
        correct = 0
        total = len(df)
        for _, row in df.iterrows():
            pred = row["predicted_status"]
            actual = row["actual_status"]
            is_pred_cancel = "欠航" in pred
            is_actual_cancel = "欠航" in actual
            if is_pred_cancel == is_actual_cancel:
                correct += 1

        accuracy = (correct / total) * 100
        return f"""
        <div style="background:#e9ecef; padding:15px; border-radius:5px; margin-bottom:20px;">
            <strong>【累積予測評価・傾向】</strong><br>
            ・解析データ件数: <strong>{total} 件</strong><br>
            ・欠航判定の適合率（精度）: <strong>{accuracy:.1f}%</strong><br>
            <small style="color:#555;">※毎日の実績と気象モデル数値を照合し、自動計算されています。</small>
        </div>
        """
    except Exception as e:
        return f"<p>精度レポート生成エラー: {e}</p>"

def get_past_records_table_html():
    """過去の実績テーブルを出力"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT date AS 日付, 
                   predicted_status AS システム予測, 
                   actual_status AS 公式実績, 
                   max_wind_speed AS '最大風速(m/s)', 
                   max_wave_height AS '最大波高(m)'
            FROM ferry_records 
            ORDER BY date DESC LIMIT 10
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return "<p>過去の実績データはまだ蓄積されていません。</p>"
        return df.to_html(index=False, classes="past-records", border=1, justify="center")
    except Exception as e:
        return f"<p>実績データの表示スキップ: {e}</p>"

# ==========================================
# 6. HTMLダッシュボード生成
# ==========================================
def generate_html(df_summary):
    """Webダッシュボード (index.html) の出力"""
    if df_summary.empty: return

    forecast_rows = ""
    for _, row in df_summary.iterrows():
        v1_str = f"{row['vis1']/1000:.1f}km" if row["vis1"] is not None else "-"
        w1_str = f"{row['wave1']:.1f}m" if row["wave1"] is not None else "-"
        wind1_str = f"{row['wind1']:.1f}m/s" if row["wind1"] is not None else "-"
        s1 = f'<span style="color:{row["tc1"]}; background-color:{row["bg1"]}; padding: 4px 8px; border-radius: 4px; font-weight:bold; display: inline-block;">{row["status1"]}</span>'

        v2_str = f"{row['vis2']/1000:.1f}km" if row["vis2"] is not None else "-"
        w2_str = f"{row['wave2']:.1f}m" if row["wave2"] is not None else "-"
        wind2_str = f"{row['wind2']:.1f}m/s" if row["wind2"] is not None else "-"
        s2 = f'<span style="color:{row["tc2"]}; background-color:{row["bg2"]}; padding: 4px 8px; border-radius: 4px; font-weight:bold; display: inline-block;">{row["status2"]}</span>'

        forecast_rows += f"""
        <tr>
            <td><strong>{row['date']}</strong></td>
            <td>波高: {w1_str} / 風速: {wind1_str} <br><small style="color:#666;">視界: {v1_str}</small></td>
            <td>{s1}</td>
            <td>波高: {w2_str} / 風速: {wind2_str} <br><small style="color:#666;">視界: {v2_str}</small></td>
            <td>{s2}</td>
        </tr>
        """

    accuracy_html = generate_accuracy_report_html()
    past_records_html = get_past_records_table_html()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>羽幌沿海フェリー 便別欠航予測・自動精度学習ダッシュボード</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 20px; line-height: 1.6; color: #212529; background-color: #f8f9fa; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: #ffffff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #1a252f; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; }}
            .notice {{ background-color: #e7f5ff; padding: 12px 15px; border-left: 5px solid #1c7ed6; margin-bottom: 20px; border-radius: 4px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background-color: #fff; }}
            th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: center; vertical-align: middle; }}
            th {{ background-color: #f1f3f5; color: #495057; font-weight: 600; }}
            .past-records table {{ width: 100%; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>羽幌沿海フェリー 便別欠航予測ダッシュボード</h1>
            <div class="notice">
                <strong>【システム運用状態】</strong><br>
                ・気象モデル: <strong>ECMWF (Windy互換) 多重冗長取得</strong><br>
                ・実績学習機能: <strong>有効（毎日の公式結果と気象数値を紐付け保存中）</strong>
            </div>
            <p style="color:#6c757d; font-size:0.9em;">最終更新: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (JST)</p>
            
            {accuracy_html}

            <h2>フェリー週間欠航予測（便別）</h2>
            <table>
                <thead>
                    <tr>
                        <th rowspan="2" style="width: 15%;">日付</th>
                        <th colspan="2" style="width: 42.5%;">第1便（午前便）</th>
                        <th colspan="2" style="width: 42.5%;">第2便（午後便）</th>
                    </tr>
                    <tr>
                        <th>気象データ</th>
                        <th>予測判定</th>
                        <th>気象データ</th>
                        <th>予測判定</th>
                    </tr>
                </thead>
                <tbody>
                    {forecast_rows}
                </tbody>
            </table>
            
            <h2>過去の運航実績・答え合わせ (直近10件)</h2>
            <div class="past-records">
                {past_records_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# ==========================================
# 7. メイン実行エントリーポイント
# ==========================================
def main():
    print("1. データベース初期化中...")
    setup_database()
    
    print("2. 羽幌沿海フェリー公式サイトから本日の実績を取得中...")
    actual_status, raw_text = fetch_actual_ferry_status()
    
    print("3. Open-Meteo API から気象・波浪予測データを多重取得中...")
    df_hourly = fetch_weather_data()
    
    if df_hourly.empty:
        print("エラー: 気象データの取得に失敗しました。")
        return

    print("4. 便別運航リスクの判定計算中...")
    df_summary = process_forecast_data(df_hourly)
    
    print("5. 予測結果と実績データをデータベースへ紐付け保存中...")
    save_and_evaluate_today(df_summary, actual_status, raw_text)
    
    print("6. ダッシュボード (index.html) 生成中...")
    generate_html(df_summary)
    
    print("処理が正常に完了しました！")

if __name__ == "__main__":
    main()
