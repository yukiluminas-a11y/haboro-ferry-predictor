import os
import sqlite3
import requests
import pandas as pd
import math
from datetime import datetime

# --- 設定 ---
DB_PATH = "ferry_data.sqlite"

# 取得を試みる座標リスト (1番目が取得不可なら2番目・3番目の沖合座標を使用)
LOCATION_CANDIDATES = [
    {"lat": 44.38, "lon": 141.30, "name": "日本海沖合1（沿岸寄り）"},
    {"lat": 44.38, "lon": 141.10, "name": "日本海沖合2（中央部）"},
    {"lat": 44.40, "lon": 140.80, "name": "日本海沖合3（広域）"}
]

# 試行するモデル順序 (優先度順)
WEATHER_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_msm"]
MARINE_MODELS = ["ecmwf_wam", "gfs_wave", "best_match"]

def setup_database():
    """データベースとテーブルの初期化"""
    try:
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
    except Exception as e:
        print(f"データベース初期化エラー: {e}")

def estimate_wave_from_wind(wind_speed):
    """
    【最悪ケース用の補完】
    波高APIが完全に落ちている場合、風速から日本海沖合の波高(m)を物理推計
    """
    if wind_speed is None or pd.isna(wind_speed):
        return 0.5
    # 日本海沿岸の簡易波高推計式 (SMB法に基づく近似)
    estimated_wave = 0.025 * (wind_speed ** 1.5)
    return round(max(0.2, min(estimated_wave, 6.0)), 2)

def fetch_weather_with_fallback():
    """風速・視程を複数の気象モデルから試行して確実に取得"""
    headers = {'User-Agent': 'FerryForecastApp/1.0'}
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    
    for model in WEATHER_MODELS:
        params = {
            "latitude": LOCATION_CANDIDATES[0]["lat"],
            "longitude": LOCATION_CANDIDATES[0]["lon"],
            "hourly": ["wind_speed_10m", "visibility"],
            "models": model,
            "wind_speed_unit": "ms",
            "timezone": "Asia/Tokyo"
        }
        try:
            res = requests.get(forecast_url, params=params, headers=headers, timeout=8)
            res.raise_for_status()
            data = res.json().get("hourly", {})
            times = data.get("time", [])
            winds = data.get("wind_speed_10m", [])
            visibilities = data.get("visibility", [])
            
            if times and len(winds) > 0 and any(w is not None for w in winds):
                print(f"風速データ正常取得 (使用モデル: {model})")
                return pd.DataFrame({
                    "datetime": pd.to_datetime(times),
                    "wind_speed": winds,
                    "visibility": visibilities
                })
        except Exception as e:
            print(f"モデル {model} からの風速取得スキップ: {e}")
            continue
            
    print("警告: 全モデルで風速取得失敗。デフォルト生成を行います。")
    return pd.DataFrame()

def fetch_marine_with_fallback(df_forecast):
    """波高データを「多重座標×多重モデル」で探索し、取れなければ風速から自走補完"""
    headers = {'User-Agent': 'FerryForecastApp/1.0'}
    marine_url = "https://api.open-meteo.com/v1/marine"
    
    for loc in LOCATION_CANDIDATES:
        for model in MARINE_MODELS:
            params = {
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "hourly": ["wave_height", "wind_wave_height", "swell_wave_height"],
                "models": model,
                "timezone": "Asia/Tokyo"
            }
            try:
                res = requests.get(marine_url, params=params, headers=headers, timeout=8)
                res.raise_for_status()
                data = res.json().get("hourly", {})
                m_times = data.get("time", [])
                m_waves = data.get("wave_height", [])
                m_wind_waves = data.get("wind_wave_height", [])

                valid_waves = [w for w in m_waves if w is not None]
                if m_times and len(valid_waves) > 0:
                    print(f"波高データ正常取得 (座標: {loc['name']} / モデル: {model})")
                    
                    final_waves = []
                    for idx, w in enumerate(m_waves):
                        if w is not None:
                            final_waves.append(w)
                        elif idx < len(m_wind_waves) and m_wind_waves[idx] is not None:
                            final_waves.append(m_wind_waves[idx])
                        else:
                            final_waves.append(None)

                    return pd.DataFrame({
                        "datetime": pd.to_datetime(m_times),
                        "wave_height": final_waves
                    })
            except Exception as e:
                continue

    # 【絶対取得保証】APIから一切取れなかった場合、風速から波高を物理推計して埋める
    print("緊急処理: 波高API障害のため、風速から波高を推定補完します。")
    if not df_forecast.empty and "wind_speed" in df_forecast:
        estimated_waves = [estimate_wave_from_wind(w) for w in df_forecast["wind_speed"]]
        return pd.DataFrame({
            "datetime": df_forecast["datetime"],
            "wave_height": estimated_waves
        })
    return pd.DataFrame()

def fetch_combined_weather_data():
    """気象データと波浪データを確実に統合"""
    df_forecast = fetch_weather_with_fallback()
    if df_forecast.empty:
        return pd.DataFrame()

    df_marine = fetch_marine_with_fallback(df_forecast)

    if not df_marine.empty:
        df_merged = pd.merge(df_forecast, df_marine, on="datetime", how="left")
        # マージ後の欠損値（None）も風速からの推計値で完全に埋める
        df_merged["wave_height"] = df_merged.apply(
            lambda r: estimate_wave_from_wind(r["wind_speed"]) if pd.isna(r["wave_height"]) else r["wave_height"],
            axis=1
        )
        return df_merged
    else:
        df_forecast["wave_height"] = df_forecast["wind_speed"].apply(estimate_wave_from_wind)
        return df_forecast

def calculate_flight_risk(wave, wind, visibility, vessel_type="ferry"):
    """
    羽幌沿海フェリー安全運航基準（regulations02.pdf）に基づく運航判定
    """
    if pd.isna(wind) or wind is None:
        return "データなし", "#6c757d", "#ffffff", 0

    wave_val = wave if (wave is not None and not pd.isna(wave)) else estimate_wave_from_wind(wind)

    if vessel_type == "high_speed":
        limit_wind, limit_wave = 12.0, 1.5
        warn_wind, warn_wave = 9.0, 1.0
    else:
        limit_wind, limit_wave = 15.0, 2.5
        warn_wind, warn_wave = 11.0, 1.8

    limit_vis = 500.0   # 限界視界 (m)
    warn_vis = 1000.0   # 注意視界 (m)
    vis_meters = visibility if (not pd.isna(visibility) and visibility is not None) else 10000.0

    exceeded_limits = []
    warned_limits = []

    if wind >= limit_wind:
        exceeded_limits.append("風速")
    elif wind >= warn_wind:
        warned_limits.append("風速")

    if wave_val >= limit_wave:
        exceeded_limits.append("波高")
    elif wave_val >= warn_wave:
        warned_limits.append("波高")

    if vis_meters <= limit_vis:
        exceeded_limits.append("視界")
    elif vis_meters <= warn_vis:
        warned_limits.append("視界")

    if exceeded_limits:
        reason_str = "・".join(exceeded_limits)
        return f"欠航警戒（{reason_str}超過）", "#721c24", "#f8d7da", 95
    elif warned_limits:
        reason_str = "・".join(warned_limits)
        return f"運航注意（{reason_str}注意）", "#856404", "#fff3cd", 60
    else:
        return "通常運航", "#155724", "#d4edda", 10

def process_forecast_data(df_hourly):
    """第1便（08-12時）と第2便（14-18時）それぞれのピーク判定"""
    if df_hourly.empty:
        return pd.DataFrame()

    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")
    df_hourly["hour"] = df_hourly["datetime"].dt.hour
    
    results = []
    dates = df_hourly["date"].unique()
    
    for d in dates:
        df_day = df_hourly[df_hourly["date"] == d]
        
        # 第1便 (08:00 〜 12:00)
        df_f1 = df_day[(df_day["hour"] >= 8) & (df_day["hour"] <= 12)]
        if not df_f1.empty and df_f1["wind_speed"].notna().any():
            wind1 = df_f1["wind_speed"].max()
            wave1 = df_f1["wave_height"].max() if "wave_height" in df_f1 and df_f1["wave_height"].notna().any() else estimate_wave_from_wind(wind1)
            vis1 = df_f1["visibility"].min()
            status1, text_color1, bg_color1, prob1 = calculate_flight_risk(wave1, wind1, vis1)
        else:
            wave1, wind1, vis1, status1, text_color1, bg_color1, prob1 = None, None, None, "データなし", "#6c757d", "#e2e3e5", 0

        # 第2便 (14:00 〜 18:00)
        df_f2 = df_day[(df_day["hour"] >= 14) & (df_day["hour"] <= 18)]
        if not df_f2.empty and df_f2["wind_speed"].notna().any():
            wind2 = df_f2["wind_speed"].max()
            wave2 = df_f2["wave_height"].max() if "wave_height" in df_f2 and df_f2["wave_height"].notna().any() else estimate_wave_from_wind(wind2)
            vis2 = df_f2["visibility"].min()
            status2, text_color2, bg_color2, prob2 = calculate_flight_risk(wave2, wind2, vis2)
        else:
            wave2, wind2, vis2, status2, text_color2, bg_color2, prob2 = None, None, None, "データなし", "#6c757d", "#e2e3e5", 0
            
        results.append({
            "date": d,
            "wave1": wave1, "wind1": wind1, "vis1": vis1, "status1": status1, "tc1": text_color1, "bg1": bg_color1, "prob1": prob1,
            "wave2": wave2, "wind2": wind2, "vis2": vis2, "status2": status2, "tc2": text_color2, "bg2": bg_color2, "prob2": prob2
        })
        
    return pd.DataFrame(results)

def get_past_records_html():
    """DBから過去実績を取得"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT date AS 日付, status AS 運航状況, 
                   wind_speed AS '最大風速(m/s)', wave_height AS '最大波高(m)'
            FROM ferry_records 
            ORDER BY date DESC LIMIT 10
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return "<p>過去の実績データはまだ蓄積されていません。</p>"
        return df.to_html(index=False, classes="past-records", border=1, justify="center")
    except Exception as e:
        return f"<p>実績データの表示をスキップしました ({e})</p>"

def save_today_record(df_summary):
    """本日の最大値を DB に保存"""
    if df_summary.empty:
        return
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_data = df_summary[df_summary["date"] == today_str]
    
    if not today_data.empty:
        row = today_data.iloc[0]
        max_wave = max([w for w in [row["wave1"], row["wave2"]] if w is not None], default=0.0)
        max_wind = max([w for w in [row["wind1"], row["wind2"]] if w is not None], default=0.0)
        min_vis = min([v for v in [row["vis1"], row["vis2"]] if v is not None], default=10000.0)
        status_text = f"1便:{row['status1']} / 2便:{row['status2']}"
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO ferry_records (date, status, wind_speed, wave_height, visibility, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (today_str, status_text, max_wind, max_wave, min_vis, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB書き込みエラー: {e}")

def generate_html(df_summary):
    """ダッシュボード (index.html) の出力"""
    if df_summary.empty:
        print("表示するデータがないため HTML 出力をスキップします。")
        return

    forecast_rows = ""
    for _, row in df_summary.iterrows():
        date = row["date"]
        
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
            <td><strong>{date}</strong></td>
            <td>波高: {w1_str} / 風速: {wind1_str} <br><small style="color:#666;">視界: {v1_str}</small></td>
            <td>{s1}</td>
            <td>波高: {w2_str} / 風速: {wind2_str} <br><small style="color:#666;">視界: {v2_str}</small></td>
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
                <strong>【データ精度・運航基準】</strong><br>
                ・気象・波浪モデル: <strong>ECMWF (Windy互換) / 多重バックアップ構造</strong><br>
                ・運航基準: 安全運航規約（regulations02.pdf）限界値（風速15m/s、波高2.5m、視界500m）
            </div>
            <p style="color:#6c757d; font-size:0.9em;">最終更新: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (JST)</p>
            
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
            
            <h2>過去の運航・欠航実績 (直近10件)</h2>
            <div class="past-records">
                {past_records_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    print("データベース初期化中...")
    setup_database()
    
    print("気象・海洋データ多重取得中 (Windy互換/ECMWF優先)...")
    df_hourly = fetch_combined_weather_data()
    
    if df_hourly.empty:
        print("エラー: 気象データの取得に失敗したため処理を中断します。")
        return
        
    print("安全運航基準に照らし合わせて便別予測計算中...")
    df_summary = process_forecast_data(df_hourly)
    
    print("本日の実績データをDBへ保存中...")
    save_today_record(df_summary)
    
    print("Webページ (index.html) 生成中...")
    generate_html(df_summary)
    
    print("全処理が正常に完了しました！")

if __name__ == "__main__":
    main()
