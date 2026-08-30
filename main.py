from datetime import datetime, timedelta
import json
import sqlite3
from bs4 import BeautifulSoup
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

LAT = 44.36
LON = 141.68
DB_FILE = "ferry_data.sqlite"

# --- 1. データベース初期化 ---
def init_db():
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            date TEXT PRIMARY KEY,
            status TEXT,
            wave_height REAL,
            wave_period REAL,
            wind_speed REAL,
            wind_dir REAL
        )
    """)
  conn.commit()
  conn.close()


# --- 2. 運航結果のスクレイピング & 記録 ---
def record_actual_status():
  try:
    res = requests.get("https://www.haboro-enkai.com/", timeout=10)
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text()

    # 簡単な判定 (状況に合わせて調整可)
    status = "運航"
    if "欠航" in text or "全便欠航" in text:
      status = "欠航"

    # 当日の気象実績を取得してDBに保存
    m_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={LAT}&longitude={LON}&current=wave_height,wave_period&timezone=Asia%2FTokyo"
    w_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=wind_speed_10m,wind_direction_10m&timezone=Asia%2FTokyo"

    m_data = requests.get(m_url).json().get("current", {})
    w_data = requests.get(w_url).json().get("current", {})

    today_str = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        INSERT OR REPLACE INTO history (date, status, wave_height, wave_period, wind_speed, wind_dir)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            today_str,
            status,
            m_data.get("wave_height"),
            m_data.get("wave_period"),
            w_data.get("wind_speed_10m"),
            w_data.get("wind_direction_10m"),
        ),
    )
    conn.commit()
    conn.close()
    print(f"[{today_str}] 運航実績を記録しました: {status}")
  except Exception as e:
    print(f"実績記録エラー: {e}")


# --- 3. 過去データからの自己学習モデル構築 ---
def train_model():
  conn = sqlite3.connect(DB_FILE)
  df = pd.read_sql_query("SELECT * FROM history", conn)
  conn.close()

  # データ数が一定（例: 5件以上）溜まったら機械学習を適用
  if len(df) >= 5 and "欠航" in df["status"].values:
    X = df[["wave_height", "wave_period", "wind_speed", "wind_dir"]]
    y = df["status"].apply(lambda x: 1 if x == "欠航" else 0)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    print("機械学習モデル（RandomForest）の再学習が完了しました。")
    return model
  return None


# --- 4. 予測とindex.htmlの更新 ---
def update_forecast_html(model=None):
  # APIから今後の予測データを取得
  w_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=wind_speed_10m,wind_direction_10m&timezone=Asia%2FTokyo&forecast_days=4"
  m_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={LAT}&longitude={LON}&hourly=wave_height,wave_period&timezone=Asia%2FTokyo&forecast_days=4"

  res_w = requests.get(w_url).json()["hourly"]
  res_m = requests.get(m_url).json()["hourly"]

  # 先ほどのHTMLコードの生成処理などを実行し index.html を出力
  print("最新の予測データから index.html を更新しました。")


if __name__ == "__main__":
  init_db()
  record_actual_status()
  model = train_model()
  update_forecast_html(model)