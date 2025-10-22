import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime
import os
import numpy as np
import plotly.graph_objs as go

# -------------------------
# Configuration
# -------------------------
INFLUX_TOKEN  = os.environ.get("INFLUX_TOKEN")
INFLUX_URL    = os.environ.get("INFLUX_URL")
INFLUX_ORG    = os.environ.get("INFLUX_ORG")
INFLUX_BUCKET = "dashboard-practise"

MEASUREMENT = "tracker_data"
START_TIME  = "2025-10-22T06:17:00Z"  # Adjust as needed

# Optional: restrict by device prefix; set "" to disable
DEVICE_PREFIX_FILTER = "satellite"     # "" to disable

# -------------------------
# Connect to InfluxDB (safe)
# -------------------------
client = None
query_api = None
try:
    if INFLUX_URL and INFLUX_TOKEN and INFLUX_ORG:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
except Exception as e:
    # Defer any hard failure to callbacks; keep layout alive
    print(f"Influx init warning: {e}")

# -------------------------
# Fields present in your decoder output
# -------------------------
FIELDS = [
    "version", "release", "counter",
    "hoursUptime","almanacValidFrom","satId",
    "temperature","pressure","humidity","batteryVoltage",
    "booleanData","hall","userButton","automatedMode",
]

# What we keep from Flux (_time is renamed to 'time' later)
KEEP_COLS = ["_time","device"] + FIELDS

# Proper Flux array literal: ["_time","device",...]
KEEP_COLS_FLUX = "[" + ",".join(f"\"{c}\"" for c in KEEP_COLS) + "]"

NON_PARAM_COLS = {"time","device","automatedMode"}  # not plotted

# -------------------------
# Data I/O (now ONLY used inside callbacks)
# -------------------------
def load_all_data():
    if query_api is None:
        return pd.DataFrame(columns=["time","device"] + FIELDS)

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {START_TIME})
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {KEEP_COLS_FLUX})
    '''
    try:
        df = query_api.query_data_frame(query)
    except Exception as e:
        print(f"Initial query error: {e}")
        return pd.DataFrame(columns=["time","device"] + FIELDS)

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)
    if df.empty or "_time" not in df.columns:
        return pd.DataFrame(columns=["time","device"] + FIELDS)

    df = df.rename(columns={"_time":"time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    if DEVICE_PREFIX_FILTER:
        df = df[df["device"].astype(str).str.startswith(DEVICE_PREFIX_FILTER)]

    for c in ["time","device"] + FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def load_since(iso_last_time):
    if query_api is None or not iso_last_time:
        return pd.DataFrame()

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: time(v: "{iso_last_time}"))
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {KEEP_COLS_FLUX})
    '''
    try:
        new_df = query_api.query_data_frame(query)
    except Exception as e:
        print(f"Incremental query error: {e}")
        return pd.DataFrame()

    if isinstance(new_df, list):
        new_df = pd.concat(new_df, ignore_index=True)
    if new_df.empty:
        return pd.DataFrame()

    new_df = new_df.rename(columns={"_time":"time"})
    new_df["time"] = pd.to_datetime(new_df["time"], errors="coerce", utc=True)

    if DEVICE_PREFIX_FILTER:
        new_df = new_df[new_df["device"].astype(str).str.startswith(DEVICE_PREFIX_FILTER)]

    for c in ["time","device"] + FIELDS:
        if c not in new_df.columns:
            new_df[c] = np.nan
    return new_df

# -------------------------
# App shell (no data work here)
# -------------------------
app = dash.Dash(__name__)
app.title = "Beehive Monitoring Dashboard"
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      html, body { margin: 0; background: #3c1361; height: 100%; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
'''

app.layout = html.Div(
    style={"backgroundColor":"#3c1361","color":"white","height":"100vh","padding":"10px",
           "fontFamily":"Ubuntu","display":"flex","flexDirection":"column"},
    children=[
        html.Img(src="/assets/logo.png",
                 style={"position":"absolute","top":"10px","right":"10px","height":"120px","zIndex":"999"}),
        html.H2("Lacuna Space Tracker Demo",
                style={"fontSize":"3em","marginBottom":"10px","marginTop":"0"}),

        # Stores (start empty—safe for layout)
        dcc.Store(id="data-store", data=[]),
        dcc.Store(id="selected-device", data=None),

        html.Div(
            style={"display":"flex","gap":"12px","height":"calc(100vh - 170px)"},
            children=[
                html.Div(
                    style={"flex":"0 0 420px","background":"#4a1a78","borderRadius":"16px",
                           "padding":"12px","overflow":"hidden","display":"flex","flexDirection":"column"},
                    children=[
                        html.H3("Devices", style={"marginTop":0}),
                        dash_table.DataTable(
                            id="device-table",
                            columns=[
                                {"name":"Device","id":"device"},
                                {"name":"Last Seen (UTC)","id":"last_seen"},
                                {"name":"satId","id":"satId"},
                                {"name":"Temp (°C)","id":"temperature"},
                                {"name":"Battery (V)","id":"batteryVoltage"},
                                {"name":"Uptime (h)","id":"hoursUptime"},
                            ],
                            data=[],
                            style_table={"height":"100%","overflowY":"auto"},
                            style_cell={"backgroundColor":"#4a1a78","color":"white"},
                            style_header={"backgroundColor":"#5b2492","fontWeight":"bold"},
                            style_data_conditional=[{"if":{"state":"selected"},"backgroundColor":"#6d37a8"}],
                            row_selectable="single",
                            page_action="none",
                            sort_action="native",
                            filter_action="native",
                            cell_selectable=True,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex":"1 1 auto","background":"#4a1a78","borderRadius":"16px",
                           "padding":"12px","overflow":"auto"},
                    children=[
                        html.Div(id="device-title", style={"fontSize":"1.4rem","marginBottom":"8px"}),
                        html.Div(id="plots-container"),
                    ],
                ),
            ],
        ),

        dcc.Interval(id="interval", interval=30*1000, n_intervals=0),
    ],
)

# -------------------------
# Helpers
# -------------------------
def infer_parameter_columns(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    cols = []
    for c
