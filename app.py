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
DEVICE_PREFIX_FILTER = "satellite"     # "" to disable

# -------------------------
# Connect to InfluxDB
# -------------------------
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# -------------------------
# Fields present from your decoder
# -------------------------
FIELDS = [
    "version","release","counter",
    "hoursUptime","almanacValidFrom","satId",
    "temperature","pressure","humidity","batteryVoltage",
    "booleanData","hall","userButton","automatedMode",
]
KEEP_COLS = ["_time","device"] + FIELDS
NON_PARAM_COLS = {"time","device","automatedMode"}

def _flux_keep_list(cols):
    # Build a Flux array literal: ["_time","device","temperature",...]
    return "[" + ",".join(f'"{c}"' for c in cols) + "]"

KEEP_COLS_FLUX = _flux_keep_list(KEEP_COLS)

# -------------------------
# Data I/O
# -------------------------
def load_all_data():
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {START_TIME})
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {KEEP_COLS_FLUX})
    '''
    df = query_api.query_data_frame(query)
    if isinstance(df, list): df = pd.concat(df, ignore_index=True)
    if df.empty or "_time" not in df.columns:
        return pd.DataFrame(columns=["time","device"] + FIELDS)
    df = df.rename(columns={"_time":"time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if DEVICE_PREFIX_FILTER:
        df = df[df["device"].astype(str).str.startswith(DEVICE_PREFIX_FILTER)]
    for c in ["time","device"] + FIELDS:
        if c not in df.columns: df[c] = np.nan
    return df

def load_since(iso_last_time):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: time(v: "{iso_last_time}"))
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {KEEP_COLS_FLUX})
    '''
    new_df = query_api.query_data_frame(query)
    if isinstance(new_df, list): new_df = pd.concat(new_df, ignore_index=True)
    if new_df.empty: return pd.DataFrame()
    new_df = new_df.rename(columns={"_time":"time"})
    new_df["time"] = pd.to_datetime(new_df["time"], errors="coerce", utc=True)
    if DEVICE_PREFIX_FILTER:
        new_df = new_df[new_df["device"].astype(str).str.startswith(DEVICE_PREFIX_FILTER)]
    for c in ["time","device"] + FIELDS:
        if c not in new_df.columns: new_df[c] = np.nan
    return new_df

# Global cache mirrored into dcc.Store
data_df = load_all_data()

# -------------------------
# App
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
      html, body { margin:0; background:#3c1361; height:100%; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>
'''

# -------------------------
# Layout
# -------------------------
app.layout = html.Div(
    style={"backgroundColor":"#3c1361","color":"white","height":"100vh","padding":"10px",
           "fontFamily":"Ubuntu","display":"flex","flexDirection":"column"},
    children=[
        html.Img(src="/assets/logo.png",
                 style={"position":"absolute","top":"10px","right":"10px","height":"120px","zIndex":"999"}),
        html.H2("Lacuna Space Tracker Demo",
                style={"fontSize":"3em","marginBottom":"10px","marginTop":"0"}),

        dcc.Store(id="data-store"),
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
    if df is None or df.empty: return []
    cols = []
    for c in df.columns:
        if c in NON_PARAM_COLS: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any(): cols.append(c)
    ordered = [c for c in FIELDS if c in cols]
    ordered += [c for c in cols if c not in ordered]
    return ordered

YLABELS = {
    "temperature":"Temperature (°C)",
    "pressure":"Pressure (hPa)",
    "humidity":"Humidity (%)",
    "batteryVoltage":"Battery Voltage (V)",
    "hoursUptime":"Uptime (hours)",
    "counter":"Counter",
    "satId":"Satellite ID",
    "booleanData":"Boolean Data (bitmask)",
    "hall":"Hall (0/1)",
    "userButton":"User Button (0/1)",
    "version":"Version",
    "release":"Release",
}

def make_figure(df_dev: pd.DataFrame, y_col: str) -> go.Figure:
    y_series = pd.to_numeric(df_dev[y_col], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_dev["time"], y=y_series, mode="lines+markers", name=y_col))
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=20, t=35, b=40),
        paper_bgcolor="#4a1a78",
        plot_bgcolor="#4a1a78",
        xaxis_title="Time (UTC)",
        yaxis_title=YLABELS.get(y_col, y_col),
        title=YLABELS.get(y_col, y_col),
        hovermode="x unified",
    )
    return fig

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("data-store", "data"),
    Input("interval", "n_intervals"),
    prevent_initial_call=False
)
def update_data_store(_n):
    global data_df
    if data_df is None or data_df.empty:
        data_df = load_all_data()
    else:
        try:
            last_time = data_df["time"].max()
            last_time_iso = pd.to_datetime(last_time, utc=True).isoformat()
            new_df = load_since(last_time_iso)
            if not new_df.empty:
                data_df = pd.concat([data_df, new_df], ignore_index=True)
                data_df.drop_duplicates(subset=["device","time"], inplace=True)
                data_df.sort_values(by=["device","time"], inplace=True)
        except Exception as e:
            print(f"Update error: {e}")
    df_tmp = data_df.copy()
    if not df_tmp.empty:
        df_tmp["time"] = pd.to_datetime(df_tmp["time"], utc=True).astype(str)
    return df_tmp.to_dict(orient="records")

@app.callback(
    Output("device-table", "data"),
    Input("data-store", "data")
)
def update_device_table(store_records):
    if not store_records: return []
    df = pd.DataFrame(store_records)
    if df.empty: return []
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    latest = (df.sort_values("time")
                .groupby("device", as_index=False)
                .tail(1)[["device","time","satId","temperature","batteryVoltage","hoursUptime"]]
                .sort_values("device"))
    latest["last_seen"] = latest["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col, ndp in [("temperature",2),("batteryVoltage",3),("hoursUptime",1)]:
        if col in latest.columns:
            latest[col] = pd.to_numeric(latest[col], errors="coerce").round(ndp)
    cols = ["device","last_seen","satId","temperature","batteryVoltage","hoursUptime"]
    return latest[cols].to_dict(orient="records")

@app.callback(
    Output("selected-device", "data"),
    Input("device-table", "selected_rows"),
    State("device-table", "data"),
    prevent_initial_call=False
)
def select_device(selected_rows, table_data):
    if not table_data: return None
    if not selected_rows: return table_data[0]["device"]
    idx = max(0, min(selected_rows[0], len(table_data)-1))
    return table_data[idx]["device"]

@app.callback(
    Output("device-title", "children"),
    Output("plots-container", "children"),
    Input("selected-device", "data"),
    State("data-store", "data"),
)
def render_device_plots(device_id, store_records):
    if not device_id or not store_records:
        return "No device selected.", []
    df = pd.DataFrame(store_records)
    if df.empty:
        return f"Device: {device_id}", [html.Div("No data available.")]
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df_dev = df[df["device"] == device_id].sort_values("time")
    if df_dev.empty:
        return f"Device: {device_id}", [html.Div("No data for this device (yet).")]
    params = infer_parameter_columns(df_dev)
    graphs = []
    for p in params:
        try:
            fig = make_figure(df_dev, p)
            graphs.append(dcc.Graph(figure=fig))
        except Exception as e:
            graphs.append(html.Div(f"Unable to plot {p}: {e}"))
    t0 = df_dev["time"].min()
    t1 = df_dev["time"].max()
    title = f"Device: {device_id} — samples from {t0.strftime('%Y-%m-%d %H:%M:%S')} to {t1.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    return title, graphs

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
