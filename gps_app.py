import os
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# ── Constants ─────────────────────────────────────────────────────────────────


st.set_page_config(layout='wide')
col1, col2 = st.columns([9,1])
with col1:
    st.title("GPS | FC Versailles")
with col2:
    st.image(
        'https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png',
        use_container_width=True
    )
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ── Fetch & cache data ────────────────────────────────────────────────────────
SCOPES         = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE     = 'token.pickle'
SPREADSHEET_ID = '1NfaLx6Yn09xoOHRon9ri6zfXZTkU1dFFX2rfW1kZvmw'
SHEET_NAME     = 'Feuille 1'
RANGE_NAME     = 'Feuille 1!A1:Z'   # only pull columns A through Z

def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)
    return creds

def fetch_google_sheet(spreadsheet_id, sheet_name):
    creds   = get_credentials()
    service = build('sheets','v4',credentials=creds)
    meta    = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id, includeGridData=True
    ).execute()
    for s in meta['sheets']:
        if s['properties']['title'] == sheet_name:
            data = s['data'][0]['rowData']
            break
    else:
        st.error(f"❌ Feuille {sheet_name} introuvable.")
        return pd.DataFrame()
    rows = []
    for row in data:
        rows.append([cell.get('formattedValue') for cell in row.get('values',[])])
    max_len = max(len(r) for r in rows)
    rows = [r + [None]*(max_len-len(r)) for r in rows]
    header = rows[0]
    return pd.DataFrame(rows[1:], columns=header)

@st.cache_data(ttl=600)
def load_data():
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)

    # === Use the fast values().get() endpoint ===
    result = (
        service.spreadsheets()
               .values()
               .get(spreadsheetId=SPREADSHEET_ID,
                    range=RANGE_NAME,
                    valueRenderOption='FORMATTED_VALUE')
               .execute()
    )
    rows = result.get('values', [])
    if not rows:
        st.error("❌ Aucune donnée trouvée dans la plage.")
        return pd.DataFrame()

    # first row = header, rest = data
    header, data_rows = rows[0], rows[1:]
    df = pd.DataFrame(data_rows, columns=header)

    # keep only your 24 columns
    expected = [
        "Season","Semaine","HUMEUR","PLAISIR","RPE","Date","Jour","Type","Name",
        "Duration","Distance","m/min","Distance 15km/h","M/min 15km/h",
        "Distance 15-20km/h","Distance 20-25km/h","Distance 25km/h",
        "Distance 90% Vmax","N° Sprints","Vmax","%Vmax","Acc","Dec","Amax","Dmax"
    ]
    df = df.loc[:, expected]

    # hard-code season
    df = df[df["Season"] == "2526"]

    # downstream processing...
    return df

data = load_data()


# ── Pre-process common cols ────────────────────────────────────────────────────
# Filter by season

# Duration → int (invalid → 0)
if "Duration" in data.columns:
    # 1) coerce to float (invalid → NaN)
    durations = pd.to_numeric(
        data["Duration"]
            .astype(str)
            .str.replace(",", ".", regex=False),
        errors="coerce"
    )
    # 2) replace NaN with 0 and cast to plain int
    data["Duration"] = durations.fillna(0).astype(int)

# Type → uppercase & stripped
if "Type" in data.columns:
    data["Type"] = data["Type"].astype(str).str.upper().str.strip()

# Name → title-case
if "Name" in data.columns:
    data["Name"] = (
        data["Name"].astype(str)
                 .str.strip()
                 .str.lower()
                 .str.title()
    )

# Semaine → integer
if "Semaine" in data.columns:
    data["Semaine"] = pd.to_numeric(data["Semaine"], errors="coerce").astype("Int64")

# Date → datetime
if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# ── Sidebar: page selection ───────────────────────────────────────────────────
pages = ["Entrainement","Match","Best performance","Player analysis","Minutes de jeu"]
page  = st.sidebar.selectbox("Choisissez une page", pages)

# ── PAGE: ENTRAINEMENT ────────────────────────────────────────────────────────
if page == "Entrainement":
    st.subheader("🏋️ Performances à l'entraînement")

    allowed_tasks = [
        "OPTI","MÉSO","DRILLS","COMPENSATION",
        "MACRO","OPPO","OPTI +","OPTI J-1","REATHLE"
    ]
    train_data = data[data["Type"].isin(allowed_tasks)].copy()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_d = train_data["Date"].min().date()
        max_d = train_data["Date"].max().date()
        sel_date = st.date_input("Filtrer par date", value=None, min_value=min_d, max_value=max_d)
    with col2:
        semaines = sorted(train_data["Semaine"].dropna().unique())
        sel_sem   = st.multiselect("Filtrer par semaine", semaines)
    with col3:
        tasks = sorted(train_data["Type"].dropna().unique())
        sel_task = st.multiselect("Filtrer par tâche", tasks)

    filt = train_data
    if sel_date:
        filt = filt[filt["Date"].dt.date == sel_date]
    if sel_sem:
        filt = filt[filt["Semaine"].isin(sel_sem)]
    if sel_task:
        filt = filt[filt["Type"].isin(sel_task)]

    st.dataframe(filt, use_container_width=True)

    if not filt.empty:
        # ── Distance @ speed bands
        dist_cols = [
            "Distance","Distance 15km/h","Distance 15-20km/h",
            "Distance 20-25km/h","Distance 25km/h","Distance 90% Vmax"
        ]
        present = [c for c in dist_cols if c in filt.columns]
        if present:
            melt = filt.melt(
                id_vars="Name", value_vars=present,
                var_name="Zone", value_name="mètres"
            )
            fig = px.bar(
                melt, x="Name", y="mètres", color="Zone",
                barmode="group", title="Distances par zone de vitesse"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Accélérations vs Décélérations
        accdec = [c for c in ["Acc","Dec"] if c in filt.columns]
        if accdec:
            melt2 = filt.melt(
                id_vars="Name", value_vars=accdec,
                var_name="Type", value_name="Nombre"
            )
            fig2 = px.bar(
                melt2, x="Name", y="Nombre", color="Type",
                barmode="group", title="Accélérations et Décélérations"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Vitesse max
        if "Vmax" in filt.columns:
            filt["Vmax"] = pd.to_numeric(
                filt["Vmax"].astype(str).str.replace(",", "."),
                errors="coerce"
            )
            fig3 = px.scatter(
                filt, x="Name", y="Vmax", size="Vmax", color="Vmax",
                title="Vitesse maximale par joueur"
            )
            st.plotly_chart(fig3, use_container_width=True)
            if not filt["Vmax"].dropna().empty:
                top = filt.loc[filt["Vmax"].idxmax()]
                st.success(
                    f"💡 Joueur le plus rapide : **{top['Name']}** à {top['Vmax']} km/h"
                )

# ── PAGE: MATCH ───────────────────────────────────────────────────────────────
elif page == "Match":


    st.subheader("⚽ Performances en match")

    # 1) Filter to GAME rows
    mask = (
        data["Type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper() == "GAME"
    )
    match_data = data[mask].copy()

    # 2) Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(match_data["Date"]):
        match_data["Date"] = pd.to_datetime(match_data["Date"], errors="coerce")

    # 3) Build & apply date filter
    available_dates = sorted(match_data["Date"].dt.date.dropna().unique())
    selected_dates = st.multiselect(
        "Filtrer par date",
        options=available_dates,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        default=available_dates,
    )
    if selected_dates:
        match_data = match_data[match_data["Date"].dt.date.isin(selected_dates)]
    else:
        match_data = match_data.iloc[:0]

    # 4) Prepare & clean/cast French‐formatted numbers
    cols = [
        "Name", "Duration", "Distance", "m/min",
        "Distance 15km/h", "M/min 15km/h",
        "Distance 15-20km/h", "Distance 20-25km/h",
        "Distance 25km/h", "N° Sprints", "Acc", "Dec",
        "Vmax", "Distance 90% Vmax"
    ]
    df = match_data.loc[:, cols].copy()
    stat_cols = [c for c in cols if c != "Name"]

    for c in stat_cols:
        if c in df.columns:
            cleaned = (
                df[c]
                  .astype(str)
                  .str.replace(r"[^\d\-,\.]", "", regex=True)
                  .str.replace(",", ".", regex=False)
                  .replace("", pd.NA)
            )
            num = pd.to_numeric(cleaned, errors="coerce")
        else:
            num = pd.Series(pd.NA, index=df.index)

        if c == "Vmax":
            df[c] = num.round(1)
        else:
            df[c] = num.round(0).astype("Int64")

    # 5) Render the first scrollable table
    html_table = df.to_html(index=False, classes="centered-table")
    row_h = 35
    total_h = max(300, len(df) * row_h)
    html = f"""
    <html><head>
      <style>
        .centered-table {{ border-collapse:collapse; width:100%; }}
        .centered-table th, .centered-table td {{
          text-align:center; padding:4px 8px; border:1px solid #ddd;
        }}
        .centered-table th {{ background:#f0f0f0; }}
      </style>
    </head><body>
      <div style="max-height:{total_h}px;overflow-y:auto;">
        {html_table}
      </div>
    </body></html>
    """
    components.html(html, height=total_h + 20, scrolling=True)

    # ── Référence Match ──
    match_df = data[mask].copy()
    if match_df.empty:
        st.info("Aucune donnée de match pour construire la référence.")
    else:
        # clean & numeric-cast reference data
        for c in stat_cols:
            if c in match_df.columns:
                cleaned = (
                    match_df[c]
                      .astype(str)
                      .str.replace(r"[^\d\-,\.]", "", regex=True)
                      .str.replace(",", ".", regex=False)
                      .replace("", pd.NA)
                )
                match_df[c] = pd.to_numeric(cleaned, errors="coerce")
            else:
                match_df[c] = pd.NA

        # build per-player reference
        records = []
        for name, grp in match_df.groupby("Name"):
            rec = {"Name": name}
            full = grp[grp["Duration"] >= 90]
            if not full.empty:
                for c in stat_cols:
                    rec[c] = full[c].max()
            else:
                longest = grp.loc[grp["Duration"].idxmax()]
                orig = longest["Duration"]
                rec["Duration"] = orig
                for c in stat_cols:
                    if c == "Duration":
                        continue
                    val = longest[c]
                    if c == "Vmax" or pd.isna(val) or orig <= 0:
                        rec[c] = val
                    else:
                        rec[c] = 90 * val / orig
            records.append(rec)

        Refmatch = pd.DataFrame.from_records(records, columns=["Name"] + stat_cols)
        for c in stat_cols:
            if c == "Vmax":
                Refmatch[c] = Refmatch[c].round(1)
            else:
                Refmatch[c] = Refmatch[c].round(0).astype("Int64")

        st.subheader("🏆 Référence Match")
        st.dataframe(Refmatch, use_container_width=True)

        # ── Objectifs Match ──
        st.subheader("🎯 Objectifs Match")

        # 1) Only these 10 stats get objectifs
        objective_fields = [
            "Duration", "Distance",
            "Distance 15km/h", "Distance 15-20km/h",
            "Distance 20-25km/h", "Distance 25km/h",
            "Acc", "Dec", "Vmax", "Distance 90% Vmax"
        ]

        # 2) Sliders in 2×5 grid
        row1, row2 = objective_fields[:5], objective_fields[5:]
        objectives = {}
        cols5 = st.columns(5)
        for i, stat in enumerate(row1):
            with cols5[i]:
                objectives[stat] = st.slider(f"{stat} (%)", 0, 100, 100, key=f"obj_{stat}")
        cols5 = st.columns(5)
        for i, stat in enumerate(row2):
            with cols5[i]:
                objectives[stat] = st.slider(f"{stat} (%)", 0, 100, 100, key=f"obj_{stat}")

        # 3) Compute % of personal reference for each match row
        obj_df = df.copy()
        ref_indexed = Refmatch.set_index("Name")
        for c in objective_fields:
            pct_vals = []
            for _, row in obj_df.iterrows():
                name = row["Name"]
                ref = ref_indexed.at[name, c] if name in ref_indexed.index else pd.NA
                val = row[c]
                if pd.notna(val) and pd.notna(ref) and ref > 0:
                    pct_vals.append(round(val / ref * 100, 1))
                else:
                    pct_vals.append(pd.NA)
            obj_df[f"{c} %"] = pct_vals

        # 4) Highlighting helper
        def highlight_stat(val, obj):
            if pd.isna(val):
                return ""
            diff = abs(val - obj)
            if diff <= 5:
                return "background-color: #c8e6c9;"
            elif diff <= 10:
                return "background-color: #fff9c4;"
            elif diff <= 15:
                return "background-color: #ffe0b2;"
            elif diff <= 20:
                return "background-color: #ffcdd2;"
            else:
                return ""

        # 5) Build & render styled Objectifs table
        display_cols = ["Name"] + sum([[c, f"{c} %"] for c in objective_fields], [])
        styled = (
            obj_df.loc[:, display_cols]
                  .style
                  .format({f"{c} %": "{:.1f} %" for c in objective_fields}, na_rep="—")
        )
        for c in objective_fields:
            styled = styled.applymap(
                lambda v, obj=objectives[c]: highlight_stat(v, obj),
                subset=[f"{c} %"]
            )
        styled = styled.set_table_attributes('class="centered-table"')
        html_obj = styled.to_html()
        total_h2 = total_h
        html2 = f"""
        <html><head>
          <style>
            .centered-table {{ border-collapse:collapse; width:100%; }}
            .centered-table th, .centered-table td {{
              text-align:center; padding:4px 8px; border:1px solid #ddd;
            }}
            .centered-table th {{ background:#f0f0f0; }}
          </style>
        </head><body>
          <div style="max-height:{total_h2}px;overflow-y:auto;">
            {html_obj}
          </div>
        </body></html>
        """
        components.html(html2, height=total_h2 + 20, scrolling=True)




# ── PAGE: BEST PERFORMANCE ────────────────────────────────────────────────────
elif page == "Best performance":
    st.subheader("🏅 Meilleures performances")
    cols = ["m/min", "Vmax", "Amax", "Dmax"]
    best_df = data[
        (data["Type"] == "GAME") & (data["Duration"] > 50)
    ].copy()
    for c in cols:
        if c in best_df.columns:
            best_df[c] = pd.to_numeric(
                best_df[c].astype(str).str.replace(",", "."), errors="coerce"
            )
    best = best_df.groupby("Name")[cols].max().reset_index().sort_values("m/min", ascending=False)
    st.dataframe(best, use_container_width=True)

    def top3(df, col, label):
        if col in df:
            p = df.nlargest(3, col)["Name"].tolist()
            return f"Top 3 {label} : {', '.join(p)}"
        return ""
    st.markdown(top3(best, "m/min", "endurants"))
    st.markdown(top3(best, "Vmax", "rapides"))
    st.markdown(top3(best, "Amax", "explosifs"))

# ── PAGE: PLAYER ANALYSIS ────────────────────────────────────────────────────
elif page == "Player analysis":
    st.subheader("🔎 Analyse d'un joueur")
    players = sorted(data["Name"].dropna().unique())
    sel = st.selectbox("Choisissez un joueur", players)
    p_df = data[data["Name"] == sel]
    g_df = p_df[p_df["Type"] == "GAME"]
    if not g_df.empty:
        wmin = g_df["Semaine"].min()
        wmax = g_df["Semaine"].max()
        weeks = pd.DataFrame({"Semaine": range(wmin, wmax + 1)})
        mins  = g_df.groupby("Semaine")["Duration"].sum().reset_index()
        merged = weeks.merge(mins, on="Semaine", how="left").fillna(0)
        fig = px.bar(merged, x="Semaine", y="Duration", title=f"{sel} – Minutes par semaine")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données de match pour ce joueur.")
