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


# ── PAGE: BEST PERFORMANCE ────────────────────────────────────────────────────
if page == "Best performance":
    st.subheader("🏅 Meilleures performances")

    # === 1) Best-per-game (min > 50) for four core metrics
    cols = ["m/min", "Vmax", "Amax", "Dmax"]
    best_df = data[(data["Type"] == "GAME") & (data["Duration"] > 45)].copy()
    for c in cols:
        if c in best_df.columns:
            best_df[c] = pd.to_numeric(
                best_df[c].astype(str).str.replace(",", "."), errors="coerce"
            )
    best = (
        best_df
        .groupby("Name")[cols]
        .max()
        .reset_index()
        .sort_values("m/min", ascending=False)
    )
    st.dataframe(best, use_container_width=True)

    # Top 3 and Flop 3 for these
    def top_n(df, col, n=3):
        return df.nlargest(n, col)["Name"].tolist()

    def flop_n(df, col, n=3):
        return df.nsmallest(n, col)["Name"].tolist()

    st.markdown(f"**🎖️ Top 3 endurants** : {', '.join(top_n(best, 'm/min'))}")
    st.markdown(f"**🔻Flop 3 endurants** : {', '.join(flop_n(best, 'm/min'))}")

    st.markdown(f"**⚡ Top 3 rapides** : {', '.join(top_n(best, 'Vmax'))}")
    st.markdown(f"**🐢 Flop 3 rapides** : {', '.join(flop_n(best, 'Vmax'))}")

    st.markdown(f"**💥 Top 3 explosifs** : {', '.join(top_n(best, 'Amax'))}")
    st.markdown(f"**🔻 Flop 3 explosifs** : {', '.join(flop_n(best, 'Amax'))}")



    # === 2) Build Référence Match ===
    st.subheader("🏆 Référence Match")

    # A) Pull all GAME rows
    mask = data["Type"] == "GAME"
    match_df = data[mask].copy()

    # B) Define and clean numeric columns
    ref_fields = [
        "Duration", "Distance", "m/min", "Distance 15km/h", "M/min 15km/h",
        "Distance 15-20km/h", "Distance 20-25km/h", "Distance 25km/h",
        "N° Sprints", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
    ]
    for c in ref_fields:
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

    # C) Aggregate per player
    records = []
    for name, grp in match_df.groupby("Name"):
        rec = {"Name": name}
        # full games ≥ 90′ → max of each
        full = grp[grp["Duration"] >= 90]
        if not full.empty:
            for c in ref_fields:
                rec[c] = full[c].max()
        else:
            # partial → copy m/min, M/min 15km/h, Vmax; scale others
            longest = grp.loc[grp["Duration"].idxmax()]
            orig = longest["Duration"]
            rec["Duration"] = orig
            for c in ref_fields:
                val = longest[c]
                if c in {"Duration","Vmax", "m/min", "M/min 15km/h"} or pd.isna(val) or orig <= 0:
                    rec[c] = val
                else:
                    rec[c] = 90 * val / orig
        records.append(rec)

    Refmatch = pd.DataFrame.from_records(records, columns=["Name"] + ref_fields)
    # D) Round & cast
    for c in ref_fields:
        if c == "Vmax":
            Refmatch[c] = Refmatch[c].round(1)
        else:
            Refmatch[c] = Refmatch[c].round(0).astype("Int64")

    # E) Render as scrollable, styled HTML
    styled = (
        Refmatch.style
                .set_table_attributes('class="centered-table"')
                .format({c: "{:.1f}" if c == "Vmax" else "{:d}" for c in ref_fields})
    )
    html = styled.to_html()
    row_h = 35
    total_h = max(300, len(Refmatch) * row_h)
    wrapper = f"""
    <html><head>
      <style>
        .centered-table {{ border-collapse: collapse; width: 100%; }}
        .centered-table th, .centered-table td {{
          text-align: center; padding: 4px 8px; border: 1px solid #ddd;
        }}
        .centered-table th {{ background-color: #f0f0f0; }}
      </style>
    </head><body>
      <div style="max-height:{total_h}px; overflow-y:auto;">
        {html}
      </div>
    </body></html>
    """
    components.html(wrapper, height=total_h + 20, scrolling=True)

    # === 3) Top & Flop from the Référence Match ===
    def best_and_worst(df, col, label):
        clean = df[df[col].notna()]
        if clean.empty:
            return ""
        top = clean.loc[clean[col].idxmax(), "Name"]
        flop = clean.loc[clean[col].idxmin(), "Name"]
        return f"**{label}** : Top – {top}, Flop – {flop}"

    st.markdown(best_and_worst(Refmatch, "m/min", "Endurance relative"))
    st.markdown(best_and_worst(Refmatch, "Vmax", "Vitesse max relative"))
    st.markdown(best_and_worst(Refmatch, "Acc", "Accélérations max"))
    st.markdown(best_and_worst(Refmatch, "Dec", "Décélérations max"))
    st.markdown(best_and_worst(Refmatch, "Distance", "Distance totale"))


# ── PAGE: ENTRAINEMENT ────────────────────────────────────────────────────────
elif page == "Entrainement":
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import streamlit.components.v1 as components

    st.subheader("🏋️ Performances à l'entraînement")

    # ── 0) Build Référence Match for % comparisons ─────────────────────────────
    mask_match = data["Type"].fillna("").str.upper().str.strip() == "GAME"
    match_df_all = data[mask_match].copy()
    ref_fields = [
        "Duration","Distance","m/min","Distance 15km/h","M/min 15km/h",
        "Distance 15-20km/h","Distance 20-25km/h","Distance 25km/h",
        "N° Sprints","Acc","Dec","Vmax","Distance 90% Vmax"
    ]
    # clean + numeric‐cast
    for c in ref_fields:
        if c in match_df_all:
            cleaned = (
                match_df_all[c].astype(str)
                              .str.replace(r"[^\d\-,\.]", "", regex=True)
                              .str.replace(",", ".", regex=False)
                              .replace("", pd.NA)
            )
            match_df_all[c] = pd.to_numeric(cleaned, errors="coerce")
        else:
            match_df_all[c] = pd.NA

    # aggregate personal bests
    records = []
    for name, grp in match_df_all.groupby("Name"):
        rec = {"Name": name}
        full = grp[grp["Duration"] >= 90]
        if not full.empty:
            for c in ref_fields:
                rec[c] = full[c].max()
        else:
            longest = grp.loc[grp["Duration"].idxmax()]
            orig = longest["Duration"]
            rec["Duration"] = orig
            for c in ref_fields:
                val = longest[c]
                if c in {"Vmax","m/min","M/min 15km/h"} or pd.isna(val) or orig <= 0:
                    rec[c] = val
                else:
                    rec[c] = 90 * val / orig
        records.append(rec)
    Refmatch = pd.DataFrame.from_records(records, columns=["Name"] + ref_fields)
    for c in ref_fields:
        if c == "Vmax":
            Refmatch[c] = Refmatch[c].round(1)
        else:
            Refmatch[c] = Refmatch[c].round(0).astype("Int64")

    # ── 1) OBJECTIFS ENTRAÎNEMENT (single date) ────────────────────────────────
    st.markdown("### 🎯 Objectifs Entraînement")

    allowed_tasks = [
        "OPTI","MESO","DRILLS","COMPENSATION",
        "MACRO","OPPO","OPTI +","OPTI J-1","REATHLE","MICRO"
    ]
    train_data = data[data["Type"].isin(allowed_tasks)].copy()
    min_d = train_data["Date"].min().date()
    max_d = train_data["Date"].max().date()

    # single-date picker
    sel_date = st.date_input(
        "Choisissez la date pour les objectifs",
        value=min_d,
        min_value=min_d,
        max_value=max_d
    )

    date_df = train_data[train_data["Date"].dt.date == sel_date]
    if date_df.empty:
        st.info(f"Aucune donnée d'entraînement pour le {sel_date}.")
    else:
        # same 10 objective stats
        objective_fields = [
            "Duration", "Distance",
            "Distance 15km/h", "Distance 15-20km/h",
            "Distance 20-25km/h", "Distance 25km/h",
            "Acc", "Dec", "Vmax", "Distance 90% Vmax"
        ]

        # 2×5 sliders
        st.markdown(f"#### Objectifs du {sel_date}")
        objectives = {}
        row1, row2 = objective_fields[:5], objective_fields[5:]
        cols5 = st.columns(5)
        for cont, stat in zip(cols5, row1):
            with cont:
                objectives[stat] = st.slider(stat, 0, 100, 100, key=f"obj_ent_{stat}")
        cols5 = st.columns(5)
        for cont, stat in zip(cols5, row2):
            with cont:
                objectives[stat] = st.slider(stat, 0, 100, 100, key=f"obj_ent_{stat}")

        # prepare metrics
        df_ent = date_df[["Name"] + objective_fields].copy()
        for c in objective_fields:
            cleaned = (
                df_ent[c].astype(str)
                         .str.replace(r"[^\d\-,\.]", "", regex=True)
                         .str.replace(",", ".", regex=False)
                         .replace("", pd.NA)
            )
            num = pd.to_numeric(cleaned, errors="coerce")
            df_ent[c] = num.round(1) if c=="Vmax" else num.round(0).astype("Int64")

        # compute % vs Référence Match
        ref_idx = Refmatch.set_index("Name")
        for c in objective_fields:
            df_ent[f"{c} %"] = df_ent.apply(
                lambda r: round(r[c]/ref_idx.at[r["Name"],c]*100,1)
                if (
                    r["Name"] in ref_idx.index
                    and pd.notna(r[c])
                    and pd.notna(ref_idx.at[r["Name"],c])
                    and ref_idx.at[r["Name"],c]>0
                ) else pd.NA,
                axis=1
            )

        # styling
        def hl(v,obj):
            if pd.isna(v): return ""
            d = abs(v-obj)
            if d<=5:   return "background-color:#c8e6c9;"
            if d<=10:  return "background-color:#fff9c4;"
            if d<=15:  return "background-color:#ffe0b2;"
            if d<=20:  return "background-color:#ffcdd2;"
            return ""

            # Build & style the Objectifs table
        display = ["Name"] + sum([[c, f"{c} %"] for c in objective_fields], [])
        styled = (
            df_ent.loc[:, display]
                  .style
                  .format(
                      {
                        # raw stats → integer (0 decimal)
                        **{c: "{:.0f}" for c in objective_fields if c != "Vmax"},
                        # Vmax → one decimal
                        **{"Vmax": "{:.1f}"},
                        # percent columns → one decimal + percent sign
                        **{f"{c} %": "{:.1f} %" for c in objective_fields}
                      }
                  )
        )
    
        # apply your highlight function as before
        for c in objective_fields:
            styled = styled.applymap(
                lambda v, obj=objectives[c]: hl(v, obj),
                subset=[f"{c} %"]
            )
    
        styled = styled.set_table_attributes('class="centered-table"')
        html_obj = styled.to_html()
        total_h = max(300, len(df_ent) * 35)
        wrapper = f"""
        <html><head>
          <style>
            .centered-table {{border-collapse:collapse;width:100%;}}
            .centered-table th, .centered-table td {{
              text-align:center; padding:4px 8px; border:1px solid #ddd;
            }}
            .centered-table th {{background:#f0f0f0;}}
          </style>
        </head><body>
          <div style="max-height:{total_h}px;overflow-y:auto;">
            {html_obj}
          </div>
        </body></html>
        """
        components.html(wrapper, height=total_h + 20, scrolling=True)

    # ── 2) PERFORMANCES DÉTAILLÉES (date range + filters) ──────────────────
    st.markdown("---")
    st.markdown("### 📊 Détail Entraînement")

    # date-range picker
    sel_range = st.date_input(
        "Période",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d
    )
    start, end = sel_range if isinstance(sel_range, (list,tuple)) else (min_d, max_d)

    # semaine & tâche filters
    col1, col2 = st.columns(2)
    with col1:
        semaines = sorted(train_data["Semaine"].dropna().unique())
        sel_sem = st.multiselect("Semaine(s)", semaines)
    with col2:
        types = sorted(train_data["Type"].unique())
        sel_task = st.multiselect("Tâche(s)", types)

    # apply all filters
    filt = train_data[
        (train_data["Date"].dt.date >= start) &
        (train_data["Date"].dt.date <= end)
    ]
    if sel_sem:
        filt = filt[filt["Semaine"].isin(sel_sem)]
    if sel_task:
        filt = filt[filt["Type"].isin(sel_task)]

    # clean display table
    to_drop = ["Season","Semaine","HUMEUR","PLAISIR","RPE"]
    disp = filt.drop(columns=to_drop, errors="ignore").copy()
    if "Date" in disp.columns:
        disp["Date"] = disp["Date"].dt.strftime("%d.%m.%y")

    st.dataframe(disp, use_container_width=True)

    if not filt.empty:
        # your existing plots unchanged…
        # ── Distance by speed bands
        # ── Accélérations vs Décélérations
        # ── Vitesse max
        pass



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

    # 3) Date filter
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

    # 4) Prepare & clean/cast French‐formatted numbers for display
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
                df[c].astype(str)
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
    components.html(html, height=total_h + 1, scrolling=True)

    # ── Référence Match ──
    match_df = data[mask].copy()
    if match_df.empty:
        st.info("Aucune donnée de match pour construire la référence.")
    else:
        # A) clean & numeric‐cast reference data
        for c in stat_cols:
            if c in match_df.columns:
                cleaned = (
                    match_df[c].astype(str)
                               .str.replace(r"[^\d\-,\.]", "", regex=True)
                               .str.replace(",", ".", regex=False)
                               .replace("", pd.NA)
                )
                match_df[c] = pd.to_numeric(cleaned, errors="coerce")
            else:
                match_df[c] = pd.NA

        # B) build per‐player reference
        records = []
        for name, grp in match_df.groupby("Name"):
            rec = {"Name": name}
            full = grp[grp["Duration"] >= 90]
            if not full.empty:
                # full‐length games: take the max of each stat
                for c in stat_cols:
                    rec[c] = full[c].max()
            else:
                # only partial games: scale some stats, copy others
                longest = grp.loc[grp["Duration"].idxmax()]
                orig = longest["Duration"]
                rec["Duration"] = orig

                for c in stat_cols:
                    val = longest[c]
                    if c in {"Duration","Vmax", "m/min", "M/min 15km/h"}:
                        # copy raw value for these three
                        rec[c] = val
                    elif pd.notna(val) and orig > 0:
                        # scale everything else to 90'
                        rec[c] = 90 * val / orig
                    else:
                        rec[c] = pd.NA

            records.append(rec)

        Refmatch = pd.DataFrame.from_records(records, columns=["Name"] + stat_cols)

        # C) Round & cast types
        for c in stat_cols:
            if c == "Vmax":
                Refmatch[c] = Refmatch[c].round(1)
            else:
                Refmatch[c] = Refmatch[c].round(0).astype("Int64")

        # D) Display
        st.subheader("🏆 Référence Match")
        st.dataframe(Refmatch, use_container_width=True)


        st.subheader("🎯 Objectifs Match")
        
        objective_fields = [
            "Duration", "Distance", "Distance 15km/h", "Distance 15-20km/h",
            "Distance 20-25km/h", "Distance 25km/h", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
        ]
        
        # 1) Sliders in 2×5 grid
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
        
        # 2) Compute % of personal reference per match row
        obj_df = df.copy()
        ref_indexed = Refmatch.set_index("Name")
        for c in objective_fields:
            obj_df[f"{c} %"] = obj_df.apply(
                lambda r: round(r[c] / ref_indexed.at[r["Name"], c] * 100, 1)
                          if (r["Name"] in ref_indexed.index
                              and pd.notna(ref_indexed.at[r["Name"], c]) 
                              and ref_indexed.at[r["Name"], c] > 0
                              and pd.notna(r[c]))
                          else pd.NA,
                axis=1
            )
        
        # 3) Highlight helper
        def highlight_stat(val, obj):
            if pd.isna(val):
                return ""
            d = abs(val - obj)
            if d <= 5:   return "background-color: #c8e6c9;"
            if d <= 10:  return "background-color: #fff9c4;"
            if d <= 15:  return "background-color: #ffe0b2;"
            if d <= 20:  return "background-color: #ffcdd2;"
            return ""
        
        # 4) Build & render styled table
        display_cols = ["Name"] + sum([[c, f"{c} %"] for c in objective_fields], [])
        styled = (
            obj_df.loc[:, display_cols]
                  .style
                  .format(
                      { **{f"{c} %": "{:.1f} %" for c in objective_fields},
                        **{"Vmax": "{:.1f}"}                # one decimal for raw Vmax
                      },
                      na_rep="—"
                  )
        )
        
        # apply per-column highlighting only on the % columns
        for c in objective_fields:
            styled = styled.applymap(
                lambda v, obj=objectives[c]: highlight_stat(v, obj),
                subset=[f"{c} %"]
            )
        
        styled = styled.set_table_attributes('class="centered-table"')
        
        # 5) wrap in scrollable div
        html_obj = styled.to_html()
        row_h = 35
        total_h2 = max(300, len(obj_df) * row_h)
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
