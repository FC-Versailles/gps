import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import streamlit as st
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import pickle
import ast
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF


# Constants for Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token.pickle'
SPREADSHEET_ID = '1NfaLx6Yn09xoOHRon9ri6zfXZTkU1dFFX2rfW1kZvmw'  # Replace with your actual Spreadsheet ID
RANGE_NAME = 'Feuille 1!A1:Z3000'



st.set_page_config(layout='wide')

# Display the club logo from GitHub at the top right
logo_url = 'https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png'
col1, col2 = st.columns([9, 1])
with col1:
    st.title("GPS | FC Versailles")
with col2:
    st.image(logo_url, use_container_width=True)
    

# Sélecteur de saison
season = st.radio("",
    options=["2425", "2526"],
    index=1,  # 2526 par défaut
    horizontal=True,
    key="season_selector"
)


# Add a horizontal line to separate the header
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# Function to get Google Sheets credentials
def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def fetch_google_sheet_full(spreadsheet_id, sheet_name):
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)

    sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id, ranges=[], includeGridData=True).execute()

    # Trouver la bonne feuille
    for sheet in sheet_metadata['sheets']:
        if sheet['properties']['title'] == sheet_name:
            data = sheet['data'][0]['rowData']
            break
    else:
        st.error(f"❌ Feuille {sheet_name} introuvable.")
        return pd.DataFrame()

    # Construire les lignes
    rows = []
    for row in data:
        row_values = []
        for cell in row.get('values', []):
            row_values.append(cell.get('formattedValue'))
        rows.append(row_values)

    # Ajuster pour aligner les colonnes
    max_len = max(len(r) for r in rows)
    rows = [r + [None] * (max_len - len(r)) for r in rows]

    # Créer le DataFrame
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)

    return df


# Fetch Google Sheet data
@st.cache_data
def load_data(ttl=60):
    return fetch_google_sheet_full(SPREADSHEET_ID, 'Feuille 1')
data = load_data()

# Appliquer le filtre si la colonne Season existe
if "Season" in data.columns:
    data = data[data["Season"] == season]
else:
    st.warning("⚠️ La colonne 'Season' est absente des données. Le filtre de saison ne peut pas s'appliquer.")

if "Duration" in data.columns:
    # Force en string pour remplacer les virgules, mais ensuite on convertit proprement
    data["Duration"] = pd.to_numeric(
        data["Duration"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

if "Task Name" in data.columns:
    data["Task Name"] = data["Task Name"].str.upper().str.strip()
    
# ✅ Clean and standardize names
if "Name" in data.columns:
    data["Name"] = data["Name"].astype(str).str.strip().str.lower().str.title()
    
# ✅ Convert 'SEMAINE' to integer
if "Semaine" in data.columns:
    data["Semaine"] = pd.to_numeric(data["Semaine"], errors="coerce").astype("Int64")

# --- PAGE SELECTION ---
pages = ["Entrainement", "Match", "Best performance", "Player analysis", "Minutes de jeu"]
page = st.sidebar.selectbox("Choisissez une page", pages)

# --- PAGE : ENTRAINEMENT ---
if page == "Entrainement":
    st.subheader("🏋️ Performances à l'entraînement")

    allowed_tasks = ["OPTI", "MÉSO", "DRILLS", "COMPENSATION", 
                     "MACRO", "OPPO", "OPTI +", "OPTI J-1", "REATHLE"]

    train_data = data[data["Task Name"].isin(allowed_tasks)].copy()

    if not pd.api.types.is_datetime64_any_dtype(train_data["Date"]):
        train_data["Date"] = pd.to_datetime(train_data["Date"], errors='coerce')

    min_date = train_data["Date"].min().date()
    max_date = train_data["Date"].max().date()

    # --- FILTRES ---
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_date = st.date_input("Filtrer par date", value=None, min_value=min_date, max_value=max_date)
    with col2:
        semaines = sorted(train_data["Semaine"].dropna().unique())
        selected_semaines = st.multiselect("Filtrer par semaine", semaines)
    with col3:
        task_names = sorted(train_data["Task Name"].dropna().unique())
        selected_tasks = st.multiselect("Filtrer par tâche", task_names)

    # --- FILTRAGE ---
    filtered = train_data
    if selected_date:
        filtered = filtered[filtered["Date"].dt.date == selected_date]
    if selected_semaines:
        filtered = filtered[filtered["Semaine"].isin(selected_semaines)]
    if selected_tasks:
        filtered = filtered[filtered["Task Name"].isin(selected_tasks)]

    st.dataframe(filtered, use_container_width=True)

    # --- VISUALISATIONS ---
    if not filtered.empty:
        # Graphique Distance, HSR, SPR
        value_cols = ["Distance", "HSR", "SPR"]
        present_cols = [col for col in value_cols if col in filtered.columns]
        if present_cols:
            melted = filtered.melt(id_vars="Name", value_vars=present_cols, var_name="Type", value_name="Valeur")
            fig1 = px.bar(melted, x="Name", y="Valeur", color="Type", barmode="group",
                          title="Distance, HSR, SPR par joueur",
                          color_discrete_map={"Distance": "#0031E3", "HSR": "#CFB013", "SPR": "grey"})
            st.plotly_chart(fig1, use_container_width=True)

        # Graphique Acc / Dec
        acc_dec_cols = ["Acc", "Dec"]
        present_accdec = [col for col in acc_dec_cols if col in filtered.columns]
        if present_accdec:
            melted2 = filtered.melt(id_vars="Name", value_vars=present_accdec, var_name="Type", value_name="Valeur")
            fig2 = px.bar(melted2, x="Name", y="Valeur", color="Type", barmode="group",
                          title="Accélérations / Décélérations",
                          color_discrete_sequence=["#0031E3", "#CFB013"])
            st.plotly_chart(fig2, use_container_width=True)

        # Vmax
        if "Vmax" in filtered.columns:
            # Assurez-vous que Vmax est numérique
            filtered["Vmax"] = pd.to_numeric(
                filtered["Vmax"].astype(str).str.replace(",", "."), 
                errors="coerce"
            )
            
            fig3 = px.scatter(
                filtered,
                x="Name",
                y="Vmax",
                size="Vmax",
                color="Vmax",
                title="Vitesse maximale",
                color_continuous_scale=[[0, "#CFB013"], [1, "#0031E3"]]
            )
            st.plotly_chart(fig3, use_container_width=True)
        
            if not filtered["Vmax"].dropna().empty:
                vmax_top = filtered.loc[filtered["Vmax"].idxmax()]
                st.success(f"💡 Joueur le plus rapide : **{vmax_top['Name']}** avec {vmax_top['Vmax']} km/h")
            else:
                st.info("Pas de valeur Vmax valide à afficher.")


# --- PAGE : MATCH ---
elif page == "Match":
    st.subheader("⚽ Performances en match")
    match_data = data[data["Task Name"] == "GAME"]
    st.dataframe(match_data, use_container_width=True)

# --- PAGE : MINUTES DE JEU ---
elif page == "Minutes de jeu":
    st.subheader("⏱️ Minutes de jeu")
    game_data = data[data["Task Name"] == "GAME"].copy()
    max_weekly = game_data.groupby("Semaine")["Duration"].max().sum()
    duration_data = (game_data.groupby("Name")["Duration"].sum()
                     .sort_values(ascending=False).reset_index())
    duration_data["% Played"] = (duration_data["Duration"] / max_weekly * 100).round(1)
    fig = px.bar(duration_data, x="Name", y="Duration", color="Duration", text="% Played",
                 title=f"Minutes jouées (Max possible: {int(max_weekly)} min)",
                 color_continuous_scale="Blues")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE : BEST PERFORMANCE ---
elif page == "Best performance":
    st.subheader("🏅 Meilleures performances")
    perf_cols = ["m/min", "HSR/min", "SPR/min", "HSPR/min", "Vmax", "Amax", "Dmax"]
    filtered = data[(data["Task Name"] == "GAME") & (data["Duration"] > 50)].copy()
    for col in perf_cols:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col].astype(str).str.replace(",", "."), errors="coerce")
    best = filtered.groupby("Name")[perf_cols].max().reset_index().sort_values(by="m/min", ascending=False)
    st.dataframe(best, use_container_width=True)
    # Insights
    def top3_text(df, col, label):
        if col in df.columns:
            players = df.nlargest(3, col)["Name"].tolist()
            return f"Top 3 {label}: {', '.join(players)}"
        return f"{col} non disponible"
    st.markdown(top3_text(best, "m/min", "endurants"))
    st.markdown(top3_text(best, "Vmax", "rapides"))
    st.markdown(top3_text(best, "Amax", "explosifs"))

# --- PAGE : PLAYER ANALYSIS ---
elif page == "Player analysis":
    st.subheader("🔎 Analyse d'un joueur")
    player_list = sorted(data["Name"].dropna().unique())
    selected = st.selectbox("Choisissez un joueur", player_list)
    player_df = data[data["Name"] == selected]
    game_df = player_df[player_df["Task Name"] == "GAME"]
    if not game_df.empty:
        min_week = game_df["Semaine"].min()
        max_week = game_df["Semaine"].max()
        full_weeks = pd.DataFrame({"Semaine": range(min_week, max_week + 1)})
        minutes = (game_df.groupby("Semaine")["Duration"].sum().reset_index())
        merged = full_weeks.merge(minutes, on="Semaine", how="left").fillna(0)
        fig = px.bar(merged, x="Semaine", y="Duration", title=f"{selected} - Minutes par semaine")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données de match pour ce joueur.")




