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


# Constants for Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token.pickle'
SPREADSHEET_ID = '1NfaLx6Yn09xoOHRon9ri6zfXZTkU1dFFX2rfW1kZvmw'  # Replace with your actual Spreadsheet ID
RANGE_NAME = 'Feuille 1'

st.set_page_config(layout='wide')

# Display the club logo from GitHub at the top right
logo_url = 'https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png'
col1, col2 = st.columns([9, 1])
with col1:
    st.title("GPS | FC Versailles")
with col2:
    st.image(logo_url, use_container_width=True)
    
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

# Function to fetch data from Google Sheet
def fetch_google_sheet(spreadsheet_id, range_name):
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    if not values:
        st.error("No data found in the specified range.")
        return pd.DataFrame()
    header = values[0]
    data = values[1:]
    max_columns = len(header)
    adjusted_data = [
        row + [None] * (max_columns - len(row)) if len(row) < max_columns else row[:max_columns]
        for row in data
    ]
    return pd.DataFrame(adjusted_data, columns=header)


# Add a button to refresh the data
if st.button("Actualiser les données"):
    st.cache_data.clear()  # Clear the cache to fetch new data
    st.success("Data refreshed successfully!")


# Fetch Google Sheet data
@st.cache_data
def load_data(ttl=60):
    return fetch_google_sheet(SPREADSHEET_ID, RANGE_NAME)


data = load_data()

# Affichage d’un aperçu des données
if not data.empty:
    st.subheader("Aperçu des données GPS")
    st.dataframe(data.head(50), use_container_width=True)
else:
    st.warning("Aucune donnée disponible ou échec de connexion à Google Sheets.")


# 🔁 Copie de travail
filtered_data = data.copy()

# 🧹 Nettoyage des noms de colonnes
filtered_data.columns = filtered_data.columns.str.strip()

# ✅ Nettoyage "Duration"
if "Duration" in filtered_data.columns:
    filtered_data["Duration"] = pd.to_numeric(
        filtered_data["Duration"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

# ✅ Nettoyage "Task Name"
if "Task Name" in filtered_data.columns:
    filtered_data["Task Name"] = (
        filtered_data["Task Name"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

# ✅ Liste des colonnes de performance
performance_cols = ["m/min", "HSR/min", "SPR/min", "HSPR/min", "Vmax", "Acc", "Amax", "Dec", "Dmax"]

# ✅ Conversion sûre de chaque colonne de perf en float
for col in performance_cols:
    if col in filtered_data.columns:
        filtered_data[col] = pd.to_numeric(
            filtered_data[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

# ✅ Filtrage sur les lignes valides
filtered_data = filtered_data[
    (filtered_data["Duration"] > 50) &
    (filtered_data["Task Name"] == "GAME")
]

# ✅ Calcul des meilleures performances
if "Name" in filtered_data.columns:
    best_performances = (
        filtered_data.groupby("Name")[performance_cols]
        .max()
        .reset_index()
        .sort_values(by="m/min", ascending=False)
    )

    st.subheader("🏅 Best Game Performances (Duration > 50 min)")
    st.dataframe(best_performances, use_container_width=True)
else:
    st.warning("La colonne 'Name' est manquante dans les données.")
    

# 🧠 Génération automatique d’un résumé
def top3_text(df, column, description):
    if column not in df.columns:
        return f"(Colonne '{column}' introuvable)"
    top_players = df.nlargest(3, column)["Name"].tolist()
    if not top_players:
        return f"(Pas de données pour '{column}')"
    return f"Pour {column}, les trois joueurs les plus {description} sont : {', '.join(top_players)}."

# 📝 Affichage du texte
st.markdown("### 📈 Analyse automatique des meilleurs profils")

st.markdown(top3_text(best_performances, "m/min", "endurants"))
st.markdown(top3_text(best_performances, "Vmax", "rapides"))
st.markdown(top3_text(best_performances, "Amax", "explosifs"))

