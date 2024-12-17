import pandas as pd

# Datei laden
input_file = '../data/MLWS_Movies_Datensatz.csv'
output_file = '../data/movie_data.csv'

try:
    # Daten laden mit Semikolon als Trennzeichen
    df = pd.read_csv(input_file, sep=';', encoding='utf-8')

    # Spaltennamen bereinigen
    df.columns = df.columns.str.strip()

    # Spalten bereinigen
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.strip()

    # Numerische Spalten konvertieren
    df['Running time'] = pd.to_numeric(df['Running time'], errors='coerce')
    df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
    df['Box Office'] = pd.to_numeric(df['Box Office'], errors='coerce')
    df['Release year'] = pd.to_numeric(df['Release year'], errors='coerce')

    # Datei mit Komma als Trennzeichen speichern
    df.to_csv(output_file, index=False, sep=',', encoding='utf-8')

    print("Datei erfolgreich formatiert und gespeichert!")
    print("\nErste 5 Zeilen der formatierten Daten:")
    print(df.head().to_string())

except Exception as e:
    print(f"Fehler beim Verarbeiten der Datei: {str(e)}")