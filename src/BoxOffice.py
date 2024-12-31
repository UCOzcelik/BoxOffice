import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prediction import predict_box_office_range  # Importiere die Vorhersagefunktion

# Daten laden
df = pd.read_csv('../data/movie_data.csv')
print("Daten erfolgreich geladen!")

# Überprüfen auf NaN-Werte
print("Überprüfen auf NaN-Werte:")
print(df.isnull().sum())

# NaN-Werte behandeln (z.B. durch Entfernen)
df = df.dropna()  # Entfernt alle Zeilen mit NaN-Werten

# Genre-Encoding
le = LabelEncoder()
df['Genre_encoded'] = le.fit_transform(df['Genre'])

# Genre-Mapping speichern
genre_mapping = dict(zip(df['Genre'], df['Genre_encoded']))

# Hauptprogramm für die Vorhersage
if __name__ == "__main__":
    print("\nFilm Box Office Vorhersage")
    print("--------------------------")
    print("\nVerfügbare Genres:")
    print(sorted(genre_mapping.keys()))

    while True:
        try:
            title = input("\nFilmtitel: ")
            running_time = float(input("Laufzeit (in Minuten): "))
            budget = float(input("Budget (in Dollar): "))
            genre = input("Genre (aus der Liste oben): ")
            release_year = int(input("Erscheinungsjahr: "))
            director = input("Regisseur: ")
            actor1 = input("Schauspieler 1: ")
            actor2 = input("Schauspieler 2: ")
            actor3 = input("Schauspieler 3: ")

            prediction = predict_box_office_range(
                running_time=running_time,
                budget=budget,
                genre=genre,
                release_year=release_year
            )

            if prediction is not None:
                print(f"\nVorhersage für '{title}':")
                print(f"Budget: ${budget:,.2f}")
                print("\nGeschätzte Box Office Einnahmen:")
                print(f"${prediction['box_office_range'][0]:,.2f} - ${prediction['box_office_range'][1]:,.2f}")

                print("\nGeschätzter Gewinn/Verlust:")
                if prediction['profit_range'][0] < 0 and prediction['profit_range'][1] < 0:
                    print(
                        f"Erwarteter Verlust: ${abs(prediction['profit_range'][0]):,.2f} - ${abs(prediction['profit_range'][1]):,.2f}")
                elif prediction['profit_range'][0] < 0 and prediction['profit_range'][1] > 0:
                    print(f"Möglicher Verlust bis zu: ${abs(prediction['profit_range'][0]):,.2f}")
                    print(f"Möglicher Gewinn bis zu: ${prediction['profit_range'][1]:,.2f}")
                else:
                    print(
                        f"Erwarteter Gewinn: ${prediction['profit_range'][0]:,.2f} - ${prediction['profit_range'][1]:,.2f}")

                # Risikobewertung
                risk_ratio = abs(prediction['profit_range'][1] - prediction['profit_range'][0]) / budget
                if risk_ratio < 0.5:
                    risk_level = "Niedrig"
                elif risk_ratio < 1.0:
                    risk_level = "Mittel"
                else:
                    risk_level = "Hoch"
                print(f"\nRisikobewertung: {risk_level}")

        except ValueError as e:
            print("\nFehler: Bitte geben Sie gültige Zahlen ein.")
        except Exception as e:
            print(f"\nFehler: {str(e)}")

        if input("\nWeitere Vorhersage machen? (ja/nein): ").lower() != 'ja':
            break

    print("\nProgramm beendet.")