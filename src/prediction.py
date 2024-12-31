import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression  # Import für LinearRegression

# Daten laden
df = pd.read_csv('../data/movie_data.csv')
print("Daten erfolgreich geladen!")

# Überprüfen auf NaN-Werte
print("Überprüfen auf NaN-Werte:")
print(df.isnull().sum())

# NaN-Werte behandeln (z.B. durch Entfernen)
df = df.dropna()  # Entfernt alle Zeilen mit NaN-Werten

# Feature Engineering
# 1. Logarithmische Transformationen
df['Budget_log'] = np.log1p(df['Budget'])
df['Running_time_log'] = np.log1p(df['Running time'])

# 2. Verhältnis-Features
df['Budget_per_minute'] = df['Budget'] / df['Running time']
df['Budget_per_year'] = df['Budget'] / np.maximum(1, (2024 - df['Release year']))

# 3. Kategorische Features
df['Is_High_Budget'] = (df['Budget'] > df['Budget'].mean()).astype(int)
df['Is_Long_Movie'] = (df['Running time'] > 120).astype(int)
df['Is_Recent'] = (df['Release year'] >= 2010).astype(int)

# 4. Genre und Director Features
df['Genre_Avg_Revenue'] = df.groupby('Genre')['Box Office'].transform('mean')
df['Director_Avg_Revenue'] = df.groupby('Director')['Box Office'].transform('mean')
df['Genre_Movie_Count'] = df.groupby('Genre')['Box Office'].transform('count')
df['Director_Movie_Count'] = df.groupby('Director')['Box Office'].transform('count')

# 5. Schauspieler Features
df['Actor1_Avg_Revenue'] = df.groupby('Actor 1')['Box Office'].transform('mean')

# Encoding für kategorische Variablen
le = LabelEncoder()
df['Genre_encoded'] = le.fit_transform(df['Genre'])
df['Director_encoded'] = le.fit_transform(df['Director'])
df['Actor1_encoded'] = le.fit_transform(df['Actor 1'])

# Genre-Mapping speichern
genre_mapping = dict(zip(df['Genre'], df['Genre_encoded']))

# Feature-Liste anpassen
features = [
    'Budget', 'Running time', 'Budget_log', 'Running_time_log',
    'Budget_per_minute', 'Budget_per_year', 'Genre_encoded',
    'Director_encoded', 'Release year', 'Is_High_Budget',
    'Is_Long_Movie', 'Is_Recent', 'Genre_Avg_Revenue',
    'Director_Avg_Revenue', 'Genre_Movie_Count', 'Director_Movie_Count',
    'Actor1_Avg_Revenue'
]

# Daten vorbereiten
X = df[features]
y = df['Box Office']

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelle definieren
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    ),
    'Linear Regression': LinearRegression()  # Hier wird LinearRegression importiert
}

# Modelle trainieren und evaluieren
print("\nModell Performances:")
print("-" * 50)
best_score = 0
best_model = None
best_model_name = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"\n{name}:")
    print(f"Training R² Score: {train_score:.2f}")
    print(f"Test R² Score: {test_score:.2f}")

    if test_score > best_score:
        best_score = test_score
        best_model = model
        best_model_name = name

print(f"\nBestes Modell: {best_model_name}")
print(f"Bester Test R² Score: {best_score:.2f}")

def predict_box_office(running_time, budget, genre, release_year):
    if genre not in genre_mapping:
        print(f"Warnung: Genre '{genre}' nicht im Trainingsdatensatz gefunden")
        print("Verfügbare Genres:", sorted(genre_mapping.keys()))
        return None

    # Feature-Werte erstellen
    features_dict = {
        'Budget': budget,
        'Running time': running_time,
        'Budget_log': np.log1p(budget),
        'Running_time_log': np.log1p(running_time),
        'Budget_per_minute': budget / running_time,
        'Budget_per_year': budget / max(1, (2024 - release_year)),
        'Genre_encoded': genre_mapping[genre],
        'Director_encoded': 0,  # Default Wert
        'Release year': release_year,
        'Is_High_Budget': 1 if budget > df['Budget'].mean() else 0,
        'Is_Long_Movie': 1 if running_time > 120 else 0,
        'Is_Recent': 1 if release_year >= 2010 else 0,
        'Genre_Avg_Revenue': df[df['Genre'] == genre]['Box Office'].mean(),
        'Director_Avg_Revenue': df['Box Office'].mean(),  # Gesamtdurchschnitt für unbekannte Regisseure
        'Genre_Movie_Count': df[df['Genre'] == genre].shape[0],
        'Director_Movie_Count': 1,  # Default für neue Regisseure
        'Actor1_Avg_Revenue': df['Actor1_Avg_Revenue'].mean()  # Durchschnitt für unbekannte Schauspieler
    }

    # DataFrame erstellen
    X_pred = pd.DataFrame([features_dict])
    X_pred = X_pred[features]  # Reihenfolge der Features anpassen

    # Standardisierung
    X_pred_scaled = scaler.transform(X_pred)

    # Vorhersage
    prediction = best_model.predict(X_pred_scaled)
    return prediction[0]

def predict_box_office_range(running_time, budget, genre, release_year):
    # Basis-Vorhersage
    base_prediction = predict_box_office(running_time, budget, genre, release_year)

    if base_prediction is None:
        return None

    # Konfidenzintervall berechnen (±20%)
    lower_bound = base_prediction * 0.8
    upper_bound = base_prediction * 1.2

    # Gewinnspannen berechnen
    min_profit = lower_bound - budget
    max_profit = upper_bound - budget

    return {
        'box_office_range': (lower_bound, upper_bound),
        'profit_range': (min_profit, max_profit)
    }

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