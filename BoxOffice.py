import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prediction import predict_box_office  # Hier ist die Korrektur


def create_visualizations(df):
    # Stil für die Plots setzen
    plt.style.use('seaborn')

    # Figure 1: Budget vs Box Office
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Budget'], df['Box Office'], alpha=0.5)
    plt.xlabel('Budget ($)')
    plt.ylabel('Box Office ($)')
    plt.title('Budget vs Box Office Earnings')
    plt.show()

    # Figure 2: Genre Distribution
    plt.figure(figsize=(12, 6))
    df['Genre'].value_counts().plot(kind='bar')
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Figure 3: Average Earnings by Genre
    plt.figure(figsize=(12, 6))
    df.groupby('Genre')['Box Office'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.title('Average Box Office Earnings by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Box Office ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Figure 4: Release Year Distribution
    plt.figure(figsize=(10, 6))
    df['Release year'].hist(bins=30)
    plt.title('Distribution of Release Years')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Movies')
    plt.show()

    # Figure 5: Running Time vs Box Office
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Running time'], df['Box Office'], alpha=0.5)
    plt.xlabel('Running Time (minutes)')
    plt.ylabel('Box Office ($)')
    plt.title('Running Time vs Box Office Earnings')
    plt.show()

    # Figure 6: Correlation Matrix
    plt.figure(figsize=(8, 6))
    numeric_cols = ['Budget', 'Running time', 'Box Office', 'Release year']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def display_welcome():
    print("\n=== Film Box Office Predictor ===")
    print("Dieses Programm sagt voraus, wie viel Ihr Film an den Kinokassen einspielen wird.")


def get_valid_float(prompt, min_value=0):
    while True:
        try:
            value = float(input(prompt))
            if value >= min_value:
                return value
            print(f"Bitte geben Sie einen Wert größer als {min_value} ein.")
        except ValueError:
            print("Bitte geben Sie eine gültige Zahl ein.")


def get_valid_year():
    while True:
        try:
            year = int(input("Erscheinungsjahr (2000-2030): "))
            if 2000 <= year <= 2030:
                return year
            print("Bitte geben Sie ein Jahr zwischen 2000 und 2030 ein.")
        except ValueError:
            print("Bitte geben Sie ein gültiges Jahr ein.")


def get_valid_genre():
    # Genres aus der CSV-Datei laden
    df = pd.read_csv('data/movie_data.csv')
    valid_genres = sorted(df['Genre'].unique())

    print("\nVerfügbare Genres:")
    print(", ".join(valid_genres))

    while True:
        genre = input("\nGenre: ").strip()
        if genre in valid_genres:
            return genre
        print("Ungültiges Genre. Bitte wählen Sie aus der Liste oben.")


def main():
    # Daten laden
    df = pd.read_csv('data/movie_data.csv')

    # Visualisierungen erstellen
    create_visualizations(df)

    display_welcome()

    while True:
        print("\n--- Neue Vorhersage ---")

        # Benutzereingaben sammeln
        title = input("\nFilmtitel: ").strip()
        running_time = get_valid_float("Laufzeit (in Minuten): ")
        budget = get_valid_float("Budget (in Dollar): ")
        genre = get_valid_genre()
        release_year = get_valid_year()

        # Vorhersage machen
        predicted_revenue = predict_box_office(
            running_time=running_time,
            budget=budget,
            genre=genre,
            release_year=release_year
        )

        # Ergebnisse anzeigen
        if predicted_revenue is not None:
            print(f"\nVorhersage für '{title}':")
            print(f"Budget: ${budget:,.2f}")
            print(f"Vorhergesagte Box Office Einnahmen: ${predicted_revenue:,.2f}")
            profit = predicted_revenue - budget
            print(f"Erwarteter {'Gewinn' if profit >= 0 else 'Verlust'}: ${abs(profit):,.2f}")

        # Fragen ob weitere Vorhersage gewünscht
        if input("\nWeitere Vorhersage machen? (ja/nein): ").lower() != 'ja':
            break

    print("\nProgramm beendet. Auf Wiedersehen!")


if __name__ == "__main__":
    main()