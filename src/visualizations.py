import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_visualizations():
    # Daten laden
    df = pd.read_csv('data/movie_data.csv')

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
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_visualizations()