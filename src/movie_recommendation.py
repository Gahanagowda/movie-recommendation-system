from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def recommend_movies(user_id, df):
    # Combine the watched genres and favorite actors
    df['Combined_Features'] = df['Watched_Genres'] + ',' + df['Favorite_Actors']

    # Convert text data into numerical vectors
    vectorizer = CountVectorizer().fit_transform(df['Combined_Features'])
    similarity_matrix = cosine_similarity(vectorizer)

    # Find the index of the user
    user_idx = df.index[df['User_ID'] == user_id].tolist()[0]

    # Get similarity scores for the user
    similarity_scores = list(enumerate(similarity_matrix[user_idx]))

    # Sort by similarity scores and exclude the user's own entry
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]

    # Recommend movies from top similar users
    similar_users_idx = [i[0] for i in sorted_scores[:3]]  # Top 3 similar users
    recommendations = df.iloc[similar_users_idx]
