def main():
    df = load_data('data/movies_dataset.csv')
    data = prepare_data(df)

    # For Random Forest, prepare features and target
    X = df[['User_ID', 'Movie_ID']]
    y = df['Ratings']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    knn_mse = train_knn(data)
    svd_mse = train_svd(data)
    rf_mse = train_random_forest(X_train, y_train, X_test, y_test)

    # Train AutoML Random Forest
    automl_model = train_automl(X_train, y_train)

    # Print results
    print(f"KNN Mean Squared Error: {knn_mse}")
    print(f"SVD Mean Squared Error: {svd_mse}")
    print(f"Random Forest Mean Squared Error: {rf_mse}")
    print(f"Best AutoML Model: {automl_model}")
def main():
    # Your main code logic goes here
    print("Movie recommendation system is running")

# The following block should have indented code inside
if __name__ == "__main__":
    main()  # Indent this line correctly

    
