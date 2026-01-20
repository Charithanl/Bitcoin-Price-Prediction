from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    # Create a scaler to standardize features
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize Logistic Regression model
    model = LogisticRegression(
    class_weight={0: 1, 1: 2},
    max_iter=1000,
    random_state=42
)

    
    # Train the model on scaled training data
    model.fit(X_train_scaled, y_train)

    # Return trained model and scaler
    return model, scaler
