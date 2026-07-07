# # ==========================================================
# # Customer Churn Prediction Model Training
# # ==========================================================

# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # ==========================================================
# # Load Dataset
# # ==========================================================

# print("Loading dataset...")

# df = pd.read_csv("Telco_Customer_Churn_Dataset.csv")

# # ==========================================================
# # Data Cleaning
# # ==========================================================

# df.drop("customerID", axis=1, inplace=True)

# df["TotalCharges"] = pd.to_numeric(
#     df["TotalCharges"],
#     errors="coerce"
# )

# df["TotalCharges"].fillna(
#     df["TotalCharges"].median(),
#     inplace=True
# )

# df["Churn"] = df["Churn"].map({
#     "No": 0,
#     "Yes": 1
# })

# # ==========================================================
# # Features and Target
# # ==========================================================

# X = df.drop("Churn", axis=1)
# y = df["Churn"]

# # ==========================================================
# # Identify Feature Types
# # ==========================================================

# categorical_features = X.select_dtypes(
#     include=["object"]
# ).columns.tolist()

# numerical_features = X.select_dtypes(
#     exclude=["object"]
# ).columns.tolist()

# # ==========================================================
# # Preprocessing
# # ==========================================================

# numeric_transformer = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="median"))
#     ]
# )

# categorical_transformer = Pipeline(
#     steps=[
#         (
#             "encoder",
#             OneHotEncoder(
#                 drop="first",
#                 handle_unknown="ignore"
#             )
#         )
#     ]
# )

# preprocessor = ColumnTransformer(
#     transformers=[
#         (
#             "num",
#             numeric_transformer,
#             numerical_features
#         ),
#         (
#             "cat",
#             categorical_transformer,
#             categorical_features
#         )
#     ]
# )

# # ==========================================================
# # Model Pipeline
# # ==========================================================

# model = Pipeline(
#     steps=[
#         ("preprocessor", preprocessor),
#         ("classifier", LogisticRegression(max_iter=1000))
#     ]
# )

# # ==========================================================
# # Train Test Split
# # ==========================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )

# # ==========================================================
# # Train Model
# # ==========================================================

# print("Training model...")

# model.fit(
#     X_train,
#     y_train
# )

# # ==========================================================
# # Evaluate
# # ==========================================================

# predictions = model.predict(X_test)

# accuracy = accuracy_score(
#     y_test,
#     predictions
# )

# print("\n==============================")
# print("Model Accuracy")
# print("==============================")

# print(f"Accuracy : {accuracy:.4f}")

# print("\nClassification Report")

# print(
#     classification_report(
#         y_test,
#         predictions
#     )
# )

# # ==========================================================
# # Save Model
# # ==========================================================

# joblib.dump(
#     model,
#     "churn_model.pkl"
# )

# print("\nModel saved as churn_model.pkl")

# # ==========================================================
# # Save Feature Names
# # ==========================================================

# feature_names = model.named_steps[
#     "preprocessor"
# ].get_feature_names_out()

# joblib.dump(
#     list(feature_names),
#     "feature_columns.pkl"
# )

# print("Feature columns saved as feature_columns.pkl")

# print("\nTraining Completed Successfully.")