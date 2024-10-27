import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset with a specific encoding
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')  # Adjust path and encoding as necessary

# Trim whitespace from column names
data.columns = data.columns.str.strip()

# Display the first few rows and the columns of the dataset
print(data.head())
print(data.columns)  # Check the actual column names

# Check if the necessary columns exist
if 'label' not in data.columns or 'message' not in data.columns:
    raise ValueError("Data must contain 'label' and 'message' columns.")

# Encode labels: spam as 1 and legitimate as 0
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Visualize the distribution of classes
sns.countplot(x='label', data=data)
plt.title('Distribution of Spam and Legitimate Messages')
plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Spam'])
plt.show()

# Split the data into features and target variable
X = data['message']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize classifiers
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True)
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")

    # Optional: AUC-ROC for SVM
    if model_name == 'Support Vector Machine':
        from sklearn.metrics import roc_auc_score

        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        print(f"{model_name} AUC-ROC: {roc_auc_score(y_test, y_prob):.2f}\n")
