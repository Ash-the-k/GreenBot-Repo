import os
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

# Create sample data for CountVectorizer
sample_data = ["This is a sample text.", "This is another text sample."]

# Initialize and fit the CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sample_data)

# Save the CountVectorizer
saved_state_dir = 'saved_state'
if not os.path.exists(saved_state_dir):
    os.makedirs(saved_state_dir)

intent_cv_path = os.path.join(saved_state_dir, 'IntentCountVectorizer.joblib')
dump(vectorizer, intent_cv_path)

print(f"IntentCountVectorizer has been saved to '{intent_cv_path}'")
