
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')

# Sample dataset
data = {
    "Message": [
        "Congratulations you won a prize",
        "Call me later",
        "Win money now",
        "How are you",
        "Claim your free reward",
        "Lets meet tomorrow"
    ],
    "Label": ["spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

df['Message'] = df['Message'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['Message'])

X = tokenizer.texts_to_sequences(df['Message'])
X = pad_sequences(X, maxlen=20)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['Label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=20),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Predict custom message
while True:
    text = input("Enter a message: ")
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=20)

    prediction = model.predict(padded)

    if prediction[0][0] > 0.5:
        print("Spam Message")
    else:
        print("Ham Message")
