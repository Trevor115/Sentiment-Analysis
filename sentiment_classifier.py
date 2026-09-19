"""
Sentiment Classifier

Name: Trevor Tourdot
Date: 09/16/2026

"""

import math


# Define word lists (you may expand these)
POSITIVE_WORDS = {"love", "great", "amazing", "good", "awesome", "fun"}
NEGATIVE_WORDS = {"hate", "bad", "awful", "terrible", "worst", "horrendous"}


def tokenize(text):
    """
    Convert text to lowercase and split into words.

    Example:
    "I love this!" -> ["i", "love", "this"]
    """
    # Convert text to lowercase
    tokens = text.lower()
    # Remove punctuation (.,!? etc.)
    for char in ".,!?;:":
        tokens = tokens.replace(char, " ")
    # Split into words and return
    return tokens.split()


def extract_features(text):
    """
    Convert text into a feature vector.

    REQUIRED features:
    x1 = count of positive words
    x2 = count of negative words
    x3 = 1 if "!" appears, else 0
    x4 = 1 if "not" appears, else 0

    OPTIONAL:
    Add at least one additional feature of your choice

    Return:
    A list like [x1, x2, x3, x4, ...]
    """
    # Tokenize the text
    tokens = tokenize(text)

    # Count positive words
    positive_count = sum(1 for word in tokens if word in POSITIVE_WORDS)

    # Count negative words
    negative_count = sum(1 for word in tokens if word in NEGATIVE_WORDS)

    # Check if "!" appears
    exclamation = 1 if "!" in text else 0

    # Check if "not" appears
    not_present = 1 if "not" in text else 0

    # Add one more feature
    word_count = len(tokens)

    features = [positive_count, negative_count, exclamation, not_present, word_count]
    return features


def sigmoid(z):
    """
    Apply the sigmoid function.

    Formula:
    1 / (1 + e^-z)
    """
    return 1 / (1 + math.exp(-z))


def compute_z(features, weights, bias):
    """
    Compute:
    z = w·x + b
    """
    # Multiply weights and features, then add bias
    z = sum(w * x for w,x in zip(weights, features)) + bias
    return z


def predict_probability(features, weights, bias):
    """
    Return probability that the text is positive.
    """
    # Compute z
    z = compute_z(features, weights, bias)

    # Apply sigmoid
    probability = sigmoid(z)

    return probability


def classify(probability):
    """
    Convert probability into label.

    Rule:
    > 0.5 = positive
    <= 0.5 = negative
    """
    # Return "positive" or "negative"
    return "positive" if probability > 0.5 else "negative"


def print_prediction(text, weights, bias):
    """
    Print prediction results.
    """
    features = extract_features(text)
    probability = predict_probability(features, weights, bias)
    label = classify(probability)

    print(f"Text: {text}")
    print(f"Features: {features}")
    print(f"Probability: {round(probability, 3)}")
    print(f"Prediction: {label}")
    print("-" * 50)


def main():
    # IMPORTANT:
    # Adjust weights if needed based on your features

    weights = [1.5, -2.0, 1.0, -1.0, 0.5]
    bias = 0.0

    test_sentences = [
        "I love this!",
        "This is bad",
        "This is not good",
        "This is awesome",
        "I hate this product",
        "That was so fun!",
        "They were horrendous",
        "I did not have fun",
    ]

    print("Sentiment Classifier Results")
    print("=" * 50)

    for sentence in test_sentences:
        print_prediction(sentence, weights, bias)


if __name__ == "__main__":
    main()