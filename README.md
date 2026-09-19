# Sentiment-Analysis
A lightweight Python sentiment classifier that uses simple lexical features and logistic regression–style scoring to predict whether a sentence is positive or negative.

How It Works

The classifier extracts features from text, including:

  Count of positive words

  Count of negative words

  Presence of “!”

  Presence of “not”

  Total word count

These features are combined with user‑defined weights and passed through a sigmoid function to produce a probability. Anything above 0.5 is labeled positive, otherwise negative.
