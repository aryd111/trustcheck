"""The classifier, plus the two things that make it "trustworthy":
an explanation for every prediction, and an honest "I'm not sure".

We use logistic regression on word counts on purpose. It is not the most
powerful model, but every word gets one learned weight, so we can show
exactly why the model decided what it did. That trade-off (accuracy vs.
being able to explain the model) is a big topic in trustworthy AI.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Below this confidence, we tell the user a human should double-check.
DEFAULT_THRESHOLD = 0.80


@dataclass
class Result:
    message: str
    label: str              # "spam" or "ham"
    confidence: float       # probability of the predicted label, 0.5 to 1.0
    trusted: bool           # False means "a human should check this"
    reasons: list = field(default_factory=list)  # (word, contribution) pairs


class TrustCheck:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.threshold = threshold
        # Turn text into numbers: one column per word (and 2-word phrase).
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.words = np.array(self.vectorizer.get_feature_names_out())
        return self

    def spam_probability(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1]

    def explain(self, text, top_n=5):
        """Which words pushed this message toward spam (+) or ham (-)?

        For logistic regression, each word's contribution is simply
        (how much of that word is in the message) x (the word's weight).
        Adding up all contributions plus the bias gives the model's score.
        """
        x = self.vectorizer.transform([text])
        cols = x.indices                      # words that appear in the message
        contributions = x.data * self.model.coef_[0][cols]
        order = np.argsort(-np.abs(contributions))[:top_n]
        return [(str(self.words[cols[i]]), float(contributions[i])) for i in order]

    def check(self, text, top_n=5):
        p_spam = float(self.spam_probability([text])[0])
        label = "spam" if p_spam >= 0.5 else "ham"
        confidence = p_spam if label == "spam" else 1 - p_spam
        reasons = self.explain(text, top_n)
        # If the model recognizes no words at all, its "confidence" comes only
        # from the bias (most messages are normal), not from this message.
        # High confidence with zero evidence should never be trusted.
        return Result(
            message=text,
            label=label,
            confidence=confidence,
            trusted=confidence >= self.threshold and len(reasons) > 0,
            reasons=reasons,
        )

    def top_words(self, n=10):
        """The words the model has learned are the strongest signals overall."""
        coef = self.model.coef_[0]
        spam_idx = np.argsort(-coef)[:n]
        ham_idx = np.argsort(coef)[:n]
        return (
            [(str(self.words[i]), float(coef[i])) for i in spam_idx],
            [(str(self.words[i]), float(coef[i])) for i in ham_idx],
        )
