"""Tests run on a tiny made-up sample (tests/sample.tsv), not the real dataset,
so they are fast and work offline."""

import io
import os
import zipfile

import numpy as np
import pytest

from trustcheck.data import extract_from_zip, load_messages
from trustcheck.evaluate import evaluate
from trustcheck.model import TrustCheck

SAMPLE = os.path.join(os.path.dirname(__file__), "sample.tsv")


@pytest.fixture(scope="module")
def checker():
    texts, labels = load_messages(SAMPLE)
    return TrustCheck().fit(texts, labels)


def test_load_messages_parses_labels():
    texts, labels = load_messages(SAMPLE)
    assert len(texts) == len(labels) == 30
    assert set(labels) == {0, 1}
    assert texts[0].startswith("Hey are we")


def test_extract_from_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme", "info")
        zf.writestr("SMSSpamCollection", "ham\thello\nspam\tfree prize\n")
    out = extract_from_zip(buf.getvalue(), str(tmp_path / "SMSSpamCollection"))
    texts, labels = load_messages(out)
    assert texts == ["hello", "free prize"] and labels == [0, 1]


def test_obvious_spam_and_ham(checker):
    assert checker.check("Claim your free cash prize now, text WIN").label == "spam"
    assert checker.check("Are we meeting for lunch tomorrow").label == "ham"


def test_explanation_adds_up_to_model_score(checker):
    """The word contributions plus the bias should equal the model's raw score.
    This proves the explanation is exactly what the model computed."""
    text = "Free prize, call now to claim your cash"
    reasons = checker.explain(text, top_n=1000)
    total = sum(c for _, c in reasons) + checker.model.intercept_[0]
    raw = checker.model.decision_function(checker.vectorizer.transform([text]))[0]
    assert np.isclose(total, raw)


def test_unknown_words_are_not_trusted(checker):
    """Even at 100% threshold-beating confidence, no evidence means no trust."""
    checker_low = TrustCheck(threshold=0.0)
    checker_low.vectorizer, checker_low.model, checker_low.words = (
        checker.vectorizer, checker.model, checker.words)
    result = checker_low.check("zxqv blorp")
    assert result.reasons == []
    assert not result.trusted


def test_evaluate_report(checker):
    texts, labels = load_messages(SAMPLE)
    report = evaluate(checker, texts, labels)
    assert 0 <= report["accuracy"] <= 1
    assert sum(row["count"] for row in report["calibration"]) == len(texts)
