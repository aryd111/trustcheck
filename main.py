"""Command-line entry point.

    python main.py evaluate                 # train, test, and print the trust report
    python main.py check "WIN a FREE prize, call now!"
    python main.py words                    # strongest spam / normal words
    python main.py interactive              # type messages and get explanations
"""

import argparse

from sklearn.model_selection import train_test_split

from trustcheck.data import load_messages
from trustcheck.evaluate import evaluate, print_report
from trustcheck.model import DEFAULT_THRESHOLD, TrustCheck


def build(threshold):
    texts, labels = load_messages()
    # Hold out 20% of messages the model never sees during training,
    # so the test measures how it does on new messages.
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    checker = TrustCheck(threshold=threshold).fit(X_train, y_train)
    return checker, X_test, y_test


def show(result):
    verdict = "SPAM" if result.label == "spam" else "NOT SPAM"
    print(f"\nMessage:    {result.message}")
    print(f"Prediction: {verdict}  ({result.confidence:.0%} confident)")
    if not result.reasons:
        print("Trust:      NOT SURE. It doesn't recognize any of these words.")
    elif not result.trusted:
        print("Trust:      NOT SURE. A human should double-check this one.")
    else:
        print("Trust:      confident")
    print("Why:")
    if not result.reasons:
        print("  (no known words, so the model is only guessing)")
    for word, c in result.reasons:
        direction = "toward spam" if c > 0 else "toward not spam"
        bar = "#" * max(1, min(20, int(abs(c) * 10)))
        print(f"  {word!r:<22} {bar:<20} {direction} ({c:+.2f})")


def main():
    parser = argparse.ArgumentParser(description="A spam filter that explains itself.")
    parser.add_argument("command", choices=["evaluate", "check", "words", "interactive"])
    parser.add_argument("message", nargs="?", help="message to check (for 'check')")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="confidence needed to trust a prediction (default 0.80)")
    args = parser.parse_args()

    checker, X_test, y_test = build(args.threshold)

    if args.command == "evaluate":
        print_report(evaluate(checker, X_test, y_test))
    elif args.command == "check":
        if not args.message:
            parser.error("give a message, e.g. python main.py check \"Free prize!\"")
        show(checker.check(args.message))
    elif args.command == "words":
        spam, ham = checker.top_words(10)
        print("Strongest SPAM signals:      ", ", ".join(w for w, _ in spam))
        print("Strongest NOT-SPAM signals:  ", ", ".join(w for w, _ in ham))
    else:
        print("Type a message and press Enter (blank line to quit).")
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            show(checker.check(text))


if __name__ == "__main__":
    main()
