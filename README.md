# TrustCheck

**A spam filter that explains its decisions and says when it isn't sure.**

Most AI tools give you an answer with no reason and no warning when they might be wrong. TrustCheck is a small experiment in doing it differently. For every text message it:

1. **Predicts** whether the message is spam.
2. **Explains** which words pushed it toward spam or toward normal.
3. **Flags uncertainty.** If its confidence is below a threshold (80% by default), it says a human should double-check.

It also **tests whether its own confidence can be trusted**: when the model says it's confident, is it actually right more often?

```
$ python main.py check "Call now to claim your free prize"

Message:    Call now to claim your free prize
Prediction: SPAM  (98% confident)
Trust:      confident
Why:
  'free'                 #########            toward spam (+0.92)
  'call'                 #########            toward spam (+0.91)
  'claim'                #######              toward spam (+0.79)
  'prize'                ######               toward spam (+0.66)
  'your'                 #####                toward spam (+0.51)
```

## Results

Measured on 1,115 held-out test messages (20% of the dataset) that the model never saw during training:

| Metric | Result |
|---|---|
| Accuracy | 98.3% |
| Spam precision | 93.9% (9 real messages wrongly flagged) |
| Spam recall | 93.3% (10 spam messages missed) |
| Confident predictions (≥ 80%) | 85.7% of messages, **99.7% accurate** |
| Unsure predictions (< 80%) | 14.3% of messages, 89.9% accurate |

So the "ask a human" flag does what it should: almost all of the mistakes are in the group the model flags as unsure.

**Calibration:**

| Model's confidence | Messages | Avg. confidence | Actually correct |
|---|---|---|---|
| 50–60% | 21 | 55.2% | 52.4% |
| 60–70% | 41 | 66.1% | 90.2% |
| 70–80% | 97 | 76.2% | 97.9% |
| 80–90% | 457 | 86.2% | 99.6% |
| 90–100% | 499 | 93.3% | 99.8% |

### What I found

1. **The model is *under*confident.** When it says 70–80%, it's actually right 98% of the time. This is probably a side effect of `class_weight="balanced"` and regularization, both of which pull probabilities toward the middle. An underconfident model is safer than an overconfident one, but it sends more messages to a human than it needs to. Recalibrating the probabilities (for example with Platt scaling) is a next step.
2. **High confidence doesn't always mean evidence.** `fr33 pr1ze, c4ll n0w` contains no words the model knows, yet it still said "not spam, 81% confident". That number came entirely from the bias, because most messages in the data are normal. The model was confident about the base rate, not about the message. I fixed this: a prediction with zero recognized words is never marked trusted. This is a small version of a real problem in AI, where models give confident answers to inputs unlike anything they were trained on.
3. **The data is old.** A modern delivery scam ("Your package is waiting, confirm your address here") came out as "not spam, 51%". The model was right to flag it as unsure, but it got the label wrong, because 2012 spam looked different.

## Quick start

```bash
git clone https://github.com/aryd111/trustcheck.git
cd trustcheck
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py evaluate                        # accuracy + trust report
python main.py check "WIN a FREE prize, call now!"
python main.py words                           # strongest spam / normal words
python main.py interactive                     # try your own messages
python -m pytest                               # run the tests
```

The first run downloads the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 real messages) into `data/`.

## How it works

| Step | What happens | File |
|---|---|---|
| Load data | Read messages labelled `ham` (normal) or `spam` | `trustcheck/data.py` |
| Text → numbers | TF-IDF turns each message into word scores (single words and 2-word phrases) | `trustcheck/model.py` |
| Learn | Logistic regression learns one weight per word | `trustcheck/model.py` |
| Explain | Each word's contribution = its score in the message × its weight | `trustcheck/model.py` |
| Trust | Below the confidence threshold → "a human should check" | `trustcheck/model.py` |
| Evaluate | Accuracy, precision/recall, and a calibration table on a 20% held-out test set | `trustcheck/evaluate.py` |

**Why logistic regression and not a neural network?** A linear model is less powerful, but its explanations are *exact*. The word contributions plus the bias add up to the model's score, and a test (`test_explanation_adds_up_to_model_score`) checks this. A deep model would need approximate explanation methods like LIME or SHAP, which can be wrong themselves. That trade-off between accuracy and being able to explain a model is a central question in trustworthy AI.

## Limitations

- It only knows words it saw in training, so misspellings like `fr33` fool it. Those messages are now marked untrusted, but they aren't caught as spam.
- The dataset is from 2011–2012, so modern scams (crypto, delivery-tracking links) slip through.
- The probabilities are underconfident (see Results). The `evaluate` trust numbers use confidence only and don't include the zero-evidence rule.

## Ideas for next steps

- Recalibrate the probabilities (Platt scaling) and compare the calibration tables.
- Compare against Naive Bayes and see which explains itself better.
- Normalize simple misspellings (`3`→`e`, `0`→`o`) before classifying.
- Add a tiny web page where people can paste a message.

## How this was built

Built by Aryamann Dhall with help from Claude, Anthropic's AI assistant, which I used to plan the project and write and review code. I can walk through every part of it, and the code comments explain each step.

## Credits

Dataset: Almeida, T. A., & Hidalgo, J. M. G. (2011). *SMS Spam Collection*. UCI Machine Learning Repository.

License: MIT
