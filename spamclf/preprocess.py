"""Text cleaning. Used as the `preprocessor` of TfidfVectorizer so it is
part of the saved model and runs identically at training and prediction time."""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w\-]+\.(?:com|co\.uk|net|org|info|biz)\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w.\-]+@[\w\-]+\.[\w.\-]+\b")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s\-]{6,}\d(?!\w)")
CURRENCY_RE = re.compile(r"[£$€]")
NUMBER_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, replace URLs/emails/phones/money/numbers with placeholder tokens,
    and strip punctuation. The placeholders keep strong spam signals
    (e.g. 'urltoken', 'currencytoken') without memorising specific values."""
    text = str(text).lower()
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = URL_RE.sub(" urltoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = CURRENCY_RE.sub(" currencytoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()
