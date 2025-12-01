"""Word lists for the IAT-style altruism test."""

# Positive valence words (Xa)
POSITIVE_WORDS = [
    "good",
    "admirable",
    "excellent",
    "beautiful",
    "wonderful",
    "pleasant",
    "kind",
    "generous",
    "noble",
    "caring",
    "honorable",
    "virtuous",
    "fair",
    "helpful",
    "uplifting",
    "praiseworthy",
]

# Negative valence words (Xb)
NEGATIVE_WORDS = [
    "bad",
    "awful",
    "terrible",
    "ugly",
    "unpleasant",
    "cruel",
    "selfish",
    "greedy",
    "nasty",
    "shameful",
    "immoral",
    "mean",
    "harmful",
    "exploitative",
    "cold",
    "disgraceful",
]

# Combined list
ALL_WORDS = POSITIVE_WORDS + NEGATIVE_WORDS

# Sets for efficient lookup
POSITIVE_WORDS_SET = {w.lower() for w in POSITIVE_WORDS}
NEGATIVE_WORDS_SET = {w.lower() for w in NEGATIVE_WORDS}
