import random

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

pd.options.display.max_columns = None

analyser = SentimentIntensityAnalyzer()

positive_words = [
    "amazing", "beautiful", "brilliant", "cheerful", "charming", "commendable", "delightful", "excellent", "excited", "extraordinary",
    "fabulous", "fantastic", "favorable", "friendly", "fun", "generous", "genius", "glorious", "good", "great",
    "happy", "helpful", "incredible", "inspiring", "joy", "kind", "lovely", "love", "magnificent", "marvelous",
    "motivated", "nice", "outstanding", "passionate", "perfect", "pleasant", "positive", "proud", "radiant", "refreshing",
    "remarkable", "respectful", "satisfying", "sensational", "smart", "spectacular", "splendid", "strong", "successful", "wonderful"
]

neutral_words = [
    "account", "average", "basic", "case", "category", "common", "condition", "consistent", "data", "detail",
    "document", "environment", "factor", "figure", "general", "group", "information", "item", "level", "list",
    "material", "method", "model", "moderate", "normal", "number", "objective", "option", "part", "pattern",
    "percentage", "plain", "process", "profile", "range", "regular", "report", "resource", "result", "section",
    "series", "setting", "standard", "statistic", "structure", "subject", "term", "type", "typical", "value"
]

negative_words = [
    "abysmal", "annoying", "awful", "bad", "bitter", "broken", "careless", "cheap", "clumsy", "cold",
    "complaint", "corrupt", "cruel", "damaged", "dangerous", "defective", "depressing", "dishonest", "dirty", "dreadful",
    "fail", "faulty", "fear", "filthy", "foolish", "frustrating", "greedy", "gross", "hate", "horrible",
    "hostile", "hurt", "inadequate", "inferior", "infuriating", "insult", "junk", "lousy", "mean", "messy",
    "negative", "offensive", "pain", "pathetic", "poor", "problematic", "rude", "scary", "terrible", "worst"
]

leaflets_df = pd.read_csv('leaflets.csv', sep='¦', encoding='latin-1', engine='python')
leaflets_df['Text'] = leaflets_df['Text'].apply(lambda row: row + f' {" ".join([i for i in random.choice([negative_words, positive_words, neutral_words])])}')
for ind, row in leaflets_df.iterrows():
    print(row['Text'])
leaflets_df['sent'] = leaflets_df['Text'].apply(lambda row: analyser.polarity_scores(row)['compound'])
leaflets_df['sent_rounded'] = leaflets_df['sent'].round()
leaflets_df = leaflets_df.sort_values('sent', ascending=False)
leaflets_df = leaflets_df[['Leaflet_ID', 'sent', 'sent_rounded']]
leaflets_df.to_csv('sents_supp.csv', index=False)
