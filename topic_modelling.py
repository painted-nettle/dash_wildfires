import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv('leaflets.csv', sep='¦', encoding='latin-1', engine='python')

df['Text'] = df['Text'].str.replace('fire', '', regex=False)

# nltk.download('punkt_tab')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def preprocess(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


df['Text'] = df['Text'].apply(preprocess)

dictionary = corpora.Dictionary(df['Text'])
corpus = [dictionary.doc2bow(text) for text in df['Text']]

lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=10,
                     passes=50,
                     random_state=42,
                     per_word_topics=True)

lda_model.save('lda.model')
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda_model, f)

vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'pyldavis_visualization.html')

# 3. Determine the dominant topic for each leaflet and save to a CSV
print("Determining dominant topics for each leaflet...")
topic_list = []
for i, doc_topics in enumerate(lda_model.get_document_topics(corpus, minimum_probability=0.0)):
    doc_dominant_topic = sorted(doc_topics, key=lambda x: x[1], reverse=True)[0][0]
    topic_list.append(doc_dominant_topic)

df_topics = pd.DataFrame({
    'Leaflet_ID': df['Leaflet_ID'],
    'dominant_topic': topic_list
})
df_topics.to_csv('topics.csv', index=False)
