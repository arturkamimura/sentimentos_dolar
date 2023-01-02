import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
from nltk.tokenize import wordpunct_tokenize

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum(): #checar alfanumericos
            rem += i
        else:
            rem += ' '
    return rem

def to_lower(text):
    return text.lower()


#nltk.download('stopwords')
#nltk.download('punkt')
#stopwords = nltk.corpus.stopwords.words('portuguese')

stopwords = ['a', 'à', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'às', 'até', 'como',
              'com', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'é', 'ela',
              'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'éramos', 'essa', 'essas', 'esse', 'esses', 'esta',
              'está', 'estamos', 'estão', 'estar', 'estas', 'estava', 'estavam', 'estávamos', 'este', 'esteja',
              'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram',
              'estivéramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem', 'estivéssemos', 'estou', 'eu',
              'foi', 'fomos', 'for', 'fora', 'foram', 'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos',
              'fui', 'há', 'haja', 'hajam', 'hajamos', 'hão', 'havemos', 'haver', 'hei', 'houve', 'houvemos', 'houver',
              'houvera', 'houverá', 'houveram', 'houvéramos', 'houverão', 'houverei', 'houverem', 'houveremos',
              'houveria', 'houveriam', 'houveríamos', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'isso',
              'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito',
              'na', 'não', 'nas', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o',
              'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'são',
              'se', 'seja', 'sejam', 'sejamos', 'sem', 'ser', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam',
              'seríamos', 'seu', 'seus', 'só', 'somos', 'sou', 'sua', 'suas', 'também', 'te', 'tem', 'tém', 'temos',
              'tenha', 'tenham', 'tenhamos', 'tenho', 'terá', 'terão', 'terei', 'teremos', 'teria', 'teriam',
              'teríamos', 'teu', 'teus', 'teve', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera',
              'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'tu', 'tua', 'tuas',
              'um', 'uma', 'você', 'vocês', 'vos']
#print(stopwords)

def rem_stopwords(text):
    words = wordpunct_tokenize(text)
    return [w for w in words if w not in stopwords]

def unir(text):
    return ' '.join(w for w in text)

df = pd.read_excel('compilado_noticias_dolar.xlsx')


df.Textos = df.Textos.apply(is_special)

df.Textos = df.Textos.apply(to_lower)
df.Textos = df.Textos.apply(rem_stopwords)
df.Textos = df.Textos.apply(unir)

all_words_str = ' '.join(df.Textos)

def plot_cloud(wordcloud):
    plt.figure(figsize=(30,20))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(f'teste.png')
    plt.close()

wordcloud = WordCloud(width=2000, height=1000, random_state=1, background_color='black',
                      colormap='viridis', collocations=False).generate(all_words_str)

plot_cloud(wordcloud)
#X = df.Textos['Textos']
y = np.array(df.Var.values)
cv = CountVectorizer()
X = cv.fit_transform(df.Textos).toarray()

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=0)


trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.15, random_state=0)

gnb, mnb, bnb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True)

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0, max_depth=200)

svcl = svm.SVC(max_iter=2000)
svcl.fit(trainx, trainy)
text_classifier.fit(trainx, trainy)
gnb.fit(trainx, trainy)
mnb.fit(trainx, trainy)
bnb.fit(trainx, trainy)



ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)


tes_c = text_classifier.predict(testx)
test_svm = svcl.predict(testx)

print(f'{accuracy_score(testy, ypg)},  {accuracy_score(testy, ypm)}, {accuracy_score(testy, ypb)},\n'
      f'{accuracy_score(testy, tes_c)}, {accuracy_score(testy, test_svm)}')


joblib.dump(text_classifier, 'model.pkl')

# Convert into ONNX format
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print(type(testx), testx)
initial_type = [('float_input', FloatTensorType([None, 16411]))]
onnx_model = convert_sklearn(text_classifier, initial_types=initial_type)
with open("text_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Compute the prediction with ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("text_classifier.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: testx.astype(numpy.float32)})[0]
print(type(testx), pred_onx)