# -*- coding: utf-8 -*-

# -- Sheet --

import sys, os, re, csv, codecs, numpy as np, pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
#!pip install ruwordnet
#from ruwordnet import RuWordNet
#!pip install wiki-ru-wordnet
#from wiki_ru_wordnet import WikiWordnet
#wikiwordnet = WikiWordnet()
#!pip install pymorphy2
#import pymorphy2
!pip install stop_words
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
stop_words = get_stop_words('russian')
'меня' in stop_words

import zipfile
with zipfile.ZipFile("lstm_for_NLP_10ep.tf.zip","r") as zip_ref:
   zip_ref.extractall("lstm_model")

import tensorflow as tf

!pip install scipy
from scipy.spatial import distance

!pip install wget
!pip install navec
!python -m wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
from navec import Navec
path = 'navec_lib.tar'
navec = Navec.load(path)

phrase='спасите меня, я застрял'
def split(s, delimiters):
    flag =False
    for d in delimiters:
        print(d, s)
        if d in s:
            item, s = s.split(d, 1)
            flag =True
            yield item
    if flag !=True: yield s
    
        
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
delimiters = [",", " ", ".", ";", ":"]
print(list(split(phrase, delimiters)))
for word in split(phrase, delimiters):
    word = morph.parse(word)[0]
    print(word)

import re
def splits(s):
    #s = re.split(',:-.; ', s)
    ans = [x for x in re.split(';|,| |\n|\t',phrase) if x != '']
    return ans
delimiters = [",", " ", ".", ";", ":"]
phrase='спасите меня, я застрял'
print(list(splits(phrase)))

#https://www.kaggle.com/competitions/nlp-getting-started - with disaster keywords

from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
#model.save("word2vec.model")

!pip install pymystem3
from pymystem3 import Mystem
text = "Красивая мама красиво мыла раму"
m = Mystem()
lemmas = m.lemmatize(text)
print(''.join(lemmas))

decisions = [
    {'экстренные': [
        {'медицинские': {
            'action': '103',
            'nodes': [
                'отсутствие сознания',
                'остановка дыхания и кровообращения',
                'инородные тела верхних дыхательных путей',
                'поражение молнией',
                'отравление',
                {'травмы': [
                    { 'механические травмы': [
                        'повреждения вследствие взрывов',
                        'повреждения вследствие аварий',
                        'разрушения зданий',
                        'разрушения сооружений',
                        'разрушения конструкций',
                        'стихийных бедствий'
                    ]},
                    { 'термические травмы': [
                        'ожоги',
                        'обморожения',
                        'тепловой удар',
                        'ожог',
                        'обморожение'
                    ]},
                    {'спортивные травмы':[
                        'вывихи',
                        'переломы',
                        'растяжения',
                        'разрыв связок',
                        'ушибы',
                        'утомление мышц'
                    ]},
                    'баротравмы (под действием резких изменений атмосферного давления)',
                    'переломы',
                    'огнестрельные травмы',
                    'порезы',
                    'наружные кровотечения'
                ]},
                {'укусы': [
                    'укусы животными',
                    'укусы насекомыми',
                    'укусы паукообразными',
                    'укусы змеями',
                    'укусы кровососущих'
                ]},
                {'техногенные факторы': [
                    'химические травмы',
                    'радиационные травмы',
                    'электротравмы',
                    'излучение',
                    'поражение электрическим током'
                ]}, 
                {'происшествия на воде': [
                    'утопление',
                    'затопления',
                    'наводнения',
                    'потоп',
                    'сель'
                ]}
            ]
        }},
        {'аптечные':{
            'action': 'pharmacy',
            'nodes': [
                'сердце',
                "спина",
                'голова'
            ]
        }},
        {'полицейские': {
            'action': 'swat',
            'nodes': ['терроризм']
        }}
    ]},
    {'несрочные': [
        {'полицейские': {
            'action': '102',
            'nodes': ['кража','суд','кража','хулиганство']
        }}
    ]}
]

#need to add проверки и подтверждения от юзера на каждом уровне дерева сразу после получения ответа
hierchy_nodes_actions_classes = {
    ('медицина','03'):
        {('срочно','вызов скорой'):
            [('реанимация','вызов реанимации'),
             ('морг','вызов перевозки тела'),
             ('инсульт','вызов скорой'),
             ('инфаркт','вызов скорой'),
             ('труп','вызов перевозки тела'), 
             ('приступ','вызов скорой')],
        ('не срочно','ближайшую аптеку или поликлинику'):
            {('сердце','рекомендация лекарств/запись к кардиологу'):
                [('недостаточность','запись к кардиологу'),
                 ('аритмия','запись к кардиологу'),
                 ('учащенное сердцебиение','запись к кардиологу'),
                 ('боли в средце','список сердечных лекарств')]},
            ("спина",'список лекарств и запись к хирургу'):
                [('боль в спине','список мазей'),
                 ('поясница','список мазей'),
                 ('защемление','запись к хирургу'),
                 ('грыжа','запись к хирургу')], 
            ("простуда",'рекомендация лекарств'):
                [("температура",'жаропонижающее'),
                 ("насморк",'капли для носа'),
                 ("горло","лекарства от боли в горле, народные средства"),
                 ('кашель',"лекарства от кашля, горчичники"),
                 ("озноб",'жаропонижающее'),
                 ("сонливость",'сон и отдых'),
                 ("ангина",'антибиотики'),
                 ("грипп",'жаропонижающее')],
            ("самочувствие",'рекомендация'):
                [("диабет",'инсулин'),
                 ("сахар",'инсулин'),
                 ("тошнота","лекарства от отравления"),
                 ('отравление',"лекарства от отравления"),
                 ('температура',"жаропонижающее"),
                 ('давление',"лекарства от давления"),
                 ('головокружение','запись к терапевту'),
                 ('нервы','успокоительное'),
                 ('усталось','отдых и сон'),
                 ('слабость','отдых и сон')],
            ("желудок",'ближайшая аптека, консультация гастроэнтегролога'):
                [('изжога','лекарства от изжоги'),
                 ('рвота','лекарства для водно-солевого баланса'),
                 ('отравление','лекарства от отравления'),
                 ('диарея','лекарства от диареи'),
                 ('язва','консультация гастроэнтегролога'),
                 ('вздутие','лекарства от диареи')],
            ('травмы','ближайший травмпункт или самолечение'):
                [('перелом','ближайший травмпункт'),
                 ('ушиб','список мазей от ушибов'),
                 ('синяк','список мазей от ушибов'),
                 ('вывих','ближайший травмпункт'),
                 ('растяжение','ближайший травмпункт'),
                 ('трещина','ближайший травмпункт')]
         },
    ('полиция','02'):{
        ('срочно','вызов специального наряда'):
            {('ограбление','вызов вооруженного наряда'):
                [('вооруженное','вызов вооруженного наряда'),
                 ('заложники','вызов вооруженного наряда'),
                 ('шантаж','вызов вооруженного наряда')],
            ('терроризм','вызов ОМОНа'):
                [('заложники','вызов ОМОНа'),
                 ('захват территории','вызов ОМОНа'),
                 ('детонация','вызов саперов'),
                 ('бомба','вызов саперов'),
                 ('жертвы','вызов ОМОНа')],
            ('убийство','вызов судмедэкспертов,патологоанатомов'):
                [('кровь','вызов судмедэкспертов'),
                 ('огнестрельное','вызов вооруженного наряда и реанимации'),
                 ('ножевое','вызов реанимации'),
                 ('ранение','вызов реанимации'),
                 ('смерть','вызов перевозки тела'),
                 ('выстрел','вызов вооруженного наряда')]
        },
        ('не срочно','обращение в полицию'):
            {('суд','ближайший районный суд'):
                 [('разбирательство','ближайший районный суд'),
                  ('иск','ближайший районный суд'),
                  ('компенсация','ближайшее отделение'),
                  ('полномочия','ближайшее отделение')],
            ('кража','ближайшее отделение'):
                [('мелкие','заявление о краже'),
                 ('документы','миграционная служба'),
                 ('крупные','заявление о краже'),
                 ('драгоценности','заявление о краже'),
                 ('угон','заявление об угоне')] ,
            ('взлом','ближайшее отделение'):
                 [('вынос','заявление о краже'),
                  ('драгоценности','заявление о краже'),
                  ('сейф','заявление о краже')],
            ('хулиганство','вызов полиции'):
                [('провокация','вызов полиции'),
                 ('шантаж','вызов полиции'),
                 ('порча имущества','заявление в полиции'),
                 ('вандализм','вызов полиции'),
                 ('граффити','вызов полиции')],
             ('увечия','вызов полиции и скорой'):
                [('телесные повреждения','вызов полиции, заявление в участок'),
                 ('холодное оружие','вызов скорой и полиции'),
                 ('огнестрельное раненение','вызов скорой и полиции'),
                 ('избиение','вызов полиции'),
                 ('шантажирование','обращение в ближайший участок')]
            },
    },
    ('служба спасения','04'):{
        ('срочно',"вызов МЧС"):
            {('стихийные бедствия',"вызов МЧС"):
                [('смерч',"рекомендация укрыться, вызов МЧС"),
                ('тайфун',"рекомендация по эвакуации, вызов МЧС"),
                ('цунами',"рекомендация по эвакуации, вызов МЧС"),
                ('ураган',"рекомендация укрыться, вызов МЧС"),
                ('лавина',"вызов спасателей и поисковой службы"),
                ('потоп',"вызов службы спасения на воде"),
                ('наводнение',"вызов службы спасения на воде")],
            ('эвакуация',"вызов эвакуационной службы"):
                [('лес',"спасательный вертолет"),
                ('пустыня',"спасательный вертолет"),
                ('горы',"спасательный вертолет"),
                ('вода',"вызов службы спасения на воде")]},
        ('не срочно',"рекомендации по поведению"):{
            ('погода','рекомендации укрыться'):
                [('молния',"укрыться, найти громоотвод"),
                 ('засуха',"находиться около воды"),
                 ('жара',"находиться в тени, больше пить"),
                 ('холод',"не выходить из дома"),
                 ('метель',"не выходить из дома"),
                 ('дождь',"укрыться под навесом"),
                 ('град',"укрыться под навесом")],
            ('технические неисправности','вызов нужной службы'):
                [('застрял в лифте',"вызов службы лифтов"),
                 ('заклинила дверь',"вызов службы спасения"),
                 ('сработала тревога',"следовать указаниям, не паниковать"),
                 ('вырубило электричество',"вызов службы ЖКХ"),
                 ('связь или интернет',"вызов оператора провайдера"),
                 ('течет кран, протечка',"вызов сантехника"),
                 ("проводка","перекрыть подачу электричества"),
                 ('короткое замыкание',"перекрыть подачу электричества")]
            },
    ('пожар','01'):
        {('огонь',"вызов пожарных"):
            [('горит дом',"покинуть здание,вызов пожарных"),
             ('короткое замыкание',"перекрыть подачу электричества"),
             ("взрыв","вызов пожарных,вызов саперов"),
             ("газ","перекрыть газ, избегать огня")],
         ('пожарная тревога',"не паниковать, следовать указаниям"):
            [('ложная тревога',"спокойно заниматься своими делами"),
             ('короткое замыкание',"перекрыть подачу электричества"),
             ("учебная тревога","спокойно заниматься своими делами"),
             ("гарь","проверить источник запаха"),
             ("задымление","проверить источник дыма"),
             ("дым","проверить источник дыма")]},
    },   
    ('дорожное происшествие',''):{
        ('не срочно',"рекомендации по поведению, вызов службы ДТП"):
            {('трасса',"вызов службы ДТП"):
                [('авария',""),
                ('заправка',""),
                ('АЗС',""),
                ('бензин',""),
                ('поломка',""),
                ('фура',""),
                ('встречка',""),
                ('превышение скорости',""),
                ('красный свет',"")],
            ('бездорожье',"вызов службы спасения"):
                [('увязло авто',"вытолкать авто, вызвать эвакуатор"),
                ('заглохла машина',"вызвать эвакуатор"),
                ('потеря пути',"вызов спасателей, ориентация по небу")]},
        ('срочно',"вызов службы ДТП"):{
            ('дорога',"вызов службы ДТП и скорой помощи"):
                [('авария с жертвами',"вызов полиции и реанимации"),
                ('взрыв авто',"вызов пожарных и реанимации"),
                ('авто горит',"вызов пожарных и скорой"),
                ('сбила машина',"вызов реанимации и службы ДТП"),
                ('сбили животное',"вызов службы ДТП"),
                ('лобовое столкновение',"вызов реанимации и службы ДТП"),
                ('потеря управлением',"вызов скорой и службы ДТП"),
                ('отказ тормозов',"вызов скорой и службы ДТП"),
                ('вылет за обочину',"вызов скорой и службы ДТП")],
            ('бездорожье',"вызов службы спасения"):
                [('запрос эвакуации',"вызов службы эвакуации и поисковиков"),
                ('сигнал SOS',"вызов службы спасения")]},
    },  
    ('начало или конец диалога',''):
        {('приветствие',"начать диалог, поприветсвовать пользователя"):
            [('привет',"поприветсвовать пользователя"),
             ('здравствуйте',"поприветсвовать пользователя"),
             (".",'уточнить, хочет ли человек начать диалог'),
             ('/start',"turn on"),
             ('хай',"поприветсвовать пользователя"),
             ("йоу","поприветсвовать пользователя"),
             ('ку',"поприветсвовать пользователя")],
         ('прощание',"уточнить о конце диалога, попрощаться"):
            [('спасибо',"спросить, нужна и еще помощь"),
             ('счастливо',"попрощаться, спящий режим"),
             ('до свидания',"попрощаться, спящий режим"),
             ('благодарю','спросить, нужна и еще помощь'),
             ('пока',"попрощаться, спящий режим"),
             ("/stop","turn off")]},
    ('неопознанный запрос',''):
        {('не найден класс',"уточнить запрос, подобрать класс из имеющихся"):
            [('не то',"уточнить запрос"),
             ('другое',"уточнить запрос"),
             ("не ответили",'уточнить запрос')],
         ('нет правильных рекомендаций',"уточнить запрос, подобрать рекомендации из имеющихся"):
            [('пробовал уже',"уточнить запрос, предложить выбрать из списка, кроме последнего"),
             ('другое',"уточнить запрос, предложить выбрать из списка, кроме последнего"),
             ("не ответили",'уточнить запрос')]}
}   


def lemma_extraction(phrase):
     lemmas = []
     synsets = wikiwordnet.get_synsets(phrase)
     synset1 = synsets[0]
     for w in synset1.get_words():
          lemmas.append(w.lemma())
     return lemmas
lemma_extraction('синяк')

querry = [
    (['Мне нужна помощь. У моей мамы случился сердечный приступ.','У меня болит сердце, похоже приступ'],['медицина','срочно','приступ']),
    (['Мой дом в огне.','Дым в окне на 5м этаже в соседнем доме','Здесь дом горит','соседний дом загорится тоже'],['пожар','срочно','горит дом']),
    (['Меня ограбили.','вынесли драгоценности из дома'],['полиция','несрочно','кража']),
    (['Я опять застрял в лифте','застрял в 12-этажном доме'],['служба спасения','несрочно','технические неисправности',"застрял в лифте" ]),
    (['Эй, привет. Помоги!','пока. спасибо','здрасти','Спасибо, до свидания'], ['начало или конец диалога']),
    (['Друга сбила машина'], ['ДТП','срочно','дорога','сбила машина']),
    (['Меня вынесло с обочины','Столкнулись две легковушки','в меня врезался джип'], ['ДТП','срочно','дорога','авария']),
    (['Мой коллега упал с лестницы, и у него течет кровь.'],['медицина','срочно']),
    (['Здесь грабят магазин','В аптеку ворвались и угрожают продавщице','Несколько вооруженных людей в масках зашли  соружием в продуктовый','Какие-то люди взломали банкомат и ушли с кучей денег', 'Помогите, нас ограбили, они только что ушли с рюкзаками полным денег', 'В соседнем магазине бандиты выносят наличку'],['полиция','срочно','ограбление']),
    ([''],['полиция','срочно','терроризм']),
    (['Помогите, тут только что убили человека','Я только что видел, как человек зарезал другого ножом!','Два чувака только что дрались, один упал и не встает, мне кажется он умер',],['полиция','срочно','убийство'])
]

for hypernym in wikiwordnet.get_hypernyms(synset1):
     print({w.lemma() for w in hypernym.get_words()})


morph = pymorphy2.MorphAnalyzer()
input_string = input()
input_string = input_string.split()
input_string = [morph.parse(word)[0].normalized.word for word in input_string]
print(input_string)
for i in range(len(input_string)):
    lemms = wikiwordnet.get_synsets(input_string[i])
    if len(lemms)>0:
        synset1 = lemms[0]
        for word in synset1.get_words():
            print(word.lemma())
    else: print('empty string of syns')


word1 = navec['пожар']
word2 = navec['ожог']
word3 = navec['уточка']
word4 = navec['гореть']
word5 = navec['огонь']
word_array = ['пожар','ожог','уточка','гореть','огонь']
words = [word1,word2,word3,word4,word5]
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))

plt.plot(np.arange(len(word1))[:20], word1[:20],label ='пожар')
plt.plot(np.arange(len(word2))[:20], word2[:20],label ='ожог' )
plt.plot(np.arange(len(word3))[:20], word3[:20] ,label ='уточка')
plt.legend()
plt.show()

for i in range(len(words)):
    for j in range(len(words)):
        print(word_array[i],' ',word_array[j] ,' ', distance.cosine(words[i],words[j]))

key_words_embd = {}
for phrase in key_words:
    n = 0
    key_words_embd[phrase] = np.zeros(300)
    for word in phrase.split():
        word = word.strip(',!;?')
        if word in navec and word not in stop_words:
            key_words_embd[phrase] = key_words_embd[phrase] + navec[word]
            n = n + 1
    if n == 0:
         key_words_embd[phrase] = np.ones(300)
    else:
        key_words_embd[phrase] = key_words_embd[phrase] / n
        #key_words_embd[phrase] = np.asarray([navec[word] for word in phrase.split() if word in navec]).sum(axis=0)
    #print(key_words_em
    bd[phrase].s

def flatten(tree):
    result=[]
    for key, item in tree.items():
        if isinstance(item , dict):
            result.extend(flatten(item))
        else:
            result.extend(item)
        result.append(key)

    return result

flatten(hierchy_nodes_actions_classes)

def nodes_from_tree(tree, parent, r=[]):
    r = [parent]
    for child in tree.get_children(parent):
        r.extend(nodes_from_tree(tree, child, r))
    return r
nodes_from_tree(hierchy_nodes_actions_classes, hierchy_nodes_actions_classes)

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tensorflow import keras
from sklearn.model_selection import train_test_split
print("tf version:",tf.__version__)
warnings.filterwarnings("ignore")
print("keras version:", keras.__version__)
neutr = pd.read_fwf('neutral.txt', sep='\n')
negativ = pd.read_fwf('negative.txt')
neutral = pd.read_csv('neutral.csv', on_bad_lines='skip', sep='\n')#read_fwf
negative = pd.read_csv('negative.csv', on_bad_lines='skip')
with open('neutral.csv') as f:
    neutral = f.readlines()
with open('negative.csv') as f:
    negative = f.readlines()
neutral = pd.DataFrame({'x':np.array([x.strip() for x in neutral]),'y': np.zeros(len(neutral)).astype(int)})
negative = pd.DataFrame({'x':np.array([x.strip() for x in negative]),'y': np.ones(len(negative)).astype(int)})
res_df = pd.concat([neutral, negative],axis=0).reset_index(drop=True)
BUFFER_SIZE = 10000
BATCH_SIZE = 512
train_dataset,test_dataset = train_test_split(res_df,train_size=0.8, stratify = res_df['y'])
VOCAB_SIZE = 2048
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset['x'])
lstm = tf.keras.Sequential([
encoder,
tf.keras.layers.Embedding(
input_dim=len(encoder.get_vocabulary()),
output_dim=16,
mask_zero=False),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
tf.keras.layers.Dense(16, activation='sigmoid'),
tf.keras.layers.Dense(1)
]) 
lstm= keras.models.load_model('lstm_model/lstm_for_NLP_10ep.tf', custom_objects=None, compile=True, options=None)

def go_further(ans,dicts):
    if isinstance(dicts , dict):
        for x,y in dicts.items():
            if x[0]==ans:
                print('Выберите один из наиболее подходящих Вам случаев:')
                try:print(list(y.keys()))
                except:print(list(y))
                return y
        print('ничего не подходит из вашего сообщения!')
    # elif isinstance(dicts , list):
    #     for y in dicts:
    #         if y[0]==ans:
    #             print('Выберите один из наиболее подходящих Вам случаев:')
    #             print(y[1])
    #             return y[1]
    elif isinstance(dicts , list):
        for y in dicts:
            if y[0]==ans:
                print('Достигнуто максимально возможное уточнение в данном режиме!')
                print('Предлагаем Вам следующее действие:')
                print(y[1])
                return y
def answer(dicts,phrase):
    pred = lstm.predict([phrase])
    if pred< 0.5:
        print('Введен нейтральный контекст, уточните категорию запроса: ')
        for key in list(dicts.keys())[:-2]:
            print(key[0]+'\n')
        return
    else:
        #phrase ='Пока'#'изжога и дискомфорт в животе '#input()#'меня сбила машина'#input()
        phrase_emb = np.asarray([navec[word] for word in phrase.split() if word in navec and word not in stop_words]).mean(axis=0)
        if len(phrase_emb)==0: 
            print('Введен текст без смысловой нагрузки, уточните запрос')
            return
        else:
            distances,keys_of_dist = [],[]
            flag_res = 1
            cur_dict = dicts
            nodes = []
            while flag_res:
                min_val = 2
                min_key = "неизвестно"
                for key,val in cur_dict.items():
                    if key[0] is not None:
                        result = np.asarray([navec[word] for word in key[0].split() if word in navec and word not in stop_words])
                        if len(result) != 0:
                            key_emb = result.mean(axis=0)
                            if distance.cosine(key_emb, phrase_emb) < min_val:
                                min_key = key
                                min_val = distance.cosine(key_emb, phrase_emb)
                        else:
                            print('Введен текст без смысловой нагрузки, уточните запрос')
                            return
                            
                cur_dict = cur_dict[min_key]
                nodes.append((min_key,min_val))
                if not isinstance(cur_dict, dict):
                    flag_res = 0      
                    min_val = 2
                    min_key = "неизвестно"
                    for pair in cur_dict: #уже не dict, а tuple
                        if pair[0] is not None:
                            result = np.asarray([navec[word] for word in pair[0].split() if pair[0] is not None and word in navec and word not in stop_words])
                            if len(result) != 0:
                                key_emb = result.mean(axis=0)
                                if distance.cosine(key_emb, phrase_emb) < min_val:
                                    min_key = pair[0]
                                    min_val = distance.cosine(key_emb, phrase_emb)
                    nodes.append((min_key,min_val))
    if nodes[0][0][0] == 'неопознанный запрос':
        print( 'не удалось распознать запрос, выберите категорию из:\n' + '\n'.join([key[0] for key in list(dicts.keys())[:-2]]))
        ans = input()
        #for i in range(len(dicts.items())-2):
        while 1:
        #for x,y in dicts.items():
            dicts_new = go_further(ans,dicts)
            if dicts_new is not None:
                dicts = dicts_new
            else:
                print("выберите строго из списка!")
            if  isinstance(dicts , tuple): break
            ans = input()
    else:
        return nodes


phase = 'я тону спасите помогите'
#phase = 'хахаха'
phase = 'срочно эвакуировать'
print(answer(hierchy_nodes_actions_classes,phase))

#LSTM APPLICATION FOR TEXTS
#import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tensorflow import keras
print("tf version:",tf.__version__)
warnings.filterwarnings("ignore")
print("keras version:", keras.__version__)

#here should be dataset of corpus for classification
neutr = pd.read_fwf('neutral.txt', sep='\n')
negativ = pd.read_fwf('negative.txt')
#print(neutr.head())
neutral = pd.read_csv('neutral.csv', on_bad_lines='skip', sep='\n')#read_fwf
negative = pd.read_csv('negative.csv', on_bad_lines='skip')
with open('neutral.csv') as f:
    neutral = f.readlines()
with open('negative.csv') as f:
    negative = f.readlines()
neutral = pd.DataFrame({'x':np.array([x.strip() for x in neutral]),'y': np.zeros(len(neutral)).astype(int)})
#print(neutral)
negative = pd.DataFrame({'x':np.array([x.strip() for x in negative]),'y': np.ones(len(negative)).astype(int)})
res_df = pd.concat([neutral, negative],axis=0).reset_index(drop=True)
print(res_df)
'''train_dataset, test_dataset = dataset['train'], dataset['test']
#print(info)
BUFFER_SIZE = 10000
BATCH_SIZE = 512
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

VOCAB_SIZE = 1000
data, labels = next(iter(train_dataset.take(1)))
data.numpy()[:2], labels.numpy()[:2]'''
#print(negative)#, negative.head())

from sklearn.model_selection import train_test_split
BUFFER_SIZE = 10000
BATCH_SIZE = 512
train_dataset,test_dataset = train_test_split(res_df,train_size=0.8, stratify = res_df['y'])
#train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
VOCAB_SIZE = 2048
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset['x'])
print(encoder.get_vocabulary())
#tf.convert_to_tensor(train_dataset['x'])

#create lstm model
lstm = tf.keras.Sequential([
encoder,
tf.keras.layers.Embedding(
input_dim=len(encoder.get_vocabulary()),
output_dim=16,
mask_zero=False),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True, activation="relu")),
tf.keras.layers.LSTM(16, name="lstm_layer2"),
tf.keras.layers.Dense(16, activation='sigmoid'),
tf.keras.layers.Dense(1, activation='softmax')
])
lstm.summary()
lstm.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

lstm_hist = lstm.fit(train_dataset['x'],train_dataset['y'],
                    epochs=1,verbose=1)
#results = lstm.evaluate(test_dataset, verbose=2)
 
#for name, value in zip(lstm.metrics_names, results):
#    print("%s: %.3f" % (name, value))
lstm.save('lstm_for_NLP_1ep.tf',save_format='tf')


lstm.save('lstm_for_NLP_1ep.tf',save_format='tf')
#lstm.save('lstm_for_NLP_1ep.tf',save_format='tf')

lstm = tf.keras.Sequential([
encoder,
tf.keras.layers.Embedding(
input_dim=len(encoder.get_vocabulary()),
output_dim=16,
mask_zero=False),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
tf.keras.layers.Dense(16, activation='sigmoid'),
tf.keras.layers.Dense(1)
]) #'lstm_model/lstm_for_NLP_20ep.tf'
lstm= keras.models.load_model('lstm_model/lstm_for_NLP_10ep.tf', custom_objects=None, compile=True, options=None)

pred = lstm.predict(['привет'])
#print(pred)
print(answer(hierchy_nodes_actions_classes, 'срочно эвакуатор'))
#print(lstm.predict(['давай поболтаем']))









