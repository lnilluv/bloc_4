
import io
import os
import pathlib
import warnings

import pandas as pd
import spacy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings('ignore')

df = pd.read_csv("https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv", error_bad_lines=False, encoding="ISO-8859-1")
df.head()

df.info()

def missing_values(df, norows):   # input by the df and the number of rows that you want to show  
    total = df.isnull().sum().sort_values(ascending=False)  
    percent = ((df.isnull().sum().sort_values(ascending=False)/df.shape[0])*100).sort_values(ascending=False)  
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  
    return(missing_data.head(norows))  
  
missing_values(df,20) # we use the df and the number of rows to show is 20

# Remove useless columns and rename v1 and v2 columns
df = df.rename(columns={"v1": "type", "v2": "message"})
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.head()


# change dtype of type column to category
df = df.astype({"type": "category"})


df.describe(include='all')


df.drop_duplicates(inplace=True) # drop duplicates


# data distribution
df.type.value_counts()


df.nunique()


# replace ham and spam by 0 and 1
df['type'] = df['type'].map({'ham': 0, 'spam': 1})
df.head()


# ### Preprocessing


nlp = spacy.load('en_core_web_md')


df['mail_cleaned'] = (
    df['message']
    .apply(lambda x:''.join(ch for ch in x if ch.isalnum() or ch==" " or ch=="'"))
    .str.lower()
    .str.strip()
    .str.replace('\s\s+', ' ')
    .apply(lambda x: " ".join([token.lemma_ for token in nlp(x) if (token.lemma_ not in STOP_WORDS) and (token.text not in STOP_WORDS)]))
)


df.head()


mask = df.mail_cleaned.apply(lambda x: type(x)==str)
mask.value_counts()


df = df.loc[mask,:]

import numpy as np

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="out_of_vocab") # instanciate the tokenizer
tokenizer.fit_on_texts(df.mail_cleaned) # fit the tokenizer on the text
df["mail_encoded"] = tokenizer.texts_to_sequences(df.mail_cleaned)


df.head()

messages_pad = tf.keras.preprocessing.sequence.pad_sequences(df.mail_encoded, padding="post")


# Train Test Split
xtrain, xval, ytrain, yval = train_test_split(messages_pad, df.type, test_size=0.3)


train = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
val = tf.data.Dataset.from_tensor_slices((xval, yval))


train_batch = train.shuffle(len(train)).batch(64)
val_batch = val.shuffle(len(val)).batch(64)



 # Regardons un batch 
for review, star in train_batch.take(1):
  print(review, star)

vocab_size = tokenizer.num_words
model = tf.keras.Sequential([
                  # Couche d'Input Word Embedding           
                  tf.keras.layers.Embedding(vocab_size+1, 8, input_shape=[review.shape[1],],name="embedding"),
                  # Gobal average pooling
                  tf.keras.layers.GlobalAveragePooling1D(),

                  # Couche Dense classique
                  tf.keras.layers.Dense(16, activation='relu'),

                  # Couche de sortie avec le nombre de neurones en sortie égale au nombre de classe avec fonction softmax
                  tf.keras.layers.Dense(1, activation="linear")
])

model.summary()


# define the tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")


# define the early stopping callback
# patience parameter is the number of epochs to wait before stopping training if the monitored metric does not improve.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)


history = model.fit(train_batch, 
                    epochs=100, 
                    validation_data=val_batch, callbacks=[tensorboard_callback, early_stop])




