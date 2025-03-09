# Generative AI generate new data based on training sample. Generative model can generate Image, Text, Audio, Videos e.t.c. data as output.
# Generative AI is mainly of two types :
# 1. Generative Image model
# 2. Generative Language model

# Why are generative models required ?
# 1. Understanding the complex pattern of the unstructured data.   (Unstructured data means a mixture of different types of data.)
# 2. Content Generation
# 3. Build Powerful Application

# Discriminative model :: It is a model which performs a classification task based on the data it was fed. For eg : Simple CNN dog/cat classification models.
# Generative model :: A generative model generates an image/text/audio from scratch with the help of other seen data.

# Generative models are trained on huge amounts of data. While training the generative model we don't need to provide a label data, It is not possible when we have a huge amount of data, So, it's just try to see relationship between the distribution of the data. In Generative we give unstructured data to the LLM for training purpose.
# Thus, generative models are based on unsupervised learning.

# Generative AI Pipeline ::

# (I) Data acquisition and data augmentation ::
# Some sort of data augmentations on textual data for training LLMs :
# 1. Biagram flip :: Eg :- I am Manas. --> Manas is my name.
# 2. Back translator :: In this we first change the language in which our text is then change the language using google translate and then copy that phrase and retranslate back to the text language, this will create a differences in the sentences even though the meaning may remain unchanged.
# 3. Replacing certain words with their synonyms.
# 4. Adding additional data/noise :: Eg :-  I am a data scientist --> I am a data scientist, I love my job.

# (II) Data Preprocessing ::
# 1. Stemming
# 2.Lemmatization
# 3. Stop words removal
# 4. Converting the words into numerical data.

# (III) Training the model using either paid LLM provider or LLM libraries.

# (IV) Evaluation of LLMs :
# Intrinsic Evaluations :: To be done after training the model.
# Extrinsic Evaluations :: To be done after model deployment.

# (V) Monitoring and Re-training

# Some important terms ::
# 1. Corpus : The entire text.
# 2. Vocabulary : A set of unique words found in the corpus.
# 3. Documents : Lines or all the words till the /n special character found.






