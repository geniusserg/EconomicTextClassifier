import pickle
import os
import string 
import nltk 
from nltk.corpus import wordnet
import re 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TweetClassifier:
    def __text_process(self, input: str) -> str:
        stops = stopwords.words('english')
        punct = string.punctuation + "`'‘’"
        output = re.sub(r"\S*@\S*\s?", "", input) #emails
        output = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', output) # any URL
        output = re.sub(r'[.,!()?:<>#]{2,}', '', output) # repeated punctuations
        output = re.sub(r'\s{2,}', ' ', output) # repeated spaces
        output = [t for t in output if t not in punct]
        output = "".join(output)
        pos_tagged_text = nltk.pos_tag(output.split(" "))
        wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
        lems =  [self.__lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]
        output = [token for token in output if token not in stops]
        return lems

    def __init__(self, path_to_models):
        self.model = pickle.load(os.path.join(path_to_models, "svm_classifier_tfidf5000"))
        self.vectorizer = pickle.load(os.path.join(path_to_models, "tfidf5000"))
        self.__lemmatizer = WordNetLemmatizer()
        nltk.download("wordnet")
        nltk.download("stopwords")
    
    def predict(self, input:str):
        input = " ".join(self.__text_process(input))
        input = self.vectorizer.transform(input)
        return self.model.predict(input)



