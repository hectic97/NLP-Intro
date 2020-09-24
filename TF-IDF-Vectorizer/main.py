import tqdm
import math
from tqdm import tqdm,trange
from utils import Utils

class Preprocess(Utils):
    def Calculate_TF(self, tokenized_news_data=None):
        tf=[]
        vocab=set()
        for article in tqdm(tokenized_news_data,desc='Create vocab... : '):
            article_text=article[1]
            for word in article_text:
                vocab.add(word)
        vocab_list=sorted(list(vocab))
        for article in tqdm(tokenized_news_data,desc='Calculating TF... : '):
            article_text=article[1]
            article_tf = [0]*len(vocab_list)
            for word in article_text:
                article_tf[vocab_list.index(word)] += 1
            tf.append((article[0],article[1],article_tf,article[2]))
        return tf

    def Calculate_IDF(self, tokenized_news_data=None):
        N = len(tokenized_news_data)
        idf = {}
        vocab=set()
        for article in tqdm(tokenized_news_data,desc='Create vocab... : '):
            article_text=article[1]
            for word in article_text:
                vocab.add(word)
        vocab_list=sorted(list(vocab))
        idf = {voca:0 for voca in vocab_list}
        for voca in tqdm(idf.keys(),desc='Calculating IDF... : '):
            for article in tokenized_news_data:
                article_text=article[1]
                if voca in article_text:
                    idf[voca] += 1
                    
        for token in idf:
            idf[token] = math.log(N/idf[token],2)
        
        return idf             


    def Calculate_Normalized_TF_IDF(self, tf_data=None, idf_data=None):
        vocab=set()
        for article in tqdm(tokenized_news_data,desc='Create vocab... : '):
            article_text=article[1]
            for word in article_text:
                vocab.add(word)
        vocab_list=sorted(list(vocab))
        tf_idf=[]
        for article in tqdm(tf_data,desc='Calculating normalized TF-IDF... : '):
            article_tf = article[2]
            for article_word in set(article[1]):
                wd_idf = idf_data[article_word]
                article_tf[vocab_list.index(article_word)] *= wd_idf
            tss = 0
            for val in article_tf:
                tss += val**2
            tf_idf_article= [round(word_idf*(1/math.sqrt(tss)),1) for word_idf in article_tf] 
            tf_idf.append((article[0],article[1],tf_idf_article,article[3]))
        return tf_idf        

if __name__ == "__main__":
    preprocess = Preprocess()
    news_data = preprocess.Load_Data(input_csv_file_path="BBC_News_Data.csv")
    tokenized_news_data = preprocess.Tokenize(input_news_data=news_data)
    
    tf_data = preprocess.Calculate_TF(tokenized_news_data=tokenized_news_data)
    idf_data = preprocess.Calculate_IDF(tokenized_news_data=tokenized_news_data)
    normalized_tf_idf_train_data = preprocess.Calculate_Normalized_TF_IDF(tf_data=tf_data, idf_data=idf_data)

    preprocess.Save_Result(normalized_tf_idf=normalized_tf_idf_train_data)
