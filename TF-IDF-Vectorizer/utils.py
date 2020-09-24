import pandas as pd
import tqdm
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from tqdm import tqdm
nltk.download('punkt')                      
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')                  


class Utils:
    def __init__(self):
        return
    def Calculate_TF(self, tokenized_news_data=None):        
        return
    def Calculate_IDF(self, tokenized_news_data=None):
        return
    def Calculate_Normalized_TF_IDF(self, tf_data=None, idf_data=None):
        return
    def Load_Data(self, input_csv_file_path=None):
        news_data = pd.read_csv(input_csv_file_path, encoding="UTF-8-sig", engine='python', error_bad_lines=False)
        return_news_data = [(article_id, article_text, article_categroy) \
                            for article_id, article_text, article_categroy \
                            in zip(list(news_data['ArticleId']), list(news_data['Text']), list(news_data['Category']))]
        print("\nComplete data Load : {0}".format(len(return_news_data)))
        return return_news_data.copy()

    def Tokenize(self,input_news_data=None):
        extract_specific_tags =  ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] 
        tokenized_data = list() 
        stop_words = set(stopwords.words('english'))            
        for one_article in tqdm(input_news_data, desc="Tokenizing... : "):            
            article_id = one_article[0]
            article_text = one_article[1]
            article_categroy = one_article[2]

            article_text = article_text.rstrip("\n")
            article_text = article_text.lower()                      
            pos_tokens = nltk.pos_tag(word_tokenize(article_text)) 
            tokenized_data.append((article_id, \
                                    [token[0] for token in pos_tokens if token[1] in extract_specific_tags if not token[0] in stop_words], \
                                    article_categroy))
        return tokenized_data.copy()    

    def Save_Result(self,normalized_tf_idf=None):
        path = "./results.txt"
        with open(path, mode='w', encoding="UTF-8-sig", errors="ignore") as out:
            max_length = len(normalized_tf_idf)
            out.write("Total Length : " + str(max_length) + "\n")
            for (article_id, article_text, article_normalized_tf_idf, article_category) in normalized_tf_idf:
                article_normalized_tf_idf = [value for value in article_normalized_tf_idf]
                out.write(str(article_id)+"\t")
                for tf_idf in article_normalized_tf_idf:
                    out.write(str(tf_idf) + '\t')
                out.write(article_category +"\n")
        print("\nSave the result file : {}".format(path))

