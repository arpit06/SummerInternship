#all imports
import re
import pandas
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams
from fuzzywuzzy.process import dedupe


def skill_extraction():
#just skills


def university_extraction():
#University name, major, gpa




def company_extraction():
#Companies, titles, dates, work experience



def personel_information():
#name,phone number, address, linkedin, github


#Eric’s section (name, email, phone, university, major, gpa) Done



#import path & variables
resume_path = r"/Users/yiyangzhou/Desktop/Yiyang (Eric) Zhou Resume 2017 Fall.txt"

resume_file = open(resume_path).read()
resume_file2 = open(resume_path).read()
resume_file2 = resume_file2.lower()
#change path
major_df = pandas.read_excel('majors.xlsx')
major_df.columns
major_file = major_df['Majors'].values
major_lower = [item.lower() for item in major_file]
tokenizer = RegexpTokenizer(r'\w+')
resume_token = tokenizer.tokenize(resume_file)
resume_token2 = tokenizer.tokenize(resume_file2)
major_distinct = []
dictionary = {'Name': 5}
regular_expression = re.compile(r"/BA|BS|Bachelor of Science|Bachelor of Arts|BBA |B/A|Bachelor of Business Administration/", re.IGNORECASE)
bach_major_result = re.search(regular_expression, resume_file)
regular_expression_two = re.compile(r"minor|Minor", re.IGNORECASE)
minor_result = re.search(regular_expression_two, resume_file)
regular_expression_three = re.compile(r"Master|master", re.IGNORECASE)
master_major_result = re.search(regular_expression_three, resume_file)
regular_expression_four = re.compile(r"university", re.IGNORECASE)
university_major_result = re.search(regular_expression_four, resume_file)
updated_majors1 = []
indexes_majors1 = []
updated_majors2 = []
indexes_majors2 = []
updated_majors3 = []
indexes_majors3 = []
updated_majors4 = []
indexes_majors4 = []
majors_minors_all = updated_majors1 + updated_majors2 + updated_majors3 + updated_majors4

university_df1 = pandas.read_excel('China_University.xlsx')
university_df2 = pandas.read_excel('India_University.xlsx')
university_df3 = pandas.read_excel('US_University.xlsx')
university_file1 = university_df1['Universities'].values
university_file2 = university_df2['Universities'].values
university_file3 = university_df3['Universities'].values
university_lower1 = [item.lower() for item in university_file1]
university_lower2 = [item.lower() for item in university_file2]
university_lower3 = [item.lower() for item in university_file3]
university_combined = university_lower1 + university_lower2 + university_lower3


#extract name finished
def extract_first_name(resume):
    name = resume.split('\n', 1)[0]
    first_name = name.split(' ', 1)[0]
    return (first_name)
    print (first_name)

def extract_last_name(resume):
    name = resume.split('\n', 1)[0]
    last_name = name.split(' ', 1)[-1]
    return (last_name)
    print (last_name)

def extract_name(resume):
    name = extract_first_name(resume_file) + extract_last_name(resume_file)
    print (name)

#extract email finished
def extract_email(resume):
    regular_expression = re.compile(r"(\w+[.|\w])*@(\w+[.])*\w+", re.IGNORECASE)
    result = re.search(regular_expression, resume)
    if result:
        result = result.group()
    print (result)

#extract phone number finished

def check_phone_number1(resume):
    resume2 = "".join(c for c in resume if c not in ('!','.','-','(',')',' ','+',))
    result = re.findall(r"\d{10}", resume2)
    result = ''.join(result)
    return (result)

def check_phone_number2(resume):
    resume2 = "".join(c for c in resume if c not in ('!','.','-','(',')',' ','+',))
    result = re.findall(r"\d{11}", resume2)
    result = ''.join(result)
    result = result[1:11]
    return (result)

def extract_phone_number(resume):
    try:
        return check_phone_number1(resume)
        print (check_phone_number1(resume))
    except:
        return check_phone_number2(resume)
        print (check_phone_number2(resume))


def personel_information(resume):
    print(extract_name(resume))
    print(extract_email(resume))
    print(extract_phone_number(resume))

#execution of extracting name, email, phone number
personel_information(resume_file)



#major, University, gpa
def get_bigrams(input):
    n = 2
    result = []
    bigrams = ngrams(input, n)
    for grams in bigrams:
        x = "%s %s" % grams
        result.append(x)
    return (result)
    print (result)


def get_threegrams(input):
    n = 3
    result = []
    threegrams = ngrams(input, n)
    for grams in threegrams:
        x = "%s %s %s" % grams
        result.append(x)
    return (result)
    print (result)

def get_fourgrams(input):
    n = 4
    result = []
    fourgrams = ngrams(input, n)
    for grams in fourgrams:
        x = "%s %s %s %s" % grams
        result.append(x)
    return (result)
    print (result)

def get_fivegrams(input):
    n = 5
    result = []
    fivegrams = ngrams(input, n)
    for grams in fivegrams:
        x = "%s %s %s %s %s" % grams
        result.append(x)
    return (result)
    print (result)

def get_sixgrams(input):
    n = 6
    result = []
    sixgrams = ngrams(input, n)
    for grams in sixgrams:
        x = "%s %s %s %s %s %s" % grams
        result.append(x)
    return (result)
    print (result)

def get_majors(a,b):
    majors=[]
    for x in a:
        if x in b:
            majors.append(x)
    return (majors)
    print (majors)

def get_majors2(a,b):
    unigram_major = get_majors(a, b)
    bigram_major = get_majors(get_bigrams(a), b)
    threegram_major = get_majors(get_threegrams(a), b)
    combined_majors_list = unigram_major + bigram_major + threegram_major
    for i in combined_majors_list:
        if i not in major_distinct:
            major_distinct.append(i)
    print (major_distinct)

def get_majors_index(major_distinct):
    for i, element in enumerate(major_distinct):
        x = resume_file2.find(element)
        dictionary[element] = x
    del dictionary['Name']
    print(dictionary)

def get_bach_index(bach_major_result):
    if bach_major_result:
        bach_major_result = bach_major_result.group()
    print (bach_major_result)
    if bach_major_result is not None:
        bach_major_index = resume_file.find(bach_major_result)
    return(bach_major_index)
    print(bach_major_index)

def get_minor_index(minor_result):
   if minor_result:
       minor_result = minor_result.group()
   print (minor_result)
   if minor_result is not None:
       minor_index = resume_file.find(minor_result)
   return(minor_index)
   print(minor_index)

def get_master_index(master_major_result):
    if master_major_result:
        master_major_result = master_major_result.group()
    print (master_major_result)
    if master_major_result is not None:
        master_major_index = resume_file.find(master_major_result)
    return(master_major_index)
    print(master_major_index)

def get_university_index(university_major_result):
    if university_major_result:
        university_major_result = university_major_result.group()
    print (university_major_result)
    if university_major_result is not None:
        university_major_index = resume_file.find(university_major_result)
    return(university_major_index)
    print(university_major_index)

def get_bach_major(dictionary):
    bach_major_index = get_bach_index(bach_major_result)
    upper_bound = bach_major_index +100
    for k, v in dictionary.items():
        if (bach_major_index < v < upper_bound):
            updated_majors1.append(k)
            indexes_majors1.append(v)
    print(updated_majors1)
    print(indexes_majors1)


def get_master_major(dictionary):
    master_major_index = get_master_index(master_major_result)
    upper_bound = master_major_index +100
    for k, v in dictionary.items():
        if (master_major_index < v < upper_bound):
            updated_majors2.append(k)
            indexes_majors2.append(v)
    print(updated_majors2)
    print(indexes_majors2)

def get_minor(dictionary):
    minor_index = get_minor_index(minor_result)
    upper_bound = minor_index +100
    for k, v in dictionary.items():
        if (minor_index < v < upper_bound):
            updated_majors3.append(k)
            indexes_majors3.append(v)
    print(updated_majors3)
    print(indexes_majors3)

def get_university_major(dictionary):
    university_major_index = get_university_index(university_major_result)
    upper_bound = university_major_index +100
    for k, v in dictionary.items():
        if (university_major_index < v < upper_bound):
            updated_majors4.append(k)
            indexes_majors4.append(v)
    print(updated_majors4)
    print(indexes_majors4)



def extract_major(majors_minors_all):
    majors_minors_all = updated_majors1 + updated_majors2 + updated_majors3 + updated_majors4
    majors_minors_final_list = list(dedupe(majors_minors_all))
    return (majors_minors_final_list)
    print (majors_minors_final_list)


#execution of extracting majors:

get_majors(resume_token2, major_lower)
get_majors2(resume_token2, major_lower)
get_majors_index(major_distinct)
get_bach_index(bach_major_result)
get_minor_index(minor_result)
get_master_index(master_major_result)
get_university_index(university_major_result)
get_bach_major(dictionary)
get_master_major(dictionary)
get_minor(dictionary)
get_university_major(dictionary)
print (extract_major(majors_minors_all))

#extract University:
def get_university(a,b):
    resume_university=[]
    for x in a:
        if x in b:
            resume_university.append(x)
    return (resume_university)
    print (resume_university)

def extract_university(resume_token_lower,university_combined):
    unigram_university = get_university(resume_token_lower, university_combined)
    bigram_university = get_university(get_bigrams(resume_token_lower), university_combined)
    threegram_university = get_university(get_threegrams(resume_token_lower), university_combined)
    fourgram_university = get_university(get_fourgrams(resume_token_lower), university_combined)
    fivegram_university = get_university(get_fivegrams(resume_token_lower), university_combined)
    sixgram_university = get_university(get_sixgrams(resume_token_lower), university_combined)
    combined_university_extraction = set(bigram_university + threegram_university + fourgram_university + fivegram_university + sixgram_university)
    print (combined_university_extraction)

#execution of extracting university:
extract_university(resume_token2,university_combined)

#extract GPA:
def extract_GPA(resume):
    result = re.search(r'(GPA|gpa): ?\d.\d{1,}',resume)
    if result:
        result = result.group(0)
    return (result)
    print (result)

#execution of extracting GPA:
extract_GPA(resume_file)


#HENRY

#Extracting Address

import re
import usaddress

#extract the address
def extract_address (text):
    text = text.replace('\n', ' ')
    regex = re.compile(r"[0-9]+ .*[.,-]? .*[.,-]? ([A-Z]{2}|\w+)[.,-]? [0-9]{5}(-[0-9]{4})?")
    result = re.search(regex, text)
    if result:
        result = result.group()
    return result

#Parse the address components
def parse_address(result):
    address = usaddress.tag(result)
    return address

#2. Extracting Company

import codecs
import os
import pandas as pd
from fuzzywuzzy.process import dedupe
import spacy
from nltk.corpus import stopwords


filename = 'BrandonThomasResume.txt'
#Open file
def open_file(filename):
    resume = open(filename, 'r', errors='ignore').read()
    return resume
resume = open_file(filename)

#Read the Work_Experience_List
data = pd.read_excel("Work Experience.xlsx", header=0)
experience_list = list(data['Example'])

#Find the experience header
def find_exp_header (resume):
    exp_header_list=[]
    for word in experience_list:
        if resume.find(word) != -1:
            exp_header_list.append(word)

    #remove duplicates of experience header
    exp_header = list(dedupe(exp_header_list))
    return exp_header

exp_header = find_exp_header(resume)
exp_header = (exp_header[0], resume.find(exp_header[0]))

#Find next section header
def find_next_section (resume):
    #Find all capitalized words
    next_section_upper = re.findall(r'([A-Z]{3,}( [A-Z]+)?( [A-Z]+)?( [A-Z]+)?)',
                                   resume[(exp_header[1] + len(exp_header[0])+ 1):])
    next_section_upper = list((itertools.chain.from_iterable(next_section_upper)))

    #Find all words with the first letter capitalized
    next_section_lower = re.findall(r'([A-Z]{1}\w+( [A-Z]{1}\w+)?( [A-Z]{1}\w+)?( [A-Z]{1}\w+)?)',
                                    resume[(exp_header[1] + len(exp_header[0])+ 1):])
    next_section_lower = list((itertools.chain.from_iterable(next_section_lower)))

    #Combine into a list
    next_section_list = next_section_upper + next_section_lower

    #if one of the items matches items in section list, that item is the next section header
    next_section=()
    for item in next_section_list:
        if item in section_list and (resume[resume.find(item)+len(item)]=='\n' or resume[resume.find(item)-1]=='\n'):
            next_section = (item, resume.find(item))
            break
    return next_section

next_section = find_next_section(resume)

# Get the section of Work_Experience
def get_workexp_section(resume):
    if next_section:
        workexp_section = str(resume[(exp_header[1]+ len(exp_header[0])+ 1):next_section[1]])
    else:
        workexp_section = str(resume[(exp_header[1]+ len(exp_header[0])+ 1):])
    return workexp_section

workexp_section = get_workexp_section(resume)
workexp_section = workexp_section.split('\n')

#Remove the detail and get the experience information
def get_exp_info(work_exp):
    company_info=[]
    temp_str=''
    for i, sent in enumerate(work_exp):
        if sent != '':
            #Everything before the bullet will be put into one sentence, for one company
            if not sent.startswith(('•','', u'\uf095', '§', '§')):
                temp_str += sent + ' '
            else:
                if not work_exp[i-1].startswith(('•','', u'\uf095', '§', '§')):
                    company_info.append(temp_str)
                    temp_str=''
    return company_info

company_info = get_exp_info(workexp_section)

#Print the company info
for i, company in enumerate(company_info):
    company = company.replace('\t', '')
    print('\nCompany {}:'.format(i+1), company)

nlp = spacy.load('en')

#Parse company info components
def extract_exp_info(company_info, filename):
    count = 0
    print(filename)
    for i, sent in enumerate(company_info):
        sent = sent.replace('\t', '')
        parsed_sent = nlp(sent)
        print('\nCompany {}'.format(i+1))

        company=''
        location=''
        time=''
        role=''
        for i ,token in enumerate(parsed_sent):
            if token.ent_type_ =='ORG':
                company += ' ' + str(token)
            elif token.ent_type_ =='GPE':
                location += ' ' + str(token)
            elif token.ent_type_ =='DATE' or token.ent_type_ =='TIME':
                time += ' ' + str(token)
            elif token.ent_type_ =='':
                if str(token).isalpha() and str(token) not in stopwords.words('english'):
                    role += ' ' + str(token)

        print('Company: {}'.format(company))
        print('Location: {}'.format(location))
        print('Time: {}'.format(time))
        print('Role: {}'.format(role))

extract_exp_info(company_info, filename)


#3. Extract Skills (Just Skills)

import nltk
import pandas as pd
import os
import codecs
from gensim.models import Phrases
import re

#Read the Skill_List.xlsx
data = pd.read_excel("Skills.xlsx", header=0)
skill_list = list(data['Skill Names'])
skill_list = set(skill_list)
skill_list= [skill.lower() for skill in skill_list]

filename ='all_text1.txt'
trained_resume_path = os.path.join('Trained Resumes', filename)

resume_text = open(trained_resume_path, 'r', encoding='utf_8').read()
special_characters = ['!','#', '$', '%','&','*','-', '/', '=','?',
                      '^','.','_','`', '{', '|', '}','~', "'", ',', '(',')', ':', '•', '§' ]

# Processing text
def resume_processing (resume_text):
    #tokenize sentences
    resume_sents = nltk.sent_tokenize(resume_text)

    #tokenize words
    resume_words = [nltk.word_tokenize(sent) for sent in resume_sents]

    #remove stopwords and special characters
    processed_resume=[]
    for sentence in resume_words:
        sent = [w.lower() for w in sentence
                          if w.lower() not in stopwords.words('english') and w.lower() not in special_characters]
        processed_resume.append(sent)

    return processed_resume

unigram_resume = resume_processing(resume_text)

#Create bigram model
bigram_model_path = 'bigram_model'

bigram_model = Phrases(unigram_resume)
bigram_model.save(bigram_model_path)

# Create bigram words
def create_bigram (unigram_resume):
    bigram_model = Phrases.load(bigram_model_path)
    bigram_resume = [bigram_model[sentence] for sentence in unigram_resume]
    return bigram_resume

bigram_resume = create_bigram(unigram_resume)

#Create trigram model
trigram_model_path = 'trigram_model'

trigram_model = Phrases(bigram_resume)
trigram_model.save(trigram_model_path)

# Create trigram words
def create_trigram (bigram_resume):
    trigram_model = Phrases.load(trigram_model_path)
    trigram_resume = [trigram_model[sentence] for sentence in bigram_resume]
    return trigram_resume

trigram_resume = create_trigram(bigram_resume)

#Normalize bigram/trigram words
def normalize_words (trigram_resume):
    for sentence in trigram_resume:
        for i, word in enumerate(sentence):
            if len(re.findall(r'\w+\_\w+', word))!= 0:
                sentence[i] = re.sub('_', ' ', word)
    return trigram_resume

normalized_resume = normalize_words(trigram_resume)

#label skills in the resume
def labeled_word (sentence):
    labels=[]
    for word in sentence:
        if word in skill_list:
            labels.append((word, 'skill'))
        else:
            labels.append((word, 'not skill'))
    return labels

labeled_words=[labeled_word(sentence) for sentence in normalized_resume]

#Get 25 similar words based on word2vec model
def similar_prob(word):
    count = 0
    terms = get_related_terms(word,25)
    for w in terms:
        if skill_series.isin([w]).any():
            count+=1
    return count/25

#Check if the word is in skill clusters, based on KMeans algorithm
def in_skill_cluster(word):
    if word in skills:
        return True
    return False

#extract featurres of skills
def extract_features (sentence, i):
    features={}
    #first feature: evaluate if that word is in skill list
    features["({})in_skill_list".format(sentence[i])]= (sentence[i] in skill_list)

    if sentence[i] in res2vec.wv.vocab:
        features["probality_of_similar_words_skills"] = similar_prob(sentence[i])
        features["in_skill_cluster"] = in_skill_cluster(sentence[i])

    #if the word is in begining of the sentence, return <Start> for prev_word
    if i==0 and len(sentence)-1 != 0:
        features["prev_word_in_skill_list"]= '<Start>'
        features["next_word_in_skill_list"]= (sentence[i+1] in skill_list)

    #if the word is in begining of the sentence, return <End> for next_word
    elif i == len(sentence)-1 and  i != 0:
        features["prev_word_in_skill_list"]= (sentence[i-1] in skill_list)
        features["next_word_in_skill_list"]= '<End>'

    #if the sentence has only 1 word, return False for both prev_word and next_word
    elif i==0 and len(sentence)-1 == 0:
        features["prev_word_in_skill_list"]= False
        features["next_word_in_skill_list"]= False
    else:
        features["prev_word_in_skill_list"]= (sentence[i-1] in skill_list)
        features["next_word_in_skill_list"]= (sentence[i+1] in skill_list)
    return features

featuresets=[]
for labeled_sent in labeled_words:
    unlabeled_sent = [word[0] for word in labeled_sent]
    for i, (w, label) in enumerate(labeled_sent):
        featuresets.append((extract_features(unlabeled_sent, i), label))

#Save the features in a file
featuresets_file = 'features_file.txt'
file = open(featuresets_file, 'w', encoding='utf_8')
file.write('\n'.join('%s %s' % item for item in featuresets ))

size = int(len(featuresets)*0.1)
train_set = featuresets[size:]
test_set = featuresets[:size]

#Train the data with NaiveBayes model
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Evaluate the accuracy
nltk.classify.accuracy(classifier, test_set)

#Extract the skills
def extract_skills(normalized_test_res, resume_number, filename):
    skills =[]
    for sent in normalized_test_res:
        for (i,_) in enumerate(sent):
            if classifier.classify(extract_features(sent, i))=='skill':
                skills.append(sent[i])
                extracted_skills = set(skills)
    print('\nResume {}:{} ({} skills)\n'.format(resume_number+1,filename, len(extracted_skills)), extracted_skills)


#VAIBHAV

#Import Statements
import csv
import re

#Email Address (Finished)
def check_email(string_to_search):
    regular_expression = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,3}", re.IGNORECASE)
    result = re.search(regular_expression, string_to_search)
    if result:
        result = result.group()
    return result
    #except:
     #   result=0
      #  return result

#LinkedIn Address(Finished)
def check_linkedin(string_to_search):
    regular_expression1 = re.compile(r"https://"
                                    r"[A-Z]{2,3}"
                                    r".linkedin.com/in/"
                                    r"[-_a-z 0-9]{5,30}", re.IGNORECASE)
    result = re.search(regular_expression1, string_to_search)
    try:
        result = result.group()
        return result
    except:
        regular_expression1 = re.compile(r"[A-Z]{2,3}"
                                        r".linkedin.com/in/"
                                        r"[-_a-z 0-9]{5,30}", re.IGNORECASE)
        result = re.search(regular_expression1, string_to_search)
        try:
            result=result.group()
            return result
        except:
            regular_expression1 = re.compile(r"[A-Z]{2,3}"
                                        r".linkedin.com/"
                                        r"[-_a-z 0-9]{5,30}", re.IGNORECASE)
            result = re.search(regular_expression1, string_to_search)
            try:
                result=result.group()
                return result
            except:
                return None

#GitHub Address (Finished)
def check_GitHub(string_to_search):
    regular_expression = re.compile(r"https://github.com/"
                                    r"[-_A-Z0-9]{5,30}", re.IGNORECASE)
    result = re.search(regular_expression, string_to_search)
    try:
        result = result.group()
        return result
    except:
        return None

#Contact Number (Finished)
def check_phone_number(string_to_search):
    try:
        regular_expression = re.compile(r"\(?"  # open parenthesis
                                        r"(\d{3})?"  # area code
                                        r"\)?"  # close parenthesis
                                        r"[\s\.-]{0,2}?"  # area code, phone separator
                                        r"(\d{3})"  # 3 digit exchange
                                        r"[\s\.-]{0,2}"  # separator bbetween 3 digit exchange, 4 digit local
                                        r"(\d{4})",  # 4 digit local
                                        re.IGNORECASE)
        result = re.search(regular_expression, string_to_search)
        if result:
            result = result.groups()
            result = "-".join(result)
        return result
    except:
        return None

def main():
#    with open('Resume_Test.txt', 'r',encoding="utf8") as myfile:
    with open('Resume_Test.txt', 'r') as myfile:
        data=myfile.read().replace('\n',' **** ')
    result=check_email(data)
    result_L=check_linkedin(data)
    result_P=check_phone_number(data)
    result_G=check_GitHub(data)
    print("Email Address:",result)
    print("Contact Number:",result_P)
    print("Linkedin Profile:",result_L)
    print("GitHub Profile:",result_G)
    #print(data)
main()


#Ashish (LinkedIn Profiles and Every other URL in the file)

# import all headers
import re
import os

# function to extract all URLs
# implemented using regex
def extract_URLs(parsedResume):
    parsedResume = parsedResume.replace('\n', ' ')
    regex = regex = re.compile('(?:(?:https?|ftp|file)://|www\.|ftp\.)[-A-Z0-9+&@#/%=~_|$?!:,.]*[A-Z0-9+&@#/%=~_|$]', re.IGNORECASE)
    result = re.findall(regex, parsedResume)
    #if result:
        #result = result.group()
    return result

# function to extract LinkedIN Profile
# implemented using regex
def extract_linkedin(parsedResume):
    parsedResume = parsedResume.replace('\n', ' ')
    regex = re.compile(r"https://www.linkedin.com/in/([a-zA-Z]|[0-9]|[-])+/?")
    result = re.search(regex, parsedResume)
    if result:
        result = result.group()
    return result

# TESTING
# path where all resumes are located
test_resume_path = '/Users/Ashish/Desktop/Internship/Personal/Test Resumes'
counter = 0

print("URLs in Test Resumes")
for filename in os.listdir(test_resume_path):
    # print(filename)
    if '.txt' in filename:
        counter = counter + 1
        resume_path= os.path.join('Test Resumes', filename)
        test_resume = open(resume_path, 'r').read()

        print("Resume ", (counter), ":")
        print("All URLs => ", extract_URLs(test_resume))
        print("LinkedIn Profiles => ", extract_linkedin(test_resume))
