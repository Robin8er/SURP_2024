from openai import OpenAI
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.data import find
from nltk.test.gensim_fixt import setup_module
import gensim
import gensim.downloader
import matplotlib.pyplot as plt
import json
import time
import csv
from transformers import BertTokenizer, BertModel, BertConfig, LongformerModel, LongformerTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
lemmatizer = WordNetLemmatizer()
porter = nltk.PorterStemmer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
long_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
long_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')


def parse_pubmed(json_file):
    """
    Input - JSON file. 
    Output - The data as a dictionary.
    Purpose - Convert the raw PubMedQA JSON files to a Python data structure.
    """
    pubmed_raw = open(json_file)
    data = json.load(pubmed_raw)
    pubmed_raw.close()

    return data


def convert_to_sentences(parsed_data):
    """
    Input - the parsed data outputted from the parse_pubmed function.
    Output - A list of lists comprised of all the sentences from the QUESTION, LONG_ANSWER, and CONTEXTS parts of the PubMedQA datasets.
    Purpose - Further converting into more readable data structures for later functions.
    """
    sentence_list = []
    for key in parsed_data.keys():
        for type in ["QUESTION", "LONG_ANSWER"]:
            for sentence in sent_tokenize(parsed_data[key][type]):
                sentence_list.append(word_tokenize(sentence))
        for context in parsed_data[key]["CONTEXTS"]:
            for sentence in sent_tokenize(context):
                sentence_list.append(word_tokenize(sentence))
    
    return sentence_list



def find_sentences(num_sentence, parsed_data):
    """
    Input - the number of sentences (int) to take from the data, and the data itself.
    Output - A list of sentences.
    """
    count = 0
    sentence_list = []

    for key in parsed_data.keys():
        answer_sentences = sent_tokenize(parsed_data[key]["LONG_ANSWER"])
        sentence_list += answer_sentences
        count += len(answer_sentences)
        if count >= num_sentence:
            break

    return sentence_list


def find_paragraphs(num_paragraphs, parsed_data):
    paragraph_list = []

    for count, key in enumerate(parsed_data.keys()):
        paragraph_list.append(parsed_data[key]["LONG_ANSWER"])
        if count >= num_paragraphs - 1:
            break

    return paragraph_list


def sentence_simplifier(sentences):
    """
    Simplified a list of sentences one at a time using ChatGPT4o
    """
    client = OpenAI()

    simple_sentences = []

    for count, sentence in enumerate(sentences):
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Please simplify the following sentence: " + sentence}])
        simple_sentences.append(completion.choices[0].message.content)
        print(count + 1)

    return simple_sentences


def paragraph_simplifier(paragraphs):
    """
    Simplifies a list of paragraphs one at a time using ChatGPT4o
    """
    client = OpenAI()

    simple_paragraphs = []

    for count, paragraph in enumerate(paragraphs):
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Please simplify the following paragraph:\n" + paragraph}])
        simple_paragraphs.append(completion.choices[0].message.content)
        print(count + 1)

    return simple_paragraphs


def create_sentence_file(sentences, simple_sentences):
    """
    Zips together original sentences with their corresponding simplified sentences in a .txt file.
    """
    f = open("1000_sentences_simplified.txt", "a")
    for i in range(len(sentences)):
        f.write(f"Sentence #{i + 1}:\n")
        f.write(sentences[i] + "\n")
        f.write(simple_sentences[i] + "\n")
    f.close()


def create_paragraph_file(paragraphs, simple_paragraphs):
    """
    Zips together original paragraphs with their corresponding simplified sentences in a .txt file.
    """
    f = open("1000_paragraphs_simplified.txt", "a")
    for i in range(len(paragraphs)):
        f.write(f"Paragraph #{i + 1}:\n")
        f.write(paragraphs[i] + "\n")
        f.write(simple_paragraphs[i] + "\n")
    f.close()


def tag_to_wordnet(tag):
    """
    Helper function to match tags to wordnet variables for use in lemmatization.
    """
    match tag[0]:
        case "J":
            return "a"
        case "V":
            return "v"
        case "R":
            return "r"
        case _:
            return "n"


def process_paragraph(paragraph):
    """
    Helper function to process paragraphs. Depending on what is commented out, this will remove stop words, then tokenize
    all words, tag them, lemmatize them, stem them (if wanted), and then return.
    """
    stop_words = set(stopwords.words("english") + ["(", ")", ";", ":"])
    tokenized = word_tokenize(paragraph)
    tagged = nltk.pos_tag(tokenized) # swap order
    stop_processed = [
        (word, tag) for word, tag in tagged if word.casefold() not in stop_words
        ]
    # lemmatized = [(lemmatizer.lemmatize(old_word, pos=tag_to_wordnet(tag)), tag) for old_word, tag in stop_processed]
    lemmatized = [lemmatizer.lemmatize(old_word, pos=tag_to_wordnet(tag)) for old_word, tag in stop_processed]
    # lemmatized = [lemmatizer.lemmatize(old_word, pos=tag_to_wordnet(tag)) for old_word, tag in tagged]
    stemmed = [porter.stem(word) for word in lemmatized]
    return stemmed


def process_file(in_file, out_file):
    """
    This function processes all of the paragraphs in a file using the process_paragraph function. The file input must have 
    a multiple of 3 for the number of lines. In each triplet, the first line is a marker to show which paragraph you're up to.
    The second line is the original paragraph, and the third line is the simplified paragraph.
    """
    in_open = open(in_file, "r")
    out_open = open(out_file, "a")
    for i, line in enumerate(in_open):
        if i % 3 == 0:
            out_open.write(line)
        else:
            out_open.write(" ".join(process_paragraph(line)) + "\n")
    in_open.close()
    out_open.close()


def subtract_intersections(in_file, out_file):
    """
    This function finds the number of words in each paragraph pair that were used more often (and by how much) than in the other.
    """
    in_open = open(in_file, "r")
    out_open = open(out_file, "a")
    for i, line in enumerate(in_open):
        if i % 3 == 0:
            out_open.write(line)
        elif i % 3 == 1:
            normal_set = Counter(line.strip().split(" "))
        else:
            simplified_set = Counter(line.strip().split(" "))
            subtracted_normal = normal_set - simplified_set
            subtracted_simplified = simplified_set - normal_set 
            if subtracted_normal:
                out_open.write(str(subtracted_normal.most_common()) + "\n")
            else:
                out_open.write("\n")
            if subtracted_simplified:
                out_open.write(str(subtracted_simplified.most_common()) + "\n")
            else:
                out_open.write("\n")
    in_open.close()
    out_open.close()


def find_jaccard_gen(in_file, out_file):
    """
    This function finds the generalized Jaccard value for each paragraph pair.
    """
    in_open = open(in_file, "r")
    out_open = open(out_file, "a")
    jaccard_list = [] # tuples: jaccard index, paragraph number
    for i, line in enumerate(in_open):
        if i % 3 == 1:
            normal_set = Counter(line.strip().split(" "))
        elif i % 3 == 2:
            simplified_set = Counter(line.strip().split(" "))
            
            intersect_sum = 0
            union_sum = 0
            for word in normal_set + simplified_set:
                intersect_sum += min(normal_set.get(word, 0), simplified_set.get(word, 0))
                union_sum += max(normal_set.get(word, 0), simplified_set.get(word, 0))

            jaccard_index = intersect_sum / union_sum

            jaccard_list.append((jaccard_index, i // 3 + 1))
    jaccard_list.sort(key=lambda tup: tup[0])
    for jaccard, i in jaccard_list:
        out_open.write(f"Paragraph #{i}\n")
        out_open.write(f"{jaccard}\n")
    in_open.close()
    out_open.close()


def word2vec_normalized_diff_comparison(in_file, differences_file, norm_file):
    """
    This function finds the normalized word2vec similarity score for each paragraph.
    """
    in_open = open(in_file, "r")
    # differences = open(differences_file, "a")
    norm = open(norm_file, "a")
    # norms = []
    new_model = gensim.models.Word2Vec.load("pubmedQA2.embedding")
    for i, line in enumerate(in_open):
        if i % 3 == 0:
            # differences.write(line)
            norm.write(line) # new line for unsorted
        if i % 3 == 1:
            normal_size = len(line.strip().split(" "))
            normal_set = Counter(line.strip().split(" "))
        elif i % 3 == 2:
            simplified_size = len(line.strip().split(" "))
            simplified_set = Counter(line.strip().split(" "))

            normal_words_to_remove = set()
            simplified_words_to_remove = set()

            for nword in normal_set: #nword - normal word
                for sword in simplified_set: # sword - simplified word
                    if nword in new_model.wv.key_to_index and sword in new_model.wv.key_to_index and new_model.wv.similarity(nword, sword) > 0.6:
                        normal_words_to_remove.add(nword)
                        simplified_words_to_remove.add(sword)
                
            if normal_words_to_remove:
                for word in normal_words_to_remove:
                    normal_set.pop(word)
            
            if simplified_words_to_remove:
                for word in simplified_words_to_remove:
                    simplified_set.pop(word)
                   
            final_normal_set = normal_set - simplified_set
            final_simplified_set = simplified_set - normal_set
            # differences.write(str(final_normal_set) + "\n")
            # differences.write(str(final_simplified_set) + "\n")

            final_normal_size = 0
            final_simplified_size = 0

            for word in final_normal_set:
                final_normal_size += final_normal_set[word]

            for word in final_simplified_set:
                final_simplified_size += final_simplified_set[word]
            
            normalized_reduction = round(((final_normal_size + final_simplified_size) / (normal_size + simplified_size)), 5)
            # norms.append((normalized_reduction, i // 3 + 1))
            norm.write(str(normalized_reduction) + "\n") # new for unsorted
    # norms.sort(key = lambda tup: tup[0])
    # for nr, i in norms:
    #     norm.write(f"Paragraph #{i}\n")
    #     norm.write(f"{nr}\n")
    in_open.close()
    # differences.close()
    norm.close()


def train_model():
    """
    This function was used to train a word2vec model on the PubMedQA datasets.
    """
    training_sentences = []
    for fname in ["ori_pqaa.json", "ori_pqal.json", "ori_pqau.json"]:
        print(f"Starting {fname}. Time taken: {time.perf_counter() - start_time}s")
        parsed = parse_pubmed(fname)
        training_sentences += convert_to_sentences(parsed)

    print(f"Training model. Time taken: {time.perf_counter() - start_time}s")
    model = gensim.models.Word2Vec(training_sentences)
    model.save("pubmedQA2.embedding")


def create_differences_csv(f_paragraphs, f_differences_set, f_differences_stops, f_differences, 
               f_jaccard, f_lemmatized_tagged, f_lemmatized, f_lemmatized_stops, f_word2vec_norm, f_word2vec, csv_file):
    """
    This function was used to create a CSV file of all results up to this point.
    """
    fields = ["Original Paragraph", "Simplified Paragraph", "Jaccard", "Normalized Set Difference", 
              "Original Differences Set", "Simplified Differences Set", "Original Differences with Stops",
              "Simplified Differences with Stops", "Original Differences", "Simplified Differences", 
              "Original Lemmatized Tagged", "Simplified Lemmatized Tagged", "Original Lemmatized",
              "Simplified Lemmatized", "Original Lemmatized with Stops", "Simplified Lemmatized with Stops", 
              "Original Word2Vec Differences", "Simplified Word2Vec Differences"]
    
    paragraphs = open(f_paragraphs, "r")
    differences_set = open(f_differences_set, "r")
    differences_stops = open(f_differences_stops, "r")
    differences = open(f_differences, "r")
    jaccard = open(f_jaccard, "r")
    lemmatized_tagged = open(f_lemmatized_tagged, "r")
    lemmatized = open(f_lemmatized, "r")
    lemmatized_stops = open(f_lemmatized_stops, "r")
    word2vec_norm = open(f_word2vec_norm, "r")
    word2vec = open(f_word2vec, "r")

    rows = [[] for i in range(1000)]

    for i, line in enumerate(paragraphs):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())
    
    for i, line in enumerate(jaccard):
        if i % 2 == 1:
            rows[i // 2].append(line.strip())
        
    for i, line in enumerate(word2vec_norm):
        if i % 2 == 1:
            rows[i // 2].append(line.strip())

    for i, line in enumerate(differences_set):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())

    for i, line in enumerate(differences_stops):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())

    for i, line in enumerate(differences):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())

    for i, line in enumerate(lemmatized_tagged):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())
    
    for i, line in enumerate(lemmatized):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())
    
    for i, line in enumerate(lemmatized_stops):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())
    
    for i, line in enumerate(word2vec):
        if i % 3 == 1:
            rows[i // 3].append(line.strip())
        elif i % 3 == 2:
            rows[i // 3].append(line.strip())

    with open(csv_file, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|')
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    paragraphs.close()
    differences_set.close()
    differences_stops.close()
    differences.close()
    jaccard.close()
    lemmatized_tagged.close()
    lemmatized.close()
    lemmatized_stops.close()
    word2vec_norm.close()
    word2vec.close()


### Here is where the Sentence Alignment stuff starts.


def bert_similarity(text1, text2): # each is a list of sentences
    """
    text1: list(str) - A list of sentences.
    text2: list(str) - A list of sentences.
    This function takes two lists of sentences and compares them on their similarity using a BERT model.
    return: (float) - The cosine similarity of the embeddings.
    """
    tokens1 = ["[CLS]"]
    tokens2 = ["[CLS]"]
    for sentence in text1:
        tokens1 += tokenizer.tokenize(sentence) + ["[SEP]"]
    for sentence in text2:
        tokens2 += tokenizer.tokenize(sentence) + ["[SEP]"]

    # Convert tokens to input IDs
    input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # Batch size 1
    input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # Batch size 1

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

    # Calculate similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score[0][0]


def bert_similarity_long_text(text1, text2):
    """
    The same as previous but with a different BERT model - LongFormer. Which allows for 4096 tokens instead of only 512.
    """
    tokens1 = ["[CLS]"]
    tokens2 = ["[CLS]"]
    for sentence in text1:
        tokens1 += long_tokenizer.tokenize(sentence) + ["[SEP]"]
    for sentence in text2:
        tokens2 += long_tokenizer.tokenize(sentence) + ["[SEP]"]

    # Convert tokens to input IDs
    input_ids1 = torch.tensor(long_tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # Batch size 1
    input_ids2 = torch.tensor(long_tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # Batch size 1

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = long_model(input_ids1)
        outputs2 = long_model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

    # Calculate similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score[0][0]


def create_dp_table(p1, p2): # rows are p1, cols are p2
    """
    p1: list(str) - a list of sentences in the first paragraph.
    p2: list(str) - a list of sentences in the second paragraph.
    The key creation. Takes two paragraphs and creates a dynamic programming table that can be used to find the optimal
    sentence alignments between the two paragraphs using the recursive formula described in the paper.
    return: list(list(tuple(float, tuple(int, int)))) - a 2D matrix of (score, (indices for backtracking purposes)).
    """
    SKIP_PEN = 0.35
    dp_table = []
    for i in range(len(p1) + 1):
        print(f"i = {i}")
        if i == 0:
            dp_table.append([(0, (-1, -1))] * (len(p2) + 1))
        else:
            dp_row = []
            for j in range(len(p2) + 1):
                if j == 0:
                    dp_row.append([0, (-1, -1)])
                else:
                    dp_row.append(max([(dp_table[i - 1][j][0] + SKIP_PEN, (i - 1, j)), 
                        (dp_row[-1][0] + SKIP_PEN if j > 0 else 0, (i, j - 1)),
                        (dp_table[i - 1][j - 1][0] + bert_similarity(p1[i - 1: i], p2[j - 1: j]) if j > 0 else 0, (i - 1, j - 1)), 
                        (dp_table[i - 1][j - 2][0] + (bert_similarity(p1[i - 1 : i], p2[j - 2 : j]) * 1.44) if j > 1 else 0, (i - 1, j - 2)),
                        (dp_table[i - 2][j - 1][0] + (bert_similarity(p1[i - 2: i], p2[j - 1 : j]) * 1.44) if j > 0 and i > 1 else 0, (i - 2, j - 1)),
                        (dp_table[i - 2][j - 2][0] + (bert_similarity(p1[i - 2: i], p2[j - 2 : j]) * 1.88) if j > 1 and i > 1 else 0, (i - 2, j - 2)),
                        (dp_table[i - 3][j - 1][0] + (bert_similarity(p1[i - 3 : i], p2[j - 1 : j]) * 1.88) if j > 0 and i > 2 else 0, (i - 3, j - 1)),
                        (dp_table[i - 3][j - 2][0] + (bert_similarity(p1[i - 3 : i], p2[j - 2: j]) * 2.3) if j > 1 and i > 2 else 0, (i - 3, j - 2))], 
                        key = lambda x: x[0]))
            dp_table.append(dp_row)
    return dp_table


def create_dp_table_detailed(p1, p2): # rows are p1, cols are p2
    """
    p1: list(str) - a list of sentences in the first paragraph.
    p2: list(str) - a list of sentences in the second paragraph.
    The key creation. Takes two paragraphs and creates a dynamic programming table that can be used to find the optimal
    sentence alignments between the two paragraphs using the recursive formula described in the paper.
    This has more detail than the previous function.
    return: 2D matrix of score, indices, raw similarity score, weighted similarity score, alignment type, unweighted overall score.
    """
    SKIP_PEN = 0.35
    dp_table = []
    for i in range(len(p1) + 1):
        print(f"i = {i}")
        if i == 0:
            dp_table.append([(0, (-1, -1), -1, -1, "0:0", 0)] * (len(p2) + 1))
        else:
            dp_row = []
            for j in range(len(p2) + 1):
                if j == 0:
                    dp_row.append((0, (-1, -1), -1, -1, "0:0", 0))
                else:
                    pair11 = bert_similarity(p1[i - 1: i], p2[j - 1: j]) if j > 0 else 0
                    pair12 = bert_similarity(p1[i - 1 : i], p2[j - 2 : j]) if j > 1 else 0
                    pair21 = bert_similarity(p1[i - 2: i], p2[j - 1 : j]) if j > 0 and i > 1 else 0
                    pair22 = bert_similarity(p1[i - 2: i], p2[j - 2 : j]) if j > 1 and i > 1 else 0
                    pair31 = bert_similarity(p1[i - 3 : i], p2[j - 1 : j]) if j > 0 and i > 2 else 0
                    pair32 = bert_similarity(p1[i - 3 : i], p2[j - 2: j]) if j > 1 and i > 2 else 0
                    dp_row.append(max([(dp_table[i - 1][j][0] + SKIP_PEN, (i - 1, j), 0, 0, "1:0", dp_table[i - 1][j][-1] + SKIP_PEN), 
                        (dp_row[-1][0] + SKIP_PEN if j > 0 else 0, (i, j - 1), 0, 0, "0:1", dp_row[-1][-1] + SKIP_PEN if j > 0 else 0),
                        (dp_table[i - 1][j - 1][0] + pair11 if j > 0 else 0, (i - 1, j - 1), pair11, pair11, "1:1", dp_table[i - 1][j - 1][-1] + pair11 if j > 0 else 0), 
                        (dp_table[i - 1][j - 2][0] + (pair12 * 1.44) if j > 1 else 0, (i - 1, j - 2), pair12, pair12 * 1.44 / 1.5, "1:2", dp_table[i - 1][j - 2][-1] + (pair12 * 1.5) if j > 1 else 0),
                        (dp_table[i - 2][j - 1][0] + (pair21 * 1.44) if j > 0 and i > 1 else 0, (i - 2, j - 1), pair21, pair21 * 1.44 / 1.5, "2:1", dp_table[i - 2][j - 1][-1] + (pair21 * 1.5) if j > 0 and i > 1 else 0),
                        (dp_table[i - 2][j - 2][0] + (pair22 * 1.88) if j > 1 and i > 1 else 0, (i - 2, j - 2), pair22, pair22 * 1.88 / 2, "2:2", dp_table[i - 2][j - 2][-1] + (pair22 * 2) if j > 1 and i > 1 else 0),
                        (dp_table[i - 3][j - 1][0] + (pair31 * 1.88) if j > 0 and i > 2 else 0, (i - 3, j - 1), pair31, pair31 * 1.88 / 2, "3:1", dp_table[i - 3][j - 1][-1] + (pair31 * 2) if j > 0 and i > 2 else 0),
                        (dp_table[i - 3][j - 2][0] + (pair32 * 2.3) if j > 1 and i > 2 else 0, (i - 3, j - 2), pair32, pair32 * 2.3 / 2.5, "3:2", dp_table[i - 3][j - 2][-1] + (pair32 * 2.5) if j > 1 and i > 2 else 0)], 
                        key = lambda x: x[0]))
            dp_table.append(dp_row)
    print(dp_table)
    return dp_table


def find_indices(table):
    """
    Finds the indices of all sentence pairs in the optimal solution based on the DP table. Helper function.
    """
    result = []
    indices = (len(table) - 1, len(table[0]) - 1)
    while True:
        result.append(indices)
        indices = table[indices[0]][indices[1]][1]
        if indices == (0, 0):
            break
    result.reverse()
    return result


def obtain_sentences(indices, original_sentences, simplified_sentences):
    """
    Helper function to obtain all sentence pairs using the indices returned from "find_indices".
    """
    o, s = 0, 0 # original index, simplified index
    n = 0 # number of sentences parsed
    aligned_original_sentences = []
    aligned_simplified_sentences = []
    while n < len(indices): # and o <= len(table) and s <= len(table[0]):
        print(f"o: {o}, s: {s}, n: {n}")
        original_group = " ".join(original_sentences[o : indices[n][0]])
        simplified_group = " ".join(simplified_sentences[s : indices[n][1]])
        aligned_original_sentences.append(original_group)
        aligned_simplified_sentences.append(simplified_group)
        o = indices[n][0]
        s = indices[n][1]
        n += 1
    return (aligned_original_sentences, aligned_simplified_sentences)
    

def align_sentence_files(paragraphs_file, output_file, indices_file="None"):
    """
    Takes a file name of a .txt file containing paragraphs in line-groups of 3 (marker, original text, simplified text). 
    Outputs an HTML file containing the aligned sentences of these paragraphs using the DP function.
    This can also contain the F1 score of the evaluation with a manual evaluation is given an optional indices file.
    The indices file is in groups of 2 lines - 1st line marker, second line comma-separated values for each index.
    The alignment (1, 2) - representing the first sentence of the original is aligned with the first two of the simplified would
    be written without parentheses - 1, 2.
    (1, 2), (2, 3) - 1, 2, 2, 3.
    """
    paragraphs = open(paragraphs_file, "r")
    output = open(output_file, "a")
    if indices_file != "None":
        indices = open(indices_file, "r")
        manual_indices = []
        for i, line in enumerate(indices):
            if i % 2 == 1:
                manual_indices.append(line_to_indices(line))
    output.write("<!DOCTYPE html>\n<html>\n<body>\n<h1>Aligned Paragraphs</h1>\n")
    background_colors = ["Orange", "Yellow", "LightGreen", "Cyan", "Violet", "Pink"]
    for i, line in enumerate(paragraphs):
        if i % 3 == 0:
            output.write(f"<h3>{line.strip()}</h3>\n")
            print(f"Processing: {line.strip()}")
        elif i % 3 == 1:
            original_sentences = sent_tokenize(line.strip())
        else:
            color = 0
            simplified_sentences = sent_tokenize(line.strip())
            dp_table = create_dp_table(original_sentences, simplified_sentences)
            sentence_indices = find_indices(dp_table)
            aligned_original_sentences, aligned_simplified_sentences = obtain_sentences(dp_table, sentence_indices, original_sentences, simplified_sentences)
            output.write("<p>")
            for sent in aligned_original_sentences:
                output.write(f"<span style='background-color: {background_colors[color]}'>{sent}</span> ")
                color = (color + 1) % 6
            output.write("</p>\n<p>")
            color = 0
            for sent in aligned_simplified_sentences:
                output.write(f"<span style='background-color: {background_colors[color]}'>{sent}</span> ")
                color = (color + 1) % 6
            output.write(f"</p>\n<p>Sentence Alignment Score: {2 * dp_table[-1][-1][0] / (sentence_indices[-1][0] + sentence_indices[-1][1])}.</p>\n")
            if indices_file != "None":
                f1_score = compare_indices(manual_indices[i // 3], sentence_indices)
                output.write(f"<p>F1 Alignment Score: {f1_score}.</p>\n")
    output.write("</body>\n</html>")
    paragraphs.close()
    output.close()


def align_sentence_files_to_csv(paragraphs_file, csv_file, html_file):
    """
    This takes a .txt file containing the paragraphs in the standard format (described on line 601) and outputs
    the sentences in order of weighted similarity score in an HTML file and also as a pipe | delimited CSV file.
    """
    paragraphs = open(paragraphs_file, "r")
    html = open(html_file, "a")
    rows = []
    for i, line in enumerate(paragraphs):
        if i % 3 == 0:
            print(f"Processing: {line.strip()}")
        elif i % 3 == 1:
            original_sentences = sent_tokenize(line.strip())
        else:
            simplified_sentences = sent_tokenize(line.strip())
            dp_table = create_dp_table_detailed(original_sentences, simplified_sentences)
            sentence_indices = find_indices(dp_table)
            aligned_original_sentences, aligned_simplified_sentences = obtain_sentences(dp_table, sentence_indices, original_sentences, simplified_sentences)
            for i in range(len(aligned_original_sentences)):
                row = [aligned_original_sentences[i], aligned_simplified_sentences[i]]
                details = dp_table[sentence_indices[i][0]][sentence_indices[i][1]]
                row += details[2 : 5]
                rows.append(row)
    fields = ["Original Sentences", "Simplified Sentences", "Raw Similarity Score", "Weighted Similarity Score", "Alignment Type"]
    with open(csv_file, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|')
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
    rows.sort(reverse = True, key=lambda x:x[3])
    html.write("<!DOCTYPE html>\n<html>\n<body>\n<h1>Aligned Paragraphs</h1>\n")
    i = 1
    for s1, s2, score, norm_score, alignment_type in rows:
        html.write(f"<h3>Sentences #{i} - ({alignment_type}):</h3>")
        html.write(f"<p>{s1}</p><p>{s2}</p><p>Raw Similarity Score: {score}</p><p>Weighted Similarity Score: {norm_score}</p>\n")
        i += 1
    html.write("</body>\n</html>")
    paragraphs.close()
    csvfile.close()
    html.close()


def bert_test(sentences_file, output_file):
    """
    Takes a file of sentences and runs the BERT similarity comparison on them. 
    """
    sentences = open(sentences_file, "r")
    output = open(output_file, "a")
    print("Starting...")
    print(f"Sentences: {sentences}")
    for i, line in enumerate(sentences):
        if i % 3 == 0:
            print(line.strip())
            output.write(f"{line.strip()}\n")
        elif i % 3 == 1:
            p1_tokens = sent_tokenize(line.strip())
        else:
            p2_tokens = sent_tokenize(line.strip())
            similarity = bert_similarity(p1_tokens, p2_tokens)
            output.write(f"{similarity}\n")
    print("Done!")
    sentences.close()
    output.close()


def paragraph_simplifier_by_sentence(paragraph):
    """
    Takes a paragraph as list of sentences and simplified the paragraph sentence by sentence using ChatGPT4o.
    """
    client = OpenAI()
    simple_sentences = []

    for count, sentence in enumerate(paragraph):
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Please simplify the following sentence:\n" + sentence}])
        simple_sentences.append(completion.choices[0].message.content)
        print(f"Sentence #{count} is simplifying.")

    return " ".join(simple_sentences)


def output_test_alignment_paragraphs(paragraphs_file, output_file):
    """
    Using the sentence-level simplification from before, take a file of paragraphs in a different format (groups of 2 lines -
    1st line as marker, 2nd line as actual text) and output a file of paragraphs in the standard format with marker, original
    paragraph, and sentence-level simplification.
    """
    paragraphs = open(paragraphs_file, "r")
    output = open(output_file, "a")
    for i, line in enumerate(paragraphs):
        print(f"Paragraph #{i // 2 + 1}")
        if i % 2 == 0:
            output.write(f"{line.strip()}\n")
        else:
            output.write(f"{line.strip()}\n")
            p_tokens = sent_tokenize(line.strip())
            simplified_by_sentence = paragraph_simplifier_by_sentence(p_tokens)
            output.write(f"{simplified_by_sentence}\n")
    paragraphs.close()
    output.close()


def compare_indices(indices1, indices2):
    """
    Takes two sets of indices and outputs the F1 similarity score.
    """
    correct = 0
    i1, i2 = 0, 0
    while i1 < len(indices1) and i2 < len(indices2):
        if indices1[i1] == indices2[i2]:
            i1 += 1
            i2 += 1
            correct += 1
        elif indices1[i1][0] <= indices2[i2][0] and indices1[i1][1] <= indices2[i2][1]:
            i1 += 1
        elif indices1[i1][0] >= indices2[i2][0] and indices1[i1][1] >= indices2[i2][1]:
            i2 += 1
        else:
            i1 += 1
            i2 += 1
    return 2 * correct / (len(indices1) + len(indices2))


def line_to_indices(line): # only call on even lengths
    """
    Takes a list of unzipped indices without their parentheses and zips them back together.
    """
    nums = line.split(",")
    nums = [int(num) for num in nums]
    result = []
    for i in range(0, len(nums), 2):
        result.append((nums[i], nums[i + 1]))
    return result


def get_corpus_word_frequencies(corpus_file):
    """
    Gets the frequencies of a corpus and puts them into a dictionary.
    """
    corpus = open(corpus_file, "r")
    word_freqs = {}
    for line in corpus:
        try:
            word, freq = line.strip().split("\t")
            word_freqs[word] = int(freq)
        except:
            pass
    corpus.close()
    return word_freqs


def is_nonalnum(word):
    """
    Returns False is a string contains any alphanumeric characters, and True if it does not.
    """
    for char in word:
        if char.isalnum():
            return False
    return True


def get_average_word_frequencies(text1, text2, corpus_word_freqs):
    """
    Gets the average word frequency of two texts using a dictionary of word frequencies taken from a corpus.
    """
    stop_words = set(stopwords.words("english"))
    words1 = word_tokenize(text1)
    words2 = word_tokenize(text2)
    sum1 = sum2 = 0
    count1 = count2 = 0
    good_words1 = []
    good_words2 = []
    for word in words1:
        if word.casefold() in stop_words or is_nonalnum(word):
            continue
        sum1 += corpus_word_freqs.get(word, 0)
        count1 += 1
        good_words1.append(word)
    for word in words2:
        if word.casefold() in stop_words or is_nonalnum(word):
            continue
        sum2 += corpus_word_freqs.get(word, 0)
        count2 += 1
        good_words2.append(word)
    # print(f"sum1, sum2: {sum1}, {sum2}")
    # print(f"count1, count2: {count1}, {count2}")
    # print(Counter(good_words1))
    # print(Counter(good_words2))
    # print(Counter(good_words1) - Counter(good_words2))
    # print(Counter(good_words2) - Counter(good_words1))
    avg_freqs1 = sum1 / count1
    avg_freqs2 = sum2 / count2
    return (avg_freqs1, avg_freqs2)


def get_cohesiveness(sentences):
    """ 
    Gets the cohesiveness score of a paragraph (list of sentences).
    """
    if len(sentences) == 1:
        return 0 
    total = 0
    for i in range(len(sentences) - 1):
        total += bert_similarity(sentences[i : i + 1], sentences[i + 1 : i + 2])
    return total / (len(sentences) - 1)


def output_cohesiveness_score(paragraphs_file, output_file):
    """
    Given a standard format .txt file of paragraphs, outputs the cohesiveness score for all in an HTML file.
    """
    paragraphs = open(paragraphs_file, "r")
    output = open(output_file, "a")
    output.write("<!DOCTYPE html>\n<html>\n<body>\n<h1>Paragraph Cohesiveness Scores</h1>\n")
    for i, line in enumerate(paragraphs):
        if i % 3 == 0:
            output.write(f"<h3>{line.strip()}</h3>\n")
            print(f"Processing: {line.strip()}")
        elif i % 3 == 1:
            original_sentences = sent_tokenize(line.strip())
        else:
            simplified_sentences = sent_tokenize(line.strip())
            output.write("<p>")
            for sentence in original_sentences:
                output.write(f"{sentence} ")
            output.write("</p>\n<p>")
            for sentence in simplified_sentences:
                output.write(f"{sentence} ")
            output.write("</p>\n")
            original_cohesiveness = get_cohesiveness(original_sentences)
            simplified_cohesiveness = get_cohesiveness(simplified_sentences)
            output.write(f"<p>Original Cohesiveness Score: {original_cohesiveness}</p>\n")
            output.write(f"<p>Simplified Cohesiveness Score: {simplified_cohesiveness}</p>\n")
    output.write("</body>\n</html>")
    paragraphs.close()
    output.close()


def output_simplicity_scores(paragraphs_file, output_file):
    """
    Outputs the simplicity (avg word freq) scores for paragraphs and their ratio in an HTML file.
    """
    paragraphs = open(paragraphs_file, "r")
    output = open(output_file, "a")
    output.write("<!DOCTYPE html>\n<html>\n<body>\n<h1>Text Simplicity Scores</h1>\n")
    # corpus_word_freqs = get_corpus_word_frequencies("normal.11.counts.txt")
    corpus_word_freqs = get_corpus_word_frequencies("google.1grams.txt")
    for i, line in enumerate(paragraphs):
        if i % 3 == 0:
            output.write(f"<h3>{line.strip()}</h3>\n")
            print(f"Processing: {line.strip()}")
        elif i % 3 == 1:
            original_text = line.strip()
        else:
            simplified_text = line.strip()
            avg_freqs1, avg_freqs2 = get_average_word_frequencies(original_text, simplified_text, corpus_word_freqs)
            output.write(f"<p>{original_text}</p>\n")
            output.write(f"<p>{simplified_text}</p>\n")
            output.write(f"<p>Original Average Word Frequency: {avg_freqs1}</p>\n")
            output.write(f"<p>Simplified Average Word Frequency: {avg_freqs2}</p>\n")
            output.write(f"<p>Ratio of Average Word Frequencies: {avg_freqs1 / avg_freqs2}</p>\n")
    output.write("</body>\n</html>")
    paragraphs.close()
    output.close()


def output_similarity_scores(paragraphs_file, output_file):
    """
    Outputs the similarity scores (sentence alignment and paragraph-level BERT similarity) for pairs of original and simplified
    texts.
    """
    paragraphs = open(paragraphs_file, "r")
    output = open(output_file, "a")
    output.write("<!DOCTYPE html>\n<html>\n<body>\n<h1>Paragraph Similarity Scores</h1>\n")
    background_colors = ["Orange", "Yellow", "LightGreen", "Cyan", "Violet", "Pink"]
    for i, line in enumerate(paragraphs):
        if i % 3 == 0:
            output.write(f"<h3>{line.strip()}</h3>\n")
            print(f"Processing: {line.strip()}")
        elif i % 3 == 1:
            original_sentences = sent_tokenize(line.strip())
        else:
            color = 0
            simplified_sentences = sent_tokenize(line.strip())
            dp_table = create_dp_table_detailed(original_sentences, simplified_sentences)
            sentence_indices = find_indices(dp_table)
            aligned_original_sentences, aligned_simplified_sentences = obtain_sentences(dp_table, sentence_indices, original_sentences, simplified_sentences)
            output.write("<p>")
            for sent in aligned_original_sentences:
                output.write(f"<span style='background-color: {background_colors[color]}'>{sent}</span> ")
                color = (color + 1) % 6
            output.write("</p>\n<p>")
            color = 0
            for sent in aligned_simplified_sentences:
                output.write(f"<span style='background-color: {background_colors[color]}'>{sent}</span> ")
                color = (color + 1) % 6
            paragraph_bert_similarity = bert_similarity(original_sentences, simplified_sentences)
            output.write(f"</p>\n<p>Aligned Sentences Similarity Score: {2 * dp_table[-1][-1][-1] / (sentence_indices[-1][0] + sentence_indices[-1][1])}.</p>\n")
            output.write(f"<p>Paragraph BERT Similarity Score: {paragraph_bert_similarity}.</p>\n")
    output.write("</body>\n</html>")
    paragraphs.close()
    output.close()
    

if __name__ == "__main__":
    """
    Main function - contains a lot of example function calls.
    """
    start_time = time.perf_counter()
    # pubmed_parsed = parse_pubmed()
    # paragraphs = find_paragraphs(1000, pubmed_parsed)
    # simple_paragraphs = paragraph_simplifier(paragraphs)
    # create_paragraph_file(paragraphs, simple_paragraphs)
    # process_file("1000_paragraphs_simplified.txt", "stemmed_paragraphs.txt")
    # subtract_intersections("stemmed_paragraphs.txt", "differences_stemmed_paragraphs.txt")
    # find_jaccard_gen("stemmed_paragraphs.txt", "jaccard_stemmed_paragraphs_sorted.txt")
    # word2vec_normalized_diff_comparison("lemmatized_paragraphs.txt", "word2vec_paragraphs.txt", "word2vec_norm_reduction_paragraphs_unsorted.txt")
    # create_csv("1000_paragraphs_simplified.txt", "differences_paragraphs_set.txt", "differences_paragraphs_sorted_with_stops.txt",
    #            "differences_paragraphs.txt", "jaccard_paragraphs_unsorted.txt", "lemmatized_paragraphs_tagged.txt", 
    #            "lemmatized_paragraphs.txt", "lemmatized_paragraphs_with_stops.txt", "word2vec_norm_reduction_paragraphs_unsorted.txt",
    #            "word2vec_paragraphs.txt", "paragraphs.csv")
    # new_model = gensim.models.Word2Vec.load("pubmedQA2.embedding")

    # for line in sentence_alignment_test():
    #     print(line)
    # paragraph1 = "An aorto-iliac aneurysm is a dilatation (aneurysm) of the aorta, the main large blood vessel in the body, which carries blood out from the heart to all organs and iliac arteries (distal branches of the aorta). The aneurysm can grow and burst (rupture), which leads to severe bleeding and is frequently fatal; an estimated 15,000 deaths occur each year from ruptured aortic abdominal aneurysms in the USA alone. To avoid this complication, the aorto-iliac aneurysm should be repaired when the maximum diameter of the aorta reaches 5 cm to 5.5 cm, or when the maximum diameter of the common iliac arteries reaches 3 cm to 4 cm. Endovascular repair of aorto-iliac aneurysms is one approach that is used to manage this condition: a tube (stent-graft) is placed inside the aorto-iliac aneurysm, so that blood flows through the stent-graft and no longer into the aneurysm, excluding it from the circulation. To achieve a successful deployment of the stent-graft, a good seal zone (fixation zone) is needed in the aorta (proximal) and in the common iliac arteries (distal). However, in 40% of patients, the distal seal zone in the common iliac arteries is inadequate. In these cases, most commonly the stent-graft is extended to the external iliac artery and the internal iliac artery is blocked (occluded). However, this obstruction (occlusion) is not without harms: the internal iliac artery supplies blood to the pelvic organs (rectum, bladder, and reproductive organs) and the pelvic muscles, and occlusion is associated with complications in the pelvic area such as buttock claudication (cramping pain in the buttock during exercise), sexual dysfunction, and spinal cord injury. New endovascular devices and techniques such as iliac branch devices have emerged to maintain blood flow into the internal iliac artery. These special stent-grafts position the distal seal zone within the external iliac artery, and a side branch of the graft allows for revascularisation of the internal iliac artery, while excluding the aneurysm from the circulation, promoting an adequate distal seal zone, and maintaining pelvic circulation. This may also preserve the quality of life of treated individuals and may reduce serious complications including spinal cord ischaemia, ischaemic colitis, and gluteal necrosis."
    # paragraph2 = "An aorto-iliac aneurysm is a bulge in the aorta and iliac arteries that can burst, causing severe, often fatal bleeding. Around 15,000 deaths occur each year in the USA from ruptured aortic aneurysms. To prevent this, repairs are recommended when the aorta reaches 5-5.5 cm or the iliac arteries reach 3-4 cm in diameter. One repair method is endovascular surgery, where a stent-graft is placed inside the aneurysm to redirect blood flow and exclude the aneurysm from circulation. This requires good seal zones in the aorta and iliac arteries. However, 40% of patients have inadequate distal seal zones in the iliac arteries. In such cases, the stent-graft is extended to the external iliac artery, and the internal iliac artery is blocked, which can cause complications like buttock pain, sexual dysfunction, and spinal cord injury. New devices, like iliac branch stent-grafts, maintain blood flow to the internal iliac artery. These devices position the seal zone in the external iliac artery and have a side branch for the internal iliac artery, preserving pelvic blood flow and reducing serious complications, thus improving the quality of life for patients."
    # p1_tokens = sent_tokenize(paragraph1)
    # p2_tokens = sent_tokenize(paragraph2)
    # dp_table = align_sentences_test()
    # for line in dp_table:
    #     print(line)
    # indices = find_indices(dp_table)
    # print(indices)
    # sentences = obtain_sentences(dp_table, indices, p1_tokens, p2_tokens)
    # print(sentences)

    # align_sentence_files_to_csv("sentence_alignment/ten_test_texts.txt", "sentence_alignment/ten_test_texts.csv", "sentence_alignment/ten_test_texts_detailed.html")

    # indices1 = [(1, 1), (2, 3), (4, 4), (5, 6), (7, 8)]
    # indices2 = [(1, 1), (3, 2), (4, 4), (5, 6), (6, 7), (7, 8)]
    # print(compare_indices(indices1, indices2))

    # align_sentence_files("sentence_alignment/ten_test_texts.txt", "sentence_alignment/ten_test_texts_updated3.html", "sentence_alignment/ten_test_texts_manual_indices.txt")
    # bert_test("sentence_alignment/new_trouble.txt", "sentence_alignment/new_trouble_bert_scores.txt")
    # output_similarity_scores("sentence_alignment/ten_random_texts.txt", "sentence_alignment/ten_random_texts_similarities.html")
    # output_similarity_scores("sentence_alignment/ten_test_texts2.txt", "sentence_alignment/ten_test_texts2_similarities.html")
    # output_similarity_scores("sentence_alignment/ten_new_texts2.txt", "sentence_alignment/ten_new_texts2_similarities.html")
    # output_simplicity_scores("sentence_alignment/ten_test_texts2.txt", "sentence_alignment/ten_test_texts2_simplicities.html")
    # output_simplicity_scores("sentence_alignment/simplicity_test.txt", "sentence_alignment/simplicity_test.html")
    # output_similarity_scores("sentence_alignment/ten_sample_paragraphs.txt", "sentence_alignment/ten_sample_paragraphs_similarities.html")
    # output_simplicity_scores("sentence_alignment/ten_sample_paragraphs.txt", "sentence_alignment/ten_sample_paragraphs_simplicities.html")
    # align_sentence_files_to_csv("sentence_alignment/ten_sample_paragraphs.txt", "sentence_alignment/ten_sample_paragraphs_details.csv", "sentence_alignment/ten_sample_paragraphs_details.html")
    # output_test_alignment_paragraphs("sentence_alignment/ten_test_texts_input.txt", "sentence_alignment/ten_new_texts.txt")
    # align_sentence_files("sentence_alignment/ten_new_texts.txt", "sentence_alignment/ten_new_texts_new.html")
    # output_test_alignment_paragraphs("sentence_alignment/ten_sample_paragraphs_input.txt", "sentence_alignment/ten_sample_paragraphs_sentence_simplified.txt")
    # output_similarity_scores("sentence_alignment/ten_sample_paragraphs_sentence_simplified.txt", "sentence_alignment/ten_sample_sentence_simplified_similarities.html")
    # output_simplicity_scores("sentence_alignment/ten_new_texts.txt", "sentence_alignment/ten_new_texts_simplicities.html")
    # align_sentence_files("sentence_alignment/ten_sample_paragraphs.txt", "sentence_alignment/ten_sample_paragraphs_f1scores.html", "sentence_alignment/ten_sample_paragraphs_manual_indices.txt")
    output_cohesiveness_score("sentence_alignment/ten_test_texts.txt", "sentence_alignment/ten_test_texts_cohesiveness.html")
    output_cohesiveness_score("sentence_alignment/ten_new_texts.txt", "sentence_alignment/ten_new_texts_cohesiveness.html")
    output_cohesiveness_score("sentence_alignment/ten_sample_paragraphs.txt", "sentence_alignment/ten_sample_paragraphs_cohesiveness.html")
    output_cohesiveness_score("sentence_alignment/ten_sample_paragraphs_sentence_simplified.txt", "sentence_alignment/ten_sample_paragraphs_sentence_simplified_cohesiveness.html")

    print(f"Time taken: {time.perf_counter() - start_time}s")
