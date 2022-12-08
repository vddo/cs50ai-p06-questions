import nltk
import sys
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np

FILE_MATCHES = 1 # Specifies how many files should match for query.
SENTENCE_MATCHES = 1 # How many sentences within those files should match.


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1]) # Load file into memory.
    file_words = {
        filename: tokenize(files[filename]) # Tokenize into list of words.
        for filename in files
    }
    file_idfs = compute_idfs(file_words) # Compute inverse document frequency values for each word.

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    documents = dict()
    
    for file in os.listdir(directory):
        filePath = os.path.join(directory, file)
        if os.path.isfile(filePath):
            with open(filePath, 'r') as f:
                documents[file] = f.read()
                
    return documents    


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Tokenizer for removing punctuations
    punct_tokenizer = RegexpTokenizer(r'\w+')

    # Set of stopwords that need to be removed
    to_be_removed = set(stopwords.words('english'))
    
    return list(
        map(
            lambda x: x.lower(), [word for word in punct_tokenizer.tokenize(document) \
            if not word in to_be_removed]
        )
    )


def num_doc_contain(word, documents):
    """
    Takes a word and returns in how many documents the word appears.

    Args:
        word (string): Word that appears at least once in one of the corpus' documents.
        documents (dictionary): List of all document's words mapped to the document's name
    """
    count_document_containing = 0
    
    for d in documents:
        if word in documents[d]:
            count_document_containing += 1
            
    return count_document_containing


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Extract all unique words from all documents by adding the set-converted
    # List of words to one combined list.
    set_all_words = set()
    for d in documents:
        set_all_words.update(
            set(documents[d])
        )
    
    # Convert set to a list.
    list_all_words = list(set_all_words)
    
    # Create and fill dictionary
    idfs = {}
    total_documents = len(documents.keys())
    
    for word in list_all_words:
        idfs[word] = np.log(
            total_documents / num_doc_contain(word, documents)
        )
        
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {}
    
    # First loop over all files.
    for file in files.keys():
        list_compute_current_file = []
        # Loop over words in set query.
        for word in query:
            if word in idfs and idfs[word] != 0:
                list_compute_current_file.append(idfs[word] * files[file].count(word))

        # Compute tf-idf and save in dict.
        tf_idf[file] = sum(list_compute_current_file)
    
    # Return ordered list with file names with best match first (desc).
    list_ordered_files = []
    tf_idf_sorted = sorted(tf_idf.items(), key= lambda x:x[1], reverse=True)
    
    i = 0
    while i < n:
        list_ordered_files.append(tf_idf_sorted[i][0])
        i += 1
    
    return list_ordered_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf_sentences = {}
    for sentence in sentences:    
        idf_sum = 0
        count_density = 0
        # Loop over words in set query.
        for word in query:
            if word in sentences[sentence] and idfs[word] != 0:
                idf_sum += idfs[word]
                count_density += 1

        # Compute idf sum and map in dict.
        idf_sentences[sentence] = [idf_sum, count_density/len(sentences[sentence])]
    
    # Order the sentences and return list with length n
    list_ordered_sentences = []
    sentences_idf_sorted = sorted(idf_sentences.items(), key= lambda x:(x[1][0], x[1][1]), reverse=True)

    i = 0
    while i < n:
        list_ordered_sentences.append(sentences_idf_sorted[i][0])
        i += 1
    
    return list_ordered_sentences


if __name__ == "__main__":
    main()
