import nltk
import sys
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

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


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
