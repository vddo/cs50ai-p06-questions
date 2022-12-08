# P06 language #

Functions to implement:

1. [x] ```load_files```
2. [x] ```tokenize```
3. [x] ```compute_idfs```
4. [x] ```top_files```
5. ```top_sentences```

## To-Dos ##

### function ```load_files``` ###

+ [x] input: name of ```directory```
+ [x] return: dictionary mapping the filename of each ```.txt``` file with content as a string
  + [x] {```filename```: ```content```}
    + [x] for one file

+ [x] **platform-independent**: use ```os.sep``` and ```os.path.join```
+ [x] key should just be filename, without directory name
+ [x] e.g. file: /corpus/a.txt -> file name: ```a.txt```

### function ```tokenize``` ###

+ [x] input: ```document``` (a string)
+ [x] return ```list``` of all words in that document
  + [x] in order
  + [x] in lowercase
+ [x] use ```nltk```'s ```word_tokenize``` function
+ filter out:
  + [x] punctuations: character in ```string.punctuation``` (after importing ```string```)
  + [x] stopwords: in ```nltk.corpus.stopwords.words("english")```.
+ [x] if a word appears multiple times in the ```document``` &rarr; also multiple times in returned ```list``` (unless it was filtered out)

### function ```compute_idfs``` ###

+ [x] input: dictionary of ```documents``` with words as *list*
+ [x] return: new dictionary mapping **words** to their **IDF** (invers document fequency) **values**
  + input is dictionary mapping name of documents to **list** of their words, see in ```main``` &rarr; ```file_words```
  + returned dictionary should map **every word** that appears at least once in all of the documents
  + IDF defined by taking natural logarithm of number of documents divided by number of documents in which the word appears

$$
IDF = \log \frac{TotalDocuments}{NumDocumentsContaining(word)}
$$

### function ```top_files``` ###

+ [x] input:
  + ```query``` (set of words),
  + ```files```  (a dictionary mapping names of files to a list of their words), also see in ```main``` &rarr; ```file_words```
  + and ```idfd``` (dictionary mapping words to their IDF values)
  
+ [x] return: **list** of the filenames of the ```n``` top files that match the query
  + ranked akording to tf-idf

+ [x] returned *list* of filenames with length ```n```, ordered with best match first

+ [x] files ordered according to the sum of tf-idf values for any word in the query
  + that also appears in the file
  + words in the query that do not appear in file should not contribute to file's score
+ [x] tf-idf... multiplying number of times the term appears in the document by the IDF value for that term

$$
tfidf = NumberInDocument(word) \cdot IDF(word)
$$

### function ```top_sentences``` ###

+ input:
  + ```query```
  + ```sentences``` (dictionary mapping sentences to a list of their words)
    + see in ```main``` ```sentence``` and ```nltk.sent_tokenize```
  + ```idfs``` (dictionary mapping words to IDF)
+ return: *list* of ```n``` top sentences
  + ranked according IDF
  + length ```n```
+ sentences ranking: "matching word measure" -> sum of IDF for any word in the query that also appears in sentence
  + term frequency not to be taken into account
  + only inverse document frequenzy
+ two sentences same value: prefer higher "query term density"
  + ... preportion of words in the sentence that are also words in the query
  + e.g. sentence has 10 words, 3 of which are in the query - *density* = ```0.3```
