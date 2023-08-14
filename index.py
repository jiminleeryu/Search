import math
import re
import sys
import xml.etree.ElementTree as et

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

import file_io


class Indexer:
    """
    The indexer class, which takes in a wiki and processes the documents in it
    into files that are used by the querier
    """

    def __init__(self, wiki: str, title: str, doc: str, word: str):
        """
        The constructor for the indexer.
        DO NOT MODIFY THIS CONSTRUCTOR.

        Note that the output files may be overwritten if they already exist.
        
        Parameters:
        wiki        the filename of the input wiki
        title       the output filename of the titles file
        doc         the output filename of the docs file
        word        the output filename of the words file
        """

        # defining epsilon for PageRank calculations
        self.EPSILON = 0.15
        # distance threshold for PageRank calculation
        self.DISTANCE_THRESHOLD = 0.001
        # set of stop words
        self.STOP_WORDS = set(stopwords.words("english"))
        # porter stemmer
        self.nltk_ps = PorterStemmer()

        """
        The tokenization regex has three parts, separated by pipes (|), which
        mean “or.” So we are actually matching three possible alternatives:

        1) \[\[[^\[]+?\]\]
        Meaning: Match two left brackets (\[\[) and two right brackets (\]\]),
        making sure there’s something in the middle, but also making sure there
        is not a left bracket in the middle, which would mean that somehow
        another link was starting inside this link ([^\[]+?).
        Returns: Links (e.g., “[[Some Wiki Page]]” or “[[Universities|Brown]]”)
        
        2) [a-zA-Z0-9]+'[a-zA-Z0-9]+
        Meaning: Match at least one alphanumeric character ([a-zA-Z0-9]+), then
        an apostrophe ('), and then at least one alphanumeric character
        ([a-zA-Z0-9]).
        Returns: Words with apostrophes 
        
        3) [a-zA-Z0-9]+
        Meaning: Match at least one alphanumeric character in a row.
        Returns words
        """
        self.tokenization_regex = r"\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+"

        # page id to word to num appearances
        self.words_to_doc_frequency = {}
        # page id to title
        self.ids_to_titles = {}
        # title to page id
        self.titles_to_ids = {}
        # page id to highest word counts
        self.ids_to_max_counts = {}
        # id to all the ids that page links to
        self.ids_to_links = {}

        self.wiki = wiki
        self.title = title
        self.doc = doc
        self.word = word


    def run(self):
        """
        Runs the indexer by parsing the document, computing term relevance and
        page rank, and writing the results to the titles/docs/words output
        files

        DO NOT MODIFY THIS METHOD.
        """
        try:
            self.parse()
            words_to_doc_relevance = self.compute_term_relevance()
            page_rank = self.compute_page_rank()
            
            file_io.write_title_file(self.title, self.ids_to_titles)
            file_io.write_document_file(
                self.doc, page_rank)
            file_io.write_words_file(self.word, words_to_doc_relevance)
        except FileNotFoundError:
            print("One (or more) of the files were not found")
        except IOError:
            print("Error: IO Exception")

    def stem_and_stop(self, word: str):
        """
        Checks if word is a stop word, converts it to lowercase, and stems it

        DO NOT MODIFY THIS METHOD.

        Parameters:
            word        the word to check
        Returns:
            "" if the word is a stop word, the converted word, otherwise
        """
        if word.lower() in self.STOP_WORDS:
            return ""

        return self.nltk_ps.stem(word.lower())

    def word_is_link(self, word: str) -> bool:
        """
        Checks if the word is a link (surrounded by '[[' and ']]')

        DO NOT MODIFY THIS METHOD.

        Parameters:
            word        the word to check
        Returns:
            true if the word is a link, false otherwise
        """
        link_regex = r"\[\[[^\[]+?\]\]"
        return bool(re.match(link_regex, word))

    def split_link(self, link: str) -> tuple[list[str], str]:
        """
        Splits a link (assumed to be surrounded by '[[' and ']]') into the text
        and the destination of the link

        DO NOT MODIFY THIS METHOD.

        Example usage:
        link_text, link_dest = split_link(link_str)

        Parameters:
            link        the link to split
        Returns:
            a tuple of the format (link text, link destination)
        """
        is_word_regex = r"[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+"

        # assume that the link is surrounded by '[[' and ']]'
        link_stripped_brackets = link[2:-2]

        title_raw = link_stripped_brackets
        text_raw = link_stripped_brackets

        # text and title differ
        if '|' in link_stripped_brackets:
            link_split = link_stripped_brackets.split("|")
            title_raw = link_split[0]
            text_raw = link_split[1]

        return (re.findall(is_word_regex, text_raw), title_raw.strip())

    def process_document(self, title: str, id: int, body: str) -> list[str]:
        """
        Takes in a document title, id, and body, and returns a list of the
        tokens in the document title and body.

        A "token" is a word, not including stopwords, that has been stemmed. For
        links, only the link text (not destination) are included in the returned
        list.


        Parameters:
            title       the title of the document
            id          the id of the document
            body        the text of the document
        Returns:
            a list of the tokens in the title and the body
        """
        
        val_tokens = []

        cool_tokens = re.findall(self.tokenization_regex, f"{title} {body}")
        for words in cool_tokens:
            if self.word_is_link(words):
                link_text, link_dst = self.split_link(words)
                if link_dst in self.titles_to_ids:
                    if id not in self.ids_to_links:
                        self.ids_to_links[id] = set() 
                    self.ids_to_links[id].add(self.titles_to_ids[link_dst])
                for text in link_text:
                    text_token = self.stem_and_stop(text)
                    if text_token != "":
                        val_tokens.append(text_token)
                        if text_token not in self.words_to_doc_frequency:
                            self.words_to_doc_frequency[text_token] = {}
                        if id not in self.words_to_doc_frequency[text_token]:
                            self.words_to_doc_frequency[text_token][id] = 1
                        else:
                            self.words_to_doc_frequency[text_token][id] = self.words_to_doc_frequency[text_token][id] + 1
                        
            else:
                word = self.stem_and_stop(words)
                if word != "":
                    val_tokens.append(word)
                    if word not in self.words_to_doc_frequency:
                        self.words_to_doc_frequency[word] = {}     
                    if id not in self.words_to_doc_frequency[word]:
                        self.words_to_doc_frequency[word][id] = 1
                    else:
                        self.words_to_doc_frequency[word][id] = self.words_to_doc_frequency[word][id] + 1

        
        max_num = 0
        for values in self.words_to_doc_frequency.values():
            if id in values.keys() and values[id] > max_num:
                max_num = values[id]
        self.ids_to_max_counts[id] = max_num

        return val_tokens # TODO: fixme

    def parse(self):
        """
        Reads in an xml file, parses titles and ids, tokenizes text, removes
        stop words, does stemming, and processes links.

        Updates ids_to_titles, titles_to_ids, words_to_doc_frequency,
        ids_to_max_counts, and ids_to_links
        """

        # load XML + root
        wiki_tree = et.parse(self.wiki)
        wiki_xml_root = wiki_tree.getroot()


        for wiki_page in wiki_xml_root:
            page_title = wiki_page.find("title").text.strip()
            id = (int) (wiki_page.find("id").text.strip())
            self.ids_to_titles[id] = page_title
            self.titles_to_ids[page_title] = id

        for wiki_page in wiki_xml_root:
            page_title = wiki_page.find("title").text.strip()
            page_id = (int) (wiki_page.find("id").text.strip())
            body = wiki_page.find("text").text.strip()
            self.process_document(page_title, int (page_id), body)

    def compute_tf(self) -> dict[str, dict[int, float]]:
        """
        Computes tf metric based on words_to_doc frequency

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            a dictionary mapping every word to its term frequency
        """

        word_list = {}

        for word in self.words_to_doc_frequency.keys():
            word_list[word] = {}
            for key, count in self.words_to_doc_frequency[word].items():
                if (count/self.ids_to_max_counts[key]) > 0:
                    word_list[word][key] = count/self.ids_to_max_counts[key]

        return word_list

    def compute_idf(self) -> dict[str, float]:
        """
        Computes idf metric based on words_to_doc_frequency

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            a dictionary mapping every word to its inverse term frequency
        """
        
        idf = {}
        number_documents = len(self.ids_to_titles) # total number of documents

        for word, words_dict in self.words_to_doc_frequency.items():
            idf[word] = {}

            number_docs_with_word = len(words_dict)
            idf[word] = math.log(number_documents/ number_docs_with_word)

        return idf

    def compute_term_relevance(self) -> dict[str, dict[int, float]]:
        """
        Computes term relevance based on tf and idf

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            a dictionary mapping every every term to a dictionary mapping a page
            id to the relevance metric for that term and page
        """

        #compute idf scores
        idf_scores = self.compute_idf()

        #compute tf scored 
        tf_scores = self.compute_tf()

        term_relevance = {}

        for term, id_to_frequency in tf_scores.items():
            term_relevance[term] = {}
            
            for doc_id, frequency in id_to_frequency.items():
                term_relevance[term][doc_id] = frequency * idf_scores[term]

        return term_relevance

    def distance(self, dict_a: dict[int, float], dict_b: dict[int, float]) -> float:
        """
        Computes the Euclidean distance between two PageRank dictionaries
        Only to be called by compute_page_rank

        DO NOT MODIFY THIS METHOD.

        Parameters:
            dict_a          the first dictionary
            dict_b          the second dictionary

        Returns:
            the Euclidean distance between dict_a and dict_b
        """
        # Assures that the two dictionaries have the same number of keys
        assert len(dict_a) == len(dict_b)

        curr_dist = 0
        for page in dict_a:
            # Assures that any key in dict_a is also a key in dict_b
            assert page in dict_b
            curr_dist += (dict_a[page] - dict_b[page]) ** 2

        return math.sqrt(curr_dist)

    def compute_weights(self) -> dict[int, dict[int, float]]:
        """
        Computes the weights matrix for PageRank

        Assumes parse has already been called to populate the relevant data
        structures.

        DO NOT MODIFY THIS METHOD.

        Returns
            a dictionary from page ids to a dictionaries from page ids to
            weights such that:
            weights[id_1][id_2] = weight of edge from page with id_1 to page
                                  with id_2
        """
        
        weights = {}
        num_pages = len(self.ids_to_titles)

        for j in self.ids_to_titles:
            weights[j] = {}
            num_links = num_pages - \
                1 if j not in self.ids_to_links else len(
                    self.ids_to_links[j])

            if j in self.ids_to_links and j in self.ids_to_links[j]:
                num_links = num_links - 1 if num_links > 1 else num_pages - 1

            for k in self.ids_to_titles:
                if j == k:
                    # page links to itself
                    weights[j][k] = self.EPSILON / num_pages
                elif j in self.ids_to_links and k in self.ids_to_links[j]:
                    # this page links to that page
                    weights[j][k] = (self.EPSILON / num_pages) + \
                        ((1 - self.EPSILON) / num_links)
                elif num_links == num_pages - 1:
                    weights[j][k] = (self.EPSILON / num_pages) + \
                        ((1 - self.EPSILON) / num_links)
                else:
                    weights[j][k] = self.EPSILON / num_pages

        return weights

    def compute_page_rank(self) -> dict[int, float]:
        """
        Computes PageRank for every page and fills the page_rank map

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            A dict mapping a page id to its authority, as computed by the
            PageRank algorithm
        """

        '''
        pseudocode for PageRank:
        pageRank():
            initialize every rank in r to be 0
            initialize every rank in r' to be 1/n
            while distance(r, r') > delta:
                r <- r'
                for j in pages:
                    r'(j) = sum of weight(k, j) * r(k) for all pages k

        Use self.DISTANCE_THRESHOLD for delta
        '''

        rank = {}
        rank_prime = {}

        weights = self.compute_weights()
        for id in self.ids_to_titles.keys():
            rank[id] = 0
            rank_prime[id] = 1/len(self.ids_to_titles.keys())


        while self.distance(rank, rank_prime) > self.DISTANCE_THRESHOLD:
            rank = rank_prime.copy()
            for j in self.ids_to_titles.keys():
                rank_prime[j] = sum(weights[k][j] * rank[k] for k in rank.keys())
        return rank_prime

if __name__ == "__main__":
    if len(sys.argv) == 5:
        the_indexer = Indexer(*sys.argv[1:])
        the_indexer.run()
    else:
        print("Incorrect arguments: use <wiki> <titles> <documents> <words>")