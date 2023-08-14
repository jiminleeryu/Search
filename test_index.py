import pytest
from index import Indexer

def setup_function():
    global my_indexer
    global simple_index
    my_indexer = Indexer("wiki1", "title1", "1", "This is the body")
    simple_index = Indexer("wikis/SimpleWiki.xml", "simple_titles.txt",
       "simple_docs.txt", "simple_words.txt")

# Here's an example test case to make sure your tests are working
# Remember that all test functions must start with "test"
def test_example():
    assert 2 == 1 + 1
    
def test_process_document():
    # Basic test for process document
    assert ["wiki1", "document", "boy", "jump"] == my_indexer.process_document("wiki1 document", 1, "The boys are jumping")

def test_compute_tf():
    simple_index.parse()

    actual = simple_index.compute_tf()
    assert actual.__len__() == 18
    assert 200 in actual["page"].keys() 
    assert 1.0 in actual["page"].values()
    assert 200, 300 in actual["page"].keys

    assert actual == {'exampl': {200: 0.6666666666666666, 30: 0.3333333333333333}, 'page': {200: 1.0, 30: 1.0}, 'content': {200: 0.3333333333333333}, 'may': {200: 0.6666666666666666}, 'contain': {200: 0.6666666666666666}, 'space': {200: 0.3333333333333333}, 'format': {200: 0.3333333333333333}, 'special': {200: 0.3333333333333333}, 'charact': {200: 0.3333333333333333}, 'regular': {200: 0.3333333333333333}, 'express': {200: 0.3333333333333333}, 'give': {200: 0.3333333333333333}, 'extract': {200: 0.3333333333333333}, 'word': {200: 0.3333333333333333}, 'number': {200: 0.3333333333333333}, 'document': {200: 0.3333333333333333}, 'link': {30: 0.3333333333333333}, 'see': {30: 0.3333333333333333}}

def test_compute_idf():
    simple_index.parse()
    actual = simple_index.compute_idf()
    assert actual['exampl'] == 0
    assert actual.__len__() == 18

    assert actual == {'exampl': 0.0, 'page': 0.0, 'content': 0.6931471805599453,\
                 'may': 0.6931471805599453, 'contain': 0.6931471805599453, \
            'space': 0.6931471805599453, 'format': 0.6931471805599453, \
                'special': 0.6931471805599453, 'charact': 0.6931471805599453, \
                    'regular': 0.6931471805599453, 'express': 0.6931471805599453, \
                        'give': 0.6931471805599453, 'extract': 0.6931471805599453, \
                            'word': 0.6931471805599453, 'number': 0.6931471805599453, \
                                'document': 0.6931471805599453, 'link': 0.6931471805599453, \
                                    'see': 0.6931471805599453}

def test_compute_rank():
    index_one = Indexer("wikis/PageRankWiki.xml", "simple_titles.txt",
       "simple_docs.txt", "simple_words.txt")
    
    index_two = Indexer("wikis/PageRankExample1.xml", "simple_titles.txt",
       "simple_docs.txt", "simple_words.txt")
    
    index_three = Indexer("wikis/PageRankExample2.xml", "simple_titles.txt",
       "simple_docs.txt", "simple_words.txt")
    
    index_one.parse()
    index_two.parse()
    index_three.parse()
    
    rank = index_one.compute_page_rank()

    assert rank[100] == pytest.approx(0.4606856899616172, abs=0.0001)

    rank = index_two.compute_page_rank()
    assert rank[1] == pytest.approx(0.4326, abs=0.0001)
    assert rank == {1: 0.4326427188659158, 2: 0.23402394780075067, 3: 0.33333333333333326}

    rank = index_three.compute_page_rank()
    assert rank == {1: 0.20184346250214996, 2: 0.03749999999999998, 3: 0.37396603749279056, 4: 0.3866905000050588}


def file_as_set(filename):
    """
    Returns all of the non-empty lines in the file, as strings in a set.
    """
    line_set = set()
    with open(filename, "r") as file:
        line = file.readline()
        while line and len(line.strip()) > 0:
            line_set.add(line.strip())
            line = file.readline()
    return line_set

def test_file_contents():
    simple_index.run() # run the indexer to write to the files
    titles_contents = file_as_set("simple_titles.txt")
    assert len(titles_contents) == 2
    assert "200::Example page" in titles_contents
    assert "30::Page with links" in titles_contents

def test_compute_term_relevance():
    simple_index = Indexer("wikis/SimpleWiki.xml", "simple_titles.txt",
       "simple_docs.txt", "simple_words.txt")
    simple_index.parse()
    actual = simple_index.compute_term_relevance()
    assert 200 in actual['exampl'].keys()
    assert 30 in actual['exampl'].keys()
    assert len(actual['exampl'].keys()) == 2

    assert actual == {'exampl': {200: 0.0, 30: 0.0}, 'page': {200: 0.0, 30: 0.0}, \
            'content': {200: 0.23104906018664842}, 'may': {200: 0.46209812037329684}, \
        'contain': {200: 0.46209812037329684}, 'space': {200: 0.23104906018664842}, \
    'format': {200: 0.23104906018664842}, 'special': {200: 0.23104906018664842}, \
'charact': {200: 0.23104906018664842}, 'regular': {200: 0.23104906018664842}, \
'express': {200: 0.23104906018664842}, 'give': {200: 0.23104906018664842}, \
'extract': {200: 0.23104906018664842}, 'word': {200: 0.23104906018664842}, \
'number': {200: 0.23104906018664842}, 'document': {200: 0.23104906018664842}, \
'link': {30: 0.23104906018664842}, 'see': {30: 0.23104906018664842}}
