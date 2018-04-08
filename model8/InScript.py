import os
import bs4
import collections
import dill
from configurations import Config

CORPUS_FOLDER = r'D:\OneDrive\PythonWorkspace\story_generation_1\model1\InScript_LREC2016\InScript\corpus'
path_test = r'D:\OneDrive\PythonWorkspace\InScript_Models\corpus\bath\bath_009.xml'


class Story(object):
    """
    load story and labels
    .path: the path
    .tokens: list of lists, each having [token, id, tag1, tag2 ...]
    """
    def __init__(self, path):
        # extract tokens, and their event tags.
        self.path = path
        soup = bs4.BeautifulSoup(open(path, 'r', encoding='utf-8'), "lxml")
        token_tags = soup.find_all('token')
        tokens = list()
        for token_tag in token_tags:
            tokens.append([token_tag['content'], token_tag['id']])
        label_tags = soup.find_all('label')
        event_labels = list()
        for label_tag in label_tags:
            if label_tag['type'] == 'event':
                if label_tag.has_attr('to'):
                    event_labels.append([label_tag['from'], label_tag['name'], label_tag['to']])
                else:
                    event_labels.append([label_tag['from'], label_tag['name']])
        self.tokens = tokens
        for event_label in event_labels:
            for i, token in enumerate(self.tokens):
                if token[1] == event_label[0]:
                    if len(event_label) == 3:
                        span = Story.span(event_label[0], event_label[2])
                        for j in range(span):
                            self.tokens[i + j].append(event_label[1])
                    else:
                        token.append(event_label[1])

    @property
    def text(self):
        text = ''
        for token in self.tokens:
            text += token[0] + ' '
        return text

    '''
    evaluate the span of an event label
    '''
    @staticmethod
    def span(from_pos, to_pos):
        return int(to_pos.split('-')[1]) - int(from_pos.split('-')[1]) + 1


class Corpus(object):
    def __init__(self, path, script_list=list(['bath'])):
        self.SCRIPTS_TO_LOAD = script_list
        self.scripts = list()
        for script in self.SCRIPTS_TO_LOAD:
            script_folder = os.path.join(path, script)
            stories = list()
            for file in os.listdir(script_folder):
                stories.append(Story(os.path.join(script_folder, file)))
            self.scripts.append(stories)

    def export(self, folder):
        with open(os.path.join(folder, 'corpus'), 'w') as fout:
            for story_list in self.scripts:
                for story in story_list:
                    for token in story.tokens:
                        for i, piece in enumerate(token):
                            if i != 1:
                                fout.write(piece + ' ')
                        fout.write('\n')
                    fout.write('<END_OF_STORY>\n')

    @property
    def raw_text(self):
        text = ''
        for script in self.scripts:
            for story in script:
                text += story.text
        return text

    # load exported files into a list
    @staticmethod
    def load(file='corpus'):
        data = list()
        with open(file, 'r') as fin:
            inline = fin.readline()
            while inline != '':
                data.append(inline.split())
                inline = fin.readline()
        return data


if __name__ == '__main__':
    s = Story(path_test)
    c = Corpus(CORPUS_FOLDER, script_list=Config.All_Scripts)
    c.export(r'.')

    #dataa = Corpus.load()
    #pass
