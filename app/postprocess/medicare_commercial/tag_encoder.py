import pickle

class TagEncoder(object):
    def __init__(self, tags):
        tags = list(tags)
        tags.sort()
        self.tag2index = {tags[i]:i for i in range(len(tags))}
        self.index2tag = {value:key for key, value in self.tag2index.items()}
        self.num_tags = len(self.tag2index)
    def encode(self, tag):
        return self.tag2index[tag]
    def decode(self, index):
        return self.index2tag[index]
    def __getstate__(self):
        return {
            "tag2index":self.tag2index
        }
    def __setstate__(self, state):
        self.tag2index = state["tag2index"]
        self.index2tag = {value:key for key, value in self.tag2index.items()}
        self.num_tags = len(self.tag2index)

def load_tag_encoder(path):
    with open(path, "rb") as input_file:
        return pickle.load(input_file)
