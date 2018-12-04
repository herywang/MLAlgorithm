# 字典树
class TrieNode:
    def __init__(self):
        self.map = {}
        self.isLeaf = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        root = self.root
        for ch in word:
            if ch not in root.map:
                root.map[ch] = TrieNode()
            root = root.map[ch]
        root.isLeaf = True

    def search(self, word):
        root = self.root
        i = 0
        for ch in word:
            if ch not in root.map:
                return False
            else:
                i += 1
                if i==len(word) and root.map[ch].isLeaf:
                    return True
                root = root.map[ch]
        return False

if __name__ == '__main__':
    trie = Trie()
    trie.insert("hello")
    result = trie.search("hello")
    print(result)


