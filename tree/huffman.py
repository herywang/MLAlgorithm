from collections import deque

class Node:
    def __init__(self, data=None, pre=None, left=None, right=None, isLeaf=False, name=None):
        self.data = data
        self.pre = pre
        self.left = left
        self.right = right
        self.isLeaf = isLeaf
        self.name = name
        self.code = ''

    def __str__(self):
        return str(self.name) + ":" + str(self.data) + "\tcode:" + self.code

class Huffman:
    def __init__(self, string):
        self.root = None
        self.string = string
        self.frequency = {}
        self.tmp_dict = []

        for ch in self.string:
            if ch in self.frequency.keys():
                self.frequency[ch] += 1
            else:
                self.frequency[ch] = 0
        min = float('inf')
        min_key = None
        print(self.frequency)
        while len(self.frequency) > 0:
            for key, value in self.frequency.items():
                if value < min:
                    min_key = key
                    min = value
            self.tmp_dict.append(Node(min, name=min_key, isLeaf=True))
            del self.frequency[min_key]
            min = float('inf')
        del self.frequency
        for i in self.tmp_dict:
            print(i)

    def encoded(self):
        while len(self.tmp_dict)>0:
            value1 = self.get_min_value()
            value2 = self.get_min_value()
            left = value1
            right = value2
            root = Node(value2.data + value1.data, None, left, right, False, None)
            left.pre = root
            right.pre = root
            self.root = root
            if len(self.tmp_dict) > 0:
                self.append_root_node(root)
            else:
                break
        self.post_order(self.root)
        tree = self.leverorder(self.root)
        for i in tree:
            print(i)

    def get_min_value(self):
        return self.tmp_dict.pop(0)

    def append_root_node(self, node):
        for i in range(len(self.tmp_dict)-1):
            if self.tmp_dict[i].data <= node.data <= self.tmp_dict[i + 1].data:
                self.tmp_dict.insert(i+1, node)
                return
        self.tmp_dict.append(node)

    def post_order(self, node):
        if node is None:
            return
        elif node.pre is not None and node.pre.left == node:
            node.code = node.pre.code + '0'
        elif node.pre is not None and node.pre.right == node:
            node.code = node.pre.code + '1'
        self.post_order(node.left)
        self.post_order(node.right)

    def leverorder(self, node):
        q = deque()
        q.append(node)
        tree_value = []
        while len(q) > 0:
            tmp_node = q.popleft()
            if tmp_node.isLeaf is True:
                tree_value.append(str(tmp_node))
            if tmp_node.left is not None:
                q.append(tmp_node.left)
            if tmp_node.right is not None:
                q.append(tmp_node.right)
        return tree_value

if __name__ == '__main__':
    tree = Huffman("abcabcabcaaabbdddeeeee")
    print(len(tree.tmp_dict))
    tree.encoded()