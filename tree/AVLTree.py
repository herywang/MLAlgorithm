# AVL树
from __future__ import absolute_import

from tree.binary_search_tree import TreeNode
from tree.binary_search_tree import BSTree

class AVLNode(TreeNode):
    def __init__(self, value=None):
        TreeNode.__init__(self, value)
        self.height = 1

class AVLTree(BSTree):
    def __init__(self):
        super(AVLTree, self).__init__()

if __name__ == '__main__':
    node = AVLNode(10)
    print(node.value)