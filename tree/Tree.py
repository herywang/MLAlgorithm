"""
data structure --- Tree
"""
from __future__ import absolute_import, print_function, division

from collections import deque

class Solution:

    def __init__(self):
        self.values = []
        self.array = None

    def preorderTraversal(self, node):
        if node is None:
            return
        else:
            self.values.extend(node.value)
            self.preorderTraversal(node.left)
            self.preorderTraversal(node.right)
    
    def midorderTreacersal(self, node):
        if node is None:
            return
        else:
            self.midorderTreacersal(node.left)
            self.values.extend(node.value)
            self.midorderTreacersal(node.right)

    def postorderTreacersal(self, node):
        if node is None:
            return
        else:
            self.postorderTreacersal(node.left)
            self.postorderTreacersal(node.right)
            self.values.extend(node.value)
    
    def leverorder(self, node):
        '''
            二叉树层序遍历， 借助队列的数据结构，先进先出的是popleft()函数
        '''
        q = deque()
        q.append(node)
        tree_value = []
        while len(q) > 0:
            tmp_node = q.popleft()
            tree_value.append(tmp_node.value)
            if tmp_node.left is not None:
                q.append(tmp_node.left)
            if tmp_node.right is not None:
                q.append(tmp_node.right)
        return tree_value

    '''
    二叉树的顺序存储
    '''
    def tree2array(self, root, len):
        self.array = [None] * len
        self.__toArrar(root, 0)

    def __toArrar(self, node, pos):
        if node is None:
            return
        self.array[pos] = node.value
        self.__toArrar(node.left, 2*pos + 1)
        self.__toArrar(node.right, 2*pos + 2)
            



class TreeNode:

    def __init__(self, val=None, left=None, right=None):
        self.value = val
        self.left = left
        self.right = right

if __name__ == '__main__':
    '''
            D
        B       E
      A   C   G
                 F
    '''
    # 构建树
    tree = TreeNode('D', 
        TreeNode('B', TreeNode('A'), TreeNode('C')), 
        TreeNode('E', TreeNode('G', right=TreeNode('F'))))
    solution = Solution()
    print('前序遍历：')
    solution.preorderTraversal(tree)
    print(solution.values)
    print('中序遍历：')
    solution.values = []
    solution.midorderTreacersal(tree)
    print(solution.values)
    print('后序遍历：')
    solution.values = []
    solution.postorderTreacersal(tree)
    print(solution.values)
    print('层序遍历：')
    values = solution.leverorder(tree)
    print(values)
    print('二叉树的顺序存储：')
    solution.tree2array(tree, 16)
    print(solution.array)