"""
data structure --- Tree
"""
from __future__ import absolute_import, print_function, division

class Solution:

    def __init__(self):
        self.values = []

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

