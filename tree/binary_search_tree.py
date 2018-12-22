class TreeNode:
    def __init__(self, value=None):
        self.parent = None
        self.left = None
        self.right = None
        self.value = value

class BSTree(object):
    def __init__(self):
        self.root = None

    def build_tree(self):
        while True:
            data = input("please input the value that you want to insert to the tree:")
            if data == "000":
                print("Finishing building a binary search tree!")
                break
            else:
                node = TreeNode(data)
                if self.root is None:
                    self.root = node
                else:
                    tmp = self.root
                    while True:
                        if data > tmp.value and tmp.right is None:
                            tmp.right = node
                            node.parent = tmp
                            break
                        elif data > tmp.value and tmp.right is not None:
                            tmp = tmp.right
                            continue
                        elif data < tmp.value and tmp.left is None:
                            tmp.left = node
                            node.parent = tmp
                            break
                        elif data < tmp.value and tmp.left is not None:
                            tmp = tmp.left
                            continue
                        else:
                            print("Data is already in tree!")

    def mid_order(self, root):
        if root is None:
            return
        self.mid_order(root.left)
        print(root.data, end=" ")
        self.mid_order(root.right)

    def get_maximum_depth(self, root):
        return 0 if root is None else max(self.get_maximum_depth(root.left), self.get_maximum_depth(root.right)) + 1

    def get_minimum_depth(self, root):
        if root is None:return 0
        if root.left is None and root.right is None:return 1
        if root.left is not None and root.right is None: return self.get_minimum_depth(root.left) + 1
        if root.left is None and root.right is not None: return self.get_minimum_depth(root.right) + 1
        return min(self.get_minimum_depth(root.left), self.get_minimum_depth(root.right)) + 1

# if __name__ == '__main__':
#     tree = BSTree()
#     tree.build_tree()
#     tree.mid_order(tree.root)
#     print('\n',tree.get_maximum_depth(tree.root))
#     print('\n',tree.get_minimum_depth(tree.root))


