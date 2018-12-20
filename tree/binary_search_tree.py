class TreeNode:
    def __init__(self, data=None):
        self.pre = None
        self.left = None
        self.right = None
        self.data = data

class BBTree(object):
    def __init__(self):
        self.root = None

    def build_tree(self):
        while True:
            data = input("please input the data that you want to insert to the tree:")
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
                        if data > tmp.data and tmp.right is None:
                            tmp.right = node
                            node.pre = tmp
                            break
                        elif data > tmp.data and tmp.right is not None:
                            tmp = tmp.right
                            continue
                        elif data < tmp.data and tmp.left is None:
                            tmp.left = node
                            node.pre = tmp
                            break
                        elif data < tmp.data and tmp.left is not None:
                            tmp = tmp.left
                            continue

    def mid_order(self, root):
        if root is None:
            return
        self.mid_order(root.left)
        print(root.data, end=" ")
        self.mid_order(root.right)

if __name__ == '__main__':
    tree = BBTree()
    tree.build_tree()
    tree.mid_order(tree.root)

