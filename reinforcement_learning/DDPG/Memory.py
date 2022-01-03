import numpy as np


class SumTree:
    write = 0

    def __init__(self, capacity, dim):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros((capacity, dim))

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_index = parent_idx
                break
            else:
                if v <= self.tree[left]:
                    parent_idx = left
                else:
                    v -= self.tree[right]
                    parent_idx = right
        data_idx = leaf_index - (self.capacity - 1)
        # 优先级索引, 优先级, 采样数据
        return leaf_index, self.tree[leaf_index], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class Memory:
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, dim):
        self.sample_tree = SumTree(capacity, dim)

    def store(self, transition):
        # 刚存进来的新的transition具有较大的优先级别.
        max_p = np.max(self.sample_tree.tree[-self.sample_tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sample_tree.add(max_p, transition)

    def sample(self, n):
        # batch_index, batch_memory, important sample weight
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.sample_tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.sample_tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.sample_tree.tree[-self.sample_tree.capacity:]) / self.sample_tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.sample_tree.get_leaf(v)
            prob = p / self.sample_tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.sample_tree.update(ti, p)
