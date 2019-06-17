import numpy as np


class SumTree:

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2*self.capacity-1)
        self.position = 0

    def _retrieve(self, idx, s):
        """

        :param idx:
        :param s: value sampled
        :return:
        """
        left = 2 * idx + 1
        right = 2 * idx + 2
        if idx > self.capacity-2:
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]

    def _propagate(self, idx, value):
        # Father node
        parent = (idx - 1) // 2
        self.tree[parent] += value
        if parent != 0:
            return self._propagate(parent, value)

    def update(self, idx, value):
        change = value - self.tree[idx]
        self.tree[idx] = value
        return self._propagate(idx, change)

    def add(self, value):
        idx = self.position + self.capacity - 1
        self.update(idx, value)
        self.position += 1
        if self.position > self.capacity - 1:
            self.position = 0

    def total(self):
        return self.tree[0]

tree = SumTree(capacity=8)
