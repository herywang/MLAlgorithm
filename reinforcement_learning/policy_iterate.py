# 策略迭代方法

class PolicyIterate(object):
    def __init__(self):
        self.v = []

    def policy_ietrate(self, grid_mdp):
        """
        :type grid_mdp: GridEnv
        """
        for i in range(1000):
            self.policy_evaluate(grid_mdp)
            self.policy_improve(grid_mdp)

    def policy_evaluate(self, grid_mdp):
        pass

    def policy_improve(self, grid_mdp):
        pass
