from abc import abstractmethod
from model import NNModel


class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class MyQAgent(QAgent):
    def __init__(self, model=None):
        super(MyQAgent, self).__init__()
        if model is None:
            self.model = NNModel()
        else:
            self.model = model

    def select_action(self, ob):
        pass
