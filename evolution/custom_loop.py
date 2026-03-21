from evolution.loop import EvolutionLoop
from models.dropout_wrapper import DropoutWrapper
from models.graph_net import NBAGraphNet

class CustomEvolutionLoop(EvolutionLoop):
    def __init__(self, data_dir=None, result_dir=None):
        super(CustomEvolutionLoop, self).__init__(data_dir, result_dir)
        self.model = DropoutWrapper(NBAGraphNet(tabular_dim=10, node_dim=10, num_classes=2))
