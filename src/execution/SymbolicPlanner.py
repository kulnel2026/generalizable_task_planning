from Model import Model

class SymbolicPlanner(Model):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SymbolicPlanner, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, model) -> None:
        super().__init__(model)
    
    def get_scene_predicates(self, o, num_blocks):
        """
        Get scene predicates
        - o: Point cloud observation
        - num_blocks: Number of blocks
        """
        return []