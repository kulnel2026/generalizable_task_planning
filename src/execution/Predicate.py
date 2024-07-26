class Predicate:
    def __init__(self, predicate_string, arguments, condition) -> None:
        self.predicate_string = predicate_string
        self.arguments = arguments
        self.condition = condition
    
    def __bool__(self) -> bool:
        return self.condition
    
    def __str__(self) -> str:
        return f"{self.predicate_string}({', '.join(self.arguments)}): {self.condition}"
    
    def __eq__(self, other) -> bool:
        return self.predicate_string == other.predicate_string and self.arguments == other.arguments

    def __hash__(self):
        return hash((self.predicate_string, tuple(self.arguments), self.condition))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return (self.predicate_string, self.arguments, self.condition) == (other.predicate_string, other.arguments, other.condition)
