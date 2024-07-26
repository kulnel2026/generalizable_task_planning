from Predicate import Predicate
class Skill:
    def __init__(self, name, logical_preconditions, logical_effects, visuomotor_policy, termination_condition, num_args):
        self.name = name
        self.logical_preconditions = logical_preconditions
        self.logical_effects = logical_effects
        self.visuomotor_policy = visuomotor_policy
        self.termination_condition = termination_condition
        self.num_args = num_args

    def can_apply(self, current_predicates):
        """
        Check if the preconditions for this skill are met in the current state.
        """
        needed = set([Predicate(pre, [], False) for pre in self.logical_preconditions])
        return needed.issubset(current_predicates)

    def apply(self, current_predicates):
        """
        Apply the effects of this skill to the current predicates, creating a new state.
        """
        new_state = set(current_predicates)
        for effect in self.logical_effects:
            # Adding effect if it's positive or toggling state if the predicate exists.
            new_predicate = Predicate(effect, [], True)
            if new_predicate in new_state:
                new_state.remove(new_predicate)
            else:
                new_state.add(new_predicate)
        return new_state
