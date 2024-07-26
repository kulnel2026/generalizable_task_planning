from Predicate import Predicate
from Skill import Skill
import argparse
from typing import List

DEBUG = False
colors = ["red", "blue", "green", "yellow", "purple", "orange"]

def generate_predicates(num_blocks):
    """
    Create a list of predicates given a list of blocks
    """
    predicates = []
    
    # For each block determine predicates
    for i in range(num_blocks):
        on_table_predicate = Predicate(f"on-table", [colors[i]], False)
        in_hand_predicate = Predicate(f"in-hand", [colors[i]], False)
        predicates.append(on_table_predicate)
        predicates.append(in_hand_predicate)
        for j in range(num_blocks):
            if i != j:
                on_top_predicate = Predicate(f"on-top", [colors[i], colors[j]], False)
                predicates.append(on_top_predicate)
    
    return predicates


def search(predicates, goal_predicates, skills: List[Skill], observed_states):
    current_state = set(predicates)
    plan = []
    goal_state = set(goal_predicates)

    while not goal_state.issubset(current_state):
        applicable_skills = [skill for skill in skills if skill.can_apply(current_state)]

        if not applicable_skills:
            print("No applicable skills found; planning failed.")
            return []

        # Choose a skill (simple heuristic: choose first applicable)
        chosen_skill = applicable_skills[0]
        plan.append(chosen_skill)
        current_state = chosen_skill.apply(current_state)

    return plan

def plan(current_predicates: List[Predicate], goal_predicates: List[Predicate]) -> List[Skill]:
    """
    Get sequence of skills to achieve goal conditions given
    and observation
    - current_predicates: List of predicates
    - goal_predicates: Set of goal conditions (predicates)
    """ 
    if DEBUG:
        plan = ["reach-on-table(red)", "stack(red,blue)"]
        print("Plan:")
        for p in plan:
            print(f"\t{p}")
        exit(0)
    # Convert lists to sets for set operations
    current_predicates_set = set(current_predicates)
    goal_predicates_set = set(goal_predicates)

    # Check if goal predicates are already satisfied
    if goal_predicates_set.issubset(current_predicates_set):
        print("Goal predicates already satisfied")
        exit(0)
    
    # Define skills
    skills = []
    reach_on_table = Skill(
        name="reach-on-table",
        logical_preconditions=["on-table"],
        logical_effects=["in-hand"],
        visuomotor_policy=None,
        termination_condition=None,
        num_args=1
    )
    skills.append(reach_on_table)
    
    reach_on_tower = Skill(
        name="reach-on-tower",
        logical_preconditions=["on-top"],
        logical_effects=["in-hand"],
        visuomotor_policy=None,
        termination_condition=None,
        num_args=1
    )
    skills.append(reach_on_tower)
    
    stack = Skill(
        name="stack",
        logical_preconditions=["in-hand"],
        logical_effects=["on-top"],
        visuomotor_policy=None,
        termination_condition=None,
        num_args=2
    )
    skills.append(stack)
    print("Planning...")
    search(current_predicates_set, goal_predicates_set, skills, set())
    return []

def observe(predicates):
    """
    Get current point cloud observation
    """
    # TODO: Symbolic planner should return list of predicates given point cloud
    return predicates

def execute_plan(o, lg, p, max_retrials):
    """
    Execute sequence of skills given by plan (p) to achieve goal conditions (lg) given
    an observation (o)
    """
    retrial_counter = {s: 0 for s in p}
    i = 0
    while i < len(p):
        s = p[i]
        while not s.check_preconditions(o, NUM_BLOCKS): # Procondition check
            if i == 0:
                return False
            s = p[i - 1]
        retrial_counter[s] += 1 
        if retrial_counter[s] > max_retrials:
            return False
        
        # Execute skill given obersevation
        s.execute(o)
        new_o = observe()
        if s.check_termination_policy(new_o):
            return True 
        i += 1
    return False

def execute(goal_predicates, num_blocks, max_replans, max_retrials)->int:
    """
    Execution algorithm
    - goal_predicates: Set of goal conditions (predicates)
    """
    predicates = generate_predicates(num_blocks)
    print("Initial predicates:")
    for p in predicates:
        print(f"\t{p}")
    print("Goal predicates:")
    for p in goal_predicates:
        print(f"\t{p}")
    replan_counter = 0
    while replan_counter < max_replans:
        current_predicates = observe(predicates)
        P = plan(current_predicates, goal_predicates)
        replan_counter += 1
        if execute_plan(current_predicates, goal_predicates, P, max_retrials):
            print("Goal conditions achieved")
            exit(0)
    print("Max replans reached, goal conditions not achieved")
    return exit(1)



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generizable task planning")
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of blocks")
    parser.add_argument("--max_replans", type=int, default=10, help="Max replans")
    parser.add_argument("--max_retrials", type=int, default=5, help="Max retrials")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    DEBUG = args.debug
    NUM_BLOCKS = args.num_blocks
    
    goal_predicates = []
    if DEBUG:
        goal_predicates.append(Predicate("on-top", [colors[0], colors[1]], True))
    # Execute algorithm
    execute(goal_predicates, args.num_blocks, args.max_replans, args.max_retrials)
