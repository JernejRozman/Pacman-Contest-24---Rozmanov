# baseline_team.py
# ---------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.

import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Create a balanced team: one offensive and one defensive agent.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions.
    We’ll incorporate deterministic tie-breaking in choose_action.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        Now includes deterministic tie-breaking.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Deterministic tie-breaking: choose the action leading to the shortest distance
        # back to a "safe" zone (like home) or the lexicographically smallest action if equal.
        if len(best_actions) > 1:
            # Consider distance to start if on offense carrying food, or just break ties lexicographically.
            # This ensures consistency in tie scenarios.
            best_distance = float('inf')
            chosen_action = None
            for a in sorted(best_actions):  # sort actions to have deterministic tie-break
                successor = self.get_successor(game_state, a)
                pos = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos)
                if dist < best_distance:
                    best_distance = dist
                    chosen_action = a
            return chosen_action

        return best_actions[0]

    def get_successor(self, game_state, action):
        """
        Finds the next state successor which is a grid position.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Compute a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Default features for a reflex agent.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Offensive agent:
    - If carrying ≥ 3 food, prioritize returning home.
    - Avoid enemy defenders if they are within 3 units.
    """

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)

        # If carrying ≥ 3 food, attempt to return to home territory
        if my_state.num_carrying >= 3 and my_state.is_pacman:
            # Prioritize moving towards home
            best_action = None
            best_distance = float("inf")
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_pos = successor.get_agent_position(self.index)
                # If this move gets us closer to home (or onto home side), prefer it
                if not successor.get_agent_state(self.index).is_pacman:
                    dist = self.get_maze_distance(self.start, successor_pos)
                    if dist < best_distance:
                        best_action = action
                        best_distance = dist
            if best_action is not None:
                return best_action

        # Otherwise, proceed with the normal evaluation
        return super().choose_action(game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Food-related features
        features['successor_score'] = -len(food_list)  # fewer food = better
        if len(food_list) > 0:
            min_food_dist = min(self.get_maze_distance(my_pos, f) for f in food_list)
            features['distance_to_food'] = min_food_dist

        # Distance to home
        home_x = self.start[0]
        # Determine which side is home: Red agents on left, Blue on right.
        # We'll just measure distance to the boundary line of our side.
        if self.red:
            # home boundary: positions < mid-line
            mid = (game_state.data.layout.width // 2) - 1
        else:
            mid = game_state.data.layout.width // 2
        # Find distance to a safe "home" position
        home_positions = [(mid, y) for y in range(game_state.data.layout.height)
                          if not game_state.has_wall(mid, y)]
        home_distance = min(self.get_maze_distance(my_pos, h) for h in home_positions)
        features['distance_to_home'] = home_distance

        # Enemy defenders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        if len(defenders) > 0:
            min_defender_dist = min(self.get_maze_distance(my_pos, d.get_position()) for d in defenders)
            features['defender_distance'] = min_defender_dist
            # If too close to a defender, mark danger
            if min_defender_dist < 3:
                features['danger_zone'] = 1

        # Food carrying
        features['food_carrying'] = my_state.num_carrying

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 200,         # Encourage states with fewer food (more eaten)
            'distance_to_food': -3,         # Closer to food is better
            'distance_to_home': -1,         # Returning home more easily is good when loaded
            'defender_distance': 5,         # Prefer staying away from defenders
            'danger_zone': -100,            # Strong penalty if too close to enemy defender
            'food_carrying': -10            # Slight push to deposit food when carrying a lot
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Defensive agent:
    - Patrol its own territory
    - Penalize presence of invaders heavily
    - Stay near the center line if no invaders are visible
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Determine if we're on defense
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Distance to invaders
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
            features['invader_distance'] = min(dists)
        else:
            # No invaders detected: encourage staying near center line
            # Identify center line
            width = game_state.data.layout.width
            if self.red:
                center_line = (width // 2)
            else:
                center_line = (width // 2) - 1

            # Distance from center line in terms of x-coordinate
            # We'll measure how far we are from the "front lines" where enemy can cross.
            distance_from_center = abs(my_pos[0] - center_line)
            features['distance_from_center'] = distance_from_center

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        # Add a penalty for drifting too far from the center when there are no invaders.
        # This helps keep the defensive agent from going all the way back to spawn.
        return {
            'num_invaders': -1000,     # Huge penalty for invaders present
            'on_defense': 100,         # Prefer staying on defense
            'invader_distance': -10,   # Closer to invader is better (to chase away)
            'stop': -100,              # Don't stop unnecessarily
            'reverse': -2,             # Slight penalty for reversing direction
            'distance_from_center': -5 # Encourage staying near the center if no invaders
        }
