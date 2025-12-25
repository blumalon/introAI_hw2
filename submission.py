from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    w1 = 10.0
    w2 = 5.0
    w3 = 8.0
    current_steps_remaining = env.num_steps
    estimated_total_steps = 200 
    progress = 1.0 - (current_steps_remaining / estimated_total_steps)
    early_phase = 1.0 - progress
    def evaluate_robot_state(r):
        score_val = r.credit
        
        # Battery Logic
        if r.battery < 6:
            battery_val = (r.battery - 10) * 1.5 
        elif r.battery >= current_steps_remaining:
            battery_val = 0 
        else:
            battery_val = r.battery * 0.5 * early_phase

        # Task Logic
        task_val = 0
        center = (2, 2) 
        if r.package is not None:
            dist_dest = manhattan_distance(r.position, r.package.destination)
            reward = manhattan_distance(r.package.position, r.package.destination) * 2
            dist_center = manhattan_distance(r.package.destination, center)
            center_penalty = dist_center * 0.1
            
            task_val = (reward - dist_dest) * (1.0 - center_penalty)
            
        else:
            available_packages = [p for p in env.packages if p.on_board]
            best_pkg_val = ("-inf")
            for p in available_packages:
                dist_to_pkg = manhattan_distance(r.position, p.position)
                dist_to_dest = manhattan_distance(p.position, p.destination)
                reward = dist_to_dest * 2
                dist_center = manhattan_distance(p.destination, center)
                center_penalty = dist_center * 0.1
                val = (reward - (dist_to_pkg + dist_to_dest)) * (1.0 - center_penalty)
                if val > best_pkg_val:
                    best_pkg_val = val
            
            task_val = best_pkg_val * early_phase
        return (score_val * w1) + (battery_val * w2) + (task_val * w3)

    my_utility = evaluate_robot_state(robot)
    opp_utility = evaluate_robot_state(other_robot)
    return my_utility - opp_utility


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)