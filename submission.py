from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math

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
        stations = env.charge_stations
        dist_to_station = min(manhattan_distance(r.position, s.position) for s in stations)
        critical_threshold = dist_to_station + 3
        #Battery Logic
        if r.battery < critical_threshold:
            battery_val = (r.battery - critical_threshold) * 2 
        elif r.battery >= current_steps_remaining:
            battery_val = 0 
        else:
            battery_val = r.battery * 0.5 * early_phase
        #Task Logic
        task_val = 0
        center = (2, 2)
        future_packages = [p for p in env.packages if not p.on_board]
        if r.package is not None:
            dist_dest = manhattan_distance(r.position, r.package.destination)
            reward = manhattan_distance(r.package.position, r.package.destination) * 2
            dist_center = manhattan_distance(r.package.destination, center)
            center_penalty = dist_center * 0.1
            future_bonus = 0
            if future_packages:
                min_dist_future = min(manhattan_distance(r.package.destination, fp.position) for fp in future_packages)
                future_bonus = (10 - min_dist_future) * 0.1 
            
            task_val = (reward - dist_dest) * (1.0 - center_penalty) + future_bonus
            
        else:
            available_packages = [p for p in env.packages if p.on_board]
            best_pkg_val = float("-inf")
            for p in available_packages:
                dist_to_pkg = manhattan_distance(r.position, p.position)
                dist_to_dest = manhattan_distance(p.position, p.destination)
                reward = dist_to_dest * 2
                dist_center = manhattan_distance(p.destination, center)
                center_penalty = dist_center * 0.1
                future_bonus = 0
                if future_packages:
                    min_dist_future = min(manhattan_distance(p.destination, fp.position) for fp in future_packages)
                    future_bonus = (10 - min_dist_future) * 0.2
                
                val = (reward - (dist_to_pkg + dist_to_dest)) * (1.0 - center_penalty) + future_bonus
                if val > best_pkg_val:
                    best_pkg_val = val
            
            task_val = best_pkg_val
            
        return (score_val * w1) + (battery_val * w2) + (task_val * w3)

    my_utility = evaluate_robot_state(robot)
    opp_utility = evaluate_robot_state(other_robot)
    return my_utility - opp_utility


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_state_minimax(self, env: WarehouseEnv, agent_id, am_i_max: bool,
                          depth: int, best_val, start_time, run_limit):
        if (time.time() - start_time) > run_limit:
            raise TimeoutError
        if (depth == 0):
            return (None, smart_heuristic(env, agent_id))
        if (am_i_max):
            for op in env.get_legal_operators(0):
                child_env = env.clone()
                child_env.apply_operator(0, op)
                child_val = self.run_state_minimax(child_env,(agent_id+1)%2,
                                                False, depth - 1, (None, -math.inf), start_time, run_limit)
                if (child_val[1] > best_val[1]):
                    best_val = (op, child_val[1])
            return best_val
        else:
            for op in env.get_legal_operators(1):
                child_env = env.clone()
                child_env.apply_operator(agent_id, op)
                child_val = self.run_state_minimax(child_env,(agent_id+1)%2, True, depth - 1,
                                                (None, math.inf), start_time, run_limit)
                if (child_val[1] < best_val[1]):
                    best_val = (op, child_val[1])
            return best_val


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        run_limit = time_limit - 0.1 #tenth of a second as safety margin
        i = 0
        try:
            while(True):
                i += 1
                if (agent_id):
                    best_move = self.run_state_minimax(env, agent_id, (agent_id +1)%2, i,
                                                   (None, -math.inf), start_time, run_limit)
                else:
                    best_move = self.run_state_minimax(env, agent_id, (agent_id + 1) % 2, i,
                                                       (None, math.inf), start_time, run_limit)
        except TimeoutError:
            pass
        if (best_move[0] == None):
            return random.choice(env.get_legal_operators(agent_id)) #to make sure we have default operator to return if time_out
        else:
            return best_move[0]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_state_alpha_beta(self, env: WarehouseEnv, agent_id, am_i_max: bool,
                             depth: int, alpha: float, beta: float, start_time, run_limit):
        
        if (time.time() - start_time) > run_limit:
            raise TimeoutError
            
        if depth == 0:
            return (None, smart_heuristic(env, agent_id))
            
        best_op = None
        
        if am_i_max:
            max_eval = float('-inf')
            for op in env.get_legal_operators(agent_id):
                child_env = env.clone()
                child_env.apply_operator(agent_id, op)
                _, eval_val = self.run_state_alpha_beta(child_env, (agent_id + 1) % 2, 
                                                        False, depth - 1, alpha, beta, start_time, run_limit)
                
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_op = op
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break 
                    
            return (best_op, max_eval)
            
        else:
            min_eval = float('inf')
            for op in env.get_legal_operators(agent_id):
                child_env = env.clone()
                child_env.apply_operator(agent_id, op)
                _, eval_val = self.run_state_alpha_beta(child_env, (agent_id + 1) % 2, 
                                                        True, depth - 1, alpha, beta, start_time, run_limit)
                
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_op = op
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break 
            
            return (best_op, min_eval)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        run_limit = time_limit - 0.1 
        best_move = None
        i = 0
        
        try:
            while True:
                i += 1
                current_best = self.run_state_alpha_beta(env, agent_id, True, i, 
                                                         float('-inf'), float('inf'), start_time, run_limit)
                best_move = current_best[0]
                
        except TimeoutError:
            pass
            
        if best_move is None:
            return random.choice(env.get_legal_operators(agent_id))
        else:
            return best_move


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