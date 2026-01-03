from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    # Weights
    w_score = 30.0
    w_task = 10.0  
    w_batt = 20.0   
    current_steps_remaining = env.num_steps
    estimated_total_steps = 200 
    early_phase = min(1.0, current_steps_remaining / estimated_total_steps)
    def evaluate_robot_state(r):
        score_val = r.credit
        # Battery Logic
        stations = env.charge_stations
        if stations:
            dist_to_station = min(manhattan_distance(r.position, s.position) for s in stations)
        else:
            dist_to_station = 10 
        critical_threshold = dist_to_station + 2
        if r.battery < critical_threshold and r.credit > 0:
            battery_val = (r.battery - critical_threshold - 5) * 2 
        elif r.battery >= current_steps_remaining:
            battery_val = 0 
        else:
            battery_val = r.battery * 0.5 * early_phase

        #Task Logic
        task_val = 0
        future_packages = [p for p in env.packages if not p.on_board]
        if r.package is not None:
            dist_dest = manhattan_distance(r.position, r.package.destination)
            reward = manhattan_distance(r.package.position, r.package.destination) * 2
            task_val = (reward - dist_dest) * 2.0
            
        else:
            available_packages = [p for p in env.packages if p.on_board]
            best_pkg_val = float("-inf")
            for p in available_packages:
                dist_to_pkg = manhattan_distance(r.position, p.position)
                dist_to_dest = manhattan_distance(p.position, p.destination)
                reward = dist_to_dest * 2
                future_bonus = 0
                if future_packages:
                    min_dist_future = min(manhattan_distance(p.destination, fp.position) for fp in future_packages)
                    future_bonus = (10 - min_dist_future) * 0.2
                val = reward - (dist_to_pkg + dist_to_dest) + future_bonus
                if val > best_pkg_val:
                    best_pkg_val = val
            if available_packages:
                task_val = best_pkg_val
            else:
                task_val = 0
        #print ("score val: " + str(score_val) + " battery val: " + str(battery_val) + " task val: " + str(task_val))
        return (score_val * w_score) + (battery_val * w_batt) + (task_val * w_task)

    my_utility = evaluate_robot_state(robot)
    opp_utility = evaluate_robot_state(other_robot)
    return my_utility - (opp_utility)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_state_minimax(self, env: WarehouseEnv, agent_id, am_i_max: bool,
                          depth: int, best_val, start_time, run_limit):
        if (time.time() - start_time) > run_limit:
            raise TimeoutError
        if env.done() or depth == 0:
            val = smart_heuristic(env, agent_id)
            if not am_i_max:
                val = -val
            return (None, val)

        if am_i_max:
            current_best = (None, -math.inf)
            for op in env.get_legal_operators(agent_id):
                if (time.time() - start_time) > run_limit:
                    raise TimeoutError

                child_env = env.clone()
                child_env.apply_operator(agent_id, op)

                child_res = self.run_state_minimax(child_env, (agent_id + 1) % 2,
                                                False, depth - 1, (None, math.inf), start_time, run_limit)

                if child_res[1] > current_best[1]:
                    current_best = (op, child_res[1])
            return current_best

        else:
            current_best = (None, math.inf)
            for op in env.get_legal_operators(agent_id):
                if (time.time() - start_time) > run_limit:
                    raise TimeoutError

                child_env = env.clone()
                child_env.apply_operator(agent_id, op)

                child_res = self.run_state_minimax(child_env, (agent_id + 1) % 2,
                                                True, depth - 1, (None, -math.inf), start_time, run_limit)

                if child_res[1] < current_best[1]:
                    current_best = (op, child_res[1])
            return current_best

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        run_limit = time_limit - 0.5
        best_move = None
        i = 0
        try:
            while True:
                i += 1
                current_best = self.run_state_minimax(env, agent_id, True, i,
                                                   (None, -math.inf), start_time, run_limit)
                best_move = current_best[0]
        except TimeoutError:
            pass
        if best_move is None:
            return random.choice(env.get_legal_operators(agent_id))
        else:
            return best_move


class AgentAlphaBeta(Agent):
    def run_state_alpha_beta(self, env: WarehouseEnv, agent_id, am_i_max: bool,
                             depth: int, alpha: float, beta: float, start_time, run_limit):

        if (time.time() - start_time) > run_limit:
            raise TimeoutError
        if env.done() or depth == 0:
            val = smart_heuristic(env, agent_id)
            if not am_i_max:
                val = -val
            return (None, val)

        best_op = None

        if am_i_max:
            max_eval = float('-inf')
            for op in env.get_legal_operators(agent_id):
                if (time.time() - start_time) > run_limit:
                    raise TimeoutError

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
                if (time.time() - start_time) > run_limit:
                    raise TimeoutError

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
        run_limit = time_limit - 0.5
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
    def run_state_expectimax(self, env: WarehouseEnv, am_i_max: bool, depth: int, agent_id, start_time, run_limit):
        if (time.time() - start_time) > run_limit:
            raise TimeoutError
        if env.done() or depth == 0:
            val = smart_heuristic(env, agent_id)
            if not am_i_max:
                val = -val
            return (None, val)

        if not am_i_max: 
            move_vector = env.get_legal_operators(agent_id)
            if not move_vector:
                return (None, -math.inf)

            found_left = 0
            found_pick_package = 0
            total_value = 0
            for op in move_vector:
                if op == 'move west':
                    found_left += 1
                elif op == "pick up":
                    found_pick_package += 1
            denominator = len(move_vector) + (2 * found_left) + (2 * found_pick_package)

            for op in move_vector:
                if (time.time() - start_time) > run_limit:
                    raise TimeoutError

                child_env = env.clone()
                child_env.apply_operator(agent_id, op)
                _, val = self.run_state_expectimax(child_env, True, depth - 1, (agent_id + 1) % 2, start_time, run_limit)
                
                weight = 1
                if op == 'move west':
                    weight = 3
                elif op == "pick up":
                    weight = 3
                
                total_value += val * weight

            expected_value = total_value / denominator
            return (None, expected_value)

        else:
            best_val = -math.inf
            best_op = None
            
            legal_ops = env.get_legal_operators(agent_id)
            if not legal_ops:
                return (None, -math.inf)

            for op in legal_ops:
                if (time.time() - start_time) > run_limit:
                    raise TimeoutError

                child_env = env.clone()
                child_env.apply_operator(agent_id, op)
                _, val = self.run_state_expectimax(child_env, False, depth - 1, (agent_id + 1) % 2, start_time, run_limit)
                
                if val > best_val:
                    best_val = val
                    best_op = op
                    
            return (best_op, best_val)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        run_limit = time_limit - 0.5
        best_move = (None, -math.inf)
        depth = 1
        try:
            while True:
                current_best = self.run_state_expectimax(env, True, depth, agent_id, start_time, run_limit)
                
                if current_best[0] is not None: 
                    best_move = current_best
                    
                depth += 1
        except TimeoutError:
            pass
            
        if best_move[0] is None:
            return random.choice(env.get_legal_operators(agent_id))
        return best_move[0]



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