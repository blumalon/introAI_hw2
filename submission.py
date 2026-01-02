from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    print("value robot\n")
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    def value_robot(env: WarehouseEnv, robot):
        print("value robot\n")
        min_turns_left = (env.num_steps + 1)/2
        # Weights
        w_score = 200.0
        w_task = 70.0
        w_batt = 15.0
        #TASK
        best_task = 100
        if robot.package != None:
            dist_to_award = manhattan_distance(robot.package.destination, robot.position) + 1
            task_val = best_task - dist_to_award + 50 #50 is bonus for picking up
        else:
            min_dist_package = math.inf
            best_pckg = None
            for pckg in env.packages:
                if (pckg.on_board == True):
                    dist_to_pck = manhattan_distance(pckg.position, robot.position)
                    if (dist_to_pck < min_dist_package):
                        min_dist_package = dist_to_pck
                        best_pckg = pckg
            if (best_pckg != None):
                task_val = best_task - manhattan_distance(best_pckg.destination, robot.position) - min_dist_package
            else:
                task_val = 0

        #battery&score
        charge_st = env.charge_stations
        if charge_st != None:
            min_distance_charge = math.inf
            for station in charge_st:
                if manhattan_distance(robot.position, station.position) < min_distance_charge:
                    min_distance_charge = manhattan_distance(robot.position, station.position)
        else:
            min_distance_charge = 0
        if (robot.battery >= min_turns_left):
            final_score = (robot.credit * w_score) + \
                          (robot.battery * w_batt) + \
                          (task_val * w_task)
            print("credit= " + str(robot.credit) + " battery= " + str(robot.battery) + " task= " + str(
                task_val))
            return final_score
        end_game_stall = 0
        if robot.battery < min_distance_charge + 2:
            if robot.credit <= 0:
                end_game_stall = -50000.0
            else:
                end_game_stall = 1000.0 * (robot.battery - min_distance_charge - 2)
        final_score = (robot.credit * w_score) + \
                      (robot.battery * w_batt) + \
                      (task_val * w_task) + \
                      end_game_stall
        print ("credit= "+str(robot.credit)+" battery= "+str(robot.battery)+" task= "+str(task_val)+ "end_game: "+str(end_game_stall))
        return final_score
    return value_robot(env, robot) - value_robot(env, other_robot)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_state_minimax(self, env: WarehouseEnv, agent_id, am_i_max: bool,
                          depth: int, best_val, start_time, run_limit):
        if (time.time() - start_time) > run_limit:
            raise TimeoutError
        if depth == 0:
            val = smart_heuristic(env, agent_id)
            if not am_i_max:
                val = -val
            return (None, val)
        if am_i_max:
            current_best = (None, -math.inf)
            for op in env.get_legal_operators(agent_id):
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
                child_env = env.clone()
                child_env.apply_operator(agent_id, op)

                child_res = self.run_state_minimax(child_env, (agent_id + 1) % 2,
                                                True, depth - 1, (None, -math.inf), start_time, run_limit)

                if child_res[1] < current_best[1]:
                    current_best = (op, child_res[1])
            return current_best
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        run_limit = time_limit - 0.1
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
    def run_state_expectimax(self, env: WarehouseEnv, am_i_expectimax: bool,depth, best_move, agent_id, start_time, run_limit):
        if (time.time() - start_time) > run_limit:
            raise TimeoutError
        if (depth == 1):
            if (am_i_expectimax == False):
                move_vector = env.get_legal_operators(agent_id)
                found_left = 0
                found_pick_package = 0
                move_values = {}
                for op in move_vector:
                    child_env = env.clone()
                    child_env.apply_operator(agent_id, op)
                    move_values[op] = smart_heuristic(child_env, agent_id)
                operator_num = len(move_vector)
                state_val = 0
                for op in move_vector:
                    if (op == 'move west'):
                        found_left += 1
                        state_val += move_values[op] * 3
                    elif (op == "pick up"):
                        found_pick_package += 1
                        state_val += move_values[op] * 3
                    else:
                        state_val += move_values[op]
                state_val = state_val / (operator_num + (2 * (found_left + found_pick_package)))
                best_move = (None, -state_val) #gets a '-'state_val to assure one lined num axis
                return best_move
            else:
                move_vector = env.get_legal_operators(agent_id)
                for op in move_vector:
                    child_env = env.clone()
                    child_env.apply_operator(agent_id, op)
                    move_val = smart_heuristic(child_env, agent_id)
                    if (move_val[1] > best_move[1]):
                        best_move = (op, move_val[1])
                return best_move
        else: #depth is bigger than 1
            if (am_i_expectimax == False):
                move_vector = env.get_legal_operators(agent_id)
                found_left = 0
                found_pick_package = 0
                move_values = {}
                state_val = 0
                for op in move_vector:
                    child_env = env.clone()
                    child_env.apply_operator(agent_id, op)
                    move_values[op] = self.run_state_expectimax(child_env, True, depth - 1, best_move,
                                                                (1 + agent_id)%2, start_time, run_limit)
                    if (op == 'move west'):
                        found_left += 1
                        state_val += move_values[op] * 3
                    elif (op == "pick up"):
                        found_pick_package += 1
                        state_val += move_values[op] * 3
                    else:
                        state_val += move_values[op]
                operator_num = len(move_vector)
                state_val = state_val / (operator_num + (2 * (found_left + found_pick_package)))
                best_move = (None, state_val)
                return best_move
            else:
                move_vector = env.get_legal_operators(agent_id)
                for op in move_vector:
                    child_env = env.clone()
                    child_env.apply_operator(agent_id, op)
                    move_val = self.run_state_expectimax(child_env, False, depth - 1, best_move,
                                                         (1 + agent_id)%2, start_time, run_limit)
                    if (move_val[1] > best_move[1]):
                        best_move = (op, move_val[1])
                return best_move
            return best_move



    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        run_limit = time_limit - 0.1
        best_move = (None, -math.inf)
        depth = 1
        try:
            while True:
                current_best = self.run_state_expectimax(env, False, depth,
                                                      (None, -math.inf), agent_id, start_time, run_limit)
                if (current_best[1] > best_move[1]):
                    best_move = current_best
                depth = depth + 1
        except TimeoutError:
            pass
        if (best_move[0] == None):
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