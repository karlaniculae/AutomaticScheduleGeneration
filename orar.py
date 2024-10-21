import time
import yaml
import numpy as np
import argparse
import sys
from utils import read_yaml_file
import check_constraints
import random
import copy
from collections import defaultdict
from scheduler import Scheduler
from utils import pretty_print_timetable
from math import sqrt, log
CP = 1.0 / sqrt(2.0)
N = 'N'
Q = 'Q'
STATE = 'state'
PARENT = 'parent'
ACTIONS = 'actions'

    

def calculate_constrains(schedule, preferred_days,  preferred_intervals,  materials, score, room_capacity):
    #calculate the score for each schedule
    professors_session = defaultdict(int)
    nr_stud = defaultdict(int)
    
    for day, day_schedule in schedule.scheduler_state.items():
        for interval, interval_schedule in day_schedule.items():
            professors = defaultdict(int)
            for room, session in interval_schedule.items():

                if session:
                    prof, subj = session
                    interval_str = f"{interval[0]}-{interval[1]}"
                    if day not in preferred_days[prof]:
                        score += 10
                    if interval_str not in preferred_intervals[prof]:
                        score += 10
                    professors[prof] += 1
                    professors_session[prof] += 1
                    nr_stud[subj] += room_capacity[room]
            #add higher score for the hard constraints which are not respected
            for prof in professors:
                if professors[prof] > 1:
                    score += 3000  

    for prof in professors_session:
        if professors_session[prof] > 7:
            score += 3000  

    for subj in nr_stud:
        if nr_stud[subj] < materials[subj]:  
            score += 3000  

    return score

def compute_conflicts(schedule, preferred_days,  preferred_intervals):
    """ add for the given schedule all the professors that have constrains in a list 
    along with the day, interval and room where the constrain is not respected"""
    prof_with_constrains = []
    
    for day, day_schedule in schedule.items():
        for interval, interval_schedule in day_schedule.items():
            for room, session in interval_schedule.items():
                if session:
                           prof, subj = session
                           interval_str = f"{interval[0]}-{interval[1]}"
                         
                           if day not in preferred_days[prof]:
                                prof_with_constrains.append((day, interval, room, prof, subj))
                                
                           if interval_str not in preferred_intervals[prof]: 
                                prof_with_constrains.append((day, interval, room, prof, subj))
    return prof_with_constrains
                                                          
def get_neighbors(schedule,prof_with_constrains,room_subjects,professor_assignments):
    """start creating the neighbors for the given schedule
    create more schedules for more possible solutions"""
    
    neighbors = []
    for constraint in prof_with_constrains:
       
        day, interval, room, prof, subj = constraint
        #create a new schedule by changing just the professor with a new one 
        #which can teach the subject
        #and also has less than 7 assignments
        for candidate_prof, details in schedule.professor_assignments.items():
                if details['assignments'] < 7 and subj in details['subjects']:
                    new_schedule3 = copy.deepcopy(schedule.scheduler_state)
                    new_schedule3[day][interval][room] = (candidate_prof, subj)
                    schedule.professor_assignments[prof]['assignments'] -= 1
                    schedule.professor_assignments[candidate_prof]['assignments'] += 1
                    final_sch = Scheduler(schedule.days, schedule.intervals, schedule.materials, schedule.professors, schedule.rooms, new_schedule3)
                    neighbors.append(final_sch)
                    break  
        #make all the possible changes for each interval and create a new schedule
        for target_day, day_schedule in schedule.scheduler_state.items():
            for target_interval, interval_schedule in day_schedule.items():
                for target_room, session in interval_schedule.items():
                    if subj in room_subjects[target_room] :
                        if not (target_day == day and target_interval == interval):
                            new_schedule = copy.deepcopy(schedule.scheduler_state)
                            n=0
                            if session:
                                #create a new schedule by interchanging the the professors and subjects
                                profesor, subject = session
                                if subject in room_subjects[room]:
                                    new_schedule[target_day][target_interval][target_room] = (prof, subj)
                                    new_schedule[day][interval][room] = (profesor, subject)
                                    n=1
                            else:
                                #create a new scheduler by moving the subject to a new room which if is empty
                                new_schedule[target_day][target_interval][target_room] = (prof, subj)
                                new_schedule[day][interval][room] = None
                                n=1
                            if(n==1):
                                final_sch=Scheduler(days, intervals, materials, professors, rooms,new_schedule)
                                neighbors.append(final_sch)      
    return neighbors

def hill_climbing(scheduler, max_iters=100):
    """hill climbing algorithm for the scheduler problem"""
    iters = 0
    statex = scheduler.scheduler_state.copy()
    new_sch = Scheduler(days, intervals, materials, professors, rooms,statex)
    crt_cost = calculate_constrains(new_sch, scheduler.preferred_days,  scheduler.preferred_intervals, scheduler.materials, 0, scheduler.room_capacity)

    while iters < max_iters:
        iters += 1
        professor_with_constrains = compute_conflicts(new_sch.scheduler_state, scheduler.preferred_days,  scheduler.preferred_intervals)
        neighbours = get_neighbors(new_sch, professor_with_constrains,scheduler.room_subjects,scheduler.professor_assignments)
        best_vecin, best_cost = None, float('inf')
        #choose the best neighbour
        for neigh in neighbours:
           cost_vecin = calculate_constrains(neigh, scheduler.preferred_days,  scheduler.preferred_intervals,  scheduler.materials, 0, scheduler.room_capacity)
           if cost_vecin < best_cost:
               best_vecin = neigh
               best_cost = cost_vecin
        if best_cost >= crt_cost:
            break
        else:
            new_sch = best_vecin
            crt_cost = best_cost
    return new_sch.scheduler_state

def is_final(state):
    """check if there are no conflicts in the schedule"""
   
    conflicts = compute_conflicts(state.scheduler_state, state.preferred_days, state.preferred_intervals)
    if not conflicts:
        return True
    return False

    
def init_node(state, parent = None):
    """initialize a node with the given state and parent node"""
    return {N: 0, Q: 0, STATE: state, PARENT: parent, ACTIONS: {}}

def select_action(node, c=CP):
    """select the best action for the given node"""
    N_node = node[N] 
    best_action = None
    best_score = float('-inf')

    for action, child in node[ACTIONS].items():
        N_a = child[N]  
        Q_a = child[Q]  

        if N_a > 0:
            exploitation = Q_a / N_a
            exploration = c * sqrt(2 * log(N_node) / N_a)
            score = exploitation + exploration
        else:
            score = float('inf')  

        if score > best_score:
            best_score = score
            best_action = action

    return best_action, node[ACTIONS].get(best_action)
 
def expand_node(node):
    """expand the given node by creating all the possible neighbors"""
    neighbors = get_neighbors(node[STATE], compute_conflicts(node[STATE].scheduler_state, node[STATE].preferred_days, node[STATE].preferred_intervals), node[STATE].room_subjects, node[STATE].professor_assignments)

    for state in neighbors:
        action_key = state 
        child_node = init_node(state, node)
        node[ACTIONS][action_key] = child_node  

    return node


def simulate(state, is_final):
    """simulate the given state until it reaches a final state or a maximum number of iterations"""
    prev=None
    cont=0
    while not is_final(state):

        neighbors = get_neighbors(state,compute_conflicts(state.scheduler_state, state.preferred_days, state.preferred_intervals),state.room_subjects,state.professor_assignments)
        if not neighbors:
            break

        prev=calculate_constrains(state, state.preferred_days,  state.preferred_intervals, state.materials, 0, state.room_capacity)
        state = random.choice(neighbors)
        if prev==calculate_constrains(state, state.preferred_days,  state.preferred_intervals, state.materials, 0, state.room_capacity):
            cont+=1
        if cont==30:
            break
    return state

def backpropagate(node, reward):
    """backpropagate the reward for the given node and calculate the new Q value and N value"""
    while node:
        node['N'] += 1
        node['Q'] += reward
        node = node['parent']

def monte_carlo_tree_search(state0, budget):
    """Monte Carlo Tree Search algorithm"""

    tree = init_node(state0)
    for state in get_neighbors(state0,compute_conflicts(state0.scheduler_state, state0.preferred_days, state0.preferred_intervals),state0.room_subjects,state0.professor_assignments):
        tree['ACTIONS'] = init_node(state)
    

    for x in range(budget):
        node = tree
        path = []
        
        while not is_final(node[STATE]) and node[ACTIONS] != {}:
            best_action,node = select_action(node)
            if node is None:
                break
            path.append(node)
        if node and not is_final(node['state']):
            expand_node(node)

        final_state = simulate(node['state'], is_final)

        reward = calculate_constrains(final_state, node[STATE].preferred_days,  node[STATE].preferred_intervals,  node[STATE].materials, 0, node[STATE].room_capacity)
    
        backpropagate(node, reward)
        
    return select_action(tree, 0)
            

    
def parse_days( days):
        return np.array(days)

def parse_intervals(intervals):
    interval_list = []
    for interval in intervals:
        start, end = map(int, interval.replace('(', '').replace(')', '').split(','))
        interval_list.append((start, end))
    interval_array = np.array(interval_list, dtype=[('start', int), ('end', int)])
    return interval_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Schedule')
    parser.add_argument('algorithm', choices=['mcts', 'hc'], help='Monte Carlo Tree Search/Hill Climbing')
    parser.add_argument('file_path', type=str, help='input')
    args = parser.parse_args()
    start_time = time.time()
    data = read_yaml_file(args.file_path)
    days = parse_days(data['Zile'])
    intervals = parse_intervals(data['Intervale'])
    materials = data['Materii']
    professors = data['Profesori']
    rooms = data['Sali']

    #intialize the scheduler
    initial_scheduler = Scheduler(days, intervals, materials, professors, rooms)
    #write the output to a file
    if args.algorithm == 'hc':
        state = hill_climbing(initial_scheduler)
        end_time = time.time()
        exec_time=end_time-start_time
        if args.file_path == 'inputs/dummy.yaml':
            with open('outputs/dummy_hc.txt', 'w') as f:
                f.write(pretty_print_timetable(state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_mic_exact.yaml':
            with open('outputs/orar_mic_exact_hc.txt', 'w') as f:
                f.write(pretty_print_timetable(state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_mediu_relaxat.yaml':
            with open('outputs/orar_mediu_relaxat_hc.txt', 'w') as f:
                f.write(pretty_print_timetable(state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_mare_relaxat.yaml':
            with open('outputs/orar_mare_relaxat_hc.txt', 'w') as f:
                f.write(pretty_print_timetable(state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_constrans_incalcat.yaml':
            with open('outputs/orar_constrans_incalcat_hc.txt', 'w') as f:
                f.write(pretty_print_timetable(state, args.file_path))
                f.write("Hard constraints check result: " + str(check_constraints.check_mandatory_constraints(state, data)) + "\n")
                f.write("Soft constraints check result: " + str(check_constraints.check_optional_constraints(state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
    elif args.algorithm == 'mcts':
        state, node = monte_carlo_tree_search(initial_scheduler, 10)
        end_time = time.time()
        exec_time=end_time-start_time
        if args.file_path == 'inputs/dummy.yaml':
            with open('outputs/dummy_mcts.txt', 'w') as f:
                f.write(pretty_print_timetable(node[STATE].scheduler_state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_mic_exact.yaml':
            with open('outputs/orar_mic_exact_mcts.txt', 'w') as f:
                f.write(pretty_print_timetable(node[STATE].scheduler_state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_mediu_relaxat.yaml':
            with open('outputs/orar_mediu_relaxat_mcts.txt', 'w') as f:
                f.write(pretty_print_timetable(node[STATE].scheduler_state, args.file_path))
                f.write("Hard constraint: " + str(check_constraints.check_mandatory_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_mare_relaxat.yaml':
            with open('outputs/orar_mare_relaxat_mcts.txt', 'w') as f:
                f.write(pretty_print_timetable(node[STATE].scheduler_state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
        elif args.file_path == 'inputs/orar_constrans_incalcat.yaml':
            with open('outputs/orar_constrans_incalcat_mcts.txt', 'w') as f:
                f.write(pretty_print_timetable(node[STATE].scheduler_state, args.file_path))
                f.write("Hard constraints: " + str(check_constraints.check_mandatory_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Soft constraints: " + str(check_constraints.check_optional_constraints(node[STATE].scheduler_state, data)) + "\n")
                f.write("Execution time: " + str(exec_time) + "\n")
