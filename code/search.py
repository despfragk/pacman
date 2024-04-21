# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a successor to the current state,
        'action' is the action required to get there, and 'stepCost' is the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """ actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        return len(actions)


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



import random


def depthFirstSearch(problem, limit=10000, randomize=False, depth_limit=100):
    visited = set()
    stack = [(problem.getStartState(), [])]

    while stack:
        current_state, current_path = stack.pop()

        if len(current_path) > limit:
            continue

        if problem.isGoalState(current_state):
            return current_path

        if current_state not in visited:
            visited.add(current_state)

            successors = problem.getSuccessors(current_state)

            if randomize:  # Randomize successors if required
                random.shuffle(successors)

            for successor_info in successors:
                successor = successor_info[0]  # Extracting successor state
                action = successor_info[1]  # Extracting action taken

                updated_path = current_path + [action]  # Update the path with the new action
                stack.append((successor, updated_path))

    return []


from queue import Queue  # Import Queue from the standard library

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # Initialize an empty queue for the frontier and a set for explored nodes
    frontier = Queue()
    explored = set()

    # Add the start state to the frontier with an empty path and a unique identifier
    start_state = problem.getStartState()
    start_id = hash(start_state)  # Use a hash of the state as a unique identifier
    frontier.put((start_state, [], start_id))

    while not frontier.empty():
        # Dequeue a node from the frontier
        node, path, node_id = frontier.get()

        # If this node is the goal, then we have a solution
        if problem.isGoalState(node):
            return path

        # Mark the node as explored
        if node_id not in explored:
            explored.add(node_id)

            # Add the successors of the node to the frontier with unique identifiers
            for successor, action, _ in problem.getSuccessors(node):
                successor_id = hash(successor)  # Use a hash of the state as a unique identifier
                new_path = path + [action]
                frontier.put((successor, new_path, successor_id))

    # If no solution was found, return an empty path
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem. This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    search_queue = PriorityQueue()
    initial_state = problem.getStartState()
    search_queue.push((initial_state, [], 0), 0)

    visited_states = set()

    while not search_queue.isEmpty():
        current_node, current_actions, total_cost_so_far = search_queue.pop()

        current_state = current_node
        if current_state in visited_states:
            continue

        visited_states.add(current_state)

        if problem.isGoalState(current_state):
            return current_actions

        for next_state, action, step_cost in problem.getSuccessors(current_state):
            new_cost = total_cost_so_far + step_cost
            new_actions = current_actions + [action]
            priority = new_cost + heuristic(next_state, problem) if heuristic else new_cost
            search_queue.push((next_state, new_actions, new_cost), priority)

    return []




def uniformCostSearch(problem):

    from util import PriorityQueue

    # Initialize the priority queue with the start state
    frontier = PriorityQueue()
    frontier.push(problem.getStartState(), 0)

    # Initialize an empty set to store visited nodes
    visited = set()

    # Initialize an empty dictionary to store actions and costs
    actions = {problem.getStartState(): []}
    costs = {problem.getStartState(): 0}

    while not frontier.isEmpty():
        # Pop the node of least total cost
        node = frontier.pop()

        # If this node is the goal, then we have found a solution
        if problem.isGoalState(node):
            return actions[node]

        # Mark the node as visited
        visited.add(node)

        # Expand the node
        for successor, action, stepCost in problem.getSuccessors(node):
            # If the successor node has not been visited yet
            if successor not in visited:
                # Calculate the cost to reach the successor
                newCost = costs[node] + stepCost

                # If the successor node has not been reached yet or if a cheaper path to this successor has been found
                if successor not in costs or newCost < costs[successor]:
                    # Update the cost of the successor
                    costs[successor] = newCost

                    # Update the actions of the successor
                    actions[successor] = actions[node] + [action]

                    # Push the successor into the frontier with its new cost
                    frontier.push(successor, newCost)

    # If no solution is found, return an empty list
    return []






# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
