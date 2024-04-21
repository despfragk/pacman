# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "An agent that goes West until it can't."
    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a supplied search problem,
    then returns actions to follow that path. As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)
    Options for fn include: depthFirstSearch or dfs breadthFirstSearch or bfs
    Note: You should NOT change any code in SearchAgent
    """
    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems
        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)
            # Get the search problem type from the name
            if prob not in globals().keys() or not prob.endswith('Problem'):
                raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
            self.searchType = globals()[prob]
            print('[SearchAgent] using problem type ' + prob)

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
# Add this class to your searchAgents.py file
class CornersProblem(search.SearchProblem):
    def __init__(self, startingGameState):
        self.startingGameState = startingGameState
        self.startingPosition = startingGameState.getPacmanPosition()
        self.corners = [(1, 1), (1, startingGameState.data.layout.height - 2), (startingGameState.data.layout.width - 2, 1), (startingGameState.data.layout.width - 2, startingGameState.data.layout.height - 2)]
        self.startState = (self.startingPosition, tuple(self.corners))
        self.walls = startingGameState.getWalls()
        self._expanded = 0
    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return len(state[1]) == 0


    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.startingGameState.getWalls()[nextx][nexty]
            if not hitsWall:
                nextPosition = (nextx, nexty)
                remainingCorners = list(state[1])
                if nextPosition in remainingCorners:
                    remainingCorners.remove(nextPosition)
                nextState = (nextPosition, tuple(remainingCorners))
                cost = 1
                successors.append((nextState, action, cost))
        return successors

    def getCostOfActions(self, actions):
        # If there are no actions, return a high cost
        if actions == None: return 999999
        x, y = self.startingPosition
        cost = 0
        corners_left = self.corners[:]

        # Iterate over each action in the list
        for action in actions:
            # Compute the directional changes based on the action
            dx, dy = Actions.directionToVector(action)

            # Update the current position using the computed changes
            new_x, new_y = int(x + dx), int(y + dy)

            # Check if the updated position is among the remaining corners
            if (new_x, new_y) in corners_left:
                # Increment the cost and remove the corner from the list
                total_cost += 1
                corners_left.remove((new_x, new_y))

        return cost
def cornersHeuristic(state, problem):

    #non trivial -Maximum Manhattan Distance
    pacman_position, corners = state
    unvisited_corners = [corner for corner in corners if corner not in pacman_position]

    if not unvisited_corners:
        return 0

    max_distance = -1
    for corner in unvisited_corners:
        distance = util.manhattanDistance(pacman_position, corner)
        max_distance = max(max_distance, distance)

    return max_distance


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

from typing import List, Tuple
from util import manhattanDistance
from searchAgents import FoodSearchProblem

def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    position, foodGrid = state
    foodList = foodGrid.asList()
    if len(foodList) == 0:
        return 0

    # Create a list of all distances between all pairs of food dots
    distances = []
    for i in range(len(foodList)):
        for j in range(i+1, len(foodList)):
            distances.append((i, j, manhattanDistance(foodList[i], foodList[j])))

    # Sort the distances in ascending order
    distances.sort(key=lambda x: x[2])

    # Create a list to store the edges of the MST
    mst = []
    # Create a list to store the sets of connected nodes
    sets = [{i} for i in range(len(foodList))]

    # Kruskal's algorithm
    for i, j, d in distances:
        for s in sets:
            if i in s:
                set_i = s
            if j in s:
                set_j = s
        if set_i != set_j:
            mst.append((i, j, d))
            sets.remove(set_i)
            sets.remove(set_j)
            sets.append(set_i.union(set_j))

    # The heuristic is the sum of the distances in the MST
    # plus the distance from the current position to the nearest food dot
    heuristic = sum(d for i, j, d in mst)
    heuristic += min(manhattanDistance(position, food) for food in foodList)

    return heuristic

class ClosestDotSearchAgent(SearchAgent):
        def findPathToClosestDot(self, gameState: pacman.GameState):
            startPosition = gameState.getPacmanPosition()
            food = gameState.getFood()
            walls = gameState.getWalls()
            problem = AnyFoodSearchProblem(gameState)

            closed = set()
            fringe = util.PriorityQueue()
            priority = 0  # or some function to calculate priority
            fringe.push([(startPosition, 0)], priority)

            while not fringe.isEmpty():
                path = fringe.pop()
                pos, cost = path[-1]

                if problem.isGoalState(pos):
                    return [action for (pos, action) in path][1:]

                if pos not in closed:
                    closed.add(pos)
                    for successor, action, stepCost in problem.getSuccessors(pos):
                        if successor not in closed:
                            new_cost = cost + stepCost
                            new_path = path + [(successor, new_cost)]
                            fringe.push(new_path, new_cost)

            return []

class AnyFoodSearchProblem(PositionSearchProblem):
    def __init__(self, gameState):
        "Stores information from the gameState. You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The state is Pacman's position. Fill this in with a goal test that will complete the problem definition.
        """
        x,y = state
        return self.food[x][y]

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
