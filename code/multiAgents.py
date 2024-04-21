# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Calculate the distance to the nearest delectable byte
        foodDistances = [manhattanDistance(newPos, foodItem) for foodItem in newFood.asList()]
        closestFood = min(foodDistances) if foodDistances else 1

        # Measure the proximity to the closest spectral entity
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        closestGhost = min(ghostDistances) if ghostDistances else 1

        # If the ghost is in a haunted state, we can safely disregard it
        if newScaredTimes[0] > 0:
            closestGhost = 1

        # Avoid division by zero
        closestFood = closestFood if closestFood != 0 else 0.1
        closestGhost = closestGhost if closestGhost != 0 else 0.1

        # Return the strategic result of the student's calculations
        return successorGameState.getScore() + (1.0 / closestFood) - (1.0 / closestGhost)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """
    evaluationFunction: object

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """
    evaluationFunction: object

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


# My awesome minimax agent for Pacman
# My awesome minimax agent for Pacman
class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        # Max value helper method
        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            bestScore = -float("inf")
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = minValue(successor, 1, depth)
                bestScore = max(bestScore, score)

            return bestScore

        # Min value helper method
        # Get minimum value for state
        def minValue(state, agent, depth):

            # Check for terminal state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Start value high
            bestScore = float("inf")

            # Get actions we can take
            actions = state.getLegalActions(agent)

            # Look at results of each action
            for action in actions:

                # Get successor state for action
                successor = state.generateSuccessor(agent, action)

                # Get score based on depth
                if agent == state.getNumAgents() - 1:
                    score = maxValue(successor, depth + 1)
                else:
                    score = minValue(successor, agent + 1, depth)

                # Find minimum
                bestScore = min(bestScore, score)

            return bestScore

        bestScore = -float("inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minValue(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
# Define the AlphaBetaAgent class
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    AlphaBetaAgent is a type of MultiAgentSearchAgent. It uses alpha-beta pruning to decrease the number of nodes
    that are evaluated by the minimax algorithm in its search tree.
    """
    def getAction(self, gameState):
        """
        Returns the alpha-beta-pruned minimax action from the current gameState using self.depth and self.evaluationFunction.
        """
        # Define the maxValue function
        def maxValue(state, alpha, beta, level):
            """
            Function to calculate max value for a given state and depth.
            """
            # Base case: if game is over or depth equals the limit
            if state.isWin() or state.isLose() or level == self.depth:
                # Return the evaluation of the state
                return self.evaluationFunction(state), None
            # Initialize value to negative infinity
            v = float("-inf")
            # Iterate over all legal actions
            for action in state.getLegalActions(0):
                # Generate the successor state
                successor = state.generateSuccessor(0, action)
                # Calculate the min value for the successor state
                val, _ = minValue(successor, alpha, beta, 1, level)
                # Update value with the max of current value and min value
                v = max(v, val)
                # If value is greater than beta, prune the branch
                if v > beta:
                    return v, action
                # Update alpha
                alpha = max(alpha, v)
            return v, None

        # Define the minValue function
        def minValue(state, alpha, beta, agentIndex, level):
            """
            Function to calculate min value for a given state, agent index, and depth.
            """
            # Base case: if game is over
            if state.isWin() or state.isLose():
                # Return the evaluation of the state
                return self.evaluationFunction(state), None
            # Initialize value to positive infinity
            v = float("inf")
            # Get the number of agents
            numAgents = state.getNumAgents()
            # Iterate over all legal actions
            for action in state.getLegalActions(agentIndex):
                # Generate the successor state
                successor = state.generateSuccessor(agentIndex, action)
                # If it's the last agent's turn, calculate the max value for the successor state
                if agentIndex == numAgents - 1:
                    val, _ = maxValue(successor, alpha, beta, level + 1)
                # Otherwise, calculate the min value for the successor state
                else:
                    val, _ = minValue(successor, alpha, beta, agentIndex + 1, level)
                # Update value with the min of current value and calculated value
                if val < v:
                    v = val
                    bestAction = action
                # If value is less than alpha, prune the branch
                if v < alpha:
                    return v, bestAction
                # Update beta
                beta = min(beta, v)
            return v, bestAction

        # Initialize alpha and beta to negative and positive infinity, respectively
        alpha, beta = float("-inf"), float("inf")
        # Initialize bestScore to negative infinity and bestAction to STOP
        bestScore, bestAction = float("-inf"), Directions.STOP
        # Get all legal actions
        legalActions = gameState.getLegalActions(0)
        # Iterate over all legal actions
        for action in legalActions:
            # Generate the successor state
            successor = gameState.generateSuccessor(0, action)
            # Calculate the min value for the successor state
            score, _ = minValue(successor, alpha, beta, 1, 0)
            # Update bestScore and bestAction if score is greater than bestScore
            if score > bestScore:
                bestScore, bestAction = score, action
            # If bestScore is greater than beta, prune the branch
            if bestScore > beta:
                return bestAction
            # Update alpha
            alpha = max(alpha, bestScore)
        # Return the best action
        return bestAction
class ExpectimaxAgent(MultiAgentSearchAgent):
    # Overriding getAction method from parent class
    def getAction(self, gameState):
        # Function to calculate max value for a given state and depth
        def maxValue(state, depth):
            # Base case: if game is over or depth equals the limit
            if state.isWin() or state.isLose() or depth == self.depth:
                # Return the evaluation of the state
                return self.evaluationFunction(state), None
            # Initialize value to negative infinity
            v = float('-inf')
            # Initialize best action to None
            bestAction = None
            # Iterate over all legal actions
            for action in state.getLegalActions(0):
                # Calculate the expected value for the successor state
                val, _ = expectValue(state.generateSuccessor(0, action), 1, depth)
                # Update value with the max of current value and expected value
                if val > v:
                    v = val
                    bestAction = action
            # Return the max value and best action
            return v, bestAction

        # Function to calculate expected value for a given state, agent index, and depth
        def expectValue(state, agentIndex, depth):
            # Base case: if game is over
            if state.isWin() or state.isLose():
                # Return the evaluation of the state
                return self.evaluationFunction(state), None
            # Initialize value to 0
            v = 0
            # Get the number of legal actions for the agent
            numActions = len(state.getLegalActions(agentIndex))
            # Iterate over all legal actions
            for action in state.getLegalActions(agentIndex):
                # If it's the last agent's turn
                if agentIndex == gameState.getNumAgents() - 1:
                    # Add the max value of the successor state to the value
                    val, _ = maxValue(state.generateSuccessor(agentIndex, action), depth + 1)
                else:
                    # Add the expected value of the successor state to the value
                    val, _ = expectValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                v += val / numActions
            # Return the average value and None as best action
            return v, None

        # Call maxValue function to get the best value and action
        bestValue, bestAction = maxValue(gameState, 0)
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    # Useful information you can extract from a GameState (pacman.py)
    pacmanPosition = currentGameState.getPacmanPosition()
    activeGhosts = [ghostState for ghostState in currentGameState.getGhostStates() if ghostState.scaredTimer == 0]
    scaredGhosts = [ghostState for ghostState in currentGameState.getGhostStates() if ghostState.scaredTimer > 0]
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    # Compute distances to items
    foodDistances = [manhattanDistance(pacmanPosition, foodPos) for foodPos in food.asList()]
    activeGhostDistances = [manhattanDistance(pacmanPosition, ghostState.getPosition()) for ghostState in activeGhosts]
    scaredGhostDistances = [manhattanDistance(pacmanPosition, ghostState.getPosition()) for ghostState in scaredGhosts]

    # Initialize features to zero
    closestFoodDistance = 0
    closestActiveGhostDistance = 0
    closestScaredGhostDistance = 0

    # Compute features
    if foodDistances:
        closestFoodDistance = min(foodDistances)
    if activeGhostDistances:
        closestActiveGhostDistance = min(activeGhostDistances)
    if scaredGhostDistances:
        closestScaredGhostDistance = min(scaredGhostDistances)

    # Compute the linear combination of features
    score = currentGameState.getScore()
    score += max(1, closestActiveGhostDistance) * -1.5  # want to avoid active ghosts
    score += len(capsules) * -3  # want to eat capsules
    score += len(food.asList()) * -2  # want to eat food
    score += closestFoodDistance * -1  # want to get closer to food
    score += closestScaredGhostDistance  # want to get closer to scared ghosts

    return score

# Abbreviation
better = betterEvaluationFunction
