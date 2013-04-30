# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
 
from util import manhattanDistance
from game import Directions
import random, util
 
from game import Agent
 
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
 
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
 
 
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.
 
    getAction chooses among the best options according to the evaluation function.
 
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
 
  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.
 
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
 
    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
 
    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
     
    "*** YOUR CODE HERE ***"
    # closer to the food
    foodList = oldFood.asList();
    foodList.sort(lambda x, y: util.manhattanDistance(x, newPos) - util.manhattanDistance(y, newPos));
    foodDist = util.manhattanDistance(newPos, foodList[0]);
    # away from the ghost
    ghostPos = [x.getPosition() for x in newGhostStates];
    ghostPos.sort(lambda x, y: int(util.manhattanDistance(x, newPos) - util.manhattanDistance(y, newPos)));
    ghostDist = util.manhattanDistance(newPos, ghostPos[0]);
    if (ghostDist == 0):
      return -9999; # ghost is on this position, keep away from it. 
    if (foodDist == 0):
      # The minimum step to the food is 0.5
      return 2.0 - 1/ghostDist; 
    else:
      return 1.0/(foodDist) - 1/ghostDist;
    
    
 
def scoreEvaluationFunction(currentGameState):
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
  
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
 

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  
  def minimax(self, state, agent, depth):
      if (state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents()):
        return self.evaluationFunction(state);
      elif agent == self.index:
        return self.max_val(state, agent, depth);
      else: 
        return self.min_val(state, agent, depth);
      
  def max_val(self, state, agent, depth):
      v = float('-inf');
      actions = state.getLegalActions(agent);
      if (Directions.STOP in actions):
        actions.remove(Directions.STOP);
      for action in actions:
        v = max(v, self.minimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth + 1));
      return v
  
  def min_val(self, state, agent, depth):
      v = float('inf');
      actions = state.getLegalActions(agent);
      if (Directions.STOP in actions):
        actions.remove(Directions.STOP);
      for action in actions:
        v = min(v, self.minimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth + 1));
      return v;
    
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.
 
      Here are some method calls that might be useful when implementing minimax.
 
      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
 
      Directions.STOP:
        The stop direction, which is always legal
 
      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
 
      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    actionMap = {};
    actions = gameState.getLegalActions(self.index);
    actions.remove(Directions.STOP);
    agent = 0;
    depth = 0;
    for action in actions:
      v = self.minimax(gameState.generateSuccessor(agent, action), agent + 1, depth + 1);
      actionMap[v] = action;
    #print "value: ", max(actionMap.keys());  
    return actionMap[max(actionMap.keys())];
         
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def alphabeta(self, state, action, agent, depth, alpha, beta):
    if (state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents()):
      return [self.evaluationFunction(state), action];
    elif agent == self.index:
      return self.max_val(state, agent, depth, alpha, beta);
    else: 
      return self.min_val(state, agent, depth, alpha, beta);
     
  def max_val(self, state, agent, depth, alpha, beta):
      v = float('-inf');
      actions = state.getLegalActions(agent);
      if (Directions.STOP in actions):
        actions.remove(Directions.STOP);
      returnedAction = Directions.STOP;
      for action in actions:
        temp = self.alphabeta(state.generateSuccessor(agent, action), action, (agent + 1) % state.getNumAgents(), depth + 1, alpha, beta);
        if (temp[0] > v):
          v = temp[0];
          returnedAction = action;
        if (v >= beta):
          return [v, returnedAction];
        alpha = max(alpha, v);
      return [v, returnedAction]; 
   
  def min_val(self, state, agent, depth, alpha, beta):
      v = float('inf');
      actions = state.getLegalActions(agent);
      if (Directions.STOP in actions):
        actions.remove(Directions.STOP);
      returnedAction = Directions.STOP;
      for action in actions:
        temp = self.alphabeta(state.generateSuccessor(agent, action), action, (agent + 1) % state.getNumAgents(), depth + 1, alpha, beta);
        if (temp[0] < v):
          v = temp[0];
          returnedAction = action;
        if v <= alpha:
          return [v, returnedAction];
        beta = min(v, beta);
      return [v, returnedAction];
  
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    agent = 0;
    depth = 0;
    alpha = float('-inf');
    beta = float('inf');
    action = Directions.STOP;
    v = self.alphabeta(gameState, action, agent, depth, alpha, beta);
    print "selected v: ", v[0];
    return v[1];


 
class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def minimax(self, state, agent, depth):
      if (state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents()):
        return self.evaluationFunction(state);
      elif agent == self.index:
        return self.max_val(state, agent, depth);
      else: 
        return self.min_val(state, agent, depth);
      
  def max_val(self, state, agent, depth):
      v = float('-inf');
      actions = state.getLegalActions(agent);
      if (Directions.STOP in actions):
        actions.remove(Directions.STOP);
      for action in actions:
        v = max(v, self.minimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth + 1));
      return v
  
  def min_val(self, state, agent, depth):
      v = 0;
      actions = state.getLegalActions(agent);
      if (Directions.STOP in actions):
        actions.remove(Directions.STOP);
      for action in actions:
        v += self.minimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth + 1);
      return v/len(actions); # as choose each one randomly with 1/len(actions) chance. 
    
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
 
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    actionMap = {};
    actions = gameState.getLegalActions(self.index);
    if (Directions.STOP in actions):
      actions.remove(Directions.STOP);
    agent = 0;
    depth = 0;
    for action in actions:
      v = self.minimax(gameState.generateSuccessor(agent, action), agent + 1, depth + 1);
      actionMap[v] = action;
    #print "value: ", max(actionMap.keys());  
    return actionMap[max(actionMap.keys())];
    
 
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
 
    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  currentPos = currentGameState.getPacmanPosition()
  oldFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  foodList = oldFood.asList();
  foodList.sort(lambda x, y: util.manhattanDistance(x, currentPos) - util.manhattanDistance(y, currentPos));
  if (len(foodList) == 0):
    foodDist = 0;
  else:
    foodDist = util.manhattanDistance(currentPos, foodList[0]);
  # away from the ghost
  ghostPos = [x.getPosition() for x in newGhostStates];
  ghostPos.sort(lambda x, y: int(util.manhattanDistance(x, currentPos) - util.manhattanDistance(y, currentPos)));
  ghostDist = util.manhattanDistance(currentPos, ghostPos[0]);
  if (ghostDist == 0):
    return -9999; # ghost is on this position, keep away from it. 
  if (foodDist == 0):
    return 2.0 - 1/ghostDist; 
  else:
    return 1.0/(foodDist) - 1/ghostDist;
  
   
# Abbreviation
better = betterEvaluationFunction
 
class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
 
  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.
 
      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
