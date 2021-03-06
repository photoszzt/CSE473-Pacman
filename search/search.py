# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
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
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
     state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].
      
    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
      
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
      
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    """
    Start: (5, 5)
    Is the start a goal? False 
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    Path found with total cost of 999999 in 0.0 seconds
    Search nodes expanded: 1
    """
    if (problem.isGoalState(problem.getStartState())):
        return [];
    nodeVisited = [];
    stack = util.Stack();
    nodeVisited.append(problem.getStartState());
    stack.push([problem.getStartState(), '']);
    parentMap = {};
    while (not stack.isEmpty()):
      nextNode = stack.pop();
      nextPos = nextNode[0];
      nextAction = nextNode[1];
      if (problem.isGoalState(nextPos)):
        cur = nextPos;
        actions = [];
        while (1):
            temp = parentMap.get(cur);
            if (temp == None):
                break;
            cur = temp[0];
            actions.append(temp[1]);
        actions.reverse();
        actions.append(nextAction);
        actions.remove('');
        return actions;
      adjacent = problem.getSuccessors(nextPos);
      length = len(adjacent);
      for i in range(0, length):
        v = adjacent[i];
        if (not (v[0] in nodeVisited or [v[0], v[1]] in stack.list)):
          nodeVisited.append(v[0]);
          stack.push([v[0], v[1]]);
          parentMap[v[0]] = [nextPos, nextAction]; 


def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  "*** YOUR CODE HERE ***"
  if (problem.isGoalState(problem.getStartState())):
        return [];
  nodeVisited = [];
  queue = util.Queue();
  nodeVisited.append(problem.getStartState());
  queue.push([problem.getStartState(), '']);
  parentMap = {};
  while (not queue.isEmpty()):
    nextNode = queue.pop();
    nextPos = nextNode[0];
    nextAction = nextNode[1];
    if (problem.isGoalState(nextPos)):
      cur = nextPos;
      actions = [];
      while (1):
          temp = parentMap.get(cur);
          if (temp == None):
              break;
          cur = temp[0];
          actions.append(temp[1]);
      actions.reverse();
      actions.append(nextAction);
      actions.remove('');
#      print "actions: ", actions;
      return actions;
    adjacent = problem.getSuccessors(nextPos);
    length = len(adjacent);
    for i in range(0, length):
      v = adjacent[i];
      if (not (v[0] in nodeVisited or [v[0], v[1]] in queue.list)):
        nodeVisited.append(v[0]);
        queue.push([v[0], v[1]]);
        parentMap[v[0]] = [nextPos, nextAction]; 
  
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  nodeVisited = [];
  queue = util.PriorityQueue();
  nodeVisited.append(problem.getStartState());
  queue.push([problem.getStartState(), '', 0], 0);
  parentMap = {};
  while (not queue.isEmpty()):
    nextNode = queue.pop();
    nextPos = nextNode[0];
    nextAction = nextNode[1];
    nextCost = nextNode[2];
    if (problem.isGoalState(nextPos)):
      cur = nextPos;
      actions = [];
      while (1):
          temp = parentMap.get(cur);
          if (temp == None):
              break;
          cur = temp[0];
          actions.append(temp[1]);
      actions.reverse();
      actions.append(nextAction);
      actions.remove('');
      return actions;
    adjacent = problem.getSuccessors(nextPos);
    length = len(adjacent);
    for i in range(0, length):
      v = adjacent[i];
      searchNode = search(queue, [v[0], v[1]]);
      newCost = nextCost + v[2];
      if (not (v[0] in nodeVisited or searchNode[0])):
        nodeVisited.append(v[0]);
        queue.push([v[0], v[1], newCost], newCost);
        parentMap[v[0]] = [nextPos, nextAction];
      elif (searchNode[0] and searchNode[1] > newCost):
        del queue.heap[searchNode[2]];
        queue.push([v[0], v[1], newCost], newCost);
        parentMap[v[0]] = [nextPos, nextAction];  
        
def search(queue, node):
    """
    Search the node in the PriorityQueue. If found return the priority. 
    """
    length = len(queue.heap);
    for i in range(0, length):
        state = queue.heap[i];
        if (state[1] == node[0] and state[2] == node[1]):
            return True, state[0], i;
    return False, None, None;
          
def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  nodeVisited = [];
  queue = util.PriorityQueue();
  nodeVisited.append(problem.getStartState());
  queue.push([problem.getStartState(), '', heuristic(problem.getStartState(), problem)], 0);
#  if (heuristic(problem.getStartState(), problem) > 27):
#      print "inadmissible heuristic";
  parentMap = {};
  while (not queue.isEmpty()):
    nextNode = queue.pop();
    nextPos = nextNode[0];
    nextAction = nextNode[1];
    nextCost = nextNode[2];
    if (problem.isGoalState(nextPos)):
      cur = nextPos;
      actions = [];
      while (1):
          temp = parentMap.get(cur);
          if (temp == None):
              break;
          cur = temp[0];
          actions.append(temp[1]);
      actions.reverse();
      actions.append(nextAction);
      actions.remove('');
  #    print "actions: ", actions;
      return actions;
    adjacent = problem.getSuccessors(nextPos);
    length = len(adjacent);
    for i in range(0, length):
      v = adjacent[i];
      searchNode = search(queue, [v[0], v[1]]);
      newCost = nextCost + v[2];
      f = newCost + heuristic(v[0], problem);
#      if (heuristic(nextPos, problem) > 27):
#          print "inadmissible heuristic", nextPos, heuristic(nextPos, problem);
#      if (heuristic(nextPos, problem) > v[2] + heuristic(v[0], problem)):
#          print "inconsistent heuristic!";
#          print "Previous h: ", heuristic(nextPos, problem), "present cost: ", v[2], "new h: ", heuristic(v[0], problem); 
      if (not (v[0] in nodeVisited or searchNode[0])):
        nodeVisited.append(v[0]);
        queue.push([v[0], v[1], newCost], f);
        parentMap[v[0]] = [nextPos, nextAction];
      elif (searchNode[0] and searchNode[1] > newCost):
        del queue.heap[searchNode[2]];
        queue.push([v[0], v[1], newCost], f);
        parentMap[v[0]] = [nextPos, nextAction];  
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
