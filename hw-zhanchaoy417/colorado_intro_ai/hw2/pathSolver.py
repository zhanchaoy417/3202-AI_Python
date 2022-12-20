from collections import deque
from numpy import sqrt 

"""
 Class PathSolver

"""

# Create PathSolver Class

def euclidian(xg,xn,yg,yn):
        return ((xg-xn)**2+(yg-yn)**2)**(1/2)


              
class Frontier_PQ():
    #Instantiation arguments, note {}used for a list and []used for a list of tuple,default as empty
    #since has init function, every function has argument self
    def __init__(self, start, cost):
        self.start = start
        self.cost = cost
        #if self.states = {} self.q = [] , it will have a problem, and return NoneType
        #it must identify the state and q structure with list and turple, ":" make a list and [,] indicte a truple
        #then use start, cost into state,q init function
        self.states = {start:cost}
        self.q = [[start, cost]] 
    
    def add(self,state,cost):
        #find the cost relate to the state
        self.states[state]=cost
   # use self.q.append function 
        self.q.append([state,cost])
    
    def pop(self):
        #pop the first item like BFS alogrithm
        return self.q.pop(0)
    
    def replace(self,state,cost):
        #since self.q is a truple, its structure like 2D array so use i,j loop, and enumerate  is loop with counters
        #2-level loop in the self.q, for example, if self.q[i][j] match the cost, then  self.q[i][0] should reference the state
        for i,j in enumerate(self.q):
            if j[1]==cost:
                self.q[i][0]=state
                

class PathSolver:
    """Contains methods to solve multiple path search algorithms"""

    # init for PathSolver Class
    def __init__(self):
        """Create PathSolver"""

    def path(self, previous, s): 
        """
        `previous` is a dictionary chaining together the predecessor state that led to each state

        `s` will be None for the initial state

        otherwise, start from the last state `s` and recursively trace `previous` back to the initial state,
        constructing a list of states visited as we go
        """ 
        
        if s is None:
            return []
        else:
            return self.path(previous, previous[s])+[s]

    def pathcost(self, path, step_costs):
        """add up the step costs along a path, which is assumed to be a list output from the `path` function above"""
        
        cost = 0
        for s in range(len(path)-1):
            cost += step_costs[path[s]][path[s+1]]
        return cost
    

    def breadth_first_search(self,start: tuple, goal, state_graph, return_cost=False):
        """ find a shortest sequence of states from start to the goal """
        print("calliing BFS")
        
        frontier = deque([start]) # doubly-ended queue of states
        previous = {start: None}  # start has no previous state; other states will
        
        # Return on start is goal
        if start == goal:
            path_out = [start]
            if return_cost: return path_out, self.pathcost(path_out, state_graph)
            return path_out

        # loop through frontine searching nodes until we find a goal
        while frontier:
            s = frontier.popleft()
            for s2 in state_graph[s]:
                if (s2 not in previous) and (s2 not in frontier):
                    frontier.append(s2)
                    previous[s2] = s
                    if s2 == goal:
                        path_out = self.path(previous, s2)
                        if return_cost: return path_out, self.pathcost(path_out, state_graph)
                        return path_out
        
        # no solution
        if return_cost:
            return [], 0
        else: 
            return []


    def depth_first_search(self,start: tuple, goal, state_graph, return_cost=False):
        # copy from above and change only popleft()to pop().
        print("calliing DFS")
        
        frontier = deque([start]) # doubly-ended queue of states
        previous = {start: None}  # start has no previous state; other states will
        
        # Return on start is goal
        if start == goal:
            path_out = [start]
            if return_cost: return path_out, self.pathcost(path_out, state_graph)
            return path_out

        # loop through frontine searching nodes until we find a goal
        while frontier:
            s = frontier.pop()
            for s2 in state_graph[s]:
                if (s2 not in previous) and (s2 not in frontier):
                    frontier.append(s2)
                    previous[s2] = s
                    if s2 == goal:
                        path_out = self.path(previous, s2)
                        if return_cost: return path_out, self.pathcost(path_out, state_graph)
                        return path_out
        
        # no solution
        if return_cost:
            return [], 0
        else: 
            return []
                     
    def uniform_cost_search(self,start: tuple, goal, state_graph, return_cost=False):
        """Problem 2.a: you need to implement this function"""
        print("calliing UCS")
        found={}
        for x in state_graph:
            found[x]=False
        #visited = [False] * ((max(state_graph)) + 1)
        prev={}
        queue = Frontier_PQ(start,0)
        #append the first item and it doesn't has previous one
        #queue.append(start)
        found[start]=True    
        #previous is a dictionary of predecessors of "start"
        prev[start]=None
        while queue.q:
         # pop() is remove last itme, for all possable edges,travel all n until it is true,since previos one to last item
        #n is child node,now is current node or parent
            (now,costValue) = queue.pop()
            for n in state_graph[now]:
             #first is check if it in state, add the prev node to the queue, the add function take(state,cost) 
             #it must check 
                    if n not in queue.states:
                        prev[n]=now
                        queue.add(n,state_graph[now][n]+costValue)
             # 'NoneType' object is not subscriptable, to fix it, state_graph is 2D array, take cost repect with cuurent and prev node  
             #add takes state which node n and cost using two function path and pathcost

                    elif (n in queue.states):
                #compare if the new cost is smaller, then use new cost replace it, mark its neighbor is true,set prev is current
                        if queue.states[n] > costValue+state_graph[now][n]:
                            prev[n]=now
                            found[n]=True
                            queue.replace(n,costValue+state_graph[now][n]) 
                        else:
                 #if the new cost is lager, then use prev cost to replace it ,stay same        
                            queue.replace(costValue+state_graph[now][n],n) 

            #append until reach goal
            if (now==goal):
                solu_path=self.path(prev,now)
               #two situations: if (return_cost=true), return a turple, else only return a list     
                if(return_cost==True):
                    return(solu_path,self.pathcost(solu_path,state_graph ))
                elif return_cost:
                        return [], 0
                else:
                    return solu_path
   

    def a_star_euclidian(self,start: tuple, goal, state_graph, return_cost=False):
        """Problem 2.b: you need to implement this function"""
        found={}
        for x in state_graph:
            found[x]=False
        #visited = [False] * ((max(state_graph)) + 1)
        prev={}
        queue = Frontier_PQ(start,0)
        #append the first item and it doesn't has previous one
        #queue.append(start)
        found[start]=True    
        #previous is a dictionary of predecessors of "start"
        prev[start]=None
        while queue.q:
         # pop() is remove last itme, for all possable edges,travel all n until it is true,since previos one to last item
        #n is child node,now is current node or parent
            (now,costValue) = queue.pop()
            for n in state_graph[now]:
                    h_euclidianCost=euclidian(goal[0],start[0],goal[1],start[1])
             #first is check if it in state, add the prev node to the queue, the add function take(state,cost) 
             #it must check 
                    if n not in queue.states:
                        prev[n]=now
                        queue.add(n,state_graph[now][n]+costValue+h_euclidianCost)
             # 'NoneType' object is not subscriptable, to fix it, state_graph is 2D array, take cost repect with cuurent and prev node  
             #add takes state which node n and cost using two function path and pathcost

                    elif (n in queue.states):
                #compare if the new cost is smaller, then use new cost replace it, mark its neighbor is true,set prev is current
                        if queue.states[n] > costValue+state_graph[now][n]+h_euclidianCost:
                            prev[n]=now
                            found[n]=True
                            queue.replace(n,costValue+state_graph[now][n]+h_euclidianCost) 
                        else:
                 #if the new cost is lager, then use prev cost to replace it ,stay same        
                            queue.replace(costValue+state_graph[now][n]+h_euclidianCost,n) 

            #append until reach goal
            if (now==goal):
                solu_path=self.path(prev,now)
               #two situations: if (return_cost=true), return a turple, else only return a list     
                if(return_cost==True):
                    return(solu_path,self.pathcost(solu_path,state_graph ))
                # "in case have the error with no len"
                elif return_cost:
                        return [], 0
                else:
                    return solu_path

    
    def a_star_manhattan(self,start: tuple, goal, state_graph, return_cost=False):
        """Problem 2c: you need to implement this function"""
        def _manhattan(xg,xn,yg,yn):
            return (xg-xn)+(yg-yn)
        
        found={}
        for x in state_graph:
            found[x]=False
        #visited = [False] * ((max(state_graph)) + 1)
        prev={}
        queue = Frontier_PQ(start,0)
        #append the first item and it doesn't has previous one
        #queue.append(start)
        found[start]=True    
        #previous is a dictionary of predecessors of "start"
        prev[start]=None
        while queue.q:
         # pop() is remove last itme, for all possable edges,travel all n until it is true,since previos one to last item
        #n is child node,now is current node or parent
            (now,costValue) = queue.pop()
            for n in state_graph[now]:
                    h_manhattanCost=_manhattan(goal[0],start[0],goal[1],start[1])
             #first is check if it in state, add the prev node to the queue, the add function take(state,cost) 
             #it must check 
                    if n not in queue.states:
                        prev[n]=now
                        queue.add(n,state_graph[now][n]+costValue+h_manhattanCost)
             # 'NoneType' object is not subscriptable, to fix it, state_graph is 2D array, take cost repect with cuurent and prev node  
             #add takes state which node n and cost using two function path and pathcost

                    elif (n in queue.states):
                #compare if the new cost is smaller, then use new cost replace it, mark its neighbor is true,set prev is current
                        if queue.states[n] > costValue+state_graph[now][n]+h_manhattanCost:
                            prev[n]=now
                            found[n]=True
                            queue.replace(n,costValue+state_graph[now][n]+h_manhattanCost) 
                        else:
                 #if the new cost is lager, then use prev cost to replace it ,stay same        
                            queue.replace(costValue+state_graph[now][n]+h_manhattanCost,n) 

            #append until reach goal
            if (now==goal):
                solu_path=self.path(prev,now)
               #two situations: if (return_cost=true), return a turple, else only return a list     
                if(return_cost==True):
                    return(solu_path,self.pathcost(solu_path,state_graph ))
                elif return_cost:
                        return [], 0
                else:
                    return solu_path
