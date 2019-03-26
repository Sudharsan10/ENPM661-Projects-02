# ====================================================================================================================================================================== #
#-------------> Project 02 <---------------#
# ====================================================================================================================================================================== #
# Authors   :-> Sudharsan(UID: 116298636)
# Course    :-> ENPM661 - Planning for Autonomous Robots
# Date      :-> 26 March 2019
# ====================================================================================================================================================================== #

# ====================================================================================================================================================================== #
# Import Section
# ====================================================================================================================================================================== #
import numpy as np
import cv2 as cv
import os, sys, time, math
# ====================================================================================================================================================================== #

# ====================================================================================================================================================================== #
# Class Definition
# ====================================================================================================================================================================== #
class Nodes:
    def __init__(self, current_index: tuple, parent_index: tuple, goal_index: tuple, cost=float('inf')):
        self.index = current_index
        self.parent = parent_index
        self.heuristic_distance = ((current_index[0] - goal_index[0])**2 + (current_index[1] - goal_index[1])**2 )**.5
        self.cost = cost

class Pathfinder:
    def __init__(self, start: tuple, goal: tuple, resolution: tuple, bot_radius = 0, clearance = 0):

        # ------> Important Variables <------- #
        self.start = start                                                  # Start Co Ordinate of the Robot 
        self.goal = goal                                                    # End Co Ordinate for the Robot to reach 
        self.resolution = resolution                                        # Height and Width of the Layout         
        self.bot_radius = bot_radius                                        # Radius of the Robot  
        self.clearance = clearance + bot_radius                             # Clearance of the robot to be maintained with obstacle + Robot's radius    
        self.visited = set()                                                # Explored Nodes 
        self.unvisited = list()                                             # Nodes yets to be explored in queue                                                       
        self.robot_points = []                                              # Contains coordinates of robot area    
        self.obstacle_nodes = list()                                        # Given Obstacle space's nodes    
        self.net_obstacles_nodes = set()                                    # New Obstacle space After Minowski Sum  
        self.shortest_path = list()                                         # List of nodes leading the shortest posible path
        self.nodes = self.generate_nodes(self.resolution, start, goal)      # Generating nodes for the given layout

        # ------> Environment Setup <------- #
        self.calc_robot_points(self.clearance)                              # Calculating Robot occupied point cloud at origin
        self.calc_obstacles(resolution)                                     # Calculating Obstacles
        self.minowski_sum(self.obstacle_nodes, self.robot_points, resolution[1], resolution[0])

        # ------> Running Dijkstra <------- #
        # self.dijkstra(start, goal)

        # ------> Running Astar <------- #
        self.Astar(start, goal)

    # ================================================================================================================================================================= #
    # -----> Function to generate Nodes <----- #
    # ================================================================================================================================================================= #
    def generate_nodes(self, resolution: tuple, start: tuple, goal: tuple)-> np.array: 
        nodes = np.empty((resolution[1], resolution[0]), dtype=object)
        for y in range(resolution[1]):
            for x in range(resolution[0]):
                nodes[y][x] = Nodes((x,y),None, goal)        
        nodes[start[1]][start[0]] = Nodes(start, start, goal, 0.0)
        return nodes
        # Later can introduce obstacles in the graph      
    
    # ================================================================================================================================================================= #
    # -----> Function to Calculate Robot Point cloud <----- #
    # ================================================================================================================================================================= #
    def calc_robot_points(self, clearance: int) -> None:
        for y in range(-clearance, clearance+1):
            for x in range(-clearance, clearance+1):
                if x**2 + y**2 - clearance**2 <= 0:
                    self.robot_points.append((x, y))

    # ================================================================================================================================================================= #
    # -----> Function to perform rectangle check <----- #
    # ================================================================================================================================================================= #
    def rect_check(self, node: tuple) -> bool:
        if 50*1 <= node[0] <= 100*1 and 37*1 <= node[1] <= 83*1: return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to perform Ellipse check <----- #
    # ================================================================================================================================================================= #
    def ellipse_check(self, node: tuple) -> bool:
        if ((((node[0] - 140*1)**2)/((15*1)**2)) + (((node[1] - 30*1)**2)/((6*1)**2))) <= 1: return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to perform Circle check <----- #
    # ================================================================================================================================================================= #
    def circle_check(self, node: tuple) -> bool:
        if ((node[0] - 190*1)**2 + (node[1] - 20*1)**2) <= (15*1)**2: return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to perform poly check <----- #
    # ================================================================================================================================================================= #
    def poly_check(self, node: tuple) -> bool:
        if (13*node[1] + 37*node[0] <= 7305*1) and (25*node[1] - 41*node[0] <= -2775*1) and (19*node[1] - 2*node[0] >= 1536*1): return True
        elif (node[1] <= 135*1) and (13*node[1] + 37*node[0] >= 7305*1) and (20*node[1] + 37*node[0] <= 9101*1) and (node[1] >= 98*1): return True
        elif (7*node[1] + 38*node[0] >= 6880*1) and (23*node[1] - 38*node[0] >= -5080*1) and (node[1] <= 98*1): return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to Calculate Obstacle Points <----- #
    # ================================================================================================================================================================= #
    def calc_obstacles(self, resolution: tuple)-> list:
         for y in range(resolution[1]):
            for x in range(resolution[0]):
                if self.rect_check((x, y)) or self.ellipse_check((x, y)) or self.circle_check((x, y)) or self.poly_check((x, y)) or y == 0 or x == 0 or y == resolution[1] - 1 or x == resolution[0]-1:
                    self.obstacle_nodes.append((x, y))

    # ================================================================================================================================================================= #
    # -----> Function to perform Minowski Sum <----- #
    # ================================================================================================================================================================= #
    def minowski_sum(self, obstacles: np.array, robot_points: np.array, h: int, w: int) -> np.array:
        for x, y in obstacles:
            for j in robot_points:
                node = (x - (-1)*j[0], y - (-1)*j[1])
                if 0 <= node[0] <= w-1 and 0 <= node[1] <= h-1:
                    self.net_obstacles_nodes.add(node)
        
    # ================================================================================================================================================================= #
    # -----> Function to Explore the neighbours <----- #
    # ================================================================================================================================================================= #
    def find_neighbours(self, current_node: tuple, flag = 0)->None:
        x,y     = current_node
        left    = (x-1, y)
        right   = (x+1, y)
        top     = (x, y-1)
        down    = (x, y+1)
        t_left  = (x-1, y-1)
        t_right = (x+1, y-1)
        b_left  = (x-1, y+1)
        b_right = (x+1, y+1)
        
        # -----> Left Neighbour <----- #
        if left[0] > -1 and left not in self.visited: 
            obj = self.compute_cost(left, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)
        # -----> Right Neighbour <----- #
        if right[0] < self.resolution[0] - 1 and right not in self.visited:
            obj = self.compute_cost(right, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)
        # -----> Top Neighbour <----- #
        if top[1] > -1 and top not in self.visited:
            obj = self.compute_cost(top, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)
        # -----> Down Neighbour <----- #
        if down[1] < self.resolution[1] - 1 and down not in self.visited:
            obj = self.compute_cost(down, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)
        
        # -----> Top Left Neighbour <----- #
        if t_left[0] > -1 and t_left[1] > -1 and t_left not in self.visited:
            obj = self.compute_cost(t_left, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)
        # -----> Top Right Neighbour <----- #
        if t_right[1] > -1 and t_right[0] < self.resolution[0] - 1 and t_right not in self.visited:
            obj = self.compute_cost(t_right, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)

        # -----> Bottom Left Neighbour <----- #
        if b_left[0] > -1 and b_left[1] < self.resolution[1] - 1 and b_left not in self.visited:
            obj = self.compute_cost(b_left, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)
        # -----> Bottom Right Neighbour <----- #
        if b_right[0] < self.resolution[0] - 1 and b_right[1] < self.resolution[1] - 1 and b_right not in self.visited:
            obj = self.compute_cost(b_right, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: self.unvisited.append(obj)

    # ================================================================================================================================================================= #
    # -----> Function to Calculate Robot Point cloud <----- #
    # ================================================================================================================================================================= #
    def compute_cost(self, node: tuple, parent: tuple, step_cost: int, flag = 0)-> object:
        inv_node = node[1],node[0]
        inv_parent = parent[1],parent[0]

        # -----> Cost calculation for A* Algorithm <----- #
        if flag:
            heuristic_distance = self.nodes[inv_node].heuristic_distance
            initial_cost = self.nodes[inv_parent].cost
            node_cost = self.nodes[inv_node].cost
            if node_cost > heuristic_distance + initial_cost + step_cost:
                self.nodes[inv_node].cost = heuristic_distance + initial_cost + step_cost
                self.nodes[inv_node].parent = parent
            return self.nodes[inv_node]

        # -----> Cost calculation for Dijkstra Algorithm <----- #
        else:  
            initial_cost = self.nodes[inv_parent].cost
            node_cost = self.nodes[inv_node].cost
            if node_cost > initial_cost + step_cost:
                self.nodes[inv_node].cost = initial_cost + step_cost
                self.nodes[inv_node].parent = parent
            
            return self.nodes[inv_node]

    # ================================================================================================================================================================= #
    # -----> Dijkstra Algorithm Function <----- #
    # ================================================================================================================================================================= #
    def dijkstra(self, start_index: tuple, goal_index: tuple)-> None:
        self.visited = set()                                                # Reset Visited Nodes
        self.unvisited = list()                                             # Reset Unvisited        
        graph = np.zeros((self.resolution[1], self.resolution[0], 3))       # GUI to vizualize the exploration  
        
        self.unvisited.append(self.nodes[start_index[1]][start_index[0]])
        while self.unvisited:
            current_node = min(self.unvisited, key = lambda x: x.cost)
            self.find_neighbours(current_node.index)
            self.visited.add(current_node.index)
            graph[current_node.index[1]][current_node.index[0]] = [255, 225, 0]
            cv.imshow(" Dijkstra Algorithm", graph)
            cv.waitKey(1)
            self.unvisited.remove(current_node)


        # -----> Extracting the shortest path <----- #
        y = (goal_index[1], goal_index[0])
        x = goal_index
        while True:
            self.shortest_path.insert(0, x)
            x = self.nodes[y].parent
            cv.line(graph, (y[1],y[0]), x, (0, 0, 255), 1)
            cv.imshow(" Dijkstra Algorithm",graph)
            cv.waitKey(1)
            if x == start_index:
                self.shortest_path.insert(0, x)
                cv.line(graph, (y[1], y[0]), x, (0, 0, 255), 1)
                cv.imshow(" Dijkstra Algorithm", graph)
                cv.waitKey(1)
                break
            y = (x[1], x[0])

        cv.circle(graph, start_index, 1, [255, 0, 255], -1)
        cv.circle(graph, goal_index, 1, [255, 0, 255], -1)
        cv.imshow(" Dijkstra Algorithm", graph)
        cv.waitKey(0)
        print("cost in Dijkstra: ", self.nodes[(goal_index[1], goal_index[0])].cost)

    # ================================================================================================================================================================= #
    # -----> A* Algorithm Function <----- #
    # ================================================================================================================================================================= #
    def Astar(self, start_index: tuple, goal_index: tuple) -> None:
        self.visited = set()                                                # Reset Visited Nodes
        self.unvisited = list()                                             # Reset Unvisited        
        graph = np.zeros((self.resolution[1], self.resolution[0], 3))       # GUI to vizualize the exploration  
        
        self.unvisited.append(self.nodes[start_index[1]][start_index[0]])
        while self.unvisited:
            current_node = min(self.unvisited, key=lambda x: x.heuristic_distance)
            self.find_neighbours(current_node.index, 1)
            self.visited.add(current_node.index)
            graph[current_node.index[1]][current_node.index[0]] = [0, 225, 225]
            cv.imshow(" A* Algorithm", graph)
            cv.waitKey(1)            
            self.unvisited.remove(current_node)
            if goal_index in self.visited: break
        
        # -----> Extracting the shortest path <----- #
        y = (goal_index[1], goal_index[0])
        x = goal_index
        while True:
            self.shortest_path.insert(0, x)
            x = self.nodes[y].parent
            cv.line(graph, (y[1], y[0]), x, (0, 0, 255), 1)
            cv.imshow(" A* Algorithm", graph)
            cv.waitKey(1)
            if x == start_index:
                self.shortest_path.insert(0, x)
                cv.line(graph, (y[1], y[0]), x, (0, 0, 255), 1)
                cv.imshow(" A* Algorithm", graph)
                cv.waitKey(1)
                break
            y = (x[1], x[0])

        cv.circle(graph, start_index, 1, [255, 0, 255], -1)
        cv.circle(graph, goal_index, 1, [255, 0, 255], -1)
        cv.imshow(" A* Algorithm", graph)
        cv.waitKey(0)

        print("cost in Astar: ", self.nodes[(goal_index[1], goal_index[0])].cost)

        
if __name__ == '__main__':
    # =================================================================================================================================================================== #
    # User Input Section
    # =================================================================================================================================================================== #
    # start = tuple([int(i) for i in input("Enter the Start node (e.g,(x,y) as 'x y' seperated by space without quotes:").split()])
    # goal = tuple([int(i) for i in input("Enter the Goal node (e.g,(x,y) as 'x y' seperated by space without quotes:").split()])
    # resolution = tuple([int(i) for i in input("Enter the Grid Size of the Graph (e.g, width and height  as 'width Height' seperated by space without quotes):").split()])
    # bot_radius = int(input("Enter the bot radius:"))
    # clearance = int(input("Enter the clearance between robot and obstacles:"))
    start = (10, 10)
    goal = (230, 120)
    resolution = (250, 150)
    bot_radius = 5
    clearance = 0

    map1 = Pathfinder(start, goal, resolution, bot_radius, clearance)
    
    cv.destroyAllWindows()
