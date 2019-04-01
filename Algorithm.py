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
# Node Class Definition
# ====================================================================================================================================================================== #
class Nodes:
    def __init__(self, current_index: tuple, parent_index: tuple, goal_index: tuple, cost=float('inf')):
        self.index = current_index
        self.parent = parent_index
        self.heuristic_distance = ((current_index[0] - goal_index[0])**2 + (current_index[1] - goal_index[1])**2 )**.5
        self.cost = cost
        self.h_cost = float('inf')

# ====================================================================================================================================================================== #
# Path Finder Class Definition
# ====================================================================================================================================================================== #
class Pathfinder:
    def __init__(self, start: tuple, goal: tuple, grid_size: tuple, bot_radius = 0, clearance = 0, resolution = 1):

        # ------> Important Variables <------- #
        self.start = start                                                  # Start Co Ordinate of the Robot 
        self.goal = goal                                                    # End Co Ordinate for the Robot to reach 
        self.res = resolution                                               # Resolution of the output        
        self.grid_size = grid_size                                          # Height and Width of the Layout 
        self.bot_radius = bot_radius                                        # Radius of the Robot  
        self.clearance = clearance + bot_radius                             # Clearance of the robot to be maintained with obstacle + Robot's radius    
        self.visited = set()                                                # Explored Nodes 
        self.unvisited = list()                                             # Nodes yets to be explored in queue                                                       
        self.robot_points = []                                              # Contains coordinates of robot area    
        self.obstacle_nodes = list()                                        # Given Obstacle space's nodes    
        self.net_obstacles_nodes = set()                                    # New Obstacle space After Minowski Sum  
        self.shortest_path_Dijkstra = list()                                # List of nodes leading the shortest posible path Dijkstra
        self.shortest_path_Astar = list()                                   # List of nodes leading the shortest posible path Astar
        self.nodes = self.generate_nodes(self.grid_size, start, goal)       # Generating nodes for the given layout
        
        # ------> Environment Setup <------- #
        self.start_point_check = False
        self.goal_point_check = False
        self.nodes_Astar = []
        self.nodes_Dijkstra = []
        self.temp = []
        self.calc_robot_points(self.clearance)                              # Calculating Robot occupied point cloud at origin
        self.calc_obstacles(grid_size)                                      # Calculating Obstacles
        self.minowski_sum(self.obstacle_nodes, self.robot_points, grid_size[0], grid_size[1])

        if goal in self.net_obstacles_nodes: self.goal_point_check = True
        if start in self.net_obstacles_nodes: self.start_point_check = True

    # ================================================================================================================================================================= #
    # -----> Function to generate Nodes <----- #
    # ================================================================================================================================================================= #
    def generate_nodes(self, grid_size: tuple, start: tuple, goal: tuple)-> np.array: 
        nodes = np.empty(grid_size, dtype=object)
        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                nodes[y][x] = Nodes((y,x),None, goal)        
        nodes[start] = Nodes(start, start, goal, 0.0)
        return nodes
        # Later can introduce obstacles in the graph      
    
    # ================================================================================================================================================================= #
    # -----> Function to Calculate Robot Point cloud <----- #
    # ================================================================================================================================================================= #
    def calc_robot_points(self, clearance: int) -> None:
        for y in range(-clearance, clearance+1):
            for x in range(-clearance, clearance+1):
                if x**2 + y**2 - clearance**2 <= 0:
                    self.robot_points.append((y, x))

    # ================================================================================================================================================================= #
    # -----> Function to Populate Obstacles in the GUI <----- #
    # ================================================================================================================================================================= #
    def populate_obstacles(self)-> None:
        
        for node in self.net_obstacles_nodes:
            if node in self.obstacle_nodes: self.graph[node] = [255, 0, 0]
            else: self.graph[node] = [255, 255, 255]

    # ================================================================================================================================================================= #
    # -----> Function to perform rectangle check <----- #
    # ================================================================================================================================================================= #
    def rect_check(self, node: tuple) -> bool:
        if 50*self.res <= node[1] <= 100*self.res and 37*self.res <= node[0] <= 83*self.res: return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to perform Ellipse check <----- #
    # ================================================================================================================================================================= #
    def ellipse_check(self, node: tuple) -> bool:
        if ((((node[1] - 140*self.res)**2)/((15*self.res)**2)) + (((node[0] - 30*self.res)**2)/((6*self.res)**2))) <= 1: return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to perform Circle check <----- #
    # ================================================================================================================================================================= #
    def circle_check(self, node: tuple) -> bool:
        if ((node[1] - 190*self.res)**2 + (node[0] - 20*self.res)**2) <= (15*self.res)**2: return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to perform poly check <----- #
    # ================================================================================================================================================================= #
    def poly_check(self, node: tuple) -> bool:
        if (13*node[0] + 37*node[1] <= 7305*self.res) and (25*node[0] - 41*node[1] <= -2775*self.res) and (19*node[0] - 2*node[1] >= 1536*self.res): return True
        elif (node[0] <= 135*self.res) and (13*node[0] + 37*node[1] >= 7305*self.res) and (20*node[0] + 37*node[1] <= 9101*self.res) and (node[0] >= 98*self.res): return True
        elif (7*node[0] + 38*node[1] >= 6880*self.res) and (23*node[0] - 38*node[1] >= -5080*self.res) and (node[0] <= 98*self.res): return True
        return False

    # ================================================================================================================================================================= #
    # -----> Function to Calculate Obstacle Points <----- #
    # ================================================================================================================================================================= #
    def calc_obstacles(self, grid_size: tuple)-> list:
         for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                if self.rect_check((y, x)) or self.ellipse_check((y, x)) or self.circle_check((y, x)) or self.poly_check((y, x)) or y == 0 or x == 0 or y == grid_size[0] - 1 or x == grid_size[1]-1:
                    self.obstacle_nodes.append((y, x))

    # ================================================================================================================================================================= #
    # -----> Function to perform Minowski Sum <----- #
    # ================================================================================================================================================================= #
    def minowski_sum(self, obstacles: np.array, robot_points: np.array, h: int, w: int) -> np.array:
        for y, x in obstacles:
            for j in robot_points:
                node = (y - (-1)*j[1], x - (-1)*j[0])
                if 0 <= node[1] <= w-1 and 0 <= node[0] <= h-1:
                    self.net_obstacles_nodes.add(node)
        
    # ================================================================================================================================================================= #
    # -----> Function to Explore the neighbours <----- #
    # ================================================================================================================================================================= #
    def find_neighbours(self, current_node: tuple, flag = 0)->None:
        y,x     = current_node
        top     = (y-1, x)
        down    = (y+1, x)
        left    = (y, x-1)
        right   = (y, x+1)
        t_left  = (y-1, x-1)
        t_right = (y-1, x+1)
        b_left  = (y+1, x-1)
        b_right = (y+1, x+1)
        
        # -----> Left Neighbour <----- #
        if left[1] > -1 and left not in self.visited: 
            obj = self.compute_cost(left, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]
        # -----> Right Neighbour <----- #
        if right[1] < self.grid_size[1] - 1 and right not in self.visited:
            obj = self.compute_cost(right, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]
        # -----> Top Neighbour <----- #
        if top[0] > -1 and top not in self.visited:
            obj = self.compute_cost(top, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]
        # -----> Down Neighbour <----- #
        if down[0] < self.grid_size[0] - 1 and down not in self.visited:
            obj = self.compute_cost(down, current_node, 1.0, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]
        
        # -----> Top Left Neighbour <----- #
        if t_left[0] > -1 and t_left[1] > -1 and t_left not in self.visited:
            obj = self.compute_cost(t_left, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]
        # -----> Top Right Neighbour <----- #
        if t_right[0] > -1 and t_right[1] < self.grid_size[1] - 1 and t_right not in self.visited:
            obj = self.compute_cost(t_right, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]

        # -----> Bottom Left Neighbour <----- #
        if b_left[1] > -1 and b_left[0] < self.grid_size[0] - 1 and b_left not in self.visited:
            obj = self.compute_cost(b_left, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]
        # -----> Bottom Right Neighbour <----- #
        if b_right[1] < self.grid_size[1] - 1 and b_right[0] < self.grid_size[0] - 1 and b_right not in self.visited:
            obj = self.compute_cost(b_right, current_node, 2**.5, flag)
            if obj not in self.unvisited and obj.index not in self.net_obstacles_nodes: 
                self.unvisited.append(obj)
                self.temp.append(obj.index)
                self.graph[obj.index] = [0, 225, 0]

    # ================================================================================================================================================================= #
    # -----> Function to Calculate Robot Point cloud <----- #
    # ================================================================================================================================================================= #
    def compute_cost(self, node: tuple, parent: tuple, step_cost: int, flag = 0)-> object:
        initial_cost = self.nodes[parent].cost
        node_cost = self.nodes[node].cost
        if node_cost > initial_cost + step_cost:
            self.nodes[node].cost = initial_cost + step_cost
            self.nodes[node].parent = parent
            # -----> Cost calculation for Dijkstra Algorithm <----- #        
            if flag: 
                self.nodes[node].h_cost = self.nodes[node].heuristic_distance + initial_cost + step_cost        
        return self.nodes[node]

    # ================================================================================================================================================================= #
    # -----> Function to generate Shortest node path <----- #
    # ================================================================================================================================================================= #
    def print_explored_nodes(self, flag: int) -> None:
        if flag:
            nodes = [[str(j) for j in i] for i in self.nodes_Astar]
            nodes = [' '.join(i)+'\n' for i in nodes]
            text_file = open("Nodes_Explored_Astar.txt", "w")
            text_file.writelines(nodes)
            text_file.close()
        else:
            nodes = [[str(j) for j in i] for i in self.nodes_Dijkstra]
            nodes = [' '.join(i)+'\n' for i in nodes]
            text_file = open("Nodes_Explored_Dijkstra.txt", "w")
            text_file.writelines(nodes)
            text_file.close()
    
    # ================================================================================================================================================================= #
    # -----> Function to generate Shortest node path <----- #
    # ================================================================================================================================================================= #
    def print_shortest_path(self, flag: int)-> None:
        
        if flag:
            short_path = [[str(j) for j in i] for i in self.shortest_path_Astar]
            short_path = [' '.join(i)+'\n' for i in short_path]
            text_file = open("Shortest_path_Astar.txt", "w")
            text_file.writelines(short_path)
            text_file.close()
        else:
            short_path = [[str(j) for j in i] for i in self.shortest_path_Dijkstra]
            short_path = [' '.join(i)+'\n' for i in short_path]
            text_file = open("Shortest_path_Dijkstra.txt", "w")
            text_file.writelines(short_path)
            text_file.close()

    # ================================================================================================================================================================= #
    # -----> Dijkstra Algorithm Function <----- #
    # ================================================================================================================================================================= #
    def dijkstra(self, start_index: tuple, goal_index: tuple)-> None:
        self.temp = []
        self.visited = set()                                                # Reset Visited Nodes
        self.unvisited = list()                                             # Reset Unvisited        
        self.graph = np.zeros((self.grid_size[0], self.grid_size[1], 3))    # GUI to vizualize the exploration  
        self.populate_obstacles()                                           # Populating the Graph with Obstacle nodes

        cv.namedWindow("Dijkstra Algorithm", cv.WINDOW_NORMAL)
        cv.resizeWindow("Dijkstra Algorithm", 1000, 600)

        self.unvisited.append(self.nodes[start_index])                      # Initialising the node to explore with start node

        while self.unvisited:
            current_node = min(self.unvisited, key = lambda x: x.cost)
            self.find_neighbours(current_node.index)
            self.visited.add(current_node.index)
            self.graph[current_node.index] = [255, 225, 0]
            self.unvisited.remove(current_node)

            # -----> To Skip to the result comment below lines <----- #
            cv.imshow("Dijkstra Algorithm", self.graph)
            cv.waitKey(1)

        # -----> Extracting the shortest path <----- #
        y = (goal_index[1], goal_index[0])
        x = goal_index
        z = (x[1], 150 - x[0])
        while True:
            self.shortest_path_Dijkstra.insert(0, z)
            x = self.nodes[x].parent
            cv.line(self.graph, (x[1],x[0]), y, (0, 0, 255), 1)
            if x == start_index:
                z = (x[1], 150 - x[0])
                self.shortest_path_Dijkstra.insert(0, z)
                cv.line(self.graph, (x[1], x[0]), y, (0, 0, 255), 1)
                break
            y = (x[1], x[0])
            z = (x[1], 150 - x[0])

        self.nodes_Dijkstra = [(x[1],150 - x[0]) for x in list(self.visited) + self.temp]
        cv.circle(self.graph, (start_index[1], start_index[0]), 1, [255, 0, 255], -1)
        cv.circle(self.graph, (goal_index[1], goal_index[0]), 1, [255, 0, 255], -1)
        cv.imshow("Dijkstra Algorithm", self.graph)
        cv.waitKey(0)
        print("cost in Dijkstra: ", self.nodes[goal_index].cost)

    # ================================================================================================================================================================= #
    # -----> A* Algorithm Function <----- #
    # ================================================================================================================================================================= #
    def Astar(self, start_index: tuple, goal_index: tuple) -> None:
        self.temp = []
        self.visited = set()                                                # Reset Visited Nodes
        self.unvisited = list()                                             # Reset Unvisited        
        self.graph = np.zeros((self.grid_size[0], self.grid_size[1], 3))    # GUI to vizualize the exploration 
        self.populate_obstacles()                                           # Populating the Graph with Obstacle nodes

        cv.namedWindow("A* Algorithm", cv.WINDOW_NORMAL)
        cv.resizeWindow("A* Algorithm", 1000, 600)

        self.unvisited.append(self.nodes[start_index])                      # Initialising the node to explore with start node
        
        while self.unvisited:
            current_node = min(self.unvisited, key=lambda x: x.h_cost)
            self.find_neighbours(current_node.index, 1)
            self.visited.add(current_node.index)
            self.graph[current_node.index] = [0, 225, 225]  
            self.unvisited.remove(current_node)
            if goal_index in self.visited: break

            # -----> To Skip to the result comment below lines <----- #         
            cv.imshow("A* Algorithm", self.graph)
            cv.waitKey(1)                      
        
        # -----> Extracting the shortest path <----- #
        y = (goal_index[1], goal_index[0])
        x = goal_index
        z = (x[1], 150 - x[0])
        while True:
            self.shortest_path_Astar.insert(0, z)
            x = self.nodes[x].parent
            cv.line(self.graph, (x[1], x[0]), y, (0, 0, 255), 1)
            if x == start_index:
                z = (x[1], 150 - x[0])
                self.shortest_path_Astar.insert(0, z)
                cv.line(self.graph, (x[1], x[0]), y, (0, 0, 255), 1)
                break
            y = (x[1], x[0])
            z = (x[1], 150 - x[0])

        self.nodes_Astar = [(x[1],150 - x[0]) for x in list(self.visited) + self.temp]
        cv.circle(self.graph, (start_index[1], start_index[0]), 1, [255, 0, 255], -1)
        cv.circle(self.graph, (goal_index[1], goal_index[0]), 1, [255, 0, 255], -1)
        cv.imshow("A* Algorithm", self.graph)
        cv.waitKey(0)

        print("cost in Astar: ", self.nodes[goal_index].cost)

        
if __name__ == '__main__':
    # =================================================================================================================================================================== #
    # User Input Section
    # =================================================================================================================================================================== #
    # start = tuple([int(i) for i in input("Enter the Start node (e.g,(x,y) as 'x y' seperated by space without quotes:").split()])
    # goal = tuple([int(i) for i in input("Enter the Goal node (e.g,(x,y) as 'x y' seperated by space without quotes:").split()])
    # grid_size = tuple([int(i) for i in input("Enter the Grid Size of the Graph (e.g, width and height  as 'width Height' seperated by space without quotes):").split()])
    # bot_radius = int(input("Enter the bot radius:"))
    # clearance = int(input("Enter the clearance between robot and obstacles:"))
    start = (70, 100)
    goal = (50, 50)
    grid_size = (250, 150)
    bot_radius = 5
    clearance = 0
    resolution = 1

    start = (int(resolution*(150 - start[1])),int(resolution*(start[0]))) 
    goal = (int(resolution*(150 - goal[1])),int(resolution*(goal[0]))) 
    grid_size = (int(resolution*grid_size[1]),int(resolution*grid_size[0]))

    maps = Pathfinder(start, goal, grid_size, bot_radius, clearance, resolution)
    
    if not maps.start_point_check and not maps.goal_point_check:
        # ------> Running Astar <------- #
        maps.Astar(start, goal)                                             # Runs the Astar Algorithm and GUI to show the exploration
        maps.print_shortest_path(1)                                         # Prints the shortest file path found using Astar Algo
        maps.print_explored_nodes(1)                                        # Prints all the explored nodes using Astar Algo
        # ------> Running Dijkstra <------- #
        maps.dijkstra(start, goal)                                          # Runs the Dijkstra Algorithm and GUI to show the exploration
        maps.print_shortest_path(0)                                         # Prints the shortest file path found using Dijkstra Algo
        maps.print_explored_nodes(0)                                        # Prints all the explored nodes using Dijkstra Algo
    else:
        print("The Goal node: ", maps.goal_point_check, "\nThe start Node: ", maps.start_point_check)

    cv.destroyAllWindows()
