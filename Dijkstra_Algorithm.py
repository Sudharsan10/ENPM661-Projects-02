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
import os, sys, time
# ====================================================================================================================================================================== #

# ====================================================================================================================================================================== #
# Class Definition
# ====================================================================================================================================================================== #
class Nodes:
    def __init__(self, current_index: tuple, parent_index: tuple, cost = float('inf')):
        self.index = current_index
        self.cost = cost
        self.parent = parent_index

class dijkstra:
    def __init__(self, start: tuple, goal: tuple, resolution: tuple, bot_radius = 0, clearance = 0):
        self.start = start
        self.goal = goal
        self.resolution = resolution
        self.bot_radius = bot_radius
        self.clearance = clearance + bot_radius
        self.visited = set()
        self.unvisited = list()
        self.explore = []
        self.nodes = np.empty(resolution, dtype=object)
    # ================================================================================================================================================================= #
    # Function to generate Nodes
    # ================================================================================================================================================================= #
    def generate_nodes(self, resolution: tuple, start: tuple, goal: tuple)-> np.array:        
        for y in range(resolution[1]):
            for x in range(resolution[0]):
                self.nodes[y][x] = Nodes((x,y), None)
        self.nodes[start] = Nodes(start, start, 0.0)
        # Later can introduce obstacles in the graph      
     
    def rect_check(self, node: tuple) -> bool:
        if 50*self.resolution <= node[0] <= 100*self.resolution and 37*self.resolution <= node[1] <= 83*self.resolution: return True
        return False


    def ellipse_check(self, node: tuple) -> bool:
        if ((((node[0] - 140*self.resolution)**2)/((15*self.resolution)**2)) + (((node[1] - 30*self.resolution)**2)/((6*self.resolution)**2))) <= 1: return True
        return False


    def circle_check(self, node: tuple) -> bool:
        if ((node[0] - 190*self.resolution)**2 + (node[1] - 20*self.resolution)**2) <= (15*self.resolution)**2: return True
        return False


    def poly_check(self, node: tuple) -> bool:
        if (13*node[1] + 37*node[0] <= 7305*self.resolution) and (25*node[1] - 41*node[0] <= -2775*self.resolution) and (19*node[1] - 2*node[0] >= 1536*self.resolution): return True
        elif (node[1] <= 135*self.resolution) and (13*node[1] + 37*node[0] >= 7305*self.resolution) and (20*node[1] + 37*node[0] <= 9101*self.resolution) and (node[1] >= 98*self.resolution): return True
        elif (7*node[1] + 38*node[0] >= 6880*self.resolution) and (23*node[1] - 38*node[0] >= -5080*self.resolution) and (node[1] <= 98*self.resolution): return True
        return False


    def minowski_sum(self, obstacles: np.array, robot_points: np.array, h: int, w: int) -> np.array:
        new_obstacle = set()
        obset = set(obstacles)
        for x, y in obstacles:
            for j in robot_points:
                node = (x - (-1)*j[0], y - (-1)*j[1])
                if 0 < node[0] < w-1 and 0 < node[1] < h-1:
                    if node not in obset and node not in new_obstacle:
                        new_obstacle.add(node)
                        self.nodes[node[1]][node[0]] = [255, 255, 0]
        return new_obstacle


    def find_neighbours(self, current_node: tuple)->list:
        neighbours = list()
        x,y     = current_node

        left    = (x-1, y)
        right   = (x+1, y)
        top     = (x, y-1)
        down    = (x, y+1)
        t_left  = (x-1, y-1)
        t_right = (x+1, y-1)
        b_left  = (x-1, y+1)
        b_right = (x+1, y+1)
        
        if left[0] > 1 and left[0] not in self.visited: 
            self.compute_cost(left, current_node, 1)
        if right[0] < self.resolution[0] - 1 and right[0] not in self.visited: neighbours.append((right, 1))
        if top[0] > 1 and top[0] not in self.visited: neighbours.append((top, 1))
        if down[0] > 1 and down[0] not in self.visited: neighbours.append((down, 1))
        if left[0] > 1 and left[0] not in self.visited: neighbours.append((left, 1.4))
        if left[0] > 1 and left[0] not in self.visited: neighbours.append((left, 1.4))
        if left[0] > 1 and left[0] not in self.visited: neighbours.append((left, 1.4))
        if left[0] > 1 and left[0] not in self.visited: neighbours.append((left, 1.4))
        
        return neighbours

    def compute_cost(self, node: tuple, parent: tuple, step_cost: int):
        initial_cost = self.nodes[parent].cost
        node_cost = self.nodes[node].cost
        if node_cost > initial_cost + step_cost:
            self.nodes[node].cost = initial_cost + step_cost

if __name__ == '__main__':
    # =================================================================================================================================================================== #
    # User Input Section
    # =================================================================================================================================================================== #
    start = tuple([int(i) for i in input("Enter the Start node (e.g,(x,y) as 'x y' seperated by space without quotes:").split()])
    goal = tuple([int(i) for i in input("Enter the Goal node (e.g,(x,y) as 'x y' seperated by space without quotes:").split()])
    resolution = tuple([int(i) for i in input("Enter the Grid Size of the Graph (e.g, width and height  as 'width Height' seperated by space without quotes):").split()])
    bot_radius = int(input("Enter the bot radius:"))
    clearance = int(input("Enter the clearance between robot and obstacles:"))
    
