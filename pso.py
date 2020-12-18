#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This code clears the console and removes all variables present on the namespace at the same time
#If spyder version is below 3.3.3 generates an error
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import numpy as np
import math
import shapely
import matplotlib.pyplot as plt
from shapely.geometry import box
from DynNeighborPSO import DynNeighborPSO   
from WoodProblemDefinition import Stock, Order1, Order2, Order3
from demo_shapely import plotShapelyPoly
global rotation
import time

def compute_shifted(order, solution, rotation):
    # This function computes the shifted coordinations
    # Takes the started coordinations and adds the new coordinations calculated by the algorithm
    
    new_order = []
    # The values of variables are saved in a array 1D, so it is needed to do some pre process.
    # For each order we take 3 values (x,y,theta)
    # If we had an array with 6 values, the first 3 are from shape1 and the other 3 from shape2
    for i in range(solution.shape[0]):
        if(i%3==0):
            x = solution[i]
            y = solution[i+1]
            theta = solution[i+2]
            temp = [x, y, theta]
            new_order.append(temp)
    
    shifted = order.copy()
    for i in range(0,len(shifted)):
        if rotation == 0 :
            shifted[i]=shapely.affinity.translate(shifted[i], new_order[i][0],new_order[i][1])
        else:
            #shifted[i]=shapely.affinity.rotate( shapely.affinity.translate(shifted[i], new_order[i][0],new_order[i][1]),new_order[i][2], origin='center')
            shifted[i]=shapely.affinity.rotate(shifted[i],new_order[i][2],origin=(0,0))
            shifted[i]=shapely.affinity.translate(shifted[i],abs(shifted[i].bounds[0]),0)
            shifted[i]=shapely.affinity.translate(shifted[i], new_order[i][0],new_order[i][1])
            
    return shifted

def ObjectiveFcn(particle, stock , order):
    # Fitness Function
    
    remaining = stock
    shifted = compute_shifted(order, particle, rotation)

    # Calculate the remaining area after orders placement
    # The goal is to maximize this. A big remaining area means that the stock can be used again
    for p in shifted:
        remaining = remaining.difference(p)
    
    # Add a penalty
    # With this penalty we try to solve the problem of overlaps
#    c2 = []
#    for p in range(len(shifted)):
#        for k in range(p+1,len(shifted)):
#            if(shifted[p].disjoint(shifted[k])):
#                c2.append(0)
#            else:
#                c2.append(1)
  
    c1 = []
    # Add a penalty 
    # This for loop calculates the sum of bounds (sum of coordinations)
    # If the sum is big then the probability that the shapes are out of box are high.
    # With this penalty we try to keep the shapes as much as possible at the axis origin.
    for p in shifted:
        c1.append(sum(p.bounds))

    return(1000*(remaining.area)+1*sum(c1)) #+10*sum(c2)

class FigureObjects:
    """ Class for storing and updating the figure's objects.
        
        The initializer creates the figure given only the lower and upper bounds (scalars, since the bounds are 
        typically equal in both dimensions).
        
        The update member function accepts a DynNeighborPSO object and updates all elements in the figure.
        
        The figure has a top row of two subplots. The left one is a 3D plot of the peaks function with only the global 
        best-so-far solution (red dot). The right one is the peaks function contour, together with the best-so-far 
        solution (red dot) and the positions of all particles in the current iteration's swarm (smaller black dots).
        The bottom row shows the best-so-far global finess value achieved by the algorithm.
    """
    
    def __init__(self, LowerBound, UpperBound):
        """ Creates the figure that will be updated by the update member function.
            
        All line objects (best solution, swarm, global fitness line) are initialized with NaN values, as we only 
        setup the style. Best-so-far fitness 
        
        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
        """
        
        assert np.isscalar(LowerBound), "The input argument LowerBound must be scalar."
        assert np.isscalar(UpperBound), "The input argument LowerBound must be scalar."
        
        
        # figure
        self.fig = plt.figure()

        # global best fitness line
        self.axBestFit = plt.subplot(121)#121
        self.axBestFit.set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
        self.lineBestFit, = self.axBestFit.plot([], [])
        
        #print positions of polygons for each iteration
        self.axBestPack = plt.subplot(122)#122
        self.axBestPack.set_title('Best-so-far Packing')
        #self.lineBestPack, = self.axBestPack.plot([], [])
        self.lineBestPack = self.axBestPack.plot([], [])
        
        # auto-arrange subplots to avoid overlappings and show the plot
        self.fig.tight_layout()
    
    
    def update(self, pso):
        """ Updates the figure in each iteration provided a PSODynNeighborPSO object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        if pso.Iteration == -1:
            xdata = np.arange(pso.MaxIterations+1)-1
            self.lineBestFit.set_xdata(xdata)
            self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)

        
        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
        self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
        self.axBestFit.relim()
        self.axBestFit.autoscale_view()
        self.axBestFit.title.set_text('Best-so-far global best fitness: {:g}'.format(pso.GlobalBestFitness))
        
        #update the global best packing
        shifted = compute_shifted(pso.order, pso.GlobalBestPosition, rotation)
                

        self.axBestPack.clear()
        plotShapelyPoly(self.axBestPack, [pso.stock]+shifted) #Stock[6:7]
        self.axBestPack.set_title('Best-so-far Packing')
        self.axBestPack.relim()
        self.axBestPack.autoscale_view()
        self.axBestPack.set_aspect('equal')
        
        # because of title and particles positions changing, we cannot update specific artists only (the figure
        # background needs updating); redrawing the whole figure canvas is expensive but we have to
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def OutputFcn(pso, figObj):
    """ Our output function: updates the figure object and prints best fitness on terminal.
        
        Always returns False (== don't stop the iterative process)
    """
    if pso.Iteration == -1:
        print('Iter.    Global best')
    print('{0:5d}    {1:.5f}'.format(pso.Iteration, pso.GlobalBestFitness))
    
    figObj.update(pso)
    
    return False


def compute_remaining_opening(stock, shifted_order):
    # Compute the remaining after opening. 
    # The remaining is the free space left after the placement of order.
    
    shifted = shifted_order
    
    remaining = stock
    for p in shifted:
        remaining = remaining.difference(p)
        
    # Opening at the remaining Stock and then plot it
    joinStyle = shapely.geometry.JOIN_STYLE.mitre
    opening = remaining.buffer(-0.3, join_style=joinStyle).buffer(0.3, join_style=joinStyle)  

    return [remaining, opening]

def plot_final_one_order_one_stock(stock, order, roation, best_solution):
    
    # best_solution = pso.GlobalBestPosition
    shifted = compute_shifted(order, best_solution, rotation)
        
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Shifted Order3 pieces for better viewing')
    plotShapelyPoly(ax, [stock]+shifted) 
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal')

    remaining, opening = compute_remaining_opening(stock, shifted)
    
    if remaining.is_empty :
        print('Remaining is empty 4')
    else :
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Remaining Stock without Opening')
        plotShapelyPoly(ax, remaining) #Stock[6:7]
        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('equal')        
    
    if opening.is_empty :
        print('Remaining is empty 2')
    else:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Remaining Stock with Opening')
        plotShapelyPoly(ax, opening) #Stock[6:7]
        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('equal')
    
    return 0


def calculate_area_of_order(shape):
    # Calculate the area of the order given for the FitOrderingStock.
    
    temp = []
    for p in shape:
        temp.append(p.area)        
    return sum(temp)


def calculate_area_of_order2(shape):
    # Calculate the area of the order given for the FitOrderingStock1.

    temp = []
    for p in shape:
        temp.append(box(*p.bounds).area)       
    return sum(temp)

           
def check_if_order_is_inside_and_disjoint(order,initial_stock):

    remaining = initial_stock
    for p in order:
        remaining = remaining.difference(p)
        
    # 0.05 is a value chosen as accepted tolerance
    if(abs(calculate_area_of_order(order)-(initial_stock.area - remaining.area)) <= 0.005):
        return True
    else:
        return False

def FitOrderingStock(all_orders,all_stock):

    global opening_stock
    opening_stock = all_stock.copy()
    i=0
    stop_limit_i = len(all_orders)
    
    # For every order
    while i < stop_limit_i :
        print(f'Current Order is {i}')

        order = all_orders[i]
        j=0
        stop_limit_j = len(opening_stock)
        flag_split_order = 0
        
        # For every stock
        while j < stop_limit_j :

            # Check if the area of Order fits to Stock
            if(calculate_area_of_order(order) <= opening_stock[j].area):
                print(f'Current Stock is {j}')
                stock = opening_stock[j]

                nVars = 3*len(order)

                # Set Upper and Lower Bounds
                LowerBounds = [0]*(nVars)
                UpperBounds = []
                for k in range(nVars):
                    if(k%3==0):
                        UpperBounds.extend([stock.bounds[2]-1]+[stock.bounds[3]-1]+[360])
    
                # Set Plots and call PSO Algorithm for a specific order and stock
                figObj = FigureObjects(LowerBounds[0], UpperBounds[0])
                outFun = lambda x: OutputFcn(x, figObj)

                pso = DynNeighborPSO(ObjectiveFcn, nVars, LowerBounds=LowerBounds, UpperBounds=UpperBounds, 
                         OutputFcn=outFun, UseParallel=False, MaxStallIterations=20, stock = stock, order = order)
                pso.optimize()
                      
                shifted = compute_shifted(order, pso.GlobalBestPosition, rotation)
                remaining, opening = compute_remaining_opening(stock, shifted)

                # If the Order doesnt fit in Stock
                if opening.is_empty or check_if_order_is_inside_and_disjoint(shifted,stock) :
                    print(f'Polygons of Order {i} fitted to Stock {j}-----------------------------')
                    opening_stock[j] = opening

                    # Check if after the placement of order the type is MultiPolygon
                    # A Multipolygon has bigger area than a Polygon
                    #if opening_stock[j].geom_type == 'MultiPolygon':
                        #print('From Multipolygon to Polygons')
                    if opening_stock[j].geom_type == 'MultiPolygon':
                        print('From Multipolygon to Polygons')
                        temp = list(opening_stock[j])
                    
                        del opening_stock[j]
                        
                        for h in temp:
                            # Shift the Stocks occured by multipolygon at axis origin
                            temp2 = shapely.affinity.translate(h, -h.bounds[0],-h.bounds[1])
                            opening_stock.insert(j,temp2)
                            
                    stop_limit_j = len(opening_stock)
                    flag_split_order = 0
                    break 

            if(j == len(opening_stock)-1):
                flag_split_order = 1

            if(flag_split_order):
                # If an order dont fit in stock then the order needs to be splitted
                print(f'Polygons NOT fitted SPLIT ORDER {i}-----------------------------')
                # Split the order and repeat the above process
                if(len(order)>2):
                    all_orders.insert(i+1,order[0:2])
                    all_orders.insert(i+2,order[2:len(order)])
                else:
                    all_orders.insert(i+1,order[0:1])
                    all_orders.insert(i+2,order[1:len(order)])
                    
            stop_limit_i = len(all_orders)

            j = j + 1
        i=i+1
           
    return 0


def FitOrderingStock1(all_orders, all_stocks_remaining_area):
    
    opening_stock = Stock.copy()
    i=0
    stop_limit = len(all_orders)
    # For every Order
    while i < stop_limit :

        order = all_orders[i]

        # For every Stock
        for j in range(len(Stock)):

            # Check if the area of Order fits to Stock
            if(calculate_area_of_order2(order) <= opening_stock[j].area):
         
                # Check if every shape in order fits in Stock
                # There is an occasion when the area of order is smaller than the area of stock
                # So we check and every shape separately that fits in
                flag = 0 
                for p in order:

                    if opening_stock[j].contains(p) == False:
                        flag = 1
                        break
                if flag == 1:
                    continue

                stock = opening_stock[j]
                order = all_orders[i]
                
                nVars = 3*len(order)
                
                # Set Upper and Lower Bounds
                LowerBounds = [0]*(nVars)
                UpperBounds = []
                for k in range(nVars):
                    if(k%3==0):
                        UpperBounds.extend([stock.bounds[2]-1]+[stock.bounds[3]-1]+[360])
                        
                # Set Plots and call PSO Algorithm for a specific order and stock
                figObj = FigureObjects(LowerBounds[0], UpperBounds[0])
                outFun = lambda x: OutputFcn(x, figObj)
                
                pso = DynNeighborPSO(ObjectiveFcn, nVars, LowerBounds=LowerBounds, UpperBounds=UpperBounds, 
                         OutputFcn=outFun, UseParallel=False, MaxStallIterations=20, stock = stock, order = order)
                pso.optimize()
        
                shifted = compute_shifted(order, pso.GlobalBestPosition, rotation)
                remaining, opening = compute_remaining_opening(stock, shifted)
                
                #global best
                #best = pso.GlobalBestPosition
                opening_stock[j] = opening
                
                if opening.is_empty :
                    print('Remaining is empty 2')

                # At this point means the order fits in a stock so it is not need to continue search
                break           

            # If the order doesnt fit in any Stock then we split the order and try the above process again
            # until the order fits in a Stock.
            # When j == 7 means that the algorithm tried to fit the Order in all Stocks unsuccesfully
            if(calculate_area_of_order2(order) > opening_stock[j].area and j == 7):
                split = math.ceil(len(order)/2)
                all_orders.append(all_orders[i][:(split-1)])
                all_orders.append(all_orders[i][(split-1):])
                stop_limit = len(all_orders)

        i=i+1
               
    return 0

if __name__ == "__main__":
    """ Executed only when the file is run as a script. """
    
    # in case somebody tries to run it from the command line directly...
    plt.ion()

    all_orders = [Order1 , Order2, Order3]

    # Rotation = 0 -> No Rotation
    # Rotation = 1 -> Yes Rotation
    
    #np.random.seed(20)
    rotation = 1
    start = time.time()
    FitOrderingStock(all_orders,Stock)
    end = time.time()
    print("Execution Time in Secs: ", end - start)

