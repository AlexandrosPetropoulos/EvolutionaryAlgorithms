#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
import math 
import shapely
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from demo_shapely import plotShapelyPoly
from shapely.geometry import Polygon,box,polygon
from WoodProblemDefinition import Stock, Order1, Order2, Order3
global rotation
import time

#--- COST FUNCTION ---------------------------------------------------+
def ObjectiveFunction(individual, stock , order):
    # Fitness Function

    remaining = stock
    shifted = compute_shifted(order, individual)

    # Calculate the remaining area after orders placement
    # The goal is to maximize this. A big remaining area means that the stock can be used again
    for p in shifted:
        remaining = remaining.difference(p)

    c1 = []
    # Add a penalty 
    # This for loop calculates the sum of bounds (sum of coordinations)
    # If the sum is big then the probability that the shapes are out of box are high.
    # With this penalty we try to keep the shapes as much as possible at the axis origin.
    for p in shifted:
        c1.append(sum(p.bounds))
  
    return(1000*remaining.area+10*sum(c1))

def compute_shifted(order, solution):
    # This function computes the shifted coordinations
    # Takes the started coordinations and adds the new coordinations calculated by the algorithm
    
    shifted = order.copy()
    for i in range(0,len(shifted)):
        if rotation == 0 :
            shifted[i]=shapely.affinity.translate(shifted[i], solution[i][0],solution[i][1])
        else:
            shifted[i]=shapely.affinity.rotate(shifted[i],solution[i][2],origin=(0,0))
            shifted[i]=shapely.affinity.translate(shifted[i],abs(shifted[i].bounds[0]),0)
            shifted[i]=shapely.affinity.translate(shifted[i], solution[i][0],solution[i][1])
       
    return shifted   

#--- FUNCTIONS ----------------------------------------------------------------+

def ensure_bounds(vec, bounds):
    # Check the values of variables that are inside the BOUNDS
    vec_new = []
    
    # Cycle through each variable in vector 
    for i in range(len(vec)):
        x = vec[i][0][0]
        y = vec[i][0][1]
        theta = vec[i][0][2]
        
        # variable X exceedes the minimum boundary
        if vec[i][0][0] < bounds[i][0][0]:
            x = bounds[i][0][0] 
          
         # variable X exceedes the maximum boundary
        if vec[i][0][0]> bounds[i][0][1]: 
            x = bounds[i][0][1] 
        
        
        # variable Y exceedes the minimum boundary
        if vec[i][0][1] < bounds[i][1][0]:
            y = bounds[i][1][0] 
               
         # variable Y exceedes the maximum boundary
        if vec[i][0][1] > bounds[i][1][1]:
            y = bounds[i][1][1]
        
        
        # variable Theta exceedes the minimum boundary
        if vec[i][0][2] < bounds[i][2][0]:
            theta = bounds[i][2][0]
               
         # variable Theta exceedes the maximum boundary
        if vec[i][0][2] > bounds[i][2][1]:
            theta = bounds[i][2][1]

        vec_new.append(((x), (y), (theta)))
        
    return vec_new


#--- MAIN ---------------------------------------------------------------------+
    
def DEGL(cost_func, bounds, popsize, recombination, maxiter, stock, order):

    #--- INITIALIZE A POPULATION (step #1) ----------------+
    wmin = 0.4
    wmax = 0.8
    a = b = 0.8
    w = 0
    population = []
    # Rotation -2, 18
    # Without -3, 10
    FunctionTolerance = 1.0e-3
    MaxStallIterations = 20
    counter = 0
    #random.seed(2)
    
    for i in range(0,popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(((random.uniform(bounds[j][0][0],bounds[j][0][1])), (random.uniform(bounds[j][1][0],bounds[j][1][1])), (random.uniform(bounds[j][2][0],bounds[j][2][1]))))
        population.append(indv)

    #--- SOLVE --------------------------------------------+
    # Cycle through each generation (step #2)
    for i in range(1,maxiter+1):
        print ('GENERATION:',i)

        gen_scores = [] # score keeping
        
        # Neighborhood size
        k = math.ceil(0.1 * popsize)
        
        # Calculate weight w
        w = wmin + (wmax - wmin) * ( (i-1) / (maxiter - 1))

        # Calculate Fitness Function for every Generetion 
        fitFunction = []
        for j in range(0, popsize):
            fitFunction.append(cost_func(population[j], stock,order))
            
        for j in range(0,popsize):
            #print ('POPULATION:',j)
            v_donor = []

            # cycle through each individual in the population
            for idx in range(0, len(population[0])):
               # print ('SHAPE:',idx)
    
                #--- MUTATION (step #3.A) ---------------------+
                # Find the neighbors
                x_t = population[j]     # target individual
                indecies = []
                for z in range(1,k+1):
                    indecies.append((j + z) % popsize)
                    if j - z >= 0:
                        indecies.append((j - z) % popsize)
                    else:
                        indecies.append((popsize - z) % popsize)
                    
                # Random_index holds q,p
                random_index = random.sample(indecies, 2)
                while random_index[0] ==  random_index[1]:
                    random_index = random.sample(indecies, 2)
    
                Zrandom = []
                Zrandom.append(((b * (population[random_index[0]][idx][0] - population[random_index[1]][idx][0])), (b * (population[random_index[0]][idx][1] - population[random_index[1]][idx][1])), ( b * (population[random_index[0]][idx][2] - population[random_index[1]][idx][2]) )))

                maxFitfunction = 100000
                maxIndex = 0;
                # Find the best in neighborhood
                for z in range(0,k + k ):
                    if fitFunction[indecies[z]] < maxFitfunction:
                        maxFitfunction = fitFunction[indecies[z]]
                        maxIndex = indecies[z]
                           
                zbest = population[maxIndex]

                Zb = []
                Zb.append(((a * (zbest[idx][0]- x_t[idx][0])), (a * (zbest[idx][1] - x_t[idx][1])), ( a * (zbest[idx][2] - x_t[idx][2]) )))         

                Li = []
                Li.append(( ((1-w) * (x_t[idx][0] + Zb[0][0] + Zrandom[0][0] )), ( (1-w) * (x_t[idx][1] + Zb[0][1]  + Zrandom[0][1] )), ( (1-w) * (x_t[idx][2] + Zb[0][2] + Zrandom[0][2])) ))
                
                r1 = randint(0, popsize-1)
                r2 = randint(0, popsize-1)
                # r1, r2 are random values inside population
                while r1 == r2:
                     r1 = randint(0, popsize-1)
                     r2 = randint(0, popsize-1)
    
                Zrandom = []
                Zrandom.append(((b * (population[r1][idx][0] - population[r2][idx][0])), (b * (population[r1][idx][1] - population[r2][idx][1])), ( b * (population[r1][idx][2] - population[r2][idx][2]) )))
    
                maxFitfunction = 100000
                maxIndex = 0;
                # Find the global best 
                for z in range(0,popsize):
                    if fitFunction[z] < maxFitfunction:
                        maxFitfunction = fitFunction[z]
                        maxIndex = z
                           
                zbest = population[maxIndex]
                
                Zb = []
                Zb.append(((a * (zbest[idx][0]- x_t[idx][0])), (a * (zbest[idx][1] - x_t[idx][1])), ( a * (zbest[idx][2] - x_t[idx][2]) )))         
                gi = []

                gi.append(( (w*(x_t[idx][0] + Zb[0][0] + Zrandom[0][0])),  (w* (x_t[idx][1] + Zb[0][1] + Zrandom[0][1])), ( w * (x_t[idx][2] + Zb[0][2] + Zrandom[0][2])) ))
                
                v_don = []
                v_don.append((  (Li[0][0] + gi[0][0]), (Li[0][1] + gi[0][1]), (Li[0][2] + gi[0][2])  )) 
                # V donor is the new mutated donor
                v_donor.append(v_don)
            
            # Chech the bounds of variables
            v_donor = ensure_bounds(v_donor, bounds)
    
            #--- RECOMBINATION (step #3.B) ----------------+
    
            v_trial = []

            # cycle through each variable in our target vector
            for r in range(len(x_t)):
    
                crossover = random.random()
                    # recombination occurs when crossover <= recombination rate
                if crossover <= recombination:
                    v_trial.append(v_donor[r])
                    # recombination did not occur
                else:
                    v_trial.append(x_t[r])
                  
            
            #--- GREEDY SELECTION (step #3.C) -------------+
            score_trial  = cost_func(v_trial,stock,order)
            score_target = cost_func(x_t,stock,order)
    
            if score_trial < score_target:
                population[j] = v_trial
                fitFunction[j] = score_trial
                gen_scores.append(score_trial)
                #print ('   Swap: >',score_trial, v_trial)
    
            else:
                #print ('   Same: >',score_target, x_t)
                gen_scores.append(score_target)
                    
        # Terminantion Criteria - Convergence
        if i > 1:
            if ((abs(gen_best - min(fitFunction)) <= FunctionTolerance)):
                counter = counter + 1

                if MaxStallIterations == counter:
                    i = maxiter + 1 
                    break
            else:
                counter=0
                    
        #--- SCORE KEEPING --------------------------------+
        gen_avg = sum(fitFunction) / popsize                            # current generation avg. fitness
        gen_best = min(fitFunction)                                     # fitness of best individual
        gen_sol = population[fitFunction.index(min(fitFunction))]       # solution of best individual
        
        # Print shapes in order
        global fig,axBestPack
        shifted = compute_shifted(order, gen_sol)
        axBestPack.clear()
        plotShapelyPoly(axBestPack, [stock]+shifted) 
        axBestPack.set_title('Best-so-far Packing')
        axBestPack.relim()
        axBestPack.autoscale_view()
        axBestPack.set_aspect('equal')
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        #print ('      > GENERATION AVERAGE:',gen_avg)
        print ('      > GENERATION BEST:',gen_best)
       # print ('      > BEST SOLUTION:',gen_sol,'\n')

    return gen_sol

#-------- ASSISTANT CODE ---------------------------------------------------+
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
    while i <stop_limit_i :
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
                bounds = [] 
                for k in range(nVars):
                    if(k%3==0):
                        bounds.append(((0,stock.bounds[2]-1),(0,stock.bounds[3]-1), (0,360)))
                        
                # Set Plots and call PSO Algorithm for a specific order and stock  
                cost_func = ObjectiveFunction                   # Cost function
                popsize = 0                                     # Population size, must be >= 4
                recombination = 0.9                             # Recombination rate [0,1]
                   
                global fig,axBestPack
                fig = plt.figure()
                axBestPack = plt.subplot(111)
                Np = 3*len(order)
                popsize = min(200, 10*Np)  
                maxiter = 100*Np
                global sol
                sol = DEGL(cost_func, bounds, popsize, recombination, maxiter,stock,order)

                shifted = compute_shifted(order, sol)
                remaining, opening = compute_remaining_opening(stock, shifted)

                # If the Order doesnt fit in Stock
                if opening.is_empty or check_if_order_is_inside_and_disjoint(shifted,stock) :
                    print(f'Polygons of Order {i} fitted to Stock {j}-----------------------------')
                    opening_stock[j] = opening
                    
                    # Check if after the placement of order the type is MultiPolygon
                    # A Multipolygon has bigger area than a Polygon
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
                nVars = 3*len(order)

                bounds = [] 
                # Set Upper and Lower Bounds
                for k in range(nVars):
                    if(k%3==0):
                        bounds.append(((0,stock.bounds[2]-1),(0,stock.bounds[3]-1), (0,360)))
     
                # Set Plots and call PSO Algorithm for a specific order and stock  
                Np = 3*len(order)
                cost_func = ObjectiveFunction                   # Cost function
                popsize = 0                                     # Population size, must be >= 4
                recombination = 0.8                             # Recombination rate [0,1]
                maxiter = 100*Np                                   # Max number of generations (maxiter)
                   
                global fig,axBestPack
                fig = plt.figure()
                axBestPack = plt.subplot(111)               
                popsize = min(200, 10*Np)  
                global sol
                sol = DEGL(cost_func, bounds, popsize, recombination, maxiter,stock,order)
                
                shifted = compute_shifted(order, sol)              
                remaining, opening = compute_remaining_opening(stock, shifted)
                opening_stock[j] = opening

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

    orders = [Order1 , Order2, Order3]
    
    # Rotation = 0 -> No Rotation
    # Rotation = 1 -> Yes Rotation
    rotation = 1
    
    random.seed(2)
    
    start = time.time()
    FitOrderingStock1(orders,Stock)
    end = time.time()
    print("Execution Time in Secs: ", end - start)
    