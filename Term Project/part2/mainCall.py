################################################################
## This program is developed by A. Razi on 2022/4/19 to test  ##
## Value iteration and Qlearning Algorithm for CPSC 4820AI-UV ##
## Course Project. The code is only for education purpose and ##
## is not supported or maintained.                            ##
################################################################


import MDP
import matplotlib.pyplot as plt

print("This program simulates a robot that moves on a gris")
print("The grid has terminating states (goal + game over) and some block states")
print("We assume that the robot can move one sopt to up, down, left, and right at each step.")
print("We will check value iteration and Q-val iteration for known MDP")

print("entering the corresponding letter")
print("\tP: Print Details")
print("\tT: Test Actions")
print("\tV: Known MDP: Value Iteration ")
print("\tQ: Known MDP: QValue Iteration ")
print("\tL: Unknown MDP: Q Learning ")
print("\tG: Unknown MDP: Q Learning Greedy ")



val = input('Enter: ').upper()
MDP.initializeGRid()


if val == "P":
    MDP.printGrid(0,False)
    MDP.printGrid(1,False)
    

if val == "T":
    MDP.testActions()
    
if val == "V":
    MDP.value_iteration(True)
    
    
if val == "Q":
    MDP.Qvalue_iteration(True)
    
if val == "L":
    MDP.Q_learning(False)
    
if val == "G":
    MDP.Q_learning_greedy(False)
    
plt.plot(MDP.c1_1, label='(1,1)')
plt.plot(MDP.c1_4, label='(1,4)')
plt.plot(MDP.c3_3, label='(3,3)')
plt.title("Q-Learning Greedy: Alpha = 0.5; Epsilon 0.9")
plt.legend()
plt.show()

