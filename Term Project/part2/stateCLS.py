################################################################
## This program is developed by A. Razi on 2022/4/19 to test  ##
## Value iteration and Qlearning Algorithm for CPSC 4820AI-UV ##
## Course Project. The code is only for education purpose and ##
## is not supported or maintained.                            ##
################################################################

class State:
    def __init__(self, row, col, stype=0, value=0, reward=0):
        self.row = row
        self.col = col
        self.stype = stype  #0: normal   1:blocked  2:terminal
        self.value = value  #value of this state [dynamic and changes with value iteration] 
        self.reward = reward  #reward for landing on this state, later we may need to fix it and it shouldn;t be necassary to have this, but current value iteration does not who the values properly 
        self.position = (self.row,self.col)
        self.possibleActions = []
        
    def updatePosition(self):
        self.position = (self.row,self.col)
    
    def setVal(self, stype, value, reward=0):
        self.stype = stype
        self.value = value
        self.reward = reward  #reward for being in this state, later we may need to fix it and it shouldn't be necassary to have this, but current value iteration does not show the values properly 

    def getVal(self):
        return self.stype, self.value, self.reward
    
    
    def display(self):
        print('State[',self.row,',',self.col,']  val=', self.value, ',  stype=', self.stype, '  reward=', self.reward)
        
        
    
    
                
        
