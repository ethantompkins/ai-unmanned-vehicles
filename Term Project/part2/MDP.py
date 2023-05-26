################################################################
## This program is developed by A. Razi on 2022/4/19 to test  ##
## Value iteration and Qlearning Algorithm for CPSC 4820AI-UV ##
## Course Project. The code is only for education purpose and ##
## is not supported or maintained.                            ##
################################################################

import random
import numpy as np
from stateCLS import State

# lists to save values over iterations
c1_1 = []
c1_4 = []
c2_2 = []
c2_4 = []
c3_3 = []

#Params
big=100000
small = 0.00001

gamma = 0.90  #discount factor
alpha = 0.5
nRows, nCols = (4, 6)
T = 2000

noise = 0.0  #0.2   #Noise factor: take best action by 1-nF, others by nF/(1-nPossibleActions)

#define Actions
Actions = {0:'^',1:'v',2:'<',3:'>',4:'x'}
nActions = 4
DoNothing = 4

#define States 
States = np.array([[State(i,j,0,0,0) for j in range(nCols)] for i in range(nRows)])

#define Policies: Policy[i,j,k] is the policy for row i, col j, time t
Policy = np.array([[DoNothing for j in range(nCols)] for i in range(nRows)])
PolicyStr = np.array([['-' for j in range(nCols)] for i in range(nRows)], dtype=object)
Values = np.array([[0 for j in range(nCols)] for i in range(nRows)])
QValues = np.array([[[0 for a in range(nActions)] for j in range(nCols)] for i in range(nRows)])

#define time indexed maps for mostly debugging purpose
PolicyT = np.array([[[DoNothing for t in range(T)] for j in range(nCols)] for i in range(nRows)])
ValuesT = np.array([[[0 for t in range(T)] for j in range(nCols)] for i in range(nRows)])
QValuesT = np.array([[[[0 for t in range(T)] for a in range(nActions)] for j in range(nCols)] for i in range(nRows)])

#Flatten States if needed
if False:
    StatesF = np.ndarray.flatten(States)
    print('States[2,3].position:', States[2,3].position)
    print('StatesF[0,3,5]=', StatesF[0].position, StatesF[3].position, StatesF[5].position)



#define Reward Values
def initializeGRid():
    #terminal states
    States[1,5].setVal(2,100)    #Goal State
    States[0,5].setVal(2,-100)   #Terminal
    States[2,3].setVal(0,0,-10)   #burden [negative reward for landing]

    #block state
    blocked = [[0,1],[1,3],[2,0],[2,1],[2,5],[3,4]]
    for i in range(len(blocked)):
        States[blocked[i][0],blocked[i][1]].setVal(1,0)  #stype=1:blocked

 
    #Update Value Array Accordingly
    for i in range (nRows):
        for j in range(nCols):
            Values[i,j,]=States[i,j].value
            ValuesT[i,j,0]=States[i,j].value
            
    #define Possible Actions for each state            
    for i in range(0, nRows):
        for j in range(0, nCols):
            States[i,j].possibleActions=[]
            for action in range(0,nActions):
                (os,flag) = checkAction(States[i,j],action,False)
                if flag:
                    States[i,j].possibleActions.append(action)
    
#initializeGRid()

def printGrid(mode=0,pause=True):
    
    if mode == 0: #print board
        
        print('\nDetails: ')
        for i in range(0, nRows):
            for j in range(0, nCols):
                print('S(', i, ',', j, ')=pos:', States[i,j].position, ' value:', States[i,j].value, '  reward:', States[i,j].reward, '  type:', States[i,j].stype)
                    
        input("Press Enter to continue...") 
        
        print('\nBoard: ')
        for i in range(0, nRows):
            for j in range(0, nCols):
                stype,val,reward=States[i,j].getVal() 
                if  stype==1: #blocked
                    print('|  B  ', end=' ')    
                elif  stype==2: #terminal
                    print('|T:{} '.format(val), end=' ')    
                elif stype==0 and val!=0: #normal with value
                    print('|{} '.format(val), end=' ')    
                elif stype==0 and reward!=0: #normal with value
                    print('|r:{} '.format(reward), end=' ')    
                else:  #normal
                    print('|     ', end=' ')    
                
            print('\n')
                
    else:
        if mode==1 or mode==2:  #print Policy, State Values
            print('\nPolicy: ')
            for i in range(0, nRows):
                for j in range(0, nCols):
                    print(Actions[Policy[i,j]], end='   ,')
                print('\n')
    
            print('\n State Values: ')
            for i in range(0, nRows):
                for j in range(0, nCols):
                    print(Values[i,j], end='   ,')
                    if (i == 1 and j == 1):
                        c1_1.append(Values[i,j])
                    elif (i == 2 and j == 2):
                        c2_2.append(Values[i,j])
                    elif(i == 2 and j == 4):
                        c2_4.append(Values[i,j])
                    elif(i == 1 and j == 4):
                        c1_4.append(Values[i,j])
                    elif(i == 3 and j == 3):
                        c3_3.append(Values[i,j])
                    
                print('\n')

        if mode==2:  #State Q Values
           print('\n State Q-Values: ')
           for i in range(0, nRows):
               for j in range(0, nCols):
                    print('Q:',QValues[i,j,0], '-', QValues[i,j,1], '-',QValues[i,j,2], '-',QValues[i,j,3],  end='   |')
               print('\n|')

    if pause:
        input("Press Enter to continue...")    
                        
#define state transition matrix
def applyAction(state,action,noise=0,debug=False):
    #noise: a number between 0 to 1: uncertainty in the action

    Probs=np.array([0 for i in range(nActions)], dtype = float)
    nextStates = {}

    if noise==0:   #deterministic
        
        print(state.possibleActions)
        if action in state.possibleActions:
            Success=True
            #print('success')
            Probs[action]=1
        else:
            #Probs[4]=1   #action is staying or doing nothing!!!
            #print('fail')
            Success=False

    else:

        Success = True
        nPossibleActions = len(state.possibleActions)
        if nPossibleActions == 0:
            Success = False
        
        else: 
            #split noise among n-1 uninetended actions, and assign 1-noise to intended 
            if action in state.possibleActions:
                for a in range(nActions):
                    if (a != action) and (a in state.possibleActions):
                        Probs[a]= noise / (nPossibleActions-1)
                    if (a == action):
                        Probs[a]= 1 - noise 
                
            else:  #split all noise amonng uninetnded actions
                for a in range(nActions):
                    if (a in state.possibleActions):
                        Probs[a]= noise / (nPossibleActions)

        #note: the probs do not sum up to one, [because the rest of Prob. is just for doing nothing]
        #for example if the intended action is impossible, it sums up to noise < 1!!!

    for a in range(nActions):
        if Probs[a]>0:
            (next_st,success)=checkAction(state,a,False)
            if success:
                nextStates[a]=next_st
        else:
            nextStates[a]=state  #remain in the same state
    return Success,Probs,nextStates




    

def checkAction(state,action,debug):

    Success = True
    outstate=state   
    (row,col)=state.position

    if debug:
        print('Check Action--> row:',row, ' col:', col, '  action: ', action)
    if Actions[action] == 'v':   #was is 's'
        row = row+1
    elif Actions[action] == '^':
        row = row-1
    elif Actions[action] == '>':
        col = col+1
    elif Actions[action] == '<':
        col = col-1

    if debug:
        print('Trying to go to row:',row, ' col:', col, ' by action: ', action)

    if row<0 or row>=nRows or col<0 or col>=nCols:  #out of range
        Success = False
    elif States[row,col].stype==1:  #impossible acction to blocked state
        Success = False

    if Success:
        outstate=States[row,col]
        if debug:
            print('Success: Transaction from state:',state.position ,' by action:',Actions[action] , '  to state:', outstate.position) 
    if not(Success) and debug:
        print('Failed: Transaction from state:',state.position ,' by action:',Actions[action] , '  to state:', outstate.position) 
    return outstate,Success







def testActions(): #test some actions
    sc=np.array([-1 for i in range(6)], dtype=object)
    sc[0]={'row':0,'col':0,'action':1,'noise':0,'msg':'Determinsitic MDP: Impossible Action: hitting blocked cell'}
    sc[1]={'row':0,'col':0,'action':3,'noise':0,'msg':'Determinsitic MDP: Possible Action'}
    sc[2]={'row':3,'col':0,'action':0,'noise':0.2,'msg':'Probabilisitic MDP: Impossible Desired Action'}
    sc[3]={'row':0,'col':0,'action':1,'noise':0.2,'msg':'Probabilisitic MDP: Desired Action is the Only Possible Action'}
    sc[4]={'row':2,'col':3,'action':3,'noise':0.2,'msg':'Probabilisitic MDP: Desired Action and other Actions Possible Action'}
    sc[5]={'row':3,'col':5,'action':3,'noise':0.2,'msg':'Probabilisitic MDP: No Possible Action Exist'}

    
    for i in range(len(sc)):
        print('\n\n\nState:',sc[i]['row'],sc[i]['col'],'  Action:',sc[i]['action'], ' Noise:',sc[i]['noise']) 
        (Success,Probs,nextStates) = applyAction(States[sc[i]['row'],sc[i]['col']],sc[i]['action'],  sc[i]['noise'],True)
        print('Possible Next States with Probs: ')
        for j in range(len(nextStates)):
            print(nextStates[j].position,' a:', Actions[j], ':P=', Probs[j])
        input(sc[i]['msg'])
        
    
#testActions()




        
def value_iteration(debug=False):
    for t in range (1, T):
        for i in range (0, nRows):
            for j in range (0, nCols):

                state = States[i,j]
                if state.stype != 0:  
                    ValuesT[i,j,t]=ValuesT[i,j,t-1]  #frozen states
                
                
                else: #find the best policy only for normal states
                    best_action = Policy[i,j] #Previous
                    best_value = Values [i,j] #used to be -big, later check
                    
                    for action in state.possibleActions:
                        Success,Probs,nextStates= applyAction(state, action, noise, False)
                        if np.sum(Probs) > 0:  #some actions, even noisy ones, are possible

                            #calculate E[R(s,a,s') + gamma*V(s')]
                            value=0
                            for ii in range(len(Probs)):
                                value = value + Probs[ii]*(nextStates[ii].reward+ gamma * Values[nextStates[ii].row,nextStates[ii].col])
                            
                            if value > best_value:
                                best_value=value
                                best_action=action


                    state.value=best_value
                    ValuesT[i,j,t]=best_value
                    PolicyT[i,j,t]=best_action
                    
                    #print('State: ', state.position ,'  Best Action:', Actions[best_action], ' Value change:',  Values[state.value,best_value])
                    
        for i in range (0, nRows):
            for j in range (0, nCols):
                if States[i,j].stype == 0:
                    Values[i,j]=ValuesT[i,j,t]
                    Policy[i,j]=PolicyT[i,j,t]
                

        if debug:
            printGrid(1,True)
            




def Qvalue_iteration(debug):
    for t in range (1, T):
        for i in range (0, nRows):
            for j in range (0, nCols):
                state = States[i,j]
                for a in range(0, nActions):    
                    if state.stype != 0:  
                        QValuesT[i,j,a,t]=QValuesT[i,j,a,t-1]  #frozen states
                    
                    
                    else: #find the best policy only for normal states
                    
                        #lets keep doing what we were doing by default
                        prev_best_action = Policy[i,j] #Previous
                        prev_best_value = Values [i,j] #used to be -big
                        
                        if a in state.possibleActions:  #valid action
                            #print('checking (i,j),a,t:',i,j,a,t)

                            Success,Probs,nextStates= applyAction(state, a, noise, False)
                            if np.sum(Probs) > 0:  #some actions, even noisy ones, are possible

                                #calculate E[R(s,a,s') + max Q(s', a')]
                                Qvalue=0
                                for ii in range(len(Probs)):
                                    next_st = nextStates[ii]
                                    if next_st.stype == 0:
                                        nextStateQvalue = gamma*np.max(QValuesT[next_st.row,next_st.col,0:nActions,t-1])
                                        nextStateQvalue = nextStateQvalue + next_st.reward
                                    else:
                                        nextStateQvalue = gamma*next_st.value + next_st.reward
                                    
                                    Qvalue = Qvalue + Probs[ii]*nextStateQvalue
                                    
                                #was within the for loop!!!, later check    
                                if Qvalue > QValues [i,j,a]:   #This check might not be necessary or even correct!!!!
                                    QValuesT[i,j,a,t]= Qvalue
                                        #print('update: ', Qvalue)
                                else:
                                    QValuesT[i,j,a,t]=QValuesT[i,j,a,t-1]
                                    #print('not update: ', QValuesT[i,j,a,t])
                                    

                                #XXXprint('State: ', state.position ,'  Action:', Actions[action], '  NextState:',next_st.position, ' Value change:',  Values[state.value,value])
                                #XXXinput("Press Enter to continue...")

                    
        for i in range (0, nRows):
            for j in range (0, nCols):
                if States[i,j].stype == 0: #regular cells
                    for a in range (0, nActions):
                        QValues[i,j,a]=QValuesT[i,j,a,t]
                    Qvec = QValues[i,j,0:nActions]
                    best_action = np.argmax(Qvec)
                    best_val = max(Qvec)
                    if best_val<= prev_best_value or  best_action < 0 or best_action >nActions -1 or best_action not in States[i,j].possibleActions:
                        best_action=DoNothing
                    Policy[i,j]=best_action
                    Values[i,j]=best_val
                    States[i,j].value = max(Qvec)#Later check
                

        if debug:
            printGrid(2,True)

def Q_learning(debug):
    for t in range(1,T):
        # start at random state
        while(True): 
            r = np.random.randint(0,nRows)
            c = np.random.randint(0,nCols)
            current_state = States[r,c]
            if current_state.stype == 0:
                break
        while current_state.stype == 0:
            best_action = None
            possible_actions = current_state.possibleActions
            if len(possible_actions) == 0:
                break
            next_action = possible_actions[np.random.randint(0,len(possible_actions))]
            Success,Probs,nextStates= applyAction(current_state, next_action, .5, False)
            
            true_action = random.choices([0,1,2,3], weights=Probs, k=1)[0]
            next_state = nextStates[true_action]
            
            QValues[r,c,true_action] = (1 - alpha) * QValues[r,c,true_action] + alpha * (gamma * next_state.value + next_state.reward)
            
            QValuesT[r,c,true_action,t] = QValues[r,c,true_action]
            
            for a in range (0, nActions):
                QValues[r,c,a]=QValuesT[r,c,a,t]
            Qvec = QValues[r,c,0:nActions]
            best_action = np.argmax(Qvec)
            best_val = max(Qvec)
            if best_val > Values[r,c]:
                if best_action < 0 or best_action > nActions -1 or best_action not in States[r,c].possibleActions:
                    best_action=DoNothing
                Policy[r,c]=best_action
                Values[r,c]=best_val     
                current_state.value = best_val       
      
            # print(f"Current state = {r},{c}")
            # print(f"Selected action: {true_action} : {Actions[true_action]}")
            # print(f"Next State = {next_state.row},{next_state.col}")
            
            current_state = next_state
            r = current_state.row
            c = current_state.col
            
        
        
        if debug:
            printGrid(2,True)
        else:
            for i in range(nRows):
                for j in range(nCols):
                    if (i == 1 and j == 1):
                        c1_1.append(Values[i,j])
                    elif(i == 1 and j == 4):
                        c1_4.append(Values[i,j])
                    elif(i == 3 and j == 3):
                        c3_3.append(Values[i,j])

def Q_learning_greedy(debug):
    for t in range(1,T):
        # start at random state
        while(True): 
            r = np.random.randint(0,nRows)
            c = np.random.randint(0,nCols)
            current_state = States[r,c]
            if current_state.stype == 0:
                break;
        while current_state.stype == 0:
            best_action = []
            possible_actions = current_state.possibleActions
            if len(possible_actions) == 0:
                break
                        
            best_q = -10**10
            # find the best action based on highest q_value
            # if there are multiple of the same highest q_values, take a random one of them as the highest
            for a in possible_actions:
                if QValues[r,c,a] > best_q:
                    best_q = QValues[r,c,a]
                    best_action = [a]
                elif QValues[r,c,a] == best_q:
                    best_action.append(a)
                # pick best action based on q value, with some randomness (if all q values are the same, do random action)
            
            # if all actions are the same q_value, pick one at random
            # else, pick a random action, giving the highest-valued action the highest priority
            if len(best_action) == len(possible_actions):
                next_action = np.random.randint(0, len(possible_actions))
            else:
                b_index = np.random.randint(0,len(best_action))
                best_action = best_action[b_index]
                probs = []
                for a in possible_actions:
                    if a == best_action:
                        probs.append(0.9)
                    else:
                        probs.append(0.1/(len(possible_actions)-1))
                next_action = random.choices(possible_actions, weights=probs, k=1)[0]

                
            Success,Probs,nextStates= applyAction(current_state, next_action, .5, False)
            
            true_action = random.choices([0,1,2,3], weights=Probs, k=1)[0]
            next_state = nextStates[true_action]
            current_state.value = (1 - alpha) * current_state.value + alpha * (gamma * next_state.value + next_state.reward)
            
            QValues[r,c,true_action] = current_state.value
            QValuesT[r,c,true_action,t] = current_state.value
            
            for a in range (0, nActions):
                QValues[r,c,a]=QValuesT[r,c,a,t]
            Qvec = QValues[r,c,0:nActions]
            best_action = np.argmax(Qvec)
            best_val = max(Qvec)
            if best_val > Values[r,c]:
                if best_action < 0 or best_action > nActions -1 or best_action not in States[r,c].possibleActions:
                    best_action=DoNothing
                Policy[r,c]=best_action
                Values[r,c]=best_val            
      
            # print(f"Current state = {r},{c}")
            # print(f"Selected action: {true_action} : {Actions[true_action]}")
            # print(f"Next State = {next_state.row},{next_state.col}")
            
            current_state = next_state
            r = current_state.row
            c = current_state.col
            
             
        
            
        
        
        if debug:
            printGrid(2,True)
        else:
            for i in range(nRows):
                for j in range(nCols):
                    if (i == 1 and j == 1):
                        c1_1.append(Values[i,j])
                    elif(i == 1 and j == 4):
                        c1_4.append(Values[i,j])
                    elif(i == 3 and j == 3):
                        c3_3.append(Values[i,j])
            




