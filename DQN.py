
import time
import sys
import random
import numpy as np
import tensorflow as tf
from Environment import Env
from collections import deque
from tensorflow.keras.layers import Dense, Conv2D, Reshape, MaxPool2D, Activation ,Flatten
from tensorflow.keras.optimizers import Adam, RMSprop ,SGD
from tensorflow.keras.models import Sequential
import time

EPISODES = 5000
STATE_SZIE = (128,15, 15, 1)
#(batch_size,HEIGHT,WIDTH,depth))

class DQNAgent:
    def __init__(self):

        self.render = False
        self.load = False
        self.save_loc = './DQN'
        self.action_size = 4
        self.discount_factor = 0.98
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.99997
        self.epsilon_min = 0.01
        self.batch_size =128
        self.train_start = 500
        self.replace_target=200
        self.learn_step_counter = 1
        self.memory = deque(maxlen=2000)
        self.state_size = STATE_SZIE
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model(self)



    #define model
    def build_model(self):
        model=Sequential()
        model.add(Conv2D(64, (5, 5),  strides=1,  padding='same', input_shape= self.state_size[1:],activation='relu'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same',activation='relu'))
        model.add(Conv2D(256, (3, 3), strides=1, padding='same',activation='relu'))


        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))

        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
        
        return model
        
    def select_action(self, state):
        # select action using epsilon-greedy
       
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        else:
            #expand dims to 4
            state = np.expand_dims(state, axis=0)

            act = self.model.predict(state)
            return np.argmax(act[0])
      

    def MEMORY(self, state, action, reward, next_state, goal ,wumpus):
        # save elf, state, action, reward, next_state, goal ,wumpus  to the memory
        self.memory.append((state,action,reward,next_state,goal,wumpus))

       


    def update_target_model(self,goal_):
        # update target model if goal = True or counter=200
        if self.learn_step_counter %  self.replace_target == 0 or goal_ ==True:
            print('Updating target network...')
            self.target_model.set_weights(self.model.get_weights())
            self.learn_step_counter=1
         


    def train_replay(self,batch_size):
    #train model

        sample_data=random.sample(self.memory,batch_size)   #random sample from memory

        state = np.zeros((self.batch_size,15,15,1))     #define state matrix to process
        next_state = np.zeros((self.batch_size, 15,15,1))       #define next_state matrix to process
        action, reward, goal = [], [], []  #define list for append action reward goal


        for i in range(self.batch_size):        #loop for gather date in batch_size len
            state[i] = sample_data[i][0]        #fill state matrrix from sample data 
            action.append(sample_data[i][1])     #action apped to list from sample data 
            reward.append(sample_data[i][2])    #reward apped to list from sample data 
            next_state[i] = sample_data[i][3]   #fill next_state matrrix from sample data
            goal.append(sample_data[i][4])      #goal apped to list from sample data


        target = self.model.predict(state)   #predict target from model
        target_next = self.model.predict(next_state)   #predict target_next from model
        target_val = self.target_model.predict(next_state)  #validation target_next predict By predict from  target_model

        for i in range(len(sample_data)):
            if goal[i]:
                target[i][action[i]] = reward[i]  #if goal True target=reward    else target =rewrd+gamma*amax
            else:

                    a_max = np.argmax(target_next[i])  
                    target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a_max])   #belman eq


        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        self.learn_step_counter += 1

        #update epsilon 
    def update_epsilon(self):
         if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
             #   print(self.epsilon)


    # save the model which is under training
    def save_model(self):

        self.model.save('model_dqn')  
       # self.target_model.save('target_model') 
        print('model saved :)')
        

if __name__ == "__main__":

    # create environment
    env = Env()
    agent = DQNAgent()
    scores = []
    state = env.reset()
    summary=agent.model.summary()
    print(" model summary is",summary)

    for e in range(1,EPISODES):
        score = 0
        goal=False
        wumpus=False
        while (not goal) and (not wumpus):
            if agent.render:
                env.render()

            action=agent.select_action(state)
            next_state, reward, goal,wumpus = env.step(action)

            agent.MEMORY(state,action,reward,next_state,goal,wumpus)
            state=next_state
            score += reward
            if len(agent.memory) > agent.batch_size:
                 agent.train_replay(agent.batch_size)
      
            agent.update_epsilon()
            agent.update_target_model(goal_=False)
            

            if goal == True:
                goal_=True
                print("episode: {:3}   score: {:8.6} epsilon {:.5}".format(e, float(score), float(agent.epsilon)))
                agent.update_target_model(goal_)
                state = env.reset()

                break
               
               

            elif wumpus == True:
                print("episode: {:3}   score: {:8.6}    epsilon {:.5}".format(e, float(score), float(agent.epsilon)))
                state = env.reset()

                break





        scores.append(score)
       
        # save the model every 100 episodes
        
 
        if e % 100== 0:
            agent.save_model()
    env.close() 
