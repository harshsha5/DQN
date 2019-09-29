#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.env = gym.make(environment_name)
        self.model = self.make_model()   

    def make_model(self):
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        model = keras.models.Sequential([
        keras.layers.Dense(30, input_dim=num_states, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(num_actions, activation='linear')]) 
        model.compile(loss=keras.losses.mean_squared_error,
                     optimizer=tf.train.AdamOptimizer(),
                     metrics=['accuracy'])
        return model

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        # self.model.save_weights("model.h5")
        self.model.save_weights(suffix)
        print("Saved model in HDF5 format")

    def load_model(self, model_file):
        # Helper function to load an existing model.
        # e.g.: torch.save(self.model.state_dict(), model_file)
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        # e.g.: self.model.load_state_dict(torch.load(model_file))
        sel.model.load_weights(weight_file)
        print("Loaded model")

def transform_state(state,size):
    return np.reshape(state, [1, size])

class Replay_Memory():

    def __init__(self, env,policy,memory_size=50000, burn_in=40):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        
        # Hint: you might find this useful:
        #         collections.deque(maxlen=memory_size)
        self.max_mem = memory_size
        self.mem = np.empty(shape=[0, 5])       #The 5 corresponds to state,action,reward,new_state and done
        #self.generate_burn_in_memory(burn_in,env,policy)

    def generate_burn_in_memory(self,burn_in,env,policy):
        state = env.reset()
        state = transform_state(state,env.observation_space.shape[0])
        for i in range(burn_in):
            action = np.argmax(policy.predict(state), axis=1)[0]
            new_state, reward, done, info = env.step(action)
            new_state = transform_state(new_state,env.observation_space.shape[0])
            transition = np.array([state,action,reward,new_state,done])
            self.append(transition)
            if(done):
                state = transform_state(env.reset(),env.observation_space.shape[0])
            else:
                state = new_state

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        return self.mem[np.random.choice(self.mem.shape[0], batch_size, replace=False), :]
        
    def append(self, transition):
        # Appends transition to the memory.     
        if(self.mem is None):
            print("Initializing memory")
            self.mem = transition
        elif(self.mem.shape[0]<self.max_mem):
            self.mem = np.vstack((self.mem,transition))
        else:
            print("Memory full")

class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #        (a) Epsilon Greedy Policy.
    #        (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, render=False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env = gym.make(environment_name)
        self.q_net = QNetwork(environment_name)
        self.replay_mem = Replay_Memory(self.env,self.q_net.model)
        self.burn_in_memory()
        self.num_episodes = 1000
        self.num_epoch = 5
        #self.learning_rate = 0.05
        self.discount_factor = 0.9
        self.epsilon = 0.5
        self.train_frequency = 1
        self.num_test_episodes = 100 #This is for final testing- how many episodes should we average our rewards over
        self.evaluate_curr_policy_frequency = 10
        self.num_episodes_to_evaluate_curr_policy = 20

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.  
        random_number = np.random.rand()
        if(random_number<self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(q_values,axis=1)[0]

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        return np.argmax(q_values,axis=1)[0]

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # When use replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        for episode in range(self.num_episodes):
            # print("Training episode: ",episode)
            done = False
            state = self.env.reset()
            state = transform_state(state,self.env.observation_space.shape[0])
            while (not done):
                action = self.epsilon_greedy_policy(self.q_net.model.predict(state))
                new_state, reward, done, info = self.env.step(action)
                new_state = transform_state(new_state,self.env.observation_space.shape[0])
                transition = np.array([state,action,reward,new_state,done])
                self.replay_mem.append(transition)
            if(episode%self.train_frequency==0):
                # print("Training sample batch")
                self.train_batch()
            if((episode+1)%self.evaluate_curr_policy_frequency==0):
                print("Evaluating current policy", episode+1)
                print("Average reward over ",self.num_episodes_to_evaluate_curr_policy," episodes: ",test_present_policy(self.env,self.num_episodes_to_evaluate_curr_policy,self.q_net.model))
        self.q_net.save_model_weights("cart_pole_weights") #Change name/pass as argument

    def train_batch(self):
        data = self.replay_mem.sample_batch()
        for i in range(data.shape[0]):
            if(data[i][4]):
                target = data[i][2]
            else:
                target = data[i][2] + self.discount_factor*np.max(self.q_net.model.predict(data[i][3]), axis=1)[0]

            #Change the target for only the action's q value. This way only that get's updated
            present_output = self.q_net.model.predict(data[i][0])
            present_output[0][data[i][1]] = target
            self.q_net.model.fit(data[i][0],present_output,self.num_epoch,verbose=0)

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        self.q_net.load_model_weights(model_file)
        return test_present_policy(self.env,self.num_test_episodes,self.q_net.model)

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        burn_in = 40
        self.replay_mem.generate_burn_in_memory(burn_in,self.env,self.q_net.model)

def test_present_policy(env,num_episodes,policy):
    net_average_reward = 0
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        state = transform_state(state,env.observation_space.shape[0])
        present_reward = 0
        while (not done):
            action = np.argmax(policy.predict(state), axis=1)[0]
            new_state, reward, done, info = env.step(action)
            new_state = transform_state(new_state,env.observation_space.shape[0])
            state = new_state
            present_reward = present_reward + reward
        net_average_reward = net_average_reward + present_reward
    return net_average_reward/num_episodes

# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
# def test_video(agent, env, epi):
#     # Usage: 
#     #     you can pass the arguments within agent.train() as:
#     #         if episode % int(self.num_episodes/3) == 0:
#     #           test_video(self, self.environment_name, episode)
#     save_path = "./videos-%s-%s" % (env, epi)
#     if not os.pa th.exists(save_path):
#         os.mkdir(save_path)
#     # To create video
#     env = gym.wrappers.Monitor(agent.env, save_path, force=True)
#     reward_total = []
#     state = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action = agent.epsilon_greedy_policy(state, 0.05)
#         next_state, reward, done, info = env.step(action)
#         state = next_state
#         reward_total.append(reward)
#     print("reward_total: {}".format(np.sum(reward_total)))
#     agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)
    agent = DQN_Agent(environment_name) 
    agent.train() 

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
    main(sys.argv)

