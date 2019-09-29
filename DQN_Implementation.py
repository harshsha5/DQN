#!/usr/bin/env python
import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.env = gym.make(environment_name)
        self.env.reset()
        # print("Observation space",self.env.observation_space)
        # print("Action",self.env.action_space)
        self.model = self.make_model()   

    def make_model(self):
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        model = keras.models.Sequential([
        keras.layers.Dense(30, input_dim=num_states, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(num_actions, activation='relu')]) 
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
        loaded_model.load_weights(weight_file)
        print("Loaded model")

class Replay_Memory():

    def __init__(self, env,policy,memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        
        # Hint: you might find this useful:
        #         collections.deque(maxlen=memory_size)
        self.max_mem = memory_size
        self.mem = generate_burn_in_memory(burn_in,env,policy)

    def transform_state(state,size):
        return npy.reshape(state, [1, size])

    def generate_burn_in_memory(self,burn_in,env,policy):
        state = env.reset()
        state = transform_state(state,self.env.observation_space.shape[0])
        for i in range(burn_in):
            action = np.argmax(model.predict(state), axis=1)[0]
            new_state, reward, done, info = env.step(action)
            new_state = transform_state(new_state)
            transition = np.array([state,action,reward,new_state,done])
            self.append(transition)
            state = new_state

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        return self.mem[np.random.choice(self.mem.shape[0], batch_size, replace=False), :]
        
    def append(self, transition):
        # Appends transition to the memory.     
        if(self.mem.shape[0]==0):
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
        self.env.reset()

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.             
        pass

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        pass 

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # When use replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        pass

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        pass

    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        pass

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
    q = QNetwork(environment_name)     #edit it for environment_name
    state = q.env.reset()
    state = npy.reshape(state, [1, 4])
    print("Hello")
    print(state.shape)
    print(type(state))
    print(q.model.predict(state))

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
    main(sys.argv)

