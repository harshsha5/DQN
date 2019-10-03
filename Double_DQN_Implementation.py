#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import matplotlib.pyplot as plt 

from tensorboardX import SummaryWriter

test_step = 0

def get_huber_loss_fn(**huber_loss_kwargs):

    def custom_huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)

    return custom_huber_loss

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name, lr=0.0001):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.lr = lr
        self.env = gym.make(environment_name)
        self.model = self.make_model()   

    def make_model(self):
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        model = keras.models.Sequential([
        keras.layers.Dense(128, input_dim=num_states, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_actions, activation='linear')]) 
        model.compile(loss=get_huber_loss_fn(delta=0.5),
                     optimizer=tf.train.AdamOptimizer(learning_rate=self.lr),
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
        self.model.load_weights(weight_file)
        print("Loaded model")

def transform_state(state,size):
    return np.reshape(state, [1, size])

class Replay_Memory():

    def __init__(self, env,policy,memory_size=60000, burn_in=10000):

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
            #action = np.argmax(policy.predict(state), axis=1)[0]
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            new_state = transform_state(new_state,env.observation_space.shape[0])
            transition = np.array([state,int(action),reward,new_state,int(done)])
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
        # else:
        #     print("Memory full")

class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #        (a) Epsilon Greedy Policy.
    #        (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, args, render=False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.environment_name = environment_name
        self.env = gym.make(environment_name)
        self.q_net = QNetwork(environment_name, args.lr)
        self.copy_q_net = copy.deepcopy(self.q_net)
        self.replay_mem = Replay_Memory(self.env,self.q_net.model)
        self.burn_in_memory()
        self.num_episodes = args.num_episodes
        self.num_epoch = 1
        self.discount_factor = 1
        self.epsilon = args.eps
        self.epsilon_min = args.eps_min
        self.epsilon_decay = args.eps_decay
        self.eps_update_freq = args.eps_update_freq
        self.train_frequency = args.train_freq
        self.frame_skip = args.frame_skip
        self.num_test_episodes = 100 #This is for final testing- how many episodes should we average our rewards over
        self.evaluate_curr_policy_frequency = args.eval_pol_freq
        self.num_episodes_to_evaluate_curr_policy = 20
        self.target_policy_update_frequency = args.target_pol_update_freq
        self.reward_list = []
        self.reward_episode_nums = []
        self.td_error_list = []
        self.writer = SummaryWriter()

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.  
        random_number = np.random.rand()
        if(random_number<=self.epsilon):
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
        step_C = 0
        print("Begun training")
        for episode in range(self.num_episodes):
            # print("Training episode: ",episode)
            done = False
            state = self.env.reset()
            state = transform_state(state,self.env.observation_space.shape[0])
            while (not done):
                step_C+=1
                action = self.epsilon_greedy_policy(self.q_net.model.predict(state))
                new_state, reward, done, info = self.env.step(action)
                new_state = transform_state(new_state,self.env.observation_space.shape[0])
                transition = np.array([state,int(action),reward,new_state,int(done)])
                if(step_C%self.frame_skip == 0):
                    self.replay_mem.append(transition)
                state = new_state

                if(step_C % self.train_frequency==0):                    
                    self.train_batch(step_C)
                
                if(self.epsilon>self.epsilon_min and step_C%self.eps_update_freq == 0):
                    self.epsilon*=self.epsilon_decay
                
                if(episode % self.target_policy_update_frequency==0):
                    self.copy_q_net = copy.deepcopy(self.q_net)
                print("Step: ", step_C)

            print("Episode done: ", episode)
            
            if((episode)%self.evaluate_curr_policy_frequency==0):
                print("Evaluating current policy", episode+1)
                present_average_reward,average_td_loss = test_present_policy(self.env,self.num_episodes_to_evaluate_curr_policy,self.q_net.model,self.discount_factor,self.copy_q_net.model,self.writer)
                print("Average reward over ",self.num_episodes_to_evaluate_curr_policy," episodes: ",present_average_reward)
                self.writer.add_scalar("test/td-error", average_td_loss, int((episode)/self.evaluate_curr_policy_frequency))
                self.writer.add_scalar("test/reward", present_average_reward, int((episode)/self.evaluate_curr_policy_frequency))
                self.reward_list.append(present_average_reward)
                self.reward_episode_nums.append((episode+1)/self.evaluate_curr_policy_frequency)
                self.td_error_list.append(average_td_loss)

        self.q_net.save_model_weights(environment_name+"-weights") #Change name/pass as argument
        plot_graph(self.reward_episode_nums,self.reward_list,"reward")
        plot_graph(self.reward_episode_nums,self.td_error_list,"td_error")

    def train_batch(self, step):
        data = self.replay_mem.sample_batch()
        loss = 0
        acc = 0

        target = get_target_value_batch(data, self.copy_q_net.model, self.discount_factor)
        present_output_batch = self.q_net.model.predict(np.array(data[:,0].tolist()).squeeze())
        for i in range(data.shape[0]):
            present_output_batch[i,data[i][1]] = target[i]

        present_output_batch = np.squeeze(np.array(present_output_batch))
        data_batch = np.array(data[:,0].tolist()).squeeze()
        history = self.q_net.model.fit(data_batch,present_output_batch,self.num_epoch,verbose=0)
        loss +=history.history['loss'][-1]
        acc +=history.history['accuracy'][-1]

        # for i in range(data.shape[0]):
        #     target = get_target_value(data[i][2],data[i][4],data[i][3],self.copy_q_net.model,self.discount_factor)

        #     #Change the target for only the action's q value. This way only that get's updated
        #     present_output = self.q_net.model.predict(data[i][0])
        #     present_output[0][data[i][1]] = target
            
        #     present_output_batch.append(present_output)
        #     data_batch.append(data[i][0])
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.add_scalar('train/accuracy', acc, step)


    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        self.q_net.load_model_weights(model_file)
        reward,td_error = test_present_policy(self.env,self.num_test_episodes,self.q_net.model,self.discount_factor,self.q_net.model)  #VERIFY
        return reward

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        burn_in = 20000
        self.replay_mem.generate_burn_in_memory(burn_in,self.env,self.q_net.model)

def test_present_policy(env,num_episodes,policy,discount_factor,copy_policy,writer=None):
    global test_step
    test_step += 1
    net_average_reward = 0
    net_avg_td_error_per_episode = 0
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        state = transform_state(state,env.observation_space.shape[0])
        present_reward = 0
        num_steps = 0
        present_td_error = 0
        while (not done):
            prediction = policy.predict(state)
            action = np.argmax(prediction, axis=1)[0]
            new_state, reward, done, info = env.step(action)
            new_state = transform_state(new_state,env.observation_space.shape[0])
            present_td_error = present_td_error + abs(np.max(prediction, axis=1)[0] - get_target_value(reward,done,new_state,copy_policy,discount_factor))
            state = new_state
            present_reward = present_reward + reward
            num_steps+=1
        net_avg_td_error_per_episode = net_avg_td_error_per_episode + (present_td_error/num_steps)
        net_average_reward = net_average_reward + present_reward
    print("Average TD Error for:,",num_episodes," episodes ",net_avg_td_error_per_episode/num_episodes)
    
    return ((net_average_reward/num_episodes),(net_avg_td_error_per_episode/num_episodes))

def get_target_value_batch(data, copy_policy, discount_factor):
    d = np.array(data[:,3].tolist()).squeeze()
    target_batch = data[:,2] + (1-data[:,4])*discount_factor*np.max(copy_policy.predict(d), axis=1)
    return target_batch

def get_target_value(reward,done,new_state,policy,discount_factor):
    if(done):
        return reward
    else:
        return reward + discount_factor*np.max(policy.predict(new_state), axis=1)[0]

def plot_graph(x,y,y_axis): 
    plt.plot(x, y) 
    plt.xlabel('iterations') 
    if(y_axis=="td_error"):
        plt.ylabel('Average TD error per episode per time_step')
        plt.title('Average TD error VS No. of iterations') 
    elif(y_axis=="reward"):
         plt.ylabel('Average reward per episode')
         plt.title('Average reward VS No. of iterations') 
    plt.show() 



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
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)

    parser.add_argument('--lr',dest='lr',type=float,default=0.0001)
    parser.add_argument('--num_episodes',dest='num_episodes',type=int,default=5000)
    parser.add_argument('--eps',dest='eps',type=float,default=0.5)
    parser.add_argument('--eps_decay',dest='eps_decay',type=float,default=0.9995)
    parser.add_argument('--eps_min',dest='eps_min',type=float,default=0.05)
    parser.add_argument('--eps_update_freq',dest='eps_update_freq',type=int,default=10000)
    parser.add_argument('--train_freq',dest='train_freq',type=int,default=2)
    parser.add_argument('--discount_factor',dest='discount_factor',type=float,default=1)
    parser.add_argument('--eval_pol_freq',dest='eval_pol_freq',type=int,default=50)
    parser.add_argument('--target_pol_update_freq',dest='target_pol_update_freq',type=int,default=1000)
    parser.add_argument('--frame_skip',dest='frame_skip',type=int,default=2)

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
    agent = DQN_Agent(environment_name, args) 
    agent.train() 
    print("Final epsilon: ",agent.epsilon)
    net_avg_reward = agent.test(environment_name+"-weights")
    print("Net average test reward is: ",net_avg_reward)

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
    main(sys.argv)
