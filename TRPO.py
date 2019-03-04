import sys

from utility import *
from Critic_Network_TRPO import *
from Actor_Network_TRPO import *
import tensorflow as tf
from PioneerP3DX_env import PioneerP3DX

if __name__ == "__main__":
    print("start")
    env = PioneerP3DX()
    action_space = env.action_space

    action_dim = env.get_action_space()

    action_bound = env.action_space.high[0]
    state_dim = env.observation_space.shape
    batch_size = 32
    learning_rate = 0.001
    discount_factor = 0.98
    num_episodes = 500 #the one we need
    len_episode = 1
    epsilon = 0.1
    load = False
    save = True
    model_name = "Saved_models/TROP/model"

    g_stat = []

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 8
    config.gpu_options.per_process_gpu_memory_fraction = 0.33

    with tf.Session(config=config)as sess:

        print("building networks...")

        with tf.variable_scope("actor"):
            actor = Actor_Net(action_dim, "actor", action_bound, state_dim,
                              learning_rate=learning_rate, batch_size=batch_size)

        with tf.variable_scope("critic"):
            critic = Critic_Net(action_dim, "critic", action_bound, state_dim,
                                learning_rate=learning_rate)

        with tf.variable_scope("actor_target"):
            target_actor = Actor_Target_Network(action_dim, "actor_target", action_bound, state_dim,
                                                learning_rate=learning_rate)
        with tf.variable_scope("critic_target"):
            target_critic = Critic_Target_Network(action_dim, "critic_target", action_bound, state_dim,
                                                  learning_rate=learning_rate)

        print("networks built.")

        if load:
            print("loading networks parameters....")
            actor.load(sess)
            critic.load(sess)
            target_actor.load(sess)
            target_critic.load(sess)
            print("done.")
        else:
            sess.run(tf.global_variables_initializer())

        """
        Trpo
        """
        loss_episodes = []
        stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
        buffer = ReplayBuffer()
        observation = env.reset()

        for i_episode in range(num_episodes):
            # Also print reward for last episode
            last_reward = stats.episode_rewards[i_episode - 1]
            print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
            sys.stdout.flush()

            done = False
            g_r = 0



            first = True
            if i_episode != 0:
                first = False
                sess.graph.clear_collection("theta_sff")

            old_observation = observation
            action = actor.predict(sess, observation)
            observation, reward, done, info = env.step(action[0])

            buffer.add_transition(old_observation, action, observation, reward, done)
            s, a, ns, r, d = buffer.next_batch(batch_size)

            q_values = critic.predict(sess, ns)

            r = np.reshape(r,[-1,1])
            y = q_values - r

            g_r += reward
            g_stat.append(int(np.round(g_r)))

            loss_critic = critic.update(sess, s, y, None)


            sys.stdout.flush()

            actor.update(sess, s, y, None, first)

            stats.episode_rewards[i_episode] += reward

            g_stat.append(int(np.round(g_r)))


            loss_episodes.append(loss_critic)

            stats.episode_lengths[i_episode] = 1

        plot_episode_stats(stats)
        plot_stats(loss_episodes)

        if save:
            actor.save(sess)
            critic.save(sess)
            target_actor.save(sess)
            target_critic.save(sess)
