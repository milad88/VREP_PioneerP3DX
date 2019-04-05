import sys

from utility import *
from Critic_Network_TRPO import *
from Actor_Network_TRPO import *
import tensorflow as tf
from Pioneer_interface import imshow
from PioneerP3DX_env import PioneerP3DX
from Exploration_Noise import ExplorationNoise

if __name__ == "__main__":
    print("start")
    env = PioneerP3DX()
    action_space = env.action_space

    action_dim = env.get_action_space()

    action_bound = env.action_space.high[0]
    state_dim = env.observation_space.shape
    batch_size = 32
    learning_rate = 0.01
    discount_factor = 0.98
    num_episodes = 500 #the one we need
    len_episode = 15
    epsilon = 0.1
    load = True
    save = True
    model_name = "Saved_models/TROP/model"

    # Ornstein-Uhlenbeck variables
    OU_THETA = 0.15
    OU_MU = 0.
    OU_SIGMA = 0.3
    # MAX_STEPS_EPISODE = 1000
    EXPLORATION_TIME = int(0.75 * len_episode)

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
            target_actor = Actor_Target_Network(action_dim, "actor_target", action_bound, state_dim, actor,
                                                learning_rate=learning_rate)
        with tf.variable_scope("critic_target"):
            target_critic = Critic_Target_Network(action_dim, "critic_target", action_bound, state_dim, critic,
                                                  learning_rate=learning_rate)

        print("networks built.")
        # sess.run(tf.global_variables_initializer())

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
        observation = env.first_reset()

        first = True
        for i_episode in range(num_episodes):
            # Also print reward for last episode
            last_reward = stats.episode_rewards[i_episode - 1]
            # print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")

            done = False
            g_r = 0
            i = 0

            noise = ExplorationNoise.ou_noise(OU_THETA, OU_MU, OU_SIGMA, len_episode)
            noise = ExplorationNoise.exp_decay(noise, EXPLORATION_TIME)

            while i < len_episode and not done:
                loss = []

                old_observation = observation
                action = actor.predict(sess, observation) [0] + noise[i]

                observation, reward, done, info = env.step(action)
                print("\rEpisode {}/{} ({}) action {} ".format(i_episode + 1, num_episodes, last_reward, action))

                buffer.add_transition(old_observation[0], action, observation[0], [reward], done)
                s, a, ns, r, d = buffer.next_batch(batch_size)

                acts = target_actor.predict(sess, ns)
                q_values = target_critic.predict(sess, ns, acts)# q values are fine
                y = []
                j = 0
                for rew, don in zip(r, d):
                    if don:
                        y.append(rew)
                    else:
                        y.append(rew + discount_factor * q_values[j])
                    j += 1
                g_r += reward
                g_stat.append(int(np.round(g_r)))

                loss_critic = critic.update(sess, s, a, y, None)# critic loss i sfine

                loss.append(loss_critic)
                actor.update(sess, s, y, None, first)

                stats.episode_rewards[i_episode] += reward

                g_stat.append(int(np.round(g_r)))
                i += 1

                print(i, " episode is done")

                first = False

            loss_episodes.append(sum(loss))

            stats.episode_lengths[i_episode] = i
            observation = env.reset()

            if (i_episode + 1) % 30 == 0:
                actor.save(sess)
                critic.save(sess)
                target_actor.save(sess)
                target_critic.save(sess)

        plot_episode_stats(stats)
        plot_stats(loss_episodes)

        if save:
            actor.save(sess)
            critic.save(sess)
            target_actor.save(sess)
            target_critic.save(sess)
