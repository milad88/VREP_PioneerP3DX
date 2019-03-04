import sys

from utility import *
from Critic_Network_DDPG import *
from Actor_Network_DDPG import *
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
    num_episodes = 500
    len_episode = 1
    epsilon = 0.1
    load = True
    save = True
    model_name = "Saved_models/DDPG/model"


    # policies = [make_epsilon_greedy_decay_policy, make_epsilon_greedy_policy, make_ucb_policy]

    g_stat = []

    with tf.Session() as sess:
        # name = policy.__name__  # No reason to save more than one from each policy atm
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

        # writer = tf.summary.FileWriter('./DDPG/log/DDPG_loss', sess.graph)
        # summ_critic_loss = tf.summary.scalar('loss_critic', critic.get_loss())
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
        DDPG
        """
        loss_episodes = []
        stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
        buffer = ReplayBuffer()


        for i_episode in range(num_episodes):

            last_reward = stats.episode_rewards[i_episode - 1]

            g_r = 0

            observation = env.reset()

            old_observation = observation

            action = actor.predict(sess, observation)
            observation, reward, done, info = env.step(action[0])

            buffer.add_transition(old_observation, action, observation, reward, done)
            s, a, ns, r, d = buffer.next_batch(batch_size)

            q_values = target_critic.predict(sess, ns)

            y = r + (discount_factor * q_values)
            # y = r + np.multiply(discount_factor, q_values)


            g_r += reward
            g_stat.append(int(np.round(g_r)))
            loss_critic = critic.update(sess, s, y, None)

            loss = loss_critic[0]
            #target_critic_out = target_critic.predict(sess,s,a)
            a_grads = critic.action_gradients(sess, s)

            print("\rEpisode {}/{} ({}) action {} Q-value {}".format(i_episode + 1, num_episodes, last_reward, action, q_values))
            sys.stdout.flush()
            #actor.update(sess, s, target_critic_out, None)
            actor.update(sess, s, a_grads, None)

            stats.episode_rewards[i_episode] += reward

            target_critic.update(sess)
            # target_actor.update(sess)

            g_stat.append(int(np.round(g_r)))

            summ_critic_loss = tf.Summary(value=[tf.Summary.Value(tag="loss_critic",
                                                                  simple_value=loss)])
            # writer.add_summary(summ_critic_loss, i_episode)

            # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_rewards",
            #                                                      simple_value=stats.episode_rewards[i_episode])]), i_episode)


            # writer.flush()
            loss_episodes.append(loss)

            stats.episode_lengths[i_episode] = 1

        plot_episode_stats(stats)
        plot_stats(loss_episodes)
        if save:
            actor.save(sess)
            critic.save(sess)
            target_actor.save(sess)
            target_critic.save(sess)

        # print(tf.get_variable("first").eval())
        # return stats, loss_episodes
