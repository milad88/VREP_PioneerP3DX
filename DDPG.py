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
    learning_rate = 0.0001
    discount_factor = 0.98
    num_episodes = 300
    len_episode = 10
    epsilon = 0.995
    decay = 0.994
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
                                learning_rate=learning_rate, batch_size=batch_size)

        with tf.variable_scope("actor_target"):
            target_actor = Actor_Target_Network(action_dim, "actor_target", action_bound, state_dim, actor,
                                                learning_rate=learning_rate, batch_size=batch_size)
        with tf.variable_scope("critic_target"):
            target_critic = Critic_Target_Network(action_dim, "critic_target", action_bound, state_dim, critic,
                                                  learning_rate=learning_rate, batch_size=batch_size)


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
        observation = env.first_reset()


        for i_episode in range(num_episodes):

            last_reward = stats.episode_rewards[i_episode - 1]

            print("\rEpisode {}/{} total rewrd {}".format(i_episode + 1, num_episodes, last_reward))
            sys.stdout.flush()
            g_r = 0
            i = 0
            done = False
            while i < len_episode and not done:

                old_observation = observation
                loss = []
                action = actor.choose(sess, observation, epsilon)

                observation, reward, done, info = env.step(action[0])

                buffer.add_transition(old_observation[0], action[0], observation[0], [reward], done)
                s, a, ns, r, d = buffer.next_batch(batch_size)

                acts = actor.choose(sess, ns, epsilon)
                q_values = target_critic.predict(sess, ns, acts)

                y = []
                j = 0
                for rew, don in zip(r, d):
                    if don:
                        y.append(rew)
                    else:
                        y.append(rew + discount_factor * q_values[j])
                    j += 1
                g_r += reward
                g_stat.append(np.round(g_r))
                loss_critic = critic.update(sess, s, a, y, None)

                loss.append(loss_critic[1])
                #target_critic_out = target_critic.predict(sess,s,a)
                a_grads = loss_critic[-1]

                print("\rGradient {} action {} reward {}".format(a_grads[0], action, reward))
                sys.stdout.flush()
                #actor.update(sess, s, target_critic_out, None)
                actor.update(sess, s, a_grads, None)

                stats.episode_rewards[i_episode] += reward

                target_critic.update(sess)
                # target_actor.update(sess)
                i+=1
            epsilon = epsilon * decay

            # summ_critic_loss = tf.Summary(value=[tf.Summary.Value(tag="loss_critic",
            #                                                       simple_value=sum(loss))])
            # writer.add_summary(summ_critic_loss, i_episode)

            # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_rewards",
            #                                                      simple_value=stats.episode_rewards[i_episode])]), i_episode)


            # writer.flush()
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

        # print(tf.get_variable("first").eval())
        # return stats, loss_episodes
