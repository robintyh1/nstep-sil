import numpy as np
import os
import tensorflow as tf
import gym
import time
import random
import core
import nstep_wrapper
from core import get_vars
from spinup.utils.logx import EpochLogger
import delay_wrapper
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0.0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


class BaseReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, ret):
        data = (obs_t, action, reward, obs_tp1, done, ret)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def store(self, obs, act, rew, next_obs, done, ret):
        self.add(obs, act, rew, next_obs, done, ret)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, rets = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, ret = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            rets.append(ret)
        return dict(obs1=np.array(obses_t), acts=np.array(actions), rews=np.array(rewards), obs2=np.array(obses_tp1), done=np.array(dones), ret=np.array(rets))

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_batch(self, batch_size):
        return self.sample(batch_size)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, weights, idxes

    def sample_batch(self, batch_size, beta):
        return self.sample(batch_size, beta)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

"""
TD3 (Twin Delayed DDPG)
"""
def td3(env_fn, env_fn_test, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1, logdir=None, nstep=None, alpha=None, beta=None, sil_weight=None):
    """
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TD3.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        pi_lr (float): Learning rate for policy.
        q_lr (float): Learning rate for Q-networks.
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)
        target_noise (float): Stddev for smoothing noise added to target 
            policy.
        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.
        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """
    assert logdir is not None
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    sess = tf.Session()

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn_test()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
    x_ph_sil, a_ph_sil, x2_ph_sil, r_ph_sil, d_ph_sil = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
    
    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    with tf.variable_scope('main', reuse=True):
        _, q1_sil, q2_sil, _ = actor_critic(x_ph_sil, a_ph_sil, **ac_kwargs)

    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
    
    with tf.variable_scope('target', reuse=True):
        pi_targ_sil, _, _, _  = actor_critic(x2_ph_sil, a_ph_sil, **ac_kwargs)

    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)

    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ_sil), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ_sil + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, q1_targ_sil, q2_targ_sil, _ = actor_critic(x2_ph_sil, a2, **ac_kwargs)

    # Experience buffer
    replay_buffer = BaseReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Prioritized replay for expert data
    sil_replay_buffer = PrioritizedReplayBuffer(size=replay_size, alpha=alpha)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    backup_discount = gamma
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + backup_discount*(1-d_ph)*min_q_targ)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss = q1_loss + q2_loss

    # sil q loss
    weights_ph = tf.placeholder(tf.float32, [None])
    ret_ph = tf.placeholder(tf.float32, [None])
    backup_sil = ret_ph

    # TD3 losses
    gains_1 = tf.nn.relu(backup_sil-q1_sil)
    gains_2 = tf.nn.relu(backup_sil-q2_sil)
    q1_loss_sil = tf.reduce_mean(weights_ph * tf.square(gains_1))
    q2_loss_sil = tf.reduce_mean(weights_ph * tf.square(gains_2))
    q_loss_sil = q1_loss_sil + q2_loss_sil
    gains = gains_1 + gains_2

    # add to the q loss
    q_loss += sil_weight * q_loss_sil

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        # test recorder
        ep_ret_list = []
        # set up
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0).flatten())
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            ep_ret_list.append(ep_ret)
        return ep_ret_list

    olist, alist, rlist, o2list, dlist = [], [], [], [], []    

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # record training
    ep_ret_record = []
    time_step_record = []

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a = get_action(o, act_noise).flatten()
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d
        if 'nstep_data_1' in info.keys():
            info['nstep_data_1'][-1] = d
        if 'nstep_data_{}'.format(nstep) in info.keys():
            info['nstep_data_{}'.format(nstep)][-1] = d

        # Store experience to replay buffer
        if 'nstep_data_1' in info.keys():
            replay_buffer.store(*info['nstep_data_1'])
            if nstep == 1:
                try:
                    assert info['nstep_data_1'] == [o, a, r, o2, d]
                except:
                    import pdb
                    pdb.set_trace()
        olist.append(o)
        alist.append(a)
        rlist.append(r)
        o2list.append(o2)
        dlist.append(d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all TD3 updates at the end of the trajectory
            (in accordance with source code of TD3 published by
            original authors).
            """
            retlist = list(discount_with_dones(rlist, dlist, gamma))
            for o, a, r, o2, d, ret in zip(olist, alist, rlist, o2list, dlist, retlist):
                sil_replay_buffer.store(o, a, r, o2, d, ret)

            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                batch_sil, weights, batch_idxes = sil_replay_buffer.sample_batch(batch_size, beta=beta)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             x_ph_sil: batch_sil['obs1'],
                             x2_ph_sil: batch_sil['obs2'],
                             a_ph_sil: batch_sil['acts'],
                             r_ph_sil: batch_sil['rews'],
                             d_ph_sil: batch_sil['done'],
                             ret_ph: batch_sil['ret'],
                             weights_ph: weights
                            }
                q_step_ops = [q_loss, q1, q2, train_q_op] + [gains]
                outs = sess.run(q_step_ops, feed_dict)
                logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                # get the priorities
                new_priorities = outs[-1] + 1e-8
                sil_replay_buffer.update_priorities(batch_idxes, new_priorities)
                #print_stats('new priorities', new_priorities)

                if j % policy_delay == 0:
                    # Delayed policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            olist, alist, rlist, o2list, dlist = [], [], [], [], []

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            ep_rets = test_agent()
            ep_ret_record.append(np.mean(ep_rets))
            time_step_record.append(t)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # save the records
            np.save(logdir + '/ep_rets', ep_ret_record)
            np.save(logdir + '/timesteps', time_step_record)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--nstep', type=int, default=1)
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--sil-weight', type=float, default=0.1)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    assert args.nstep == 1
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    if args.delay > 1:
        env_name = args.env + 'delay{}'.format(args.delay)
    else:
        env_name = args.env
    logdir = 'td3_sil_return_{}/seed_{}nstep_{}gamma_{}hid_{}l_{}alpha_{}beta_{}silweight_{}'.format(env_name, args.seed, args.nstep, args.gamma, args.hid, args.l,
        args.alpha, args.beta, args.sil_weight)
    
    def env_fn():
        env = gym.make(args.env)
        env = delay_wrapper.DelayedRewardEnv(env, nstep=args.delay)
        return nstep_wrapper.NstepWrapper(env, nstep=args.nstep, gamma=args.gamma)

    def env_fn_test():
        return gym.make(args.env)

    td3(env_fn, env_fn_test, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, logdir=logdir, nstep=args.nstep, alpha=args.alpha, beta=args.beta, sil_weight=args.sil_weight)