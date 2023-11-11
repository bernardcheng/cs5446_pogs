def custom_step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        terminated = []

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):

            c_x, c_y = self.grid.positions_xy[agent_idx]
            f_x, f_y = self.grid.finishes_xy[agent_idx]

            #d = math.sqrt((c_x - f_x) ** 2 + (c_y - f_y) ** 2)    
            #reward = 1 - (d / (math.sqrt(2) * GRID_LEN))
            reward = 1 - ( (abs(c_x - f_x) + abs(c_y - f_y)) / (2 * GRID_LEN) )
            #print(f"[CURR] {c_x}, {c_y} [FINISH] {f_x}, {f_y} [DIST] {d} [REWARD] {reward}")


            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                print("FINISH", reward)
                rewards.append(reward)
            else:
                rewards.append(reward)
            terminated.append(on_goal)

        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.grid.is_active[agent_idx] = False

        infos = self._get_infos()

        observations = self._obs()
        truncated = [False] * self.grid_config.num_agents
        return observations, rewards, terminated, truncated, infos



def evaluate_success_rate(model, env, num_episodes=1000):
    success_count = 0
    step_array = []
    rewards_arr = []
    all_rewards_arr = []
    for i in range(num_episodes):
        print(f'---{i}---')
        obs = env.reset()

        # Check if observation is a tuple and extract the first element if true.
        if isinstance(obs, tuple):
            obs = obs[0]
        max_step = 64
        steps_taken = 0
        reward_acc = 0 
        done = truncated = False
        while not done and max_step > 0:
            action, _ = model.predict(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            print(action,max_step,success_count,done, reward)
            reward_acc += reward
            max_step -= 1
            steps_taken += 1
            # Check if next_obs is a tuple and extract the first element if true.
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            obs = next_obs

            # Check if agent was successful in that episode.
            if done:
                success_count += 1
                print("acc reward", reward_acc)
                step_array.append(steps_taken)
                rewards_arr.append(reward_acc)
                env.save_animation(f"media_d/render{i}.svg", AnimationConfig(egocentric_idx=0))
                break
        
        all_rewards_arr.append(reward_acc)

    success_rate = success_count / num_episodes
    return success_rate, step_array, rewards_arr, all_rewards_arr