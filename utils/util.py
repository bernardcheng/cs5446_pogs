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