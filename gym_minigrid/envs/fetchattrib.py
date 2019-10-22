from gym_minigrid.minigridattrib import *
import random


class FetchGame(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(
            self,
            color_to_idx,
            shade_to_idx,
            size_to_idx,
            size,
            numObjs,
            manual,
            oneobject,
            random_target,
            reward_if_wrong_object,
            wrong_object_terminal,
            use_defined_missions,
            shuffle_attrib,
            missions,
            num_attrib=4
    ):
        self.numObjs = numObjs
        self.color_to_idx = color_to_idx
        self.color_names = sorted(list(color_to_idx.keys()))
        self.shade_to_idx = shade_to_idx
        self.shade_names = sorted(list(shade_to_idx.keys()))
        self.size_to_idx = size_to_idx
        self.size_names = sorted(list(size_to_idx.keys()))
        self.manual = manual
        self.oneobject = oneobject
        self.random_target = random_target
        self.reward_if_wrong_object = reward_if_wrong_object
        self.wrong_object_terminal = wrong_object_terminal
        self.use_defined_missions = use_defined_missions
        self.shuffle_attrib = shuffle_attrib
        self.missions = missions
        self.num_attrib = num_attrib
        random.seed(self.seed)
        super().__init__(
            grid_size=size,
            max_steps=5 * size ** 2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        types = ['key', 'ball']

        objs = []
        if self.manual:
            objColor = self._rand_elem(self.color_names)
            obj = Key(objColor)
            self.place_obj(obj)
            objs.append(obj)
            obj = Ball(objColor)
            self.place_obj(obj)
            objs.append(obj)

        elif self.oneobject:
            objColor = self._rand_elem(self.color_names)
            obj = Key(objColor)
            self.place_obj(obj)
            objs.append(obj)

        else:
            # For each object to be generated
            while len(objs) < self.numObjs:
                if self.use_defined_missions:
                    mission = random.choice(self.missions)
                    objColor = mission[0]
                    objType = mission[1]
                    objSize = mission[2]
                    objShade = mission[3]

                else:
                    objType = self._rand_elem(types)
                    objColor = self._rand_elem(self.color_names)
                    objShade = self._rand_elem(self.shade_names)
                    objSize = self._rand_elem(self.size_names)

                if objType == 'key':
                    obj = Key(objColor, objShade, objSize)
                elif objType == 'ball':
                    obj = Ball(objColor, objShade, objSize)

                self.place_obj(obj)
                objs.append(obj)

        # Usefuk for stats on objects
        self.objs = objs
        # Randomize the player start position and orientation
        self.place_agent()

        if self.random_target:
            # Choose a random object to be picked up
            target = objs[self._rand_int(0, len(objs))]
        else:
            # Choose always the same object
            target = objs[0]

        self.targetType = target.type
        self.targetColor = target.color
        self.targetShade = target.shade
        self.targetSize = target.size

        attrib = [self.targetColor, self.targetSize, self.targetShade]

        if self.shuffle_attrib:
            random.shuffle(attrib)

        attrib += [self.targetType]

        descStr = " ".join(attrib)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        is_carrying, hindsight_reward, hindsight_target = 0, 0, 0

        if self.carrying:
            is_carrying = 1
            if self.carrying.color == self.targetColor \
                    and self.carrying.type == self.targetType \
                    and self.carrying.shade == self.targetShade \
                    and self.carrying.size == self.targetSize:
                # reward = self._reward()
                reward = 1
                done = True
            else:
                # hindsight_reward = self._reward()
                hindsight_reward = 1
                hindsight_target = {
                    "color": self.carrying.color,
                    "type": self.carrying.type,
                    "shade": self.carrying.shade,
                    "size": self.carrying.size
                }

                reward = self.reward_if_wrong_object

                if self.wrong_object_terminal:
                    done = True
                else:
                    # Get the position in front of the agent
                    fwd_pos = self.front_pos

                    # Get the contents of the cell in front of the agent
                    fwd_cell = self.grid.get(*fwd_pos)

                    # Drop the object
                    if not fwd_cell and self.carrying:
                        self.grid.set(*fwd_pos, self.carrying)
                        self.carrying.cur_pos = fwd_pos
                        self.carrying = None
                    done = False

                    obs = self.gen_obs()

        return obs, reward, done, is_carrying, hindsight_reward, hindsight_target

    def step_continual(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        can_pickup = 0

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell and fwd_cell.can_pickup():
            can_pickup = 1
        if self.carrying:
            # Get the position in front of the agent
            fwd_pos = self.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Drop the object
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

            done = False

            obs = self.gen_obs()

        return obs, reward, done, can_pickup
