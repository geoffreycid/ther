from gym_minigrid.minigrid_4_actions import *
from gym_minigrid.register import register


class FetchGame(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(
        self,
        color_to_idx,
        size,
        numObjs,
        manual,
        oneobject,
        random_target,
        use_her
    ):
        self.numObjs = numObjs
        self.color_to_idx = color_to_idx
        self.color_names = sorted(list(color_to_idx.keys()))
        self.manual = manual
        self.oneobject = oneobject
        self.random_target = random_target
        self.use_her = use_her
        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

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
                objType = self._rand_elem(types)
                objColor = self._rand_elem(self.color_names)

                if objType == 'key':
                    obj = Key(objColor)
                elif objType == 'ball':
                    obj = Ball(objColor)

                self.place_obj(obj)
                objs.append(obj)

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

        descStr = '%s %s' % (self.targetColor, self.targetType)

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
        return_her = 0
        is_carrying = 0
        if self.carrying:
            is_carrying = 1
            if self.carrying.color == self.targetColor and \
               self.carrying.type == self.targetType:
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True
                if self.use_her:
                    hindsight_reward = self._reward()
                    hindsight_target = {
                        "color": self.carrying.color,
                        "type": self.carrying.type
                    }
                    return_her = 1
        if return_her:
            return obs, reward, done, return_her, is_carrying, hindsight_reward, hindsight_target
        else:
            return obs, reward, done, return_her, is_carrying
