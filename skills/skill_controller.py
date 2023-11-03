import collections
import copy
import numpy as np
from .skills import InsertSkill, PushSkill, SweepSkill


class SkillController:

    SKILL_MAPS = {
        'push': PushSkill,
        'insert': InsertSkill,
        'sweep': SweepSkill,
    }
    KEYFRAME_COUNT = 3

    def __init__(
            self,
            args,
            config={},
    ):
        self._setup_config(args, config)

        self._cur_skill = None
        self._num_ac_calls = None
        self._max_ac_calls = None
        self._pos_is_delta = None
        self._ori_is_delta = None
        self.keyframe_counter = SkillController.KEYFRAME_COUNT

    def _setup_config(self, args, config):
        default_config = dict(
            success_penalty_fac=1.0,
            aff_penalty_fac=1.0,
            skills=['atomic'],
        )
        self._config = copy.deepcopy(default_config)
        self._config.update(config)

        skill_names = self._config['skills']
        for skill_name in skill_names:
            assert skill_name in SkillController.SKILL_MAPS
        skill_names = [
            skill_name for skill_name in SkillController.SKILL_MAPS.keys()
            if skill_name in skill_names
        ]

        assert len(skill_names) >= 1
        assert self._config['success_penalty_fac'] >= 1.0

        if skill_names == ['atomic']:
            self._config['ignore_aff_rew'] = True
        elif 'ignore_aff_rew' not in self._config:
            self._config['ignore_aff_rew'] = False

        assert self._config['aff_penalty_fac'] >= 0.0

        self._skills = collections.OrderedDict()
        for skill_name in skill_names:
            base_skill_config = self._config.get('base_config', {})
            skill_config = copy.deepcopy(base_skill_config)
            if skill_name in SkillController.SKILL_MAPS:
                skill_class = SkillController.SKILL_MAPS[skill_name]
            else:
                raise ValueError
            skill_config.update(
                self._config.get(skill_name + '_config', {})
            )
            self._skills[skill_name] = skill_class(
                args, skill_name, **skill_config)

        self._param_dims = None

    def get_skill_dim(self):
        num_skills = len(self._skills)
        if num_skills <= 1:
            return 0
        else:
            return num_skills

    def get_param_dim(self):
        if self._param_dims is None:
            self._param_dims = collections.OrderedDict()
            for skill_name, skill in self._skills.items():
                dim = skill.get_param_dim()
                self._param_dims[skill_name] = dim
        return np.max(list(self._param_dims.values()))

    def get_skill_names(self):
        return list(self._skills.keys())

    def reset_param(self, param, obs=None, env_config=None):
        self._num_ac_calls = 0
        self._cur_skill.update_param(param, obs, env_config)
        return param

    def reset_to_skill(self, skill_name):
        self._cur_skill = self._skills[skill_name]

        self._num_ac_calls = 0
        self._max_ac_calls = self._cur_skill.get_max_ac_calls()
        self._pos_is_delta = None
        self._ori_is_delta = None

    def random_hl_param(self, obs=None, env_config=None):
        for _ in range(5000):
            random_param = np.random.uniform(-1, 1,
                                            self._cur_skill.get_param_dim())
            if self._cur_skill.check_valid_param(random_param, obs):
                return self.reset_param(random_param, obs, env_config)
        assert False, 'Failed to find a valid random param for skill: {self._cur_skill._skill_name}'        
        # return self.reset_param(random_param, obs)

    def step_ll_action(self, robot_state):
        ''' given the proprio_state, return the low-level action'''
        skill = self._cur_skill
        cur_ee_pose = {
            'cur_pos': robot_state['robot0_eef_pos'], 'cur_quat': robot_state['robot0_eef_quat']}

        skill.update_state_machine(cur_ee_pose)

        ll_action = skill.get_ll_action()

        self._num_ac_calls += 1

        return ll_action

    def is_success(self):
        return self._cur_skill.is_success()

    def done(self):
        return self.is_success()  or (self._num_ac_calls >= self._max_ac_calls)
    
    def is_keyframe_reached(self):
        # if self._cur_skill._state_transition:
        #     self.keyframe_counter = SkillController.KEYFRAME_COUNT
            
        # if self._cur_skill._skill_name == 'insert' and self.keyframe_counter > 0:    
        #     self.keyframe_counter -= 1
        #     return self._cur_skill.is_keyframe_reached(self._cur_skill._state, True)
        if self._cur_skill.is_keyframe_reached(self._cur_skill._state, self._cur_skill._state_transition):
            
            cur_pose_state_idx=self._cur_skill.__class__.STATES.index(self._cur_skill._state)-1
            cur_pose_state = self._cur_skill.__class__.STATES[cur_pose_state_idx]
            expected_cur_pose = self._cur_skill.subgoal_poses[cur_pose_state]
            return expected_cur_pose
        else:
            return None

        # return self._cur_skill.is_keyframe_reached(self._cur_skill._state, self._cur_skill._state_transition)

    def get_num_ac_calls(self):
        """Number of action controller calls

        Returns:
            int: Number of action controller calls
        """
        return self._num_ac_calls

    def get_cur_skill_name(self):

        return self._cur_skill._skill_name

    def get_skill_code(self, skill_name, default=None):
        '''return the one-hot encoding of the skill_name'''
        skill_names = self.get_skill_names()
        if skill_name not in skill_names:
            skill_name = default
        if skill_name is None:
            return None
        skill_idx = skill_names.index(skill_name)
        skill_dim = self.get_skill_dim()
        if skill_dim > 0:
            skill_code = np.zeros(skill_dim)
            skill_code[skill_idx] = 1.0
            return skill_code
        else:
            return None

    def get_skill_names_and_colors(self):
        skill_color_map = dict(
            atomic='gold',
            reach='dodgerblue',
            reach_osc='dodgerblue',
            grasp='green',
            push='orange',
            open='darkgoldenrod',
            close='gray',
        )

        skill_names = self.get_skill_names()
        return skill_names, [skill_color_map[skill_name] for skill_name in skill_names]

    def get_full_skill_name_map(self):
        map = dict(
            atomic='Atomic',
            reach='Reach',
            reach_osc='Reach OSC',
            grasp='Grasp',
            push='Push',
            open='Release',
            close='Close',
        )

        return map
