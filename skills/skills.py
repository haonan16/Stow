import copy
import numpy as np
import robosuite.utils.transform_utils as trans
import utils.data_utils
import utils.utils3d as utils3d
import time
import torch
from utils.data_utils import *
import pdb
from scipy.spatial.transform import Rotation

# Constants
DEFAULT_MAX_AC_CALLS = 150
DEFAULT_GRIPPER_OPEN = -1
DEFAULT_GRIPPER_CLOSED = 1
DEFAULT_GRASP_STAY_COUNT = 5
DEFAULT_STATE_COUNT = 35
PUSH_OFFSET = 0.01
SLIDE_OFFSET = 0.1
INSERT_Y_OFFSET = 0.01
SHELF_OFFSET = 0.025
SWEEP_Y_OFFSET = 0.03

class BaseSkill:
    GRIPPER_OPEN = DEFAULT_GRIPPER_OPEN
    GRIPPER_CLOSED = DEFAULT_GRIPPER_CLOSED
    GRASP_STAY_COUNT = DEFAULT_GRASP_STAY_COUNT
    STATE_COUNT = DEFAULT_STATE_COUNT

    def __init__(
            self,
            args,
            skill_name,

            ### common settings ###
            param_bounds=np.array([
                [-1, -1, -1],
                [1, 1, 1]
            ]),
            reach_threshold=0.03,
            angle_threshold=0.1,


            **config
    ):
        self._skill_name = skill_name
        self._config = dict(
            param_bounds=param_bounds,
            reach_threshold=reach_threshold,
            angle_threshold=angle_threshold,
            **config
        )
        self.args = args

        for k in [ 'param_bounds']:
            assert self._config[k] is not None
            self._config[k] = np.array(self._config[k])
            
        self.KEY_STATES = []
        self.STATES = []
        self.grasp_counter = BaseSkill.GRASP_STAY_COUNT
        self.state_counter = BaseSkill.STATE_COUNT

    def reset(self, params, config_update={}):
        self._params = params
        self._state = None
        self._config.update(config_update)

    def _get_unnormalized_param(self, pos, bounds):

        pos = clamp(pos, -1, 1)
        pos = (pos + 1) / 2
        low, high = bounds[0], bounds[1]
        return low + (high - low) * pos

    def get_max_ac_calls(self):
        return 20 if self.args.debug else self._config['max_ac_calls']

    def update_param(self, params, obs=None, env_config=None):
        '''update normalized parameters for the behavior primitive'''
        # the last dimension is parameter dimension
        if self._skill_name in ['sweep', 'insert']:
            self._update_shelf_config(env_config)

        self._params = params[..., :self.get_param_dim()]
        if isinstance(self._params, torch.Tensor):
            self._config = {key: value  if isinstance(value ,torch.Tensor) else torch.tensor(value, device=self._params.device).float() for key, value in self._config.items()}
        else:
            self._config = {key: np.array(value) for key, value in self._config.items() }

        self.reset(self._params)

        self._get_action_subgoals(obs, env_config)
        self._state = 'SWEEP_INIT' if self._skill_name == 'sweep' else 'INIT' 
        self.subgoal_pose = self.subgoal_poses[self._state]
        
        
    def _update_shelf_config(self, env_config):
        self._config['shelf_y_range'][0] = SHELF_OFFSET - env_config['shelf_bottom']['size'][1] / 2

    @staticmethod
    def get_param_dim():
        assert NotImplementedError

    def update_state_machine(self, cur_ee_pose):
        '''update the state machine of the push primitive'''
        if self.reached_subgoal(cur_ee_pose, self.subgoal_pose) or self.state_counter == 0:
            self.state_counter = BaseSkill.STATE_COUNT
            if self._state == 'FINISHED':
                pass
            elif self._state == 'PRE_GRASP' or self._state == 'GRASPED'  or \
            self._state == 'STANDED' or self._state == 'REGRASP_1' or \
            self._state == 'PRE_PUSH' or self._state == 'SWEEP_INIT' or \
                self._state == 'PUSHED':
                self.grasp_counter -= 1
                if self.grasp_counter == 0:
                    self._state = self.__class__.STATES[self.__class__.STATES.index(self._state)+1]
                    self._state_transition = True
                    self.grasp_counter = BaseSkill.GRASP_STAY_COUNT
            else:
                if self.args.debug:
                    print('change state from ', self._state, ' to ',
                        self.__class__.STATES[self.__class__.STATES.index(self._state)+1])
                self._state = self.__class__.STATES[self.__class__.STATES.index(self._state)+1]
                self._state_transition = True
        else:
            self.state_counter -= 1
            self._state_transition = False


        self.subgoal_pose = self.subgoal_poses[self._state]
        if self.args.debug:
            print('state: ', self._state, 'cur_ee_pos: ',
                cur_ee_pose['cur_pos'], 'subgoal_pos: ', self.subgoal_pose['pos'])
            print('state: ', self._state, 'cur_axisangle: ',
                trans.quat2axisangle(np.array(cur_ee_pose['cur_quat'])), 'subgoal_axisangle: ', trans.quat2axisangle(np.array(self.subgoal_pose['quat'])))

        assert self._state in self.__class__.STATES

    def reached_subgoal(self, cur_ee_pose, subgoal_pose):
        '''check if the robot has reached the subgoal pose of the push primitive'''
        reached_subgoal_pos = (np.linalg.norm(
            subgoal_pose['pos'] - cur_ee_pose['cur_pos']) < self._config['reach_threshold'])
        
        reached_subgoal_ori = trans.get_quat_error(
            subgoal_pose['quat'], cur_ee_pose['cur_quat']) < self._config['angle_threshold']
        if self.args.debug:
            print('pos error: ', np.linalg.norm(subgoal_pose['pos'] - cur_ee_pose['cur_pos']),
                  'angle error: ', trans.get_quat_error(subgoal_pose['quat'], cur_ee_pose['cur_quat']))

        return reached_subgoal_pos and reached_subgoal_ori

    def action_dict_to_array(self, action_dict):
        '''convert action dict to array'''
        return np.concatenate((action_dict['pos'].squeeze().copy(), action_dict['axisangle'].squeeze().copy(), [action_dict['gripper']]))

    def get_ll_action(self):
        '''get the low-level action for the behavior primitive'''
        ll_action = self.action_dict_to_array(self.subgoal_pose)
        return ll_action

    def _get_start_action(self):
        pass

    def _get_target_action(self):
        pass

    def is_success(self):
        assert NotImplementedError

    @classmethod
    def is_keyframe_reached(cls, state, state_transition):
        if state in cls.KEY_STATES and state_transition:
            return True
        else:
            return False

    @classmethod
    def num_keyframes(cls):
        return len(cls.KEY_STATES)

    def get_keyposes(self):
        return [self.subgoal_poses[self.__class__.STATES[max(self.__class__.STATES.index(state)-1, 0)]] for state in self.__class__.ALL_KEY_STATES]

    def check_valid_param(self, random_param, obs):
        return True
    
    


class PushSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'REACH_START', 'RETURN_BACK', 'FINISHED']
    KEY_STATES = ['REACH_START', 'RETURN_BACK']

    ALL_KEY_STATES = ['LIFTED', 'REACH_START']

    def __init__(
            self,
            args,
            skill_name,
            max_ac_calls=150,
            **config
    ):
        super().__init__(
            args,
            skill_name,
            max_ac_calls=max_ac_calls,
            **config
        )
        
    @staticmethod
    def get_param_dim():
        return 3  # x, y, pushing distance

    def _get_action_subgoals(self, obs=None, env_config=None):

        unnormalized_param = self._get_unnormalized_param(
            self._params, self._config['param_bounds'])
        pos = (unnormalized_param[..., :2])
 

        lifted_pos = concat_tensors_or_arrays(((pos), self._config['hover_z']), dim=-1)
        start_pos = concat_tensors_or_arrays(((pos), self._config['push_z']), dim=-1)
        push_quat = broadcast_quat_to_pos_shape(self._config['push_quat'], lifted_pos)
        self.subgoal_poses = {}
        

        self.subgoal_poses['INIT'] = {'pos': (lifted_pos),
                                      'quat': push_quat,
                                      'axisangle': trans.quat2axisangle(push_quat),
                                      'gripper': BaseSkill.GRIPPER_OPEN}

        self.subgoal_poses['LIFTED'] = {'pos': (start_pos),
                                        'quat': push_quat,
                                        'axisangle': trans.quat2axisangle(push_quat),
                                        'gripper': BaseSkill.GRIPPER_OPEN}

        # push in negative y direction
        target_pos = copy_variable(start_pos)
        target_pos[..., 1] -= unnormalized_param[..., -1]
        
        
        target_pos[..., 1] = clamp(target_pos[..., 1], 
                                   -env_config['table']['size'][1]/2+env_config['cubeA_size'][1]/2+PUSH_OFFSET, 
                                   env_config['table']['size'][1]/2-env_config['cubeA_size'][1]/2-PUSH_OFFSET)
        self.subgoal_poses['REACH_START'] = {'pos': target_pos,
                                             'quat': push_quat,
                                             'axisangle': trans.quat2axisangle(push_quat),
                                             'gripper': BaseSkill.GRIPPER_OPEN}
        final_pos = lifted_pos # copy_variable(target_pos)
        self.subgoal_poses['RETURN_BACK'] = {'pos': final_pos,
                                          'quat': push_quat,
                                          'axisangle': trans.quat2axisangle(push_quat),
                                          'gripper': BaseSkill.GRIPPER_OPEN}


        self.subgoal_poses['FINISHED'] = self.subgoal_poses['RETURN_BACK']
        
    def is_success(self):
        return self._state == 'FINISHED'


class InsertSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'PRE_GRASP', 'GRASPED', 'PRE_STAND', 'STANDED',
            'PRE_PUSH', 'PUSH_TO_SHELF', 'SLIDE_DOWN', 'ROTATE_SHELF', 'SLIDE_SIDE', 'SLIDE_FINISHED', 'FINISHED']

    KEY_STATES = [  'PUSH_TO_SHELF', 'SLIDE_DOWN',  'ROTATE_SHELF', 'SLIDE_SIDE', 'SLIDE_FINISHED']
    ALL_KEY_STATES = [ 'PRE_PUSH', 'PUSH_TO_SHELF', 'SLIDE_DOWN', 'ROTATE_SHELF', 'SLIDE_SIDE']


    def __init__(
            self,
            args,
            skill_name,
            max_ac_calls=350,
            **config
    ):
        super().__init__(
            args,
            skill_name,
            max_ac_calls=max_ac_calls,
            **config
        )

    @staticmethod
    def get_param_dim():
        return 4  # offset in y direction, height, sliding distance, rolling angle

    def _get_action_subgoals(self, obs=None, env_config=None):
        
        unnormalized_param = self._get_unnormalized_param(
            self._params, self._config['param_bounds'])

        insert_y = env_config['shelf_bottom']['size'][1]/2 - env_config['cubeA_size'][2]/2 - INSERT_Y_OFFSET
        y_distance = insert_y - unnormalized_param[...,0]
        
        height = unnormalized_param[...,1]
        
        slide_bound = env_config['shelf_bottom']['size'][1]/2 - env_config['cubeA_size'][2] - sum([env_config[f'cubeB_{i}_size'][1] for i in range(env_config['num_obj_on_shelf'])])
        max_slide_distance = -slide_bound - y_distance
        

        slide_distance = unnormalized_param[...,2] * max_slide_distance
        slide_bound = abs(slide_bound)

        quat_percentage = unnormalized_param[...,3]


        if isinstance(self._config['insert_start_x'], torch.Tensor):
            device = self._config['insert_start_x'].device
            obj_pos = torch.from_numpy(obs['cubeA_pos']).to(device)
            obj_side_pos = obj_pos - torch.tensor([env_config['cubeA_size'][0]/2, 0, 0], device=device)
            lift_pos = obj_side_pos + self._config['reach_offset']
            lift_pos[2] = self._config['lift_z']
            reach_pos = obj_side_pos + self._config['reach_offset']
            grasp_pos = obj_side_pos + self._config['grasp_offset']
            pre_stand_pos = obj_side_pos + self._config['grasp_offset']
            pre_stand_pos[2] = self._config['lift_z']

        else:
            obj_pos = np.array(obs['cubeA_pos'])
            obj_side_pos = obj_pos - np.array([env_config['cubeA_size'][0]/2, 0, 0])
            lift_pos = obj_side_pos + np.array(self._config['reach_offset'])
            lift_pos[2] = self._config['lift_z']
            reach_pos = obj_side_pos + np.array(self._config['reach_offset'])
            grasp_pos = obj_side_pos + np.array(self._config['grasp_offset'])
            pre_stand_pos = obj_side_pos + np.array(self._config['grasp_offset'])
            pre_stand_pos[2] = self._config['lift_z']
        
        stand_z = env_config['table']['pos'][2] + env_config['cubeA_size'][1]/2 + 0.15 - self._config['grasp_offset'][1]
        slide_z = env_config['table']['pos'][2] + env_config['cubeA_size'][1]/2 + 0.07 - self._config['grasp_offset'][1]
        if isinstance(self._config['insert_start_x'], torch.Tensor):
            insert_z = stand_z.expand(height.size(0), 1) + height

            stand_pos = torch.cat((self._config['insert_start_x'].expand(y_distance.size(0), 1), 
                                   y_distance, 
                                   stand_z.expand(y_distance.size(0), 1)), dim=1)

            insert_start_pos = torch.cat((self._config['insert_start_x'].expand(y_distance.size(0), 1), 
                                   y_distance, 
                                   insert_z), dim=1)
            insert_target_pos = torch.cat((self._config['insert_target_x'].expand(y_distance.size(0), 1), 
                                   y_distance, 
                                   insert_z), dim=1)
            slide_down_pos = torch.cat((self._config['insert_target_x'].expand(y_distance.size(0), 1), 
                                   y_distance, 
                                   slide_z.expand(y_distance.size(0), 1)), dim=1)
            slide_target_pos = torch.cat((self._config['insert_target_x'].expand(y_distance.size(0), 1), 
                                   clamp(y_distance+slide_distance, -slide_bound, slide_bound), 
                                   slide_z.expand(y_distance.size(0), 1)), dim=1)
            slide_finish_pos = torch.cat((self._config['insert_target_x'].expand(y_distance.size(0), 1), 
                                   clamp(y_distance+slide_distance, -slide_bound, slide_bound), 
                                   (slide_z).expand(y_distance.size(0), 1)), dim=1)


        else:
            height = height.squeeze()
            y_distance = y_distance.squeeze()
            slide_distance = slide_distance.squeeze()

            insert_z = stand_z + height
            
            stand_pos = np.array(
                [self._config['insert_start_x'], y_distance, stand_z])

            insert_start_pos = np.array(
                [self._config['insert_start_x'], y_distance, insert_z])

            insert_target_pos = np.array(
                [self._config['insert_target_x'], y_distance, insert_z])
            slide_down_pos = np.array(
                [self._config['insert_target_x'], y_distance, slide_z])

            slide_target_pos = np.array([self._config['insert_target_x'],
                                        clamp(
                                            y_distance+slide_distance, -slide_bound, slide_bound),
                                        slide_z])
            slide_finish_pos = np.array([self._config['insert_target_x'],
                                        clamp(
                                            y_distance+slide_distance, -slide_bound, slide_bound),
                                        slide_z])
            
        if isinstance(self._config['grasp_quat'], torch.Tensor):
            lift_pos = lift_pos.float().to(self._config['grasp_quat'].device)
        grasp_quat = self._config['grasp_quat']
        stand_quat = broadcast_quat_to_pos_shape(self._config['stand_quat'], stand_pos)
        if isinstance(quat_percentage, torch.Tensor):
            rotate_quat = trans.quaternion_lerp(torch.tensor(self._config['quat_bounds'][0]).to(quat_percentage.device), torch.tensor(
                self._config['quat_bounds'][1]).to(quat_percentage.device), quat_percentage).float()
        else:
            rotate_quat = trans.quaternion_lerp(np.array(self._config['quat_bounds'][0]), np.array(
                self._config['quat_bounds'][1]), quat_percentage)
            
        if isinstance(grasp_quat, torch.Tensor):
            align_quat = trans.quaternion_lerp(grasp_quat, 
                stand_quat, 0.9).float()
        else:
            align_quat = trans.quaternion_lerp(np.array(grasp_quat), np.array(
                stand_quat), 0.9)


        self.subgoal_poses = {}

        self.subgoal_poses['INIT'] = {'pos': (lift_pos),
                                      'quat': grasp_quat,
                                      'axisangle': trans.quat2axisangle(grasp_quat),
                                      'gripper': BaseSkill.GRIPPER_OPEN}

        self.subgoal_poses['LIFTED'] = {'pos': (reach_pos),
                                        'quat': grasp_quat,
                                        'axisangle': trans.quat2axisangle(grasp_quat),
                                        'gripper': BaseSkill.GRIPPER_OPEN}

        self.subgoal_poses['PRE_GRASP'] = {'pos': (grasp_pos),
                                         'quat': grasp_quat,
                                         'axisangle': trans.quat2axisangle(grasp_quat),
                                         'gripper': BaseSkill.GRIPPER_OPEN}

        
        self.subgoal_poses['GRASPED'] = {'pos': (grasp_pos),
                                         'quat': grasp_quat,
                                         'axisangle': trans.quat2axisangle(grasp_quat),
                                         'gripper': BaseSkill.GRIPPER_CLOSED}

        self.subgoal_poses['PRE_STAND'] = {'pos': (pre_stand_pos),
                                         'quat': grasp_quat,
                                         'axisangle': trans.quat2axisangle(grasp_quat),
                                         'gripper': BaseSkill.GRIPPER_CLOSED}


        self.subgoal_poses['STANDED'] = {'pos': (stand_pos),
                                         'quat': stand_quat,
                                         'axisangle': trans.quat2axisangle(stand_quat),
                                         'gripper': BaseSkill.GRIPPER_CLOSED}

        self.subgoal_poses['PRE_PUSH'] = {'pos': (insert_start_pos),
                                          'quat': stand_quat,
                                          'axisangle': trans.quat2axisangle(stand_quat),
                                          'gripper': BaseSkill.GRIPPER_CLOSED}


        self.subgoal_poses['PUSH_TO_SHELF'] = {'pos': (insert_target_pos),
                                               'quat': stand_quat,
                                               'axisangle': trans.quat2axisangle(stand_quat),
                                               'gripper': BaseSkill.GRIPPER_CLOSED}
        
        self.subgoal_poses['SLIDE_DOWN'] = {'pos': (slide_down_pos),
                                               'quat': align_quat,
                                               'axisangle': trans.quat2axisangle(align_quat),
                                               'gripper': BaseSkill.GRIPPER_CLOSED}
        self.subgoal_poses['ROTATE_SHELF'] = {'pos': (slide_down_pos),
                                               'quat': rotate_quat,
                                               'axisangle': trans.quat2axisangle(rotate_quat),
                                               'gripper': BaseSkill.GRIPPER_CLOSED}

        self.subgoal_poses['SLIDE_SIDE'] = {'pos': (slide_target_pos),
                                       'quat': stand_quat,
                                       'axisangle': trans.quat2axisangle(stand_quat),
                                       'gripper': BaseSkill.GRIPPER_CLOSED}

        self.subgoal_poses['SLIDE_FINISHED'] = {'pos': (slide_target_pos),
                                       'quat': stand_quat,
                                       'axisangle': trans.quat2axisangle(stand_quat),
                                       'gripper': BaseSkill.GRIPPER_CLOSED}

        self.subgoal_poses['FINISHED'] = self.subgoal_poses['SLIDE_FINISHED']

    def is_success(self):
        return self._state == 'FINISHED'



class SweepSkill(BaseSkill):
    STATES = ['SWEEP_INIT', 'PUSHED', 'SWEEP_DOWN', 'SWEEPED', 'SWEEP_BACK', 'FINISHED']

    KEY_STATES = ['PUSHED', 'SWEEP_DOWN', 'SWEEPED', 'SWEEP_BACK']
    ALL_KEY_STATES = ['SWEEP_INIT', 'PUSHED', 'SWEEP_DOWN', 'SWEEPED']

    def __init__(
            self,
            args,
            skill_name,
            max_ac_calls=200,
            **config
    ):
        super().__init__(
            args,
            skill_name,
            max_ac_calls=max_ac_calls,
            **config
        )

    @staticmethod
    def get_param_dim():
        return 3  # offset in y direction, sweep height, sliding distance

    def _get_action_subgoals(self, obs=None, env_config=None):

        unnormalized_param = self._get_unnormalized_param(
            self._params, self._config['param_bounds'])

        # y_distance, height, slide_distance = (unnormalized_param)
        sweep_y = env_config['shelf_bottom']['size'][1]/2 - SWEEP_Y_OFFSET
        y_distance = sweep_y - unnormalized_param[...,0]

        height = self._config['sweep_z'] + unnormalized_param[...,1]
        
        slide_bound = env_config['shelf_bottom']['size'][1]/2 - sum([env_config[f'cubeB_{i}_size'][1] for i in range(env_config['num_obj_on_shelf'])])

        max_slide_distance = -slide_bound - y_distance
        
        slide_y = (-slide_bound - y_distance)  * unnormalized_param[...,2] + y_distance # max_slide_distance
        slide_bound = abs(slide_bound) + SLIDE_OFFSET

        if isinstance(self._config['push_x'], torch.Tensor):
            push_pos = torch.cat((self._config['push_x'].expand(y_distance.size(0), 1), y_distance, height), dim=1)
            sweep_start_pos = torch.cat((self._config['sweep_x'].expand(y_distance.size(0), 1), y_distance, height), dim=1)
            sweep_down_pos = torch.cat((self._config['sweep_x'].expand(y_distance.size(0), 1),
                                            y_distance, 
                                        self._config['sweep_z'].expand(y_distance.size(0), 1)), dim=1)

            sweep_target_pos = torch.cat((self._config['sweep_x'].expand(y_distance.size(0), 1), clamp(
                                            slide_y, -slide_bound,  slide_bound), 
                                          self._config['sweep_z'].expand(y_distance.size(0), 1)), dim=1)
            
            
            sweep_back_pos = torch.cat((self._config['sweep_x'].expand(y_distance.size(0), 1), y_distance, height), dim=1)

        else:
            y_distance = y_distance.squeeze()
            height = height.squeeze()
            # slide_distance = slide_distance.squeeze()
            push_pos = np.array([self._config['push_x'], y_distance, height])
            sweep_start_pos = np.array([self._config['sweep_x'], y_distance, height])
            sweep_down_pos = np.array([self._config['sweep_x'], y_distance, self._config['sweep_z']])
            sweep_target_pos = np.array([self._config['sweep_x'],
                                        clamp(
                                            slide_y, -slide_bound,  slide_bound),
                                        self._config['sweep_z']])
            sweep_back_pos = np.array([self._config['sweep_x'], y_distance, height])

            
            
        sweep_quat = broadcast_quat_to_pos_shape(self._config['sweep_quat'], push_pos)
        
        self.subgoal_poses = {}

        self.subgoal_poses['SWEEP_INIT'] = {'pos': (push_pos),
                                      'quat': sweep_quat,
                                      'axisangle': trans.quat2axisangle(sweep_quat),
                                      'gripper': BaseSkill.GRIPPER_OPEN}

        self.subgoal_poses['PUSHED'] = {'pos': (sweep_start_pos),
                                        'quat': sweep_quat,
                                        'axisangle': trans.quat2axisangle(sweep_quat),
                                        'gripper': BaseSkill.GRIPPER_OPEN}
        
        self.subgoal_poses['SWEEP_DOWN'] = {'pos': (sweep_down_pos),
                                         'quat': sweep_quat,
                                         'axisangle': trans.quat2axisangle(sweep_quat),
                                         'gripper': BaseSkill.GRIPPER_OPEN}

        self.subgoal_poses['SWEEPED'] = {'pos': (sweep_target_pos),
                                         'quat': sweep_quat,
                                         'axisangle': trans.quat2axisangle(sweep_quat),
                                         'gripper': BaseSkill.GRIPPER_OPEN}

        self.subgoal_poses['SWEEP_BACK'] = {'pos': (sweep_back_pos),
                                         'quat': sweep_quat,
                                         'axisangle': trans.quat2axisangle(sweep_quat),
                                         'gripper': BaseSkill.GRIPPER_OPEN}

        self.subgoal_poses['FINISHED'] = self.subgoal_poses['SWEEP_BACK']

    def is_success(self):
        return self._state == 'FINISHED'


    def check_valid_param(self, random_param, obs):
        return True
