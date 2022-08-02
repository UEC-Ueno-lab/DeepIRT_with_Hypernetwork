# Code reused from https://github.com/ckyeungac/DeepIRT.git
import datetime
import os

class ModelConfigFactory():
    @staticmethod
    def create_model_config(args):
        if args.dataset == 'assist2009':
            return Assist2009Config(args).get_args()
        if args.dataset == 'assist2009_akt':
            return Assist2009_akt_Config(args).get_args()
        elif args.dataset == 'assist2015_akt':
            return Assist2015_akt_Config(args).get_args()
        elif args.dataset == 'Eedi':
            return Eedi_Config(args).get_args()
        elif args.dataset == 'assist2017':
            return Assist2017Config(args).get_args()
        elif args.dataset == 'assist2017_akt':
            return Assist2017_akt_Config(args).get_args()
        elif args.dataset == 'statics_akt':
            return StaticsaktConfig(args).get_args()
        elif args.dataset == 'junyi':
            return junyi_Config(args).get_args()
        else:
            raise ValueError("The '{}' is not available".format(args.dataset))


class ModelConfig():
    def __init__(self, args):
        self.default_setting = self.get_default_setting()
        self.init_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M")

        self.args = args
        self.args_dict = vars(self.args)
        for arg in self.args_dict.keys():
            self._set_attribute_value(arg, self.args_dict[arg])

        self.set_result_log_dir()
        self.set_checkpoint_dir()
        self.set_tensorboard_dir()

    def get_args(self):
        return self.args

    def get_default_setting(self):
        default_setting = {}
        return default_setting

    def _set_attribute_value(self, arg, arg_value):
        self.args_dict[arg] = arg_value \
            if arg_value is not None \
            else self.default_setting.get(arg)

    def _get_model_config_str(self):
        model_config = 'b' + str(self.args.batch_size) \
                    + '_m' + str(self.args.memory_size) \
                    + '_q' + str(self.args.key_memory_state_dim) \
                    + '_qa' + str(self.args.value_memory_state_dim) \
                    + '_f' + str(self.args.summary_vector_output_dim)
        return model_config

    def set_result_log_dir(self):
        result_log_dir = os.path.join(
            './results',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('result_log_dir', result_log_dir)

    def set_checkpoint_dir(self):
        checkpoint_dir = os.path.join(
            './models',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('checkpoint_dir', checkpoint_dir)

    def set_tensorboard_dir(self):
        tensorboard_dir = os.path.join(
            './tensorboard',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('tensorboard_dir', tensorboard_dir)


class Eedi_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 300,
            'batch_size': 256,
            'train': True,
            'show': True,
            'learning_rate': 0.01,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 27683,
            'n_skills': 1200,
            'data_dir': './data/Eedipid',
            'data_name': 'Eedi',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1,
            'delta_2': 1,
            'rounds': 2,
            'num_pattern': 0,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Assist2009Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 300,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.005,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 110,
            'data_dir': './data/assist2009_updated_remake',
            'data_name': 'assist2009_updated',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1.5,
            'delta_2': 1.5,
            'rounds': 3,
            'num_pattern': 1,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class junyi_Config(ModelConfig):
      def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 300,
            'batch_size': 64,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 800,
            'data_dir': './data/junyi',
            'data_name': 'junyi',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1,
            'delta_2': 1,
            'rounds': 2,
            'num_pattern': 5,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Assist2009_akt_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 300,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 16891,
            'n_skills': 110,
            'data_dir': './data/assist2009_akt',
            'data_name': 'assist2009_pid',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1.5,
            'delta_2': 1.5,
            'rounds': 4,
            'num_pattern': 1,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Assist2017_akt_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 300,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 102,
            'n_skills': 3162,
            'data_dir': './data/assist2017_akt',
            'data_name': 'assist2017_pid',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1.5,
            'delta_2': 1.5,
            'rounds': 4,
            'num_pattern': 1,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Assist2015_akt_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 300,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.001,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 100, 
            'data_dir': './data/assist2015_akt',
            'data_name': 'assist2015',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1.5,
            'delta_2': 1.5,
            'rounds': 3,
            'num_pattern': 0,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Assist2017Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 300,
            'batch_size': 64,
            'train': True,
            'show': True,
            'learning_rate': 0.002,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 102,
            'data_dir': './data/assist2017_remake',
            'data_name': 'assist2017',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1.5,
            'delta_2': 1.5,
            'rounds': 2,
            'num_pattern': 0,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class StaticsaktConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 300,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.001,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 1223,
            'data_dir': './data/statics_akt',
            'data_name': 'statics',
            # DKVMN param
            'memory_size': 25,
            'key_memory_state_dim': 25,
            'value_memory_state_dim': 50,
            'summary_vector_output_dim': 25,
            # hyper-parameter of hyper-network and number of pattern
            'delta_1': 1,
            'delta_2': 1.7,
            'rounds': 2,
            'num_pattern': 0,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


