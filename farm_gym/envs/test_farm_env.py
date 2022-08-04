import gym
from gym import spaces
import numpy as np
from typing import Optional
import pandas as pd
import pcse
import os
import datetime
import yaml
from pcse.fileinput import YAMLCropDataProvider, CABOFileReader, YAMLAgroManagementReader
from pcse.util import WOFOST72SiteDataProvider

#DATA_DIR = os.path.join(os.getcwd(), 'farm_gym/envs/env_data/')
DATA_DIR = "/home/tmrob2/PycharmProjects/farming-gym/farm_gym/envs/env_data/"
# Conversion constant kg / h -> kg / m ^ 2
WEIGHT_CONVERSION = 1000
PENALTY = 1
PLANTBONUS = 1

class IrrigationEnv(gym.Env):
    """An environment for OpenAI gym to study crop irrigation"""
    metadata = {'render.modes': ['human']}
    crop_planting_dict = {None: 0, 'wheat': 1, 'maize': 0}

    def __init__(
        self, 
        soil_type: str,
        fixed_location: tuple,
        start_date,
        end_date,
        intervention_interval=7,
        beta=10, 
        seed=0,
        training=False
    ):
        self.action_space = gym.spaces.Discrete(9)
        # The observation space is 11 for the model output, and 7 (days) * 5 (weather metric observations / day)
        # + 1 State for whether something is planted { 0 - nothing, 1 - wheat, 2 - maize }
        # + 1 state for the crop age
        # + 1 state for crop maturity
        # + 1 state for fallow flag
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11 + 7 * 5 + 4,), dtype=np.float32)

        crop_params_dir = os.path.join(DATA_DIR, 'crop_params/')
        crop = YAMLCropDataProvider(crop_params_dir)
        
        soil_data_course = os.path.join(DATA_DIR, 'SOILD', f'{soil_type}.NEW')
        soil_course = CABOFileReader(soil_data_course)

        sited = WOFOST72SiteDataProvider(WAV=10)

        self.parameterprovider = pcse.base.ParameterProvider(soildata=soil_course, cropdata=crop, sitedata=sited)
        self.intervention_interval = intervention_interval
        self.beta = beta
        self.amount = 4 #2*self.intervention_interval/7
        self.seed(seed)
        self.fixed_location = fixed_location
        self.weatherdataprovider = self._get_weatherdataprovider()
        self.sim_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        self.sim_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        self.train_weather_data = self._get_train_weather_data()
        self.agromanagement = self._load_agromanagement_data()
        # Input the baseline agromanagement
        agro_dir = os.path.join(DATA_DIR, "agro/")
        wheat_agromanagement_file = os.path.join(agro_dir, 'wofost_baseline_maize.yaml')
        maize_agromanagement_file = os.path.join(agro_dir, 'wofost_baseline_wheat.yaml')
        self.wheat_baseline_agromanagement = YAMLAgroManagementReader(wheat_agromanagement_file)
        self.maize_baseline_agromanagement = YAMLAgroManagementReader(maize_agromanagement_file)
        self.date = self.sim_start_date
        self.current_crop = None
        self.isplanted = False
        self.fallow = False
        self.fallow_time = 0
        self.fallow_start_days = 0
        self.crop_planted = {}
        self.model = pcse.models.Wofost72_WLP_FD(
            self.parameterprovider,
            self.weatherdataprovider,
            self.agromanagement
        )
        self.wheat_baseline_model = pcse.models.Wofost72_WLP_FD(
            self.parameterprovider,
            self.weatherdataprovider,
            self.wheat_baseline_agromanagement
        )
        self.maize_baseline_model = pcse.models.Wofost72_WLP_FD(
            self.parameterprovider,
            self.weatherdataprovider,
            self.wheat_baseline_agromanagement
        )
        self.baseline_wheat_output = self.run_baseline_to_terminate(self.wheat_baseline_model)
        self.baseline_maize_output = self.run_baseline_to_terminate(self.maize_baseline_model)
        self.crop_cycle_delta(self.baseline_maize_output, self.maize_baseline_model)
        self.crop_cycle_delta(self.baseline_wheat_output, self.wheat_baseline_model)
        self.log = self._init_log()
        self.training = training

    def step(self, action):
        """
        TODO this documentation must be made specific to this gym environment and not 
        general information relating to gym.Env but we can do this as a last step
        recording all of the documentation gathered along the way


        Execute one time step within the environment

        Parameters
        ----------
        action : integer within the range of the action space

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        irrigation_amount, apply_penalty, action_applied = self._take_action(action)
        output = self._run_simulation(self.model)
        self.date = output.index[-1]
        soil_moisture = output['SM'][-1]
        try:
            growth = output['TAGP'][-1] / WEIGHT_CONVERSION - \
                     output['TAGP'][-1 - self.intervention_interval] / WEIGHT_CONVERSION

            if self.isplanted:
                self.crop_planted[self.current_crop]['qty_tagp'] += growth
                self.crop_planted[self.current_crop]['qty_tagp'] = round(self.crop_planted[self.current_crop]['qty_tagp'], 2)
                self.crop_planted[self.current_crop]['delta'] = growth
                self.crop_planted[self.current_crop]['delta'] = round(self.crop_planted[self.current_crop]['delta'], 2)
        except:
            growth = np.nan
            if self.current_crop in self.crop_planted.keys():
                self.crop_planted[self.current_crop]['delta'] = 0.

        # get the baseline growth
        cycle_days = self.model.agromanager.ndays_in_crop_cycle
        baseline_growth = self.get_expected_baseline_growth(cycle_days)
        baseline_growth = baseline_growth / WEIGHT_CONVERSION if not np.isnan(baseline_growth) else 0.0

        base_reward = 3. if self.isplanted else 0.
        growth = growth if not np.isnan(growth) else 0
        #if baseline_growth > 0.:
        base_reward += growth - baseline_growth
        if soil_moisture <= 0.2 and self.isplanted:
            base_reward += self.beta * irrigation_amount / 1000
        else:
            base_reward -= self.beta * irrigation_amount / 1000
        #if not apply_penalty and action == 0 and not self.isplanted:
        #    base_reward = 0.5
        # crop harvest and soil management
        if self.isplanted and self._get_crop_maturity(output['DVS'][-1]) == 2:
            if action == 8:
                base_reward = 5.
            else:
                base_reward -= 2.
        if apply_penalty:
            base_reward = -1.

        if self.fallow:
            self.compute_fallow_days()
            self._fallow_period()
            if action != 0 and self.fallow_time < 180:
                base_reward = -1
            elif action == 0 and self.fallow_time < 180:
                base_reward = 3.

        observation = self._process_output(output)
        done = self.date >= self.sim_end_date

        self._log(growth, baseline_growth, irrigation_amount, base_reward, self.crop_planted)
        if self.training:
            info = {**self.log}
        else:
            info = {**output.to_dict(), **self.log}
        print(f"action: {action}, date: {self.date}, reward: {base_reward:.2f}, "
              f"grw: {growth:.2f} b grw: {baseline_growth:.2f} "
              f"fallow: {self.fallow}, fallow days: {self.fallow_time} "
              f"moist: {soil_moisture:.2f} "
              f"irr: {self.beta * irrigation_amount / 1000} "
              f"crop: {self.current_crop}, crop cycle: {self.model.agromanager.ndays_in_crop_cycle}"), #crops planted: {self.crop_planted}")
        # print(self.crop_planted)
        return observation, base_reward, done, info

    @staticmethod
    def _init_log():
        return {'growth': dict(), 'baseline_growth': dict(), 'irrigation': dict(), 'reward': dict(), 'crops_planted': dict()}

    def _load_agromanagement_data(self):
        with open(os.path.join(DATA_DIR, 'agro/wofost_wheat.yaml')) as file:
            agromanagement = yaml.load(file, Loader=yaml.SafeLoader)
        #return self._replace_year(agromanagement)
        return agromanagement

    def _log(self, growth, baseline_growth, irrigation, reward, crops_planted):
        self.log['growth'][self.date] = growth
        self.log['baseline_growth'][self.date] = baseline_growth
        self.log['irrigation'][self.date - datetime.timedelta(self.intervention_interval)] = \
            irrigation
        self.log['reward'][self.date] = reward
        self.log['crops_planted'] = crops_planted

    def _process_output(self, output):
        crop_observation = np.array(output.iloc[-1])
        # weather until next intervention time
        weather_observation = self._get_weather(self.weatherdataprovider, self.date,
                                             self.intervention_interval)
        crop_maturity = self._get_crop_maturity(output['DVS'][-1])
        add_on_states = [
            1. if self.isplanted else 0.,
            self.model.agromanager.ndays_in_crop_cycle,
            crop_maturity,
            1. if self.fallow else 0.
        ]
        observation = np.concatenate([add_on_states,
                                      crop_observation,
                                      weather_observation.flatten()], dtype=np.float32)
        observation = np.nan_to_num(observation)
        return observation

    def _fallow_period(self):
        if self.fallow_time > 180:
            self.fallow = False
            self.fallow_time = 0
            self.fallow_start_days = 0
        else:
            self.fallow = True

    @property
    def get_current_days(self):
        return self.model.timer.day_counter

    def compute_fallow_days(self):
        self.fallow_time += (self.get_current_days - self.fallow_start_days)

    def _get_crop_maturity(self, dvs):
        if dvs < 0.5:
            return 0
        elif 0.5 <= dvs < 1.5:
            return 1
        else:
            return 2
    def _get_weatherdataprovider(self):
        location = self.fixed_location
        return pcse.db.NASAPowerWeatherDataProvider(*location)

    @staticmethod
    def _get_train_weather_data():
        all_years = range(1983, 2018)
        missing_data = [2007, 2008, 2010, 2013, 2015, 2017]
        test_years = [1984, 1994, 2004, 2014]
        train_weather_data = [year for year in all_years if year not in missing_data + test_years]
        return train_weather_data

    def _run_simulation(self, model):
        model.run(days=self.intervention_interval)
        output = pd.DataFrame(model.get_output()).set_index('day')
        output = output.fillna(value=np.nan)
        return output

    def _take_action(self, action): 
        """
        Apply some irrigation to the crop. The irrigation model can be  {4, 8, 12, 16, 20} 
        cm h of irrigation
        actions:
        {0} - do nothing
        {1, 2, 3} - wheat, maize, 
        if self.isplanted then planting a crop will do nothing i.e. equivalent of 0
        {3, 4, 5, 6, 7} - irrigate [4, 8, 12, 16, 20] cm of water
        {8} - finish crop (harvest)

        """
        amount = 0
        apply_penalty = False
        if self.model.agromanager.ndays_in_crop_cycle >= 300:
            action = 8
        if action == 1:
            # plant wheat
            # copy the current format
            if not self.isplanted:
                _dict = self.agromanagement['AgroManagement'][0] # <- this is a dictionary
                # with the start of the crop campaign as the key
                # we want to copy over its values and add a new campaign based on the current date
                for k, v in _dict.items():
                    v['CropCalendar'] = {
                        'crop_name': 'wheat',
                        'variety_name': "Winter_wheat_101",
                        'crop_start_date': self.date,
                        'crop_start_type': 'sowing',
                        'crop_end_date': self.sim_end_date,
                        'crop_end_type': 'harvest',
                        'max_duration': 300
                    }
                    self.agromanagement['AgroManagement'].append({self.date: v})
                self.agromanagement['AgroManagement'].pop(0)
                self.model = pcse.models.Wofost72_WLP_FD(
                    self.parameterprovider,
                    self.weatherdataprovider,
                    self.agromanagement
                )
                self.current_crop = 'wheat'
                if 'wheat' in self.crop_planted.keys():
                    self.crop_planted['wheat']['date'].append(self.date)
                else:
                    self.crop_planted['wheat'] = {'date': [self.date], 'qty_tagp': 0., 'delta': 0.}
                self.isplanted = True
            else:
                apply_penalty = True
        elif action == 2:
            # plant maize
            if not self.isplanted:
                _dict = self.agromanagement['AgroManagement'][0]
                for k, v in _dict.items():
                    v['CropCalendar'] = {
                        'crop_name': 'maize',
                        'variety_name': "Maize_VanHeemst_1988",
                        'crop_start_date': self.date,
                        'crop_start_type': 'sowing',
                        'crop_end_date': self.sim_end_date,
                        'crop_end_type': 'harvest',
                        'max_duration': 300
                    }
                    self.agromanagement['AgroManagement'].append({self.date: v})
                self.agromanagement['AgroManagement'].pop(0)
                self.model = pcse.models.Wofost72_WLP_FD(
                    self.parameterprovider,
                    self.weatherdataprovider,
                    self.agromanagement
                )
                self.current_crop = 'maize'
                if 'maize' in self.crop_planted.keys():
                    self.crop_planted['maize']['date'].append(self.date)
                else:
                    self.crop_planted['maize'] = {'date': [self.date], 'qty_tagp': 0., 'delta': 0.}
                self.isplanted = True
            else:
                apply_penalty = True
        elif action in list(range(3, 8)):
            # irrigate
            if self.isplanted:
                irrigation_rate = (action - 2)
                amount = irrigation_rate*self.amount # in cm
                self.model._send_signal(signal=pcse.signals.irrigate, amount=amount, efficiency=0.7)
            else:
                apply_penalty = True
        elif action == 8:
            # harvest
            if self.isplanted:
                _dict = self.agromanagement['AgroManagement'][0]
                for _k, v in _dict.items():
                    v['CropCalendar'] = {
                        'crop_name': 'maize',
                        'variety_name': "Maize_VanHeemst_1988",
                        'crop_start_date': self.sim_end_date - datetime.timedelta(days=1),
                        'crop_start_type': 'sowing',
                        'crop_end_date': self.sim_end_date,
                        'crop_end_type': 'earliest',
                        'max_duration': 300
                    }
                    self.agromanagement['AgroManagement'].append({self.date: v})
                self.agromanagement['AgroManagement'].pop(0)
                self.model = pcse.models.Wofost72_WLP_FD(
                    self.parameterprovider,
                    self.weatherdataprovider,
                    self.agromanagement
                )
                self.fallow = True
                self.fallow_start_days = self.model.timer.day_counter
                self.current_crop = None
                self.isplanted = False
            elif not self.isplanted and self.model.agromanager.ndays_in_crop_cycle >= 7:
                # handle an error with progressing the model with action ending up in a crop rotation
                _dict = self.agromanagement['AgroManagement'][0]
                for _k, v in _dict.items():
                    v['CropCalendar'] = {
                        'crop_name': 'maize',
                        'variety_name': "Maize_VanHeemst_1988",
                        'crop_start_date': self.sim_end_date - datetime.timedelta(days=1),
                        'crop_start_type': 'sowing',
                        'crop_end_date': self.sim_end_date,
                        'crop_end_type': 'earliest',
                        'max_duration': 300
                    }
                    self.agromanagement['AgroManagement'].append({self.date: v})
                self.agromanagement['AgroManagement'].pop(0)
                self.model = pcse.models.Wofost72_WLP_FD(
                    self.parameterprovider,
                    self.weatherdataprovider,
                    self.agromanagement
                )
                self.fallow = True
                self.current_crop = None
                self.isplanted = False
            else:
                apply_penalty = True
        return amount, apply_penalty, action

    def reset(self):
        """
        reset the state of the environment to an initial state
        """
        self.log = self._init_log()
        self.weatherdataprovider = self._get_weatherdataprovider()
        self.date = self.sim_start_date
        self.agromanagement = self._load_agromanagement_data()
        self.current_crop = None
        self.isplanted = False
        self.fallow = False
        self.fallow_time = 0
        self.fallow_start_days = 0
        self.crop_planted = {}
        self.model = pcse.models.Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider,
                                         self.agromanagement)
        output = self._run_simulation(self.model)
        observation = self._process_output(output)
        return observation

    def render(self, mode='human', close=False):
        """
        render the environment to the screen
        """

    def seed(self, seed=None):
        """
        fix the random seed
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_weather(self, weatherdataprovider, date, days):
        """
        Get weather observations for a range of days

        Parameters
        ----------
        weatherdataprovider : pcse weatherdataprovider
        date: datetime.date, start date for requested observations
        days: int, number of days of weather observations requested

        Returns
        -------
        numpy array containing the requested weatherdata
        """
        dates = [date + datetime.timedelta(i) for i in range(0, days)]
        weather = [self._get_weather_day(weatherdataprovider, day) for day in dates]
        return np.array(weather)

    @staticmethod
    def _get_weather_day(weatherdataprovider, date):
        """
        Get weather observations for a single day

        Parameters
        ----------
        weatherdataprovider : pcse weatherdataprovider
        date: datetime.date, date for requested observations

        Returns
        -------
        numpy array containing the requested weatherdata
        """
        weatherdatacontainer = weatherdataprovider(date)
        weather_vars = ['IRRAD', 'TMIN', 'TMAX', 'VAP', 'RAIN']
        weather = [getattr(weatherdatacontainer, attr) for attr in weather_vars]
        return weather

    def run_baseline_to_terminate(self, baseline_model):
        baseline_model.run_till_terminate()
        output = pd.DataFrame(baseline_model.get_output())
        return output
    def crop_cycle_delta(self, output: pd.DataFrame, baseline_model):
        """
        Go through each crop cycle in the agromanager, if then map the current date in the
        """
        crop_cycle_days = output['day'].apply(self._apply_crop_days, args=(baseline_model.agromanager.crop_calendars,))
        output.set_index('day')
        output["crop_days"] = crop_cycle_days
        return output


    def _apply_crop_days(self, date, crop_calendar):
        for cropcal in crop_calendar:
            start_date = cropcal.crop_start_date
            end_date = cropcal.crop_end_date
            if start_date <= date < end_date:
                return (date - start_date).days
        return np.nan

    def get_expected_baseline_growth(self, production_days):
        if self.current_crop == "maize":
            max_maize_product_days = self.baseline_maize_output.iloc[-1]['crop_days']
            if production_days > max_maize_product_days:
                current = \
                    self.baseline_maize_output.iloc[-1]['TAGP']
                previous_cycle = \
                    self.baseline_maize_output.iloc[-2]['TAGP']
                baseline_growth = current - previous_cycle
            elif production_days - self.intervention_interval < 0:
                current = \
                    self.baseline_maize_output[self.baseline_maize_output.crop_days == 0]['TAGP'].values[0]
                previous_cycle = \
                    self.baseline_maize_output[self.baseline_maize_output.crop_days == 0]['TAGP'].values[0]
                baseline_growth = current - previous_cycle
            else:
                current = \
                    self.baseline_maize_output[self.baseline_maize_output.crop_days == production_days]['TAGP'].values[0]
                previous_cycle = \
                    self.baseline_maize_output[self.baseline_maize_output.crop_days == production_days - self.intervention_interval]['TAGP'].values[0]
                baseline_growth = current - previous_cycle
            return baseline_growth
        elif self.current_crop == "wheat":
            max_wheat_product_days = self.baseline_wheat_output.iloc[-1]['crop_days']
            if production_days > max_wheat_product_days:
                current = \
                    self.baseline_maize_output.iloc[-1]['TAGP']
                previous_cycle = \
                    self.baseline_maize_output.iloc[-2]['TAGP']
                baseline_growth = current - previous_cycle
            elif production_days - self.intervention_interval < 0:
                current = \
                    self.baseline_wheat_output[self.baseline_wheat_output.crop_days == 0]['TAGP'].values[0]
                previous_cycle = \
                    self.baseline_wheat_output[self.baseline_wheat_output.crop_days == 0]['TAGP'].values[0]
                baseline_growth = current - previous_cycle
            else:
                current = \
                    self.baseline_wheat_output[self.baseline_wheat_output.crop_days == production_days]['TAGP'].values[0]
                previous_cycle = \
                    self.baseline_wheat_output[self.baseline_wheat_output.crop_days == production_days - self.intervention_interval]['TAGP'].values[0]
                baseline_growth = current - previous_cycle
            return baseline_growth
        return np.nan


