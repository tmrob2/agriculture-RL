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
WEIGHT_CONVERSION = 10000

class IrrigationEnv(gym.Env):
    """An environment for OpenAI gym to study crop irrigation"""
    metadata = {'render.modes': ['human']}

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
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11 + 7 * 5,), dtype=np.float32)

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
        agromanagement_file = os.path.join(agro_dir, 'baseline_4yr_model.yaml')
        self.baseline_agromanagement = YAMLAgroManagementReader(agromanagement_file)
        self.date = self.sim_start_date
        self.baseline_date = self.sim_start_date
        self.current_crop = None
        self.isplanted = False
        self.baseline_growth = 0.
        self.crop_planted = {}
        self.model = pcse.models.Wofost72_WLP_FD(
            self.parameterprovider,
            self.weatherdataprovider,
            self.agromanagement
        )
        self.baseline_model = pcse.models.Wofost72_WLP_FD(
            self.parameterprovider,
            self.weatherdataprovider,
            self.baseline_agromanagement
        )
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
        irrigation_amount = self._take_action(action)
        baseline_date_diff = self.date - self.baseline_date
        output = self._run_simulation(self.model, baseline_date_diff)
        self.date = output.index[-1]
        if self.date > self.baseline_date:
            baseline_output = self._run_simulation(self.baseline_model, self.date - self.baseline_date)
            self.baseline_date = baseline_output.index[-1]
            baseline_growth = baseline_output['TAGP'][-1] \
                              - baseline_output['TAGP'][-1 - self.intervention_interval]
            baseline_growth = baseline_growth / WEIGHT_CONVERSION if not np.isnan(baseline_growth) else 0
            self.baseline_growth = baseline_growth
        try:
            growth = output['TAGP'][-1] / WEIGHT_CONVERSION - \
                     output['TAGP'][-1 - self.intervention_interval] / WEIGHT_CONVERSION
            if self.isplanted:
                self.crop_planted[self.current_crop]['qty_tagp'] += growth
                self.crop_planted[self.current_crop]['delta'] = growth
        except:
            growth = np.nan
            if self.current_crop in self.crop_planted.keys():
                self.crop_planted[self.current_crop]['detla'] = 0.

        growth = growth / WEIGHT_CONVERSION if not np.isnan(growth) else 0

        reward = growth - self.baseline_growth - self.beta * irrigation_amount / 1000  # cm -> m

        observation = self._process_output(output)
        done = self.date >= self.sim_end_date

        self._log(growth, self.baseline_growth, irrigation_amount, reward, self.crop_planted)
        if self.training:
            info = {**self.log}
        else:
            info = {**output.to_dict(), **self.log}
        #print(f"action: {action}, agent date: {self.date}, baseline date: {self.baseline_date}, reward: {reward}, done: {done}, crops planted: {self.crop_planted}")
        # print(self.crop_planted)
        return observation, reward, done, info

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
        observation = np.concatenate([crop_observation, weather_observation.flatten()], dtype=np.float32)
        observation = np.nan_to_num(observation)
        return observation

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

    def _run_simulation(self, model, datediff: datetime.timedelta):
        if datediff.days > self.intervention_interval:
            model.run(days=datediff.days)
        else:
            model.run(days=self.intervention_interval)
        output = pd.DataFrame(model.get_output()).set_index("day")
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
        elif action in list(range(3, 8)):
            # irrigate
            irrigation_rate = (action - 2)
            amount = irrigation_rate*self.amount # in cm
            self.model._send_signal(signal=pcse.signals.irrigate, amount=amount, efficiency=0.7)
        elif action == 8:
            # harvest
            if self.isplanted:
                _dict = self.agromanagement['AgroManagement'][0]
                for _k, v in _dict.items():
                    v['CropCalendar'] = {
                        'crop_name': 'maize',
                        'variety_name': "Maize_VanHeemst_1988",
                        'crop_start_date': self.date + datetime.timedelta(days=180)
                            if self.date + datetime.timedelta(days=180) < self.sim_end_date
                            else self.sim_end_date - datetime.timedelta(days=1),
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
                self.current_crop = None
                self.isplanted = False
        return amount

    def reset(self):
        """
        reset the state of the environment to an initial state
        """
        self.log = self._init_log()
        self.weatherdataprovider = self._get_weatherdataprovider()
        self.date = self.sim_start_date
        self.agromanagement = self._load_agromanagement_data()
        self.baseline_date = self.sim_start_date
        self.current_crop = None
        self.isplanted = False
        self.baseline_growth = 0.
        self.crop_planted = {}
        self.model = pcse.models.Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider,
                                         self.agromanagement)
        self.baseline_model = pcse.models.Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider,
                                                  self.baseline_agromanagement)
        output = self._run_simulation(self.model, datetime.timedelta(0, 0, 0))
        self._run_simulation(self.baseline_model, datetime.timedelta(0, 0, 0))
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

