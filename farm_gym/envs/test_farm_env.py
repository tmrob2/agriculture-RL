import gym
from gym import spaces
import numpy as np
from typing import Optional
import pandas as pd
import pcse
import os
import datetime
import yaml
from pcse.fileinput import YAMLCropDataProvider, CABOFileReader
from pcse.util import WOFOST72SiteDataProvider

DATA_DIR = os.path.join(os.getcwd(), 'farm_gym/envs/env_data/')

class IrrigationEnv(gym.Env):
    """An environment for OpenAI gym to study crop irrigation"""
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        soil_type: str,
        fixed_location: tuple,
        fixed_date, 
        intervention_interval=7,
        beta=10, 
        seed=0, 
    ):
        self.action_space = gym.spaces.Discrete(6)
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
        self.fixed_year = datetime.datetime.strptime(fixed_date, "%Y-%m-%d")
        self.train_weather_data = self._get_train_weather_data()
        self.agromanagement, self.crop_start_date, self.crop_end_date = self._load_agromanagement_data()
        self.date = self.crop_start_date
        # The LINTUL engine (Light Interception and Utilisation) simple general crop model
        # simulating dry matter production as the result of light interception and utilisation
        # with a constant light use efficiency
        #
        # LINTUL3 simulated crop growth under water-limited and nitrogen-limited conditions

        self.model = pcse.models.Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider,
                                         self.agromanagement)
        self.baseline_model = pcse.models.Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider,
                                                  self.agromanagement)
        self.log = self._init_log()

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
        output = self._run_simulation(self.model)
        baseline_output = self._run_simulation(self.baseline_model)
        observation = self._process_output(output)
        self.date = output.index[-1]

        growth = output['TAGP'][-1] - output['TAGP'][-1-self.intervention_interval]
        growth = growth if not np.isnan(growth) else 0
        twso = output['TWSO'][-1]
        baseline_growth = baseline_output['TAGP'][-1]\
            - baseline_output['TAGP'][-1-self.intervention_interval]
        baseline_growth = baseline_growth if not np.isnan(baseline_growth) else 0

        reward = growth - baseline_growth - self.beta * irrigation_amount
        done = self.date >= self.crop_end_date

        self._log(growth, baseline_growth, irrigation_amount, reward)

        info = {**output.to_dict(), **self.log}

        return observation, reward, done, info

    @staticmethod
    def _init_log():
        return {'growth': dict(), 'baseline_growth': dict(), 'irrigation': dict(), 'reward': dict()}

    def _load_agromanagement_data(self):
        with open(os.path.join(DATA_DIR, 'agro/wofost_wheat.yaml')) as file:
            agromanagement = yaml.load(file, Loader=yaml.SafeLoader)
        return self._replace_year(agromanagement)

    def _log(self, growth, baseline_growth, irrigation, reward):
        self.log['growth'][self.date] = growth
        self.log['baseline_growth'][self.date] = baseline_growth
        self.log['irrigation'][self.date - datetime.timedelta(self.intervention_interval)] = \
            irrigation
        self.log['reward'][self.date] = reward

    def _process_output(self, output):
        crop_observation = np.array(output.iloc[-1])
        # weather until next intervention time
        weather_observation = self._get_weather(self.weatherdataprovider, self.date,
                                             self.intervention_interval)
        observation = np.concatenate([crop_observation, weather_observation.flatten()], dtype=np.float32)
        observation = np.nan_to_num(observation)
        return observation

    def _replace_year(self, agromanagement):
        dict_ = agromanagement['AgroManagement'][0]
        old_date = next(iter(dict_.keys()))
        new_date = old_date.replace(year=self.fixed_year.year + 1)
        content = dict_[old_date]
        crop_start_date = content['CropCalendar']['crop_start_date'].replace(year=self.fixed_year.year+1)
        content['CropCalendar']['crop_start_date'] = crop_start_date
        crop_end_date = content['CropCalendar']['crop_end_date'].replace(year=self.fixed_year.year+1)
        content['CropCalendar']['crop_end_date'] = crop_end_date
        dict_[new_date] = dict_.pop(old_date)
        return agromanagement, crop_start_date, crop_end_date

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
        output = pd.DataFrame(model.get_output()).set_index("day")
        output = output.fillna(value=np.nan)
        return output

    def _take_action(self, action): 
        """
        Apply some irrigation to the crop. The irrigation model can be  {4, 8, 12, 16, 20} 
        cm h of irrigation
        """
        amount = action*self.amount # in cm
        self.model._send_signal(signal=pcse.signals.irrigate, amount=amount, efficiency=0.7)
        return amount

    def reset(self):
        """
        reset the state of the environment to an initial state
        """
        self.log = self._init_log()
        self._replace_year(self.agromanagement)
        self.weatherdataprovider = self._get_weatherdataprovider()
        self.crop_start_date = \
            list(self.agromanagement['AgroManagement'][0].values())[0]['CropCalendar']['crop_start_date']
        self.crop_end_date = \
            list(self.agromanagement['AgroManagement'][0].values())[0]['CropCalendar']['crop_end_date']
        self.date = self.crop_start_date
        self.model = pcse.models.Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider,
                                         self.agromanagement)
        self.baseline_model = pcse.models.Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider,
                                                  self.agromanagement)
        output = self._run_simulation(self.model)
        self._run_simulation(self.baseline_model)
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
