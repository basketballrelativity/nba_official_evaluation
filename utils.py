"""
utils.py

Utility functions for processing
official-related data
"""

import random
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats


def playoff_assignments():
	""" This function gathers the official IDs
	for playoff games

	Returns:
		- count_df (DataFrame): DataFrame containing
			the number of playoff games worked by official
	"""

	playoff_df_19 = pd.read_csv('data/box/box_2019_playoff.csv')
	playoff_df_20 = pd.read_csv('data/box/box_2020_playoff.csv')
	playoff_df_21 = pd.read_csv('data/box/box_2021_playoff.csv')

	official_list = list(playoff_df_19['Official 1']) + \
		list(playoff_df_19['Official 2']) + \
		list(playoff_df_19['Official 3']) + \
		list(playoff_df_20['Official 1']) + \
		list(playoff_df_20['Official 2']) + \
		list(playoff_df_20['Official 3']) + \
		list(playoff_df_21['Official 1']) + \
		list(playoff_df_21['Official 2']) + \
		list(playoff_df_21['Official 3'])

	all_df = pd.DataFrame({'official_id': official_list,
						   'id': range(len(official_list))})

	count_df = pd.DataFrame(all_df.groupby('official_id')['id'].count()).reset_index()
	count_df.columns = ['official_id', 'playoff_games']

	return count_df


def officials_to_one_hot(official_df):
	""" This function fomrats official IDs into one-hot
	encoded features

	@param official_df (pd.DataFrame): DataFrame containing
		referee IDs

	Returns:

		- official_df (DataFrame): DataFrame with one-hot
			indicators for referees working a game
	"""
	officials = list(set(list(official_df['Official 1']) +
						 list(official_df['Official 2']) +
						 list(official_df['Official 3'])))

	for official in officials:
		official_df[str(official)] = [1 if off_1 == official
									  or off_2 == official
									  or off_3 == official
									  else 0
									  for off_1, off_2, off_3 in zip(official_df['Official 1'],
									  								 official_df['Official 2'],
									  								 official_df['Official 3'])]

	return official_df


def str_to_time(time_str, period):
    """ This function converts a period and time to seconds remaining
    in the game.
    
    Overtime periods are treated as new games.

	@param time_str (str): Game time in MM:SS format
	@param period (int): Integer indicating the quarter of the
		game. Quarters greater than 4 indicate OT periods

	Returns:
		- seconds_left: Integer of seconds remaining in
			a quarter
    """

    split_time = time_str.split(':')
    seconds_left_period = int(split_time[0]) * 60 + int(split_time[1])

    if period <= 4:
        base_time = 720 * (4 - period)
    else:
        base_time = 0

    seconds_left = seconds_left_period + base_time

    return seconds_left


def str_to_time_l2m(time_str):
    """ This function converts a period and time to seconds remaining
    in the game.
    
    Overtime periods are treated as new games.

	@param time_str (str): Game time in MM:SS format

	Returns:
		- seconds_left_period - Integer of seconds remaining
			in a quarter
    """

    split_time = time_str.split(':')
    seconds_left_period = int(split_time[0]) * 60 + int(round(float(split_time[1])))

    return seconds_left_period


def read_files(year):
	""" This function reads in L2M
	and play-by-play data for the year provided

	@param year (int): Year in YYYY format

	Returns:

		- box_df (pd.DataFrame): DataFrame containing
			official and game IDs from the provided season
		- l2m_df (pd.DataFrame): DataFarme containing
			Last Two Minute report data
	"""

	if year != 2019:
		box_df = pd.read_csv(f'data/box/box_{year}.csv')

		l2m_df = pd.read_csv(f'data/l2m/l2m_{year}.csv')
	if year == 2019:
		box_df = pd.read_csv('data/box/box_2019.csv')
		box_alt_df = pd.read_csv('data/box/box_2019_alt.csv')
		box_df = pd.concat([box_df, box_alt_df], axis=0)

		l2m_df = pd.read_csv('data/l2m/l2m_2019.csv')

	l2m_ids = list(set(l2m_df['game_id']))


	return l2m_df, box_df


def date_conversion(challenge_df):
	""" This function converts the date strings from the
	challenge date to datetime objects

	@param challenge_df (pd.DataFrame): DataFrame containing
		Coach's Challenge data

	Returns:

		challenge_df (pd.DataFrame): DataFrame of challenges
			with a formatted Date column
	"""

	date_dict = {'Oct': '10',
				 'Nov': '11',
				 'Dec': '12',
				 'Jan': '1',
				 'Feb': '2',
				 'Mar': '3'}

	year_dict = {'Oct': '2019',
				 'Nov': '2019',
				 'Dec': '2019',
				 'Jan': '2020',
				 'Feb': '2020',
				 'Mar': '2020'}

	date_list = []
	for date_str in challenge_df['Date']:
		if '-' in date_str:
			date_vals = date_str.split('-')
			date_obj = datetime.strptime(date_dict[date_vals[1]] + '/' +
										 date_vals[0] + '/' +
										 year_dict[date_vals[1]], "%m/%d/%Y")
		elif "/" in date_str[-4:]:
			date_obj = datetime.strptime(date_str, "%m/%d/%y")
		else:
			date_obj = datetime.strptime(date_str, "%m/%d/%Y")

		date_list.append(date_obj)

	challenge_df['Date'] = date_list

	return challenge_df


def read_challenges():
	""" THis function reads in the challenge data
	and joins to referee assignments

	Returns:

		challenge_df (pd.DataFrame): DataFrame containig
			Coach's challenge data and referee assignments
	"""

	challenge_df = pd.read_csv('data/challenge/challenge_2019.csv', encoding = "ISO-8859-1")
	challenge_df_20 = pd.read_csv('data/challenge/challenge_2020.csv', encoding = "ISO-8859-1")
	challenge_df = pd.concat([challenge_df, challenge_df_20], axis=0)

	challenge_df = challenge_df[pd.notnull(challenge_df['Date'])]
	challenge_df = date_conversion(challenge_df)

	games_df = pd.read_csv('data/game/games_2019.csv')
	games_df_alt = pd.read_csv('data/game/games_2019_alt.csv')
	games_df_21 = pd.read_csv('data/game/games_2020.csv')
	games_df = pd.concat([games_df, games_df_alt], axis=0)
	games_df = pd.concat([games_df, games_df_21], axis=0)

	games_df['Date'] = [datetime.strptime(x[0:10], '%Y-%m-%d')
								 for x in games_df['GAME_DATE_EST']]
	games_df['Visiting'] = [x.split('/')[1][0:3] for x in games_df['GAMECODE']]
	games_df['Home'] = [x.split('/')[1][3:] for x in games_df['GAMECODE']]

	challenge_df = challenge_df.merge(games_df[['GAME_ID', 'Visiting', 'Home', 'Date']],
									  left_on=['Date', 'Visiting', 'Home'],
									  right_on=['Date', 'Visiting', 'Home'])

	return challenge_df
