from datetime import datetime, time
import pandas as pd
import numpy as np
from tqdm import tqdm

redundant = ["device", "sourceName", "HKMetadataKeyHeartRateRecoveryActivityType", "HKHeartRateEventThreshold", "HKWorkoutEventTypeMarker", "HKMetadataKeyUserMotionContext", "HKMetadataKeySyncIdentifier",
             "HKMetadataKeySyncVersion", "HKMetadataKeyHeartRateRecoveryTestType", "HKMetadataKeyHeartRateRecoveryActivityDuration", "HKMetadataKeyHeartRateRecoveryMaxObservedRecoveryHeartRate", "HKMetadataKeySessionEstimate"]


def add_date_variables(data):
    if data.startDate.isna().sum() > 0:
        print(
            f"Dropping {data.startDate.isna().sum()} entries with missing time")
        data.dropna(subset=["startDate"], inplace=True)

    data["start"] = pd.to_datetime(data.startDate)
    data["end"] = pd.to_datetime(data.endDate)
    data["date"] = data["start"].dt.date
    data["time"] = data["start"].dt.time
    data["week"] = data["start"].dt.week
    data["weekday"] = data["start"].dt.weekday
    data["day"] = data["start"].dt.day
    data["hour"] = data["start"].dt.hour
    return data

def light_preprocess(data):
    data = data.dropna(axis=1, how="all")
    # add empty rows (missing dates) and add time variables
    data = add_date_variables(data)
    # force numeric values
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    # filter for sourceName where the source contains the substring "Apple Watch"
    data = data[data["sourceName"].str.contains("Watch",regex=False)]
    return data



def add_workout_variables(data):
    """This function takes in a dataframe data and adds several columns related to workout information. 
        The function first retrieves the workouts from the data using the get_workouts function, and then iterates through each workout. 
        For each workout, the function extracts information such as the workout type, start and end times, and whether or not there are pauses during the workout. 
        It then creates a subset of the data that corresponds to the current workout and adds columns to this subset such as "workout_type", "duration_into_workout", and "identifier". 
        The subset is then added back to the original dataframe, and the function returns the modified dataframe.

    Args:
        data (dataframe): data

    Returns:
        dataframe: data
    """
    # get each workout. they are mutually exlusive time intervals
    workouts = get_workouts(data)
    # get rows in original table in that interval (ignoring breaks for now)
    for i, workout in tqdm(workouts.iterrows()):
        # get workout properties
        workout_type = workout.workoutActivityType[workout.workoutActivityType.find(
            "Type")+4:].lower()
        identifier = workout_type + str(workout.start)
        pause_start = datetime.strptime(workout.HKWorkoutEventTypePause[:-6], "%Y-%m-%d %H:%M:%S") if isinstance(
            workout.HKWorkoutEventTypePause, str) else pd.NA
        pause_end = datetime.strptime(workout.HKWorkoutEventTypeResume[:-6], "%Y-%m-%d %H:%M:%S") if isinstance(
            workout.HKWorkoutEventTypeResume, str) else pd.NA
        has_pause_start = False if pause_start is pd.NA else True
        has_pause_end = False if pause_end is pd.NA else True

        # get subset of data recorded during the workout's time interval
        indices = ((data["start"] > workout.start)
                   & (data["end"] <= workout.end))
        workout_subset = data.loc[indices]
        # get some workout related columns
        workout_subset["workout"] = True
        workout_subset["workout_type"] = workout_type
        workout_subset["duration_into_workout"] = (
            workout_subset["start"]-workout.start).dt.total_seconds()/60
        # this minuses out the time spend in pause
        workout_subset["net_duration"] = workout.duration
        workout_subset["all_duration"] = workout.end - workout.start
        workout_subset["identifier"] = identifier
        new_cols = ["workout", "workout_type", "duration_into_workout",
                    "net_duration", "all_duration", "identifier"]

        if has_pause_start:
            new_cols.append("has_pause")
            new_cols.append("paused")
            workout_subset["has_pause"] = has_pause_start
            if has_pause_end:
                workout_subset["paused"] = (workout_subset["start"] > pause_start) & (
                    workout_subset["end"] < pause_end)
                workout_subset["duration_since_last_paused"] = [max(0, x) for x in (
                    workout_subset["start"]-pause_start).dt.total_seconds()/60]
                new_cols.append("duration_since_last_paused")
            else:
                workout_subset["paused"] = (workout_subset["start"] > pause_start) & (
                    workout_subset["end"] < workout.end)

        # set identifier & type to workout entry as well
        data.loc[i, "identifier"] = identifier
        data.loc[i, "workout_type"] = workout_type
        # Overwrite subset pertaining to workout with modified version
        data.loc[indices, new_cols] = workout_subset[new_cols]

    return data


def modify_HK_units(data):
    """gets rid of the units to cast some HK columns to numeric

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    data["HKMetadataKeyBarometricPressure"] = data["HKMetadataKeyBarometricPressure"].str.extract(
        "(\d+)").astype(float)
    data["HKElevationAscended"] = data["HKElevationAscended"].str.extract(
        "(\d+)").astype(float)/100  # -> in meters
    data["HKAverageMETs"] = data["HKAverageMETs"].str.extract(
        "(\d+)").astype(float)  # -> kcal/hr*kg
    data["HKWeatherHumidity"] = data["HKWeatherHumidity"].str.extract(
        "(\d+)").astype(float)/100  # -> in %
    data["HKWeatherTemperature"] = (data["HKWeatherTemperature"].str.extract(
        "(\d+)").astype(float)-32)*(5/9)  # -> in °C
    return data


def preprocess(data, drop=False, set_start_index=False):
    """
    1) Drop empty columns 
    2) adds date variables (including day and hour)
    3) adds workout variables (matching entries to workouts and assigning new columns for these subsets)
    4) modifies units (drops the units so columns are numerical)

    Args:
        data (df): original data
        drop (bool, optional): Whether to drop "redundant" columns. Defaults to False.
        set_start_index (bool, optional): Whether to set start as indices. Defaults to False.

    Returns:
        data: modified original dataframe
    """
    data = data.dropna(axis=1, how="all")
    if drop:
        # the list needs to be updated
        data.drop(axis=1, labels=redundant, inplace=True)
    # add empty rows (missing dates) and add time variables
    data = add_date_variables(data)
    # set start and end as index
    data = add_workout_variables(data)
    # modify units
    data = modify_HK_units(data)
    # force numeric values
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    # potentially change index
    if set_start_index:
        data = data.set_index(["start"])
    return data


def select_period(data, start, end):
    """pass start and end as datetime(year,month,day). Both endpoints included"""
    period_start = start.date()
    period_end = end.date()
    assert type(period_start) == type(datetime.today().date())
    assert type(period_end) == type(datetime.today().date())

    data = data[(data["start"] >= period_start) & (data["end"] <= period_end)]

    return data


def get_workouts(data):
    """The high level workout data points are found through the "workoutActivitiyType" column not being empty.
    """
    return data[data.workoutActivityType.notna()]


def get_daily_heartrate_stats(data):
    hr = get_heartrate_data(data)
    daily = (hr[hr["first_on_wrist"]]
             .groupby("date")[["time", "off_wrist"]].min()
             .join(hr[hr["last_on_wrist"]]
                   .groupby("date")[["time"]].max(), rsuffix="_last", lsuffix="_first")
             .join(data[(data["type"] == "HeartRate") & (data["workout"] != True)]
                   .pivot_table(index="date", columns="type", values="value", aggfunc=["min", "max", "var"]))
             .join(data[(data["type"] == "HeartRate")]
                   .pivot_table(index="date", columns="type", values="value", aggfunc=["max", "var"]), lsuffix="_leisure")
             .join(data[data["type"] == "RestingHeartRate"]
                   .pivot_table(index="date", columns="type", values="value", aggfunc="min"))
             .join(get_workouts(data).groupby("date").agg({"identifier": "count", "duration": "sum"}))
             )
    daily.rename(columns={
        "time_first": "first_on_wrist",
        "time_last": "last_on_wrist",
        ('min', 'HeartRate'): "min_Heartrate",
        "('max', 'HeartRate')_leisure": "max_Heartrate_leisure",
        "('var', 'HeartRate')_leisure": "var_Heartrate_leisure",
        ('max', 'HeartRate'): "max_HeartRate",
        ('var', 'HeartRate'): "var_HeartRate"
    }, inplace=True)
    return daily


def get_heartrate_data(data):
    """auxiliary function for get_daily_heartrate_stats

    Args:
        data (df): df

    Returns:
        df: df
    """
    hr = data[data["type"] == "HeartRate"].sort_values("start")
    hr["off_wrist"] = hr.start.diff(1).dt.total_seconds()/3600
    hr["first_on_wrist"] = ((hr["off_wrist"] > 5) & (
        hr["time"] < time(13)) & (hr["time"] > time(4)))
    hr["last_on_wrist"] = hr["first_on_wrist"].shift(-1).fillna(False)
    return hr
