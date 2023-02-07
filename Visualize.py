import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_weekday(data,column,kind=None,agg_first=False, aggfunc="mean"):
    ss = data.query("type==@column")
    
    # optional: sum for each day, then average
    if agg_first:
        ss = ss.pivot_table(index="Day",values=["value","Weekday"],aggfunc={"value":aggfunc,"Weekday":"mean"})
    else:
        pass
    
    if kind is not None:
        sns.catplot(data=ss,x="Weekday",y="value",kind=kind)
    else:
        f = plt.figure(figsize=(16,10))
        sns.boxplot(data=ss,x="Weekday",y="value")
    plt.title(column)
    plt.show()
    
def plot_hour(data,column,kind=None,agg_first=False, aggfunc="sum"):
    ss = data.query("type==@column")

    # optional: sum for each hour, then average ! Might be biased due to unbalancing
    if agg_first:
        ss = ss.pivot_table(index=["Date","Hour"],values=["value"],aggfunc={"value":aggfunc}).reset_index()
    else:
        pass
    
    
    if kind is not None:
        sns.catplot(data=ss,x="Hour",y="value",kind=kind)
    else:
        f = plt.figure(figsize=(16,10))
        sns.boxplot(data=ss,x="Hour",y="value")
    plt.title(column)
    plt.show()
