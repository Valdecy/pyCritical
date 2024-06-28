###############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# pyCritical

# GitHub Repository: <https://github.com/Valdecy/pyCritical>

###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import scipy.stats

###############################################################################

# Helper Functions

# Function: Probability to Finish in Date X
def date_prob(date, mean, dp):
  z    = (date - mean)/dp
  prob = scipy.stats.norm.cdf(z)
  return prob

# Function: Date Required to Finish wit Probability Y
def date_required(prob, mean, dp):
  z    = scipy.stats.norm.ppf(prob)
  date = (dp*z) + mean
  return date

###############################################################################

# Plot Functions

# Function: Gantt Chart
def gantt_chart(dataset, dates, size_x = 15, size_y = 10, show_slack = True):
    finish_time = dates.iloc[:,1].max()
    tasks       = list(dates.index.values)
    tasks_list  = list(range(0, dates.shape[0]))
    fig, ax     = plt.subplots(figsize = (size_x, size_y))
    yticks      = [5+1.5*i for i in range(0, len(tasks_list))]
    xticks      = [i for i in range(0, int(np.ceil(finish_time)) + 1)]
    height      = 1
    colors      = ['#aec1d1' if dates.iloc[i, -1] > 0 else '#e6637d' for i in range(0, dates.shape[0])]
    colors_dict = dict(zip(tasks, colors))
    ax.set_ylabel('Tasks')
    ax.set_xlim(0, finish_time + 1)
    ax.set_xticks(xticks)
    ax.set_xlabel('Time')
    plt.gca().invert_yaxis()
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title('CPM')
    ax.grid(True)
    for i in range(0, dates.shape[0]):
        idx_s = dates.iloc[i, 0]
        idx_e = dates.iloc[i, 1]
        dur   = idx_e - idx_s
        ax.broken_barh( [(idx_s, dur) ], (yticks[i] - height/2, height), facecolors = colors_dict[tasks[i]])
        ax.text(x = idx_e - (dur)/2, y = yticks[i], s = tasks[i], ha = 'center', va = 'center', color = 'k')
        if (dates.iloc[i, -1] > 0 and show_slack == True):
          ax.hlines(y = yticks[i], xmin = dates.iloc[i, 1], xmax = dates.iloc[i, -2], linewidth = 1, color = 'r')
          ax.plot( dates.iloc[i, -2], yticks[i], 'o', markersize = 5, color = 'r')
        if (len(dataset) > 0):
          if (len(dataset[i][1]) > 0):
            dependencies = dataset[i][1]
            for item in dependencies:
              k   = tasks.index(item)
              h_0 = dates.iloc[i, 0]
              h_1 = dates.iloc[k, 0] + (dates.iloc[k, 1] - dates.iloc[k, 0] )/2
              v_0 = yticks[i]
              if (yticks[k] > yticks[i]):
                v_1 = yticks[k] - 0.5
              else:
                v_1 = yticks[k] + 0.5
              ax.hlines(y = yticks[i], xmin = h_0, xmax = h_1, linewidth = 1, color = 'k')
              ax.vlines(x = h_1,       ymin = v_0, ymax = v_1, linewidth = 1, color = 'k')
    return

###############################################################################

# CPM & PERT

# Function: CPM
def critical_path_method(dataset):
  ids = [item[0] for item in dataset]
  idx = [i for i in range(0, len(dataset))]
  idd = dict(zip(ids, idx))
  e_m = np.zeros((len(dataset), len(dataset)))
  l_m = np.zeros((len(dataset), len(dataset)))
  for i in range(0, e_m.shape[0]):
    if (len(dataset[i][1]) != 0):
      a = idd[dataset[i][0]]
      for b in dataset[i][1]:
        b         = idd[b]
        e_m[a, b] = e_m[a, b] + 1
        l_m[a, b] = l_m[a, b] + 1
  dates        = np.zeros((len(dataset), 5))
  dates[:,-2:] = float('+inf')
  early        = np.sum(e_m, axis = 1)
  flag         = True
  while (np.sum(early, axis = 0) != -early.shape[0]):
    j_lst = []
    for i in range(0, early.shape[0]):
      if (early[i] == 0):
        early[i] = -1
        j_lst.append(i)
    if (flag == True):
      for j in j_lst:
        dates[j, 0] = 0
        dates[j, 1] = dates[j, 0] + dataset[j][2]
        flag        = False
    if (flag == False):
      for j in j_lst:
        for i in range(0, early.shape[0]):
          if (e_m[i, j] == 1):
            e_m[i, j] = 0
            early[i]  = np.clip(early[i] - 1, -1, float('+inf'))
            if (dates[i, 1] < dates[j, 1] + dataset[i][2]):
              dates[i, 0] = dates[j, 1]
              dates[i, 1] = dates[j, 1] + dataset[i][2]
  finish_time = np.max(dates[:,1])
  late        = np.sum(l_m, axis = 0)
  flag        = True
  while (np.sum(late, axis = 0) > -late.shape[0]):
    i_lst = []
    for i in range(0, late.shape[0]):
      if (late[i] == 0):
        late[i] = -1
        i_lst.append(i)
    if (flag == True):
      for i in i_lst:
        dates[i, 3] = finish_time
        dates[i, 2] = dates[i, 3] - dataset[i][2]
        dates[i,-1] = dates[i, 3] - dates[i, 1]
        flag        = False
    if (flag == False):
      for i in i_lst:
        for j in range(0, late.shape[0]):
          if (l_m[i, j] == 1):
            l_m[i, j] = 0
            late[j]   = np.clip(late[j] - 1, -1, float('+inf'))
            if (dates[j, 3] > dates[i, 3] - dataset[i][2]):
              dates[j, 3] = dates[i, 3] - dataset[i][2]
              dates[j, 2] = dates[j, 3] - dataset[j][2]
              dates[j,-1] = dates[j, 3] - dates[j, 1]
  dates = pd.DataFrame(dates, index = ids, columns = ['ES', 'EF', 'LS','LF', 'Slack'])
  return dates

# Function: PERT
def pert_method(dataset):
  ids  = [item[0] for item in dataset]
  idx  = [i for i in range(0, len(dataset))]
  idd  = dict(zip(ids, idx))
  mean = [(dataset[i][2] + 4*dataset[i][3] + dataset[i][4])/6 for i in range(0, len(dataset))]
  var  = [((dataset[i][4] - dataset[i][2])/6)**2 for i in range(0, len(dataset))]
  e_m  = np.zeros((len(dataset), len(dataset)))
  l_m  = np.zeros((len(dataset), len(dataset)))
  for i in range(0, e_m.shape[0]):
    if (len(dataset[i][1]) != 0):
      a = idd[dataset[i][0]]
      for b in dataset[i][1]:
        b         = idd[b]
        e_m[a, b] = e_m[a, b] + 1
        l_m[a, b] = l_m[a, b] + 1
  dates        = np.zeros((len(dataset), 5))
  dates[:,-2:] = float('+inf')
  early        = np.sum(e_m, axis = 1)
  flag         = True
  while (np.sum(early, axis = 0) != -early.shape[0]):
    j_lst = []
    for i in range(0, early.shape[0]):
      if (early[i] == 0):
        early[i] = -1
        j_lst.append(i)
    if (flag == True):
      for j in j_lst:
        dates[j, 0] = 0
        dates[j, 1] = dates[j, 0] + mean[j]
        flag        = False
    if (flag == False):
      for j in j_lst:
        for i in range(0, early.shape[0]):
          if (e_m[i, j] == 1):
            e_m[i, j] = 0
            early[i]  = np.clip(early[i] - 1, -1, float('+inf'))
            if (dates[i, 1] < dates[j, 1] + mean[i]):
              dates[i, 0] = dates[j, 1]
              dates[i, 1] = dates[j, 1] + mean[i]
  finish_time = np.max(dates[:,1])
  late        = np.sum(l_m, axis = 0)
  flag        = True
  while (np.sum(late, axis = 0) > -late.shape[0]):
    i_lst = []
    for i in range(0, late.shape[0]):
      if (late[i] == 0):
        late[i] = -1
        i_lst.append(i)
    if (flag == True):
      for i in i_lst:
        dates[i, 3] = finish_time
        dates[i, 2] = dates[i, 3] - mean[i]
        dates[i,-1] = dates[i, 3] - dates[i, 1]
        flag        = False
    if (flag == False):
      for i in i_lst:
        for j in range(0, late.shape[0]):
          if (l_m[i, j] == 1):
            l_m[i, j] = 0
            late[j]   = np.clip(late[j] - 1, -1, float('+inf'))
            if (dates[j, 3] > dates[i, 3] - mean[i]):
              dates[j, 3] = dates[i, 3] - mean[i]
              dates[j, 2] = dates[j, 3] - mean[j]
              dates[j,-1] = dates[j, 3] - dates[j, 1]
  dates = pd.DataFrame(dates, index = ids, columns = ['ES', 'EF', 'LS','LF', 'Slack'])
  dates = dates.round(decimals = 4)
  idx   =  [i for i in range(0, dates.shape[0]) if dates.iloc[i, -1] == 0]
  dp    = sum([var[i] for i in idx])**(1/2)
  return dates, dates.iloc[:,1].max(), dp

###############################################################################