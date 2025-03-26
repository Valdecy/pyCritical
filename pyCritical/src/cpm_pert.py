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

# Function: Gantt Chart
def gantt_chart_dep(dataset, dates, size_x = 15, size_y = 10, show_slack = True):
    finish_time = dates['EF'].max()
    tasks       = list(dates.index.values)
    idd         = {task_id: i for i, task_id in enumerate(tasks)}
    yticks      = [5 + 1.5 * i for i in range(len(tasks))]
    height      = 1
    fig, ax     = plt.subplots(figsize = (size_x, size_y))
    ax.set_xlim(0, finish_time + 1)
    ax.set_xticks(range(int(finish_time) + 2))
    ax.set_ylabel('Tasks')
    ax.set_xlabel('Time')
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.gca().axes.get_yaxis().set_visible(False)
    colors     = ['#aec1d1' if dates.iloc[i, -1] > 0 else '#e6637d' for i in range(dates.shape[0])]
    color_dict = dict(zip(tasks, colors))
    for i, task in enumerate(dataset):
        task_id = task[0]
        es      = dates.loc[task_id, 'ES']
        ef      = dates.loc[task_id, 'EF']
        dur     = ef - es
        ax.broken_barh([(es, dur)], (yticks[i] - height/2, height), facecolors = color_dict[task_id])
        ax.text(x = es + dur/2, y = yticks[i], s = task_id, ha = 'center', va = 'center', color = 'k')
        if (show_slack and dates.loc[task_id, 'Slack'] > 0):
            slack_end = dates.loc[task_id, 'LF']
            ax.hlines(y = yticks[i], xmin = ef, xmax = slack_end, linewidth = 1, color = 'r')
            ax.plot(slack_end, yticks[i], 'o', markersize = 5, color = 'r')
    for i, task in enumerate(dataset):
        task_id = task[0]
        preds   = task[1]
        for pred_id, dep_type, lag in preds:
            if pred_id not in idd:
                continue
            j        = idd[pred_id]
            h_start  = dates.loc[pred_id, 'ES']
            h_end    = dates.loc[task_id, 'ES']
            v0       = yticks[j]
            v1       = yticks[i]
            h_anchor = h_start if dep_type in ['SS', 'SF'] else dates.loc[pred_id, 'EF']
            t_anchor = h_end   if dep_type in ['SS', 'FS'] else dates.loc[task_id, 'EF']
            h        = h_anchor
            t        = t_anchor
            x_mid    = (h + t) / 2
            ax.hlines(y = v0,    xmin = h,     xmax = x_mid, linewidth = 1, color = 'k')
            ax.vlines(x = x_mid, ymin = v0,    ymax = v1,    linewidth = 1, color = 'k')
            ax.hlines(y = v1,    xmin = x_mid, xmax = t,     linewidth = 1, color = 'k')
            ax.text(x = t, y = (v0+v1)/2, s = f'{dep_type}+{lag}', fontsize = 8, color = 'blue', ha = 'left')
    plt.show()
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

# CPM & PERT with Dependencies and Lags

# Function: CPM with Dependencies and Lags
def critical_path_method_dep(dataset):
    ids           = [item[0] for item in dataset]
    idd           = dict(zip(ids, range(len(dataset))))
    n             = len(dataset)
    dates         = np.zeros((n, 5))  # ES, EF, LS, LF, Slack
    dates[:, -2:] = float('+inf')
    visited       = set()
    while len(visited) < n:
        for i, (task_id, preds, duration) in enumerate(dataset):
            if task_id in visited:
                continue
            if all(pred[0] in ids and pred[0] in visited for pred in preds):
                es_candidates = []
                for pred_id, dep_type, lag in preds:
                    j = idd[pred_id]
                    pred_es, pred_ef = dates[j, 0], dates[j, 1]
                    if dep_type == 'FS':
                        es_candidates.append(pred_ef + lag)
                    elif dep_type == 'SS':
                        es_candidates.append(pred_es + lag)
                    elif dep_type == 'FF':
                        es_candidates.append(pred_ef + lag - duration)
                    elif dep_type == 'SF':
                        es_candidates.append(pred_es + lag - duration)
                    else:
                        raise ValueError(f"Unsupported dependency type: {dep_type}")
                es          = max(es_candidates) if es_candidates else 0
                ef          = es + duration
                dates[i, 0] = es
                dates[i, 1] = ef
                visited.add(task_id)
    finish_time = np.max(dates[:, 1])
    visited     = set()
    while len(visited) < n:
        for i in reversed(range(n)):
            task_id, preds, duration = dataset[i]
            if task_id in visited:
                continue
            successors = [ (succ_id, dep_type, lag, j) for j, (succ_id, succ_preds, _) in enumerate(dataset) for (pred_id, dep_type, lag) in succ_preds if pred_id == task_id ]
            if all(succ[0] in visited or succ[0] not in ids for succ in successors):
                lf_candidates = []
                for succ_id, dep_type, lag, j in successors:
                    if dep_type == 'FS':
                        lf_candidates.append(dates[j, 0] - lag)
                    elif dep_type == 'SS':
                        lf_candidates.append(dates[j, 0] - lag + duration)
                    elif dep_type == 'FF':
                        lf_candidates.append(dates[j, 3] - lag)
                    elif dep_type == 'SF':
                        lf_candidates.append(dates[j, 3] - lag + duration)
                    else:
                        raise ValueError(f"Unsupported dependency type: {dep_type}")
                lf          = min(lf_candidates) if lf_candidates else finish_time
                ls          = lf - duration
                dates[i, 3] = lf
                dates[i, 2] = ls
                dates[i, 4] = ls - dates[i, 0]
                visited.add(task_id)
    df               = pd.DataFrame(dates, index = ids, columns = ['ES', 'EF', 'LS', 'LF', 'Slack'])
    df               = df.round(4)
    predecessors_map = {task[0]: task[1] for task in dataset}
    critical_set     = set(df[np.isclose(df['EF'], finish_time, atol = 1e-4)].index)

    def trace_critical(task_id):
        for pred_id, dep_type, lag in predecessors_map.get(task_id, []):
            if pred_id not in df.index:
                continue
            pred_ef   = df.loc[pred_id, 'EF']
            curr_es   = df.loc[task_id, 'ES']
            pred_es   = df.loc[pred_id, 'ES']
            dur       = df.loc[task_id, 'EF'] - df.loc[task_id, 'ES']
            satisfied = False
            if dep_type == 'FS' and np.isclose(curr_es,         pred_ef + lag, atol = 1e-4):
                satisfied = True
            elif dep_type == 'SS' and np.isclose(curr_es,       pred_es + lag, atol = 1e-4):
                satisfied = True
            elif dep_type == 'FF' and np.isclose(curr_es + dur, pred_ef + lag, atol = 1e-4):
                satisfied = True
            elif dep_type == 'SF' and np.isclose(curr_es + dur, pred_es + lag, atol = 1e-4):
                satisfied = True
            if satisfied and pred_id not in critical_set:
                critical_set.add(pred_id)
                trace_critical(pred_id)
                
    for end_task in list(critical_set):
        trace_critical(end_task)
    df.loc[list(critical_set), 'Slack'] = 0.0
    return df

# Function: Pert with Dependencies and Lags
def pert_method_dep(dataset):
    ids           = [item[0] for item in dataset]
    idd           = dict(zip(ids, range(len(dataset))))
    n             = len(dataset)
    means         = [(o + 4*m + p)/6 for (_, _, o, m, p) in dataset]
    vars_         = [((p - o)/6)**2 for (_, _, o, m, p) in dataset]
    dates         = np.zeros((n, 5))  # ES, EF, LS, LF, Slack
    dates[:, -2:] = float('+inf')
    visited       = set()
    while len(visited) < n:
        for i, (task_id, preds, *_ ) in enumerate(dataset):
            if task_id in visited:
                continue
            if all(pred[0] in ids and pred[0] in visited for pred in preds):
                es_candidates = []
                for pred_id, dep_type, lag in preds:
                    j = idd[pred_id]
                    pred_es, pred_ef = dates[j, 0], dates[j, 1]
                    if dep_type == 'FS':
                        es_candidates.append(pred_ef + lag)
                    elif dep_type == 'SS':
                        es_candidates.append(pred_es + lag)
                    elif dep_type == 'FF':
                        es_candidates.append(pred_ef + lag - means[i])
                    elif dep_type == 'SF':
                        es_candidates.append(pred_es + lag - means[i])
                    else:
                        raise ValueError(f"Unsupported dependency type: {dep_type}")
                es          = max(es_candidates) if es_candidates else 0
                ef          = es + means[i]
                dates[i, 0] = es
                dates[i, 1] = ef
                visited.add(task_id)
    finish_time = np.max(dates[:, 1])
    visited     = set()
    while len(visited) < n:
        for i in reversed(range(n)):
            task_id, preds, *_ = dataset[i]
            if task_id in visited:
                continue
            successors = [ (succ_id, dep_type, lag, j) for j, (succ_id, succ_preds, *_ ) in enumerate(dataset) for (pred_id, dep_type, lag) in succ_preds if pred_id == task_id ]
            if all(succ[0] in visited or succ[0] not in ids for succ in successors):
                lf_candidates = []
                for succ_id, dep_type, lag, j in successors:
                    succ_ls, succ_lf = dates[j, 2], dates[j, 3]
                    if dep_type == 'FS':
                        lf_candidates.append(succ_ls - lag)
                    elif dep_type == 'SS':
                        lf_candidates.append(succ_ls - lag + means[i])
                    elif dep_type == 'FF':
                        lf_candidates.append(succ_lf - lag)
                    elif dep_type == 'SF':
                        lf_candidates.append(succ_lf - lag + means[i])
                    else:
                        raise ValueError(f"Unsupported dependency type: {dep_type}")
                lf          = min(lf_candidates) if lf_candidates else finish_time
                ls          = lf - means[i]
                dates[i, 3] = lf
                dates[i, 2] = ls
                dates[i, 4] = ls - dates[i, 0]
                visited.add(task_id)
    df               = pd.DataFrame(dates, index = ids, columns = ['ES', 'EF', 'LS', 'LF', 'Slack'])
    df               = df.round(4)
    predecessors_map = {task[0]: task[1] for task in dataset}
    critical_set     = set(df[np.isclose(df['EF'], finish_time, atol = 1e-4)].index)
    
    def trace_critical(task_id):
        for pred_id, dep_type, lag in predecessors_map.get(task_id, []):
            if pred_id not in df.index:
                continue
            pred_ef   = df.loc[pred_id, 'EF']
            curr_es   = df.loc[task_id, 'ES']
            pred_es   = df.loc[pred_id, 'ES']
            dur       = df.loc[task_id, 'EF'] - df.loc[task_id, 'ES']
            satisfied = False
            if dep_type == 'FS' and np.isclose(curr_es,         pred_ef + lag, atol = 1e-4):
                satisfied = True
            elif dep_type == 'SS' and np.isclose(curr_es,       pred_es + lag, atol = 1e-4):
                satisfied = True
            elif dep_type == 'FF' and np.isclose(curr_es + dur, pred_ef + lag, atol = 1e-4):
                satisfied = True
            elif dep_type == 'SF' and np.isclose(curr_es + dur, pred_es + lag, atol = 1e-4):
                satisfied = True
            if satisfied and pred_id not in critical_set:
                critical_set.add(pred_id)
                trace_critical(pred_id)
                
    for end_task in list(critical_set):
        trace_critical(end_task)
    df.loc[list(critical_set), 'Slack'] = 0.0
    std_dev = np.sqrt(sum([vars_[idd[task]] for task in critical_set]))
    return df, finish_time, round(std_dev, 4)

###############################################################################

