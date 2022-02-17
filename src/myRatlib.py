#!/usr/bin/env python
# coding: utf-8

# In[1]:


def clean_data(data):
    """
     Verifies data from hex maze session:
         * All times must be consecutive and descending
         * All trials must have the same target node
         * Target node must not appear in the middle of a trial
     Trials for which these conditions are not met is removed

     Parameters
     ----------
     data : df
         pandas dataframe with rat info from one day

    Returns
    -------
    pandas dataframe with suspicious or wrong trials removed

    """
    from statistics import mode
    import logging
    import pandas as pd
    import numpy as np

    
    # Initiallize lists with log info and log errors to write to logfile at the end of the function
    log_msg = []
    log_error = []
    
    log_msg.append('\n\n----------------------------------------------------------------------------\n')
    log_msg.append('Data verification and removal of bad trials\n')
    log_msg.append('----------------------------------------------------------------------------\n')

    
    # Initiallize list of bad_trials and determine target node for this session
    bad_trials = []
    tgt_node = mode(data.groupby('trial_no')['node'].last())
        
    # Check the times are all consecutive and descending
    if ~(data['time'].sort_values().sort_index() == data['time'].sort_index()).all():
        error_msg = 'ERROR: Times are not monotonically descending\n'
        print(error_msg)
        log_error.append(error_msg)
    
    # Check consistency of derived values within each trial
    # Within each trial, 
    for tr in data['trial_no'].unique():
        this_trial = data.loc[data['trial_no'] == tr]
        # check 'speed_ff' = 'distance' / 'seconds_ff'
        speed = this_trial['distance'] / this_trial['seconds_ff']
        if ~(this_trial[1:]['speed_ff'] == round(speed[1:], 3)).all():
            error_msg = 'ERROR: Speed inconsistency found in trial '+ str(tr) +'\n'
            print(error_msg)
            log_error.append(error_msg)
            bad_trials.append(tr)

        # Check 'seconds_ff' = time[x] - time[x-1]
        seconds = (this_trial['time'] - this_trial['time'].shift(1)).dt.total_seconds()
        if ~ (np.isclose(seconds[1:], this_trial[1:]['seconds_ff'], atol = 0.001)).all():
            error_msg = 'ERROR: Time (seconds_ff) inconsistency found in trial '+ str(tr) +'\n'
            print(error_msg)
            log_error.append(error_msg)
            bad_trials.append(tr)
                  
    # Check that tgt node is the last node of every trial, and doesn't appear anywhere else
    for tr in data['trial_no'].unique():
        trial_nodes = data.loc[data['trial_no'] == tr, 'node']
        if trial_nodes.iloc[-1] != tgt_node:
            info_msg = 'ERROR: trial ' + str(tr) + ' does not end in target node ' + tgt_node + '\n'
            print(info_msg)
            log_msg.append(info_msg)
            bad_trials.append(tr)
        if (trial_nodes.iloc[:-2] == tgt_node).any():
            info_msg ='ERROR: trial ' + str(tr) + ' contains target node ' + tgt_node + ' in places other than last\n'
            print(info_msg)
            log_msg.append(info_msg)
            bad_trials.append(tr)

    # Remove bad trials from dataframe
    data_cl = data[~data['trial_no'].isin(bad_trials)]
    info_msg = '\nRemoved bad trials: ' + str(bad_trials) + '\n'
    print(info_msg)
    log_msg.append(info_msg)
    
    
    log_error_txt = ''.join(log_error)
    log_info_txt = ''.join(log_msg)
    logging.error(log_error_txt)
    logging.info(log_info_txt)
    
    return(data_cl)


# In[ ]:


def import_file(rat_filename, file_path = '../data/raw/'):
    """
     Reads Rat HexMaze behavioural data from experiment logs.
     Performs basic cleaning (removes duplicates)
     Re-organises in a pandas-friendly format, and returns pandas dataframe.

     Parameters
     ----------
     arg1 : str
         Filename of txt data with experimental logs for single rat
     arg2 : str, optional
         Path to where rat_filename is stored

    Returns
    -------
    pandas dataframe with clean data

    """
    import pandas as pd
    import re # Regular expressions
    import logging


    #https://docs.python.org/3/howto/logging.html

    # Extract exp. date and Rat Id from file name (format: Rat_HM_Ephys_RatX_ratcode_YYYYMMDD.txt)
    # Info in filename is separated by '_'. Split parts accordingly:
    filename_info = rat_filename.split('_')
    # Exp date is last element in filename, but contains also .txt which we remove by splitting again
    exp_date = pd.to_datetime(filename_info[-1].split('.')[0]).date()
    rat_no = filename_info[3]
    rat_id = filename_info[4]
    info_msg = 'Importing data from ' + rat_no + ' on ' + exp_date.strftime('%Y-%m-%d') + ' ('+ rat_filename + ')'
    print(info_msg + ':')
    
    log_msg = []
    
    log_msg.append('\n\n=========================================================================\n')
    log_msg.append(info_msg)
    log_msg.append('\n---------------------------------------------------------------------------\n')
    

    # Extract fileinfo to list_all
    with open(file_path + rat_filename, 'r') as f:
        # Load all lines from file
        list_all = [x for k,x in enumerate(f.readlines())]

    # Locate trial line boundaries and first trial line
    list_tr_bb = [n for n, x in enumerate(list_all) if re.match(r'^Summary Trial', x)]
    trial_headers = [x for n, x in enumerate(list_all) if re.match(r'^Summary Trial', x)]

    # Add a final entry to list_tr_bb to mark end boundary of last trial
    list_tr_bb.append(len(list_all))

    info_msg = str(len(list_tr_bb)) + ' trials found initially\n'
    print(info_msg)
    log_msg.append(info_msg)



    ### DATA CLENANING
    # Data files contain duplicate lines and/or duplicate trials
    # Find and remove them

    # Look for duplicate trials to write info to log
    dupe_trials = [x for n, x in enumerate(trial_headers) if x in trial_headers[:n]] # Get duplicate trial headers

    info_msg = str(len(dupe_trials)) + ' duplicate trials found\n'
    log_msg.append(info_msg)
    [log_msg.append('Trial ' + x.split()[2] + '\n') for x in dupe_trials]
    print(info_msg)
    [print('Trial ' + x.split()[2]) for x in dupe_trials]


    # Look for duplicate lines in general and remove them from list_all
    dupe_lines = [n for n, x in enumerate(list_all) if x in list_all[:n]]

    # Remove duplicate lines from list_all
    info_msg = str(len(dupe_lines)) + ' duplicate lines found\n'
    print(info_msg)
    log_msg.append(info_msg)

    for x in sorted(dupe_lines, reverse=True):
        del list_all[x]

    # Update trial boundaries after deletions
    list_tr_bb = [n for n, x in enumerate(list_all) if re.match(r'^Summary Trial', x)]
    list_tr_bb.append(len(list_all))

    # Strip extraneous strings from file lines, and read data within each line into list of lists of strings
    list_sess = []
    for l,tr in enumerate(list_tr_bb[:-1]): 
        # Find all lines starting with ( between trial boundaries
        list_sing = [x for x in list_all[list_tr_bb[l]:list_tr_bb[l+1]] if x[0]=='('] # Lines starting with ( 
        # Remove parenthesis, commas and '' from string
        list_sing = [x.replace("(", "") for x in list_sing]
        list_sing = [x.replace(")", "") for x in list_sing]
        list_sing = [x.replace(",", "") for x in list_sing]
        list_sing = [x.replace("'", "") for x in list_sing]
        list_sing = [x.replace("\n", "") for x in list_sing]
        # Now separate data in each line within the trial
        list_sing = [x.split(sep=' ') for x in list_sing]
        list_sess.append(list_sing)

    info_msg = 'Final number of trials loaded: ' + str(len(list_sess)) + '\n'
    log_msg.append(info_msg)
    print(info_msg)
    
    ## TRANSFORM TO 'LONG' FORMAT AND SAVE IN PANDAS DATA FRAME
    # Transform current 'wide' data format into 'long' format, and store as a pandas 
    # dataframe.
    # Edge trajectory data will be re-ordered into a successive list of nodes and times.
    # Additional data, such as seconds, distance, and speed refers to the trajectory from
    # node[i-1] to node[i], being zero for the inital row.

    # First re-order and store as single lists for each variable
    # Variables with _ff suffix  ("from file") can be computed from other primary variables in the file. 
    # Suffix is included in case these vars need to be re-computed 

    # List size: total no. of lines (across all trials
    # with one additional line per trial to flatten structure
    # also discount empty trials in case there are any
    size = sum([len(x) for x in list_sess]) + len(list_sess) - list_sess.count([])                                                # with one additional line per trial (25 trials in total) to flatten structure
    trial_no = [None]*size 
    distance = [None]*size 
    seconds_ff = [None]*size # seconds from file 
    nodes = [None]*size #np.zeros(size, dtype=int) # nodes the rat passes through (flat)
    speed_ff = [None]*size #np.zeros(size)
    times = [None]*size #np.zeros(shape) # all timestamps (flat)
    rat_no_col = [rat_no]*size
    rat_id_col = [rat_id]*size
    exp_date_col = [exp_date]*size


    line_no = 0 # initialize counter

    for tr, trial in enumerate(list_sess):
        # First check if the trial is empty (An empty trial was found for Rat8 on 20210618)
        if(trial) == []:
            info_msg = '***Trial ' + str(tr+1) + ' is empty ****\n'
            log_msg.append(info_msg)
            continue

        # Add inital trial line for nodes and timestamps array with starting node (first value of first line in the trial)
        trial_no[line_no] = tr+1
        nodes[line_no] = trial[0][0] # First value from first row in that trial: start node
        times[line_no] = trial[0][2] # Third value from first row in that trial: start time
        # times[:, cum_line_no] = [float(x) for x in time_list]
        # Initialize distance and seconds
        distance[line_no] = 0
        seconds_ff[line_no] = 0
        speed_ff[line_no] = 0
        line_no +=1

        for row in trial:

            trial_no[line_no] = tr+1
            nodes[line_no] = row[1]
            times[line_no] = row[3]
            seconds_ff[line_no] = row[4]
            distance[line_no] = row[5]
            speed_ff[line_no] = row[6]     
            line_no += 1


    ## BUILD PD DATA FRAME

    data = pd.DataFrame(list(zip(rat_id_col, rat_no_col, exp_date_col, trial_no, nodes, times, distance, seconds_ff, speed_ff)),
                       columns = ['rat_id', 'rat_no', 'date', 'trial_no', 'node', 'time', 'distance', \
                                  'seconds_ff', 'speed_ff'])
    # Set appropriate data types for each column. Times data set to timedelta object,
    # although maybe it would better be just a time? (day data is not relevant in our case)
    data["distance"] = data["distance"].astype('float')
    data["seconds_ff"] = data["seconds_ff"].astype('float')
    data["speed_ff"] = data["speed_ff"].astype('float')
    data["time"] = pd.to_timedelta(data["time"])

    
    log_txt = ''.join(log_msg)
    logging.info(log_txt)

    return data


# In[ ]:


# def enrich_data(data):
#     """
#      Adds useful columns for posterior rat analysis:
#          * Within-trial cumulative seconds
#          * Within-trial cumulative speed
#          * Island name at each step
#          * Boolean u_turn column
#          * Boolean re-visted column
#          * Shortest distance to target at each step
#          * Actual distance to target according to path taken by rat

#      Parameters
#      ----------
#      data : df
#          pandas dataframe with rat info from one day

#     Returns
#     -------
#     pandas dataframe with additional columns

#     """
#     import pandas as pd
#     import numpy as np
#     import networkx as nx # Package for graph represenations 

    
#     # Set trial number as index for easy selecting trials
#     data.set_index('trial_no', inplace=True)

#     # Set appropriate data types for each column. Times data set to timedelta object,
#     # although maybe it would better be just a time? (day data is not relevant in our case)
#     data["distance"] = data["distance"].astype('float')
#     data["seconds_ff"] = data["seconds_ff"].astype('float')
#     data["speed_ff"] = data["speed_ff"].astype('float')
#     data["times"] = pd.to_timedelta(data["times_ff"])


#     # Add column with cumulative distance and cumulative seconds
#     data["cum_distance"] = ""
#     data["cum_seconds"] = ""
#     for tr in range(1, len(list_sess)):
#         data.loc[tr, "cum_distance"] = data.loc[tr, "distance"].cumsum()
#         data.loc[tr, "cum_seconds"] = data.loc[tr, "seconds_ff"].cumsum()

#     # Add column with node classification    
#     data["island"] = ""
#     data.loc[data["node"].astype('int') // 100 == 1, "island"] = 'Ireland'
#     data.loc[data["node"].astype('int') // 100 == 2, "island"] = 'Japan'
#     data.loc[data["node"].astype('int') // 100 == 3, "island"] = 'Hawaii'
#     data.loc[data["node"].astype('int') // 100 == 4, "island"] = 'Easter I.'

#     # Detect u_turns: moments when current node is the same as 2 nodes ago.
#     # Store as a boolean column to the main datafrmae
#     data['u_turn'] = (data['node'] == data['nodes'].shift(2))

#     data['re-visited'] = False

#     for tr in data.index.unique():
#         dup_nodes = data.loc[tr, 'node'].duplicated()
#         data.loc[tr, 're-visited'] = dup_nodes

#     data[data['re-visited'] == True] \
#                 [['node', 'speed_ff']]

#     data['act_stps_2trgt'] = np.nan
#     data['min_stps_2trgt'] = np.nan

#     ## BUILD A MAZE REPRESENTATION: (maybe this should be a different function)
#     # Load hexMaze graph and compute shortest paths

#     print("load graph")
#     G = nx.read_edgelist(edgelist_filename)  
#     print("graph: number of nodes = ",G.number_of_nodes(),", edges = ",G.number_of_edges())

#     ### pre-compute all possible shortest path lengths and save in dictionary (does not save shortest paths itself)
#     DD = nx.shortest_path_length(G) 
#     DD = dict(DD)
#     print("dictionary: len =", len(DD), ", total items =", sum([len(dv) for dv in DD.values()]))

#     for tr in data.index.unique():
#         # Number of steps taken by rat in the trial (trajectory lenght):
#         traj_len = len(data.loc[tr, 'node'])

#         # Number of steps still left until rat reaches target, at each node passed:
#         act_stps_2trgt = [traj_len-n for n in range(1, traj_len+1)]

#         # Target node
#         tgt_node = data.loc[tr, 'node'].iloc[-1]
#         # Minimum number of steps until target at each node passed by rat
#         min_stps_2trgt = [DD[str(a_node)][tgt_node] for a_node in data.loc[tr, 'nodes']]

#         data.loc[tr, 'act_stps_2trgt'] = act_stps_2trgt
#         data.loc[tr, 'min_stps_2trgt'] = min_stps_2trgt

