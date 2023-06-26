
def load_RPDR_labs_multiple(dir_data, path_labs, path_synonyms, datetime_col='Seq_Date_Time', result_col='Result', test_col='Group_Id', delim='|', clean_columns=True):
    ''' load_labs_multiple(dir_data, path_labs, path_synonyms, datetime_col='Seq_Date_Time', result_col='Result', test_col='Group_Id', delim='|', clean_columns=True):
        Sequentially loads all files from RPDR data dump when multiple files have the same name. 
        
        1. Starts in dir_data (should have trailing slash), grabs all sub-folders' names automatically, then sequentially loads: dir_data/[sub-folders]/path_labs (where path_labs is the name of the file)
        * Note for whatever reason, on a multiple-split file dump from RPDR the labs, demographics, etc files are all named the exact same, just in different zips
        2. Calls the traditional load LFTs function on each file
        3. Concatenates all results and returns 1 DF (may be VERY large. ~3-4 gb / 80,000 patients)
        
        See load_native_data for remainder of parameters which are passed to that function
        
        '''
    
    import warnings
    warnings.warn("lftlib.load_RPDR_labs_multiple is deprecated; use labs.load_RPDR_labs_multiple")
    
    import os
    import pandas as pd
    
    # get list of subdirectories
    subdirectories = [x[0] for x in os.walk(dir_data)][1:]
    
    first=True
    # for each subdir, use the traditional load function to load data and concat
    for subdir in subdirectories:
        path_to_labs=subdir+'/'+path_labs
        lfts, discarded = load_RPDR_labs(path=path_to_labs,
                                           path_synonyms=path_synonyms,
                                           datetime_col=datetime_col,
                                           result_col=result_col,
                                           test_col=test_col, 
                                           delim=delim,
                                           clean_columns=clean_columns)
        
        if first==True:
            concat_pd = lfts
            first=False
        else:
            concat_pd=pd.concat([concat_pd, lfts],ignore_index=True)
    
    return concat_pd

def load_RPDR_labs(path,path_synonyms,datetime_col='Seq_Date_Time', result_col='Result', test_col='Group_Id', delim='|', clean_columns=True):
    '''load_RPDR_labs(path,path_synonyms,datetime_col='Seq_Date_Time', result_col='Result', test_col='Group_Id', delim='|', clean_columns=True):
    
    DESC: This is the main labs loading function, for loading labs data from an RPDR data pull, and processing lab results. 
     Does MANY THINGS. 
     1. Homogenizes column names:
         rename result_col to 'result' if not already named result
         rename datetime_col to 'datetime' if not already named so; convert this to pd.DateTime format
         rename test_col to 'test_desc'
    path: path to labs file (in .csv or other text delimited file)
    path_synonyms: path to synonyms file (.csv). structure of synonyms file: first row is what you want to call each lab 
     defined in the test_col column; remainder of rows identify the exact text of labs that should be considered equivalent. 
     All of these will be homogenized to the row 1 name
    datetime_col: name of column with date/time of lab test
    result_col: name of column that contains the result of the lab test
    test_col: name of the column that contains the name of the lab test (this is what synonyms acts upon)
    delim: delimiter of the labs file (the file pointed to by path)
    clean_columns: bool. If true, removes all columns except those named the following:
     [['EMPI', 'test_desc', 'datetime', 'result_orig', 'result', 'MRN_Type', 'MRN', 'Test_Description', 'Result_Text', 'above_below_assay_limit']]
    
    RETURNS: lfts, discarded_rows
     lfts is the labs as panda database (should really be named labs, but initially developed for only liver function tests (LFTs))
     discarded_rows is a second pandas database containing anything removed by the processing
     
    Note: input result column will be returned as 'result_orig' and the processed result column will be returned as 'result'
    '''
    import warnings
    warnings.warn("lftlib.load_RPDR_labs is deprecated; use labs.load_RPDR_labs")
    
    # read data
    # 
    # WARNINGS:
    # - default behavior is to remove < and >; could add a new column to mark these events if pertinent later
    import pandas as pd
    import re
    
    # read the data
    print('Loading data from ' + path)
    lfts = pd.read_csv(path,delimiter=delim, error_bad_lines=False)
    
#     if result_col != 'result':
#         lfts['result'] = lfts.loc[:,result_col]
#     else:
#         # result col is already named result. rename it
#         lfts['result_orig'] = lfts['result']

    ## Mod 1 ##
    if result_col in lfts.columns:
        lfts.rename(columns={result_col: 'result_orig'}, inplace=True)
        lfts['result'] = lfts.loc[:,'result_orig']
    else:
        raise ValueError('Incorrect name specified for result column')
    ## Mod 1 ##
    
    # convert datetime column to datetime format
    if datetime_col != 'datetime':
        lfts['datetime'] = pd.to_datetime(lfts.loc[:,datetime_col])
        lfts.drop(labels=datetime_col, axis=1, inplace=True)
    else:
        lfts['datetime'] = pd.to_datetime(lfts['datetime'])
    
    if test_col != 'test_desc':
        lfts['test_desc'] = lfts.loc[:,test_col]
    else:
        # test_desc col is already named test_desc. rename it
        lfts['test_desc_orig'] = lfts['test_desc']
    
    ######################
    # Identify synonyms of labs and homogenize based on a file called lab_synonyms.csv
    # structure: first row is what you want to call each lab; below the col are all the synonyms to change to the header name
    syn=pd.read_csv(path_synonyms)
    correct_lab_names=syn.columns.tolist()
    for correct_lab in correct_lab_names:
        alternate_names_to_replace=syn[correct_lab].tolist()
        for alt in alternate_names_to_replace:
            # irregular length columns, if this column has nan's at the end, don't cycle through them
            if isinstance(alt,str):
                lfts.test_desc.replace(alt,correct_lab,regex=False, inplace=True)
            
    # error checking
    if set(correct_lab_names) == set(lfts.test_desc.unique().tolist()):
        print('Successful homogenization of lab names')
        print(correct_lab_names)
    else:
        got_list=set(lfts.test_desc.unique().tolist())
        expected_list=set(correct_lab_names)
        print('FAILED TO HOMOGENIZE NAMES')
        print('Expected : ')
        print(expected_list)
        print('Of these, got only : ')
        print(got_list & expected_list)
        print('Got additionally : ')
        print(got_list-expected_list)
        print('...and we are missing : ')
        print(expected_list-got_list)
    
    ######################
    # CERTIFY INDIVIDUAL RESULTS THAT REQUIRE SPECIAL HANDLING
    # NUMERICS
    list_numeric_labs = ['ALT','AST','AKP','DBILI','TBILI', 'CR','INR','IGG']

    lfts['above_below_assay_limit'] = 0

    fil = lfts.test_desc.isin(list_numeric_labs)

    # def upper and lower bound finder
    def upper_lower_bound_finder(row):
        if not isinstance(row.result, float):
            if '<' in row.result:
                return -1
            elif '>' in row.result:
                return 1
            else:
                return 0

    # def upper and lower bound finder
    def remove_gg_ll_signs(row):
        if isinstance(row, str):
            if '<' in row:
                return row.replace('<','')
            elif '>' in row:
                return row.replace('>','')
            else:
                return row

    # first mark whether a result is at the upper or lower limit (otherwise NaN)
    lfts.loc[fil, 'above_below_assay_limit'] = lfts.loc[fil, ['result', 'above_below_assay_limit']].apply(upper_lower_bound_finder, axis=1)

    # remove greater than and less than signs
    lfts.loc[fil, 'result'] = lfts.loc[fil, 'result'].apply(remove_gg_ll_signs)
    
    # now that we've removed greater than and less than signs, data should be numeric. Anything non-numeric (in this category) is crap, 
    # e.g., "REFUSED", "HEMOLYZED" etc
    
    # filter for coercing data to numeric, catching NaNs (text)
    print('Removing non-numeric result values...')
    lfts.loc[fil, 'result'] = pd.to_numeric(lfts.loc[fil, 'result'], errors='coerce')
    # hold on to removed rows -- error checking
    removed_rows=lfts[(fil) & lfts.result.isnull()].copy()
    # remove these rows
    lfts.drop(lfts[(fil) & lfts.result.isnull()].index, inplace=True)
    
    
    ######################
    # CERTIFY INDIVIDUAL RESULTS THAT REQUIRE SPECIAL HANDLING
    # ANTIBODY TITERS
    
    list_titer_labs = ['LKM','ASMA', 'ANA', 'AMA', 'SLA']

    fil_ab = lfts.test_desc.isin(list_titer_labs)
    
    if not lfts[fil_ab].empty:

        def titer_interp(row):
            import math
            import re

            bool_switched_result_resultext=False

            if isinstance(row.result, float):
                if row.result < 20: # antibody titer cutoff for LKM and works for ANA, ASMA
                    return 0
                elif math.isnan(row.result):
                    if isinstance(row.Result_Text,str):
                        this_string=row.Result_Text
                        bool_switched_result_resultext=True
                    else:
                        return row.result
                else:
                    return row.result

            # function I'm going to use a bunch of times:
            def all_numbers_within_n_chars_of_word(text, word, n):
                idx_zero_pos_of_word=text.lower().find(word)
                if len(text[idx_zero_pos_of_word:]) < n:
                    sub_string_to_return=text[idx_zero_pos_of_word:]
                else:
                    sub_string_to_return=text[idx_zero_pos_of_word:idx_zero_pos_of_word+n]

                return find_all_num_in_string(sub_string_to_return)

            # CASES AT THIS POINT. Either this_string holds a swapped text or doesn't. Any string in result hasn't been accounted for yet
            # most of the logic will be shared, nevertheless keep them separate because, e.g., a single number in result is usually the value we want
            #  while a single number in result text might not be

            if isinstance(row.result, str):
                numbers = find_all_num_in_string(row.result)

                if '<' in row.result:
                    return 0
                elif 'ANA SCREEN POSITIVE, TITRE PERFORMED' in row.result or 'POSITIVE - SEE TITER' in row.result:
                    # this category will get removed; there's always a titer separately reported at same time point
                    return 'see titer'
                
                    
                ## FIRST IDENTIFY POSITIVELY OR NEGATIVELY IDENTIFIED EXAMPLES
                elif 'negative' in row.result.lower() and not 'positive' in row.result.lower():
                    return 0
                elif 'positive' in row.result.lower() and not 'negative' in row.result.lower():
                    
                    if len(numbers) == 1:
                        return numbers[0]
                    elif len(numbers) == 2 and numbers[0]==1:
                        return numbers[1]
                    
                    elif len(numbers) == 4 and numbers[2]==1:
                        return numbers[3]
                    
                    else: 
                        result_text=row.Result_Text
                        if isinstance(result_text,str):
                            
                            ## Some cases doenst contain the word 'positive' in Result_text which is generating a random incorrect result
                            if 'positive' in result_text.lower():
                                numbers=all_numbers_within_n_chars_of_word(result_text, 'positive', n=26)
                                if len(numbers) == 1:
                                    return numbers[0]
                                elif len(numbers) == 2 and numbers[0]==1:
                                    return numbers[1]
                                elif len(numbers) > 2:
                                    this_string=row.Result_Text.lower()
                                    if isinstance(this_string,str):
                                        idx_titer=this_string.find('titer')
                                        if len(row.Result_Text[idx_titer:]) <= 17:
                                            search_text=row.Result_Text[idx_titer:]
                                        else:
                                            search_text=row.Result_Text[idx_titer:idx_titer+17]
                                        numbers=find_all_num_in_string(search_text)
                                        if len(numbers) == 1:
                                            return numbers[0]
                                        elif len(numbers) == 2 and numbers[0]==1:
                                            return numbers[1]
                            else:
                                return 20

                elif 'positive' in row.result.lower() and 'negative' in row.result.lower():
                    # both pos and neg present; find the text following the 'positive' word (26 chars)
                    numbers=all_numbers_within_n_chars_of_word(row.result, 'positive', n=26)
                    if len(numbers) == 1:
                        return numbers[0]
                    elif len(numbers) == 2 and numbers[0]==1:
                        return numbers[1]
                    else:
                        # possible it's farther out?
                        text=row.result[index_positive:]
                        numbers=find_all_num_in_string(text)
                        print('pos & neg text present but no value within 26 chars; went out on a limb and returning : ' + str(numbers[1]))
                        return numbers[1]

                # okay, so the words 'positive' and 'negative' aren't present
                elif len(numbers) == 1:
                    return numbers[0]
                elif len(numbers) == 2 and numbers[0]==1:
                    # pair of numbers returned; make sure the first value is 1
                    return numbers[1]

            # CASES AT THIS POINT. Either this_string holds a swapped text or doesn't. Any interpretable string has been returned

            if bool_switched_result_resultext:
                numbers = find_all_num_in_string(this_string)

                ## FIRST IDENTIFY POSITIVELY OR NEGATIVELY IDENTIFIED EXAMPLES
                if 'negative' in this_string.lower() and not 'positive' in this_string.lower():
                    return 0
                elif 'positive' in this_string.lower() and not 'negative' in this_string.lower():
                    ## Positive at logic is not working when there are 4 numbers. Removed the if statement
                    ## Example: Positive at 1:40 and 1:160 (endpoint)
                    ## Finding all numbers within a close proximity of 'positive' seems to be working for all cases
                    numbers=all_numbers_within_n_chars_of_word(this_string, 'positive', n=26)
                    if len(numbers) == 1:
                        return numbers[0]
                    elif len(numbers) == 2 and numbers[0]==1:
                        return numbers[1]
                    elif len(numbers) == 4 and numbers[2]==1:
                        return numbers[3]
                    elif len(numbers)>0:
                        # first try to match the expression 'positive at 1:x'
                        m = re.search('positive at 1:(\d+)', this_string.lower())
                        if m:
                            return m.group(1)
                        else:
                            # no 'positive at' expressions.. get all the numbers and return the largest one, and print
                            #  result text because this is a weird case..
                            max_num = max(numbers)
                            print('positive only in result text but neither 1 nor 2 numbers, going on limb and taking : ' + str(max_num))
                            print(this_string)
                            return max_num
                    
                elif 'positive' in this_string.lower() and 'negative' in this_string.lower():
                    # both pos and neg present; find the text following the 'positive' word (15 chars)
                    index_positive=this_string.lower().find('positive')
                    if len(this_string[index_positive:]) <= 26:
                        text=this_string[index_positive:]
                    else:
                        text=this_string[index_positive:index_positive+26]
                    numbers=find_all_num_in_string(text)
        #             print(text)
        #             print(numbers)
                    if len(numbers) == 1:
                        return numbers[0]
                    elif len(numbers) == 2 and numbers[0]==1:
                        return numbers[1]
                    elif len(numbers) == 4:
                        return max(numbers)
                    elif 'negative at' in this_string.lower():
                        return 0
                # okay, so the words 'positive' and 'negative' aren't present
                elif len(numbers) == 1:
                    return numbers[0]
                elif len(numbers) == 2 and numbers[0]==1:
                    return numbers[1]


            return row.result

        # first mark whether a result is at the upper or lower limit (otherwise NaN)
        lfts.loc[fil_ab, 'result'] = lfts.loc[fil_ab, ['result', 'Result_Text']].apply(titer_interp, axis=1)

        print('Cleaning up antibody titer results...')
        lfts.loc[fil_ab, 'result'] = pd.to_numeric(lfts.loc[fil_ab, 'result'], errors='coerce')
        # hold on to removed rows -- error checking
        removed_rows_ab=lfts[(fil_ab) & lfts.result.isnull()].copy()
        # remove these reows
        lfts.drop(lfts[(fil_ab) & lfts.result.isnull()].index, inplace=True)
    
        ######################
        # COMBINE DISCARDED DATA
        removed_rows = pd.concat([removed_rows,removed_rows_ab],axis=0)
    
    ######################
    # REMOVE UNUSED/DISTRACTING/PRECURSOR COLUMNS
    if clean_columns:
        # easier to define as what we want to keep
        lfts = lfts[['EMPI', 'test_desc', 'datetime', 'result_orig', 'result', 'MRN_Type', 'MRN', 'Test_Description', 'Result_Text', 'above_below_assay_limit']].copy()
        
#     from IPython import embed; embed()
        
    # enforce the EMPI column is strings for later
    lfts['EMPI'] = lfts.EMPI.astype(str)
        
    print('...Done')
    
    return lfts, removed_rows



def find_all_num_in_string(sentence):
    # accepts a sentence; returns an array of all numbers in the sentence
    import re
    
    import warnings
    warnings.warn("lftlib.find_all_num_in_string is deprecated; use labs.find_all_num_in_string")
    
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', sentence)]
    return s

#################################################
## lftlib starts here. DELETE ABOVE
#################################################

def get_table_traj(table,empi):
    filter_pt = (table.EMPI == empi)
    this_traj = table[filter_pt].sort_values(by='datetime').reset_index(drop=True).copy()
    
    return this_traj

def get_traj(table, empi, lab='ALT'):
    '''get_traj(table, empi, lab='ALT')
    DESC: accepts the input table and a particular unique identifier EMPI and returns a table filtered for that EMPI for the lab requested
    table: the input data table containing many patients (many EMPIs)
    empi: the empi requested
    lab: the lab requested, options are 'ALT','AST', 'INR'; it directly filters on this expression so works for arbitrary lab type
    
    WARNINGS:
    - AST and ALT have been made uniform (e.g., SGPT=>ALT, Alt=>ALT etc), but not any other values yet; need to preprocess and homogenize tbili, inr etc
    '''
    # accepts a data table of lft trajectories and returns a dict containing dataframes for each separate peak
    filter_pt = (table.EMPI == empi) & (table.test_desc == lab)
    this_traj = table[filter_pt].sort_values(by='datetime').reset_index(drop=True).copy()
    
    xlab = this_traj.result.astype(float).to_numpy()
    tseries=to_np_timeseries(this_traj)
        
    return xlab, tseries

def get_peaks(x,t,d=[], dates_dt=28, opt_prominence=25, plot_peaks=True, min_height=None, min_datapoints=0, allow_edge_peaks=False, options='peak_to_min', strict_monotonic=True, distance=None):
    '''get_peaks(x,t,d=[], dates_dt=28, opt_prominence=25, plot_peaks=True, min_height=None, min_datapoints=0, allow_edge_peaks=False, options='peak_to_min', strict_monotonic=True)
    
    Parameters:
    x-ALT or AST data, np array
    t-matched t trajectory for alt/ast data
    d-(optional) date list in epoch UTC time, same as t. If d is not empty, only return peaks within dates_dt of passed dates
    opt_prominence - parameter to be passed to scipy.signal.find_peaks()
    plot_peaks - bool
    min_height - param to be passed to scipy.signal.find_peaks()
    min_datapoints - number of datapoints per interval to be included
    allow_edge_peaks bool, false by default
    options='peak_to_min' (| 'peak_to_peak' | 'min_to_min')
    strict_monotonic=True (allow any upticks?)
    
    returns peaks_final, n_peaks, LL_vec, UL_vec
    
    WARNINGS: min_to_min probably does not work exactly right. Instead, get min_to_min from peak_min
    
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    ####################
    # First, identify the peaks
    ####################
    
    if len(x) == 0:
        print('get_peaks(): The x vector passed is empty, no peaks to assign')
        # empty - don't return peaks
        return None, 0, None, None
    
    if options=='min_to_min':
        # some restrictions if we are going min_to_min on the options
        if strict_monotonic==True:
            print("strict_monotonic must be False to get min-to-min behavior")
            return None, 0, None, None
        if min_datapoints != 0:
            print("min_to_min should have a 0 datapoint minimum or weird behavior")
            return None, 0, None, None
    
    # get the peaks
    peaks, trash = find_peaks(x, height=[min_height, None], prominence=[opt_prominence,None], distance=distance)
    
    # are we allowed to call the very first or very last value a peak? if so, algo is: add the lowest value to beginning and end
    #  if that makes a peak at beginning or end, update the peaks indices. drop the bases found this way (not relevant)
    if allow_edge_peaks:
        # add lowest value to beginning and end of signal
        x_temp = np.concatenate(([min(x)],x,[min(x)]))
        # stores peak indices but not bases; ie don't overwrite peak_dict from above
        peaks, trash = find_peaks(x_temp, height=[min_height, None], prominence=[opt_prominence,None], distance=distance)
        # peak location offset by one since we added one to beginning; subtract one to fix
        peaks=peaks-1
    
    n_peaks=len(peaks)
    
    # make sure there are peaks; if not, return None
    if n_peaks == 0:
        return None, 0, None, None
    
#     from IPython import embed; embed()
    
    # OKAY, there are peaks. If we want specific dates, apply that filter now
    # pseudocode. For each peak, check whether within dates_dt for each date
    if d:
        peaks_dates=np.array([])
        # if d just ensures d is not empty, if d empty move on with all peaks..
        for k in range(0, n_peaks):
            current_peak=peaks[k]
            current_t=t[current_peak]
            has_date_match=False
            
            for date_UTC in d:
                if date_UTC-dates_dt <= current_t and date_UTC+dates_dt >= current_t:
                    has_date_match=True
            if has_date_match:
                peaks_dates=np.append(peaks_dates, current_peak)
        
        # we could be back to zero peaks
        peaks=peaks_dates.astype(int)
        
        n_peaks=len(peaks)
        
        # make sure there are peaks; if not, return None
        if n_peaks == 0:
            print('There were peaks, but none within dt_dates')
            return None, 0, None, None
    
    ####################
    # Now identify the intervals.
    ## 'peak_to_min' behavior is LL=peak, UL=min
    ####################
    
    peaks_final=[]
    LL_vec=[]
    UL_vec=[]
              
    if options=='peak_to_min' or options=='peak_to_peak' or options=='min_to_min':
        for j in range(0,n_peaks):
            # is there another peak or is this the last one?
            if j==n_peaks-1:
                # this is last one; UL is index of last element of x
                UL=len(x)
            else:
                # use the next peak as the initial end
                UL=peaks[j+1]
            # first pass: all values between this peak and the next:
            LL = peaks[j]
            # subset xalt to this region between peaks
            xalt_sub=x[LL:UL]
            
            if options=='peak_to_min' or options=='min_to_min':
                # find min on this region (second half); 
                min_idx_sub = np.argmin(xalt_sub)
                # to get the global index w.r.t xalt (rather than xalt_sub) add the LL
                min_idx = LL + min_idx_sub
                # truncate again to this min value (inclusive, thus the +1); update UL
                UL=min_idx+1
                
            if strict_monotonic:
                # peak finder finds all of : peak > min_height AND prominence > input_prominence
                # - unfortunately this means a peak of 26 (less than min_height) over a baseline of 12 won't get called
                # this code enforces a strict monotonic decline
                continue_loop=True
                n=1
                new_x = np.array([x[LL]])
                while continue_loop==True and n+LL<=UL:
                    if x[LL+n] < x[LL+n-1]:
                        new_x = np.append(new_x,x[LL+n])
                        
                    else:
                        continue_loop=False
                    n=n+1
                UL=LL+n-1
                
            # accumulate if sufficient number of datapoints
            num_datapoints=len(x[LL:UL])
            if num_datapoints >= min_datapoints:
                peaks_final.append(peaks[j])
                LL_vec.append(LL)
                UL_vec.append(UL)
    
    if options=='min_to_min':
    # here's the trick. min to min can be extracted from the ENDS of a peak to min. Then we just have to deal with the left edge and right edge
        LL_vec=np.array(LL_vec)
        # contains all the mins
        UL_vec=np.array(UL_vec)
        
        new_LL_1=np.array([0])
        new_LL=np.append(new_LL_1,UL_vec[0:-1]-1)
        
        LL_vec=new_LL
        
        if UL_vec[-1] != len(t):
            # there is a minimum value before the end; add one more segment
            LL_vec=np.append(LL_vec,np.array([UL_vec[-1]-1]))
            UL_vec=np.append(UL_vec,np.array([len(t)]))
        
        peaks_vec=np.array(peaks_final)
        
    elif options=="peak_to_min":
        LL_vec=np.array(LL_vec)
        UL_vec=np.array(UL_vec)
        peaks_vec=np.array(peaks_final)
    
    if plot_peaks:
        plt.plot(t,x)
        plt.plot(t[peaks_final], x[peaks_final],'x')
        for i in range(0,len(LL_vec)):
            t_LL=t[LL_vec[i]]
            t_UL=t[UL_vec[i]-1]
            plt.axvspan(t_LL, t_UL, facecolor='grey', alpha=0.5)
        plt.show()
    
    return peaks_final, len(peaks_final), LL_vec, UL_vec



def plot_lfttraj(tseries,x,peaks=None, bases=None):
    import matplotlib.pyplot as plt
    
    plt.plot(tseries,x)
    
    if not peaks == None:
        plt.plot(tseries[peaks], x[peaks],'x')
    if not bases == None:
        plt.plot(tseries[peaks], x[bases],'ro')
    
    plt.show()
    
    return None

## THESE DATETIME FUNCTIONS ARE ALL OVER THE PLACE. 
# - not timezone aware
# - should probably use built in pandas functions, don't need to calculate it myself..
# - epoch_days_to_date is somewhat modernized, still no timezone awareness

def to_np_timeseries(df, dtype='df', datetime_col='datetime'):
    
    import pandas as pd
    
    if dtype=='df':
        #native behavior for backward compatibility; assumes the column is datetime
        epoch_time=pd.Timestamp('1970-01-01 00:00:00')
        t_dseries = df.apply(lambda x: (x[datetime_col]-epoch_time).total_seconds(),axis=1)
        t_days = t_dseries.to_numpy()/(3600*24)
        
    elif dtype=='ds':
        epoch_time=pd.Timestamp('1970-01-01 00:00:00')
        t_dseries = df.apply(lambda x: (x-epoch_time).total_seconds())
        t_days = t_dseries.to_numpy()/(3600*24)
    
    return t_days

def epoch_days_to_date(epoch_day):
    ''' converts UTC (time since epoch) in units of days since epoch to a datetime timestamp
    if input is scalar, returns scalar
    if input is np.array, returns datetime array
    warnings:
    - not timezone aware
    '''
    import numpy as np
    from datetime import datetime
    import pandas as pd
    
    if np.isscalar(epoch_day):
        epoch_seconds=int(epoch_day*(3600*24))
        out = datetime.utcfromtimestamp(epoch_seconds).strftime('%Y-%m-%d %H:%M:%S')
    else:
        epoch_seconds = np.rint(epoch_day*(3600*24))
        out = pd.to_datetime(epoch_seconds, unit='s')
    
    return out

def epoch_date_to_days(pd_datetime):
    from datetime import datetime
    import pandas as pd
    epoch_time=pd.Timestamp('1970-01-01 00:00:00')
    epoch_seconds = (pd_datetime-epoch_time).total_seconds()
    
    epoch_days=epoch_seconds/(3600*24)
    
    return epoch_days

def rect_trapz(t,x,delay=3):
    '''integration with rectangular integration when t(i)-t(i-1) > delay '''
    import scipy
    
    if len(t)==1:
        # only one element
        return 0
        
    cum_aoc = 0
    for i in range(0,len(x)-1):
        tau = t[i+1]-t[i]
        if tau > delay:
            # rectangular integration
            cum_aoc = cum_aoc+(tau*x[i])
        else:
            # trap integration
            cum_aoc = cum_aoc + scipy.trapz(x[i:i+1],x=t[i:i+1])
            
    return cum_aoc
            
def running_mean(x, N):
    import numpy as np
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def discrete_autocorr(x,h=1):
    # reference:
    # 1.Kuzmič, P., Lorenz, T. & Reinstein, J. Analysis of residuals from enzyme kinetic and protein folding experiments 
    #  in the presence of correlated experimental noise. Analytical Biochemistry 395, 1–7 (2009).
    import numpy as np
    
    # mean
    xbar=x.mean()
    n_d=len(x)
    
    xh=x[h:]
    C_h=(1/n_d)*np.sum((x[:-1]-xbar)*(xh-xbar))
    C_0=(1/n_d)*np.sum((x-xbar)**2)    
    
    R_h=C_h/C_0
    
    z1_alpha2=None
    
    if h==1:
        # calculate whether the residual autocorrelation could occur by random chance given the null hypothesis z=0
        z1_alpha2=R_h*np.sqrt(n_d)
    
    return R_h, z1_alpha2

def qualitative_error(t, x, residuals):
    # I am making this up
    import numpy as np
    
    model_vals = x+residuals
       
    window_range_min = np.min(np.append(model_vals,x))
    window_range_max = np.max(np.append(model_vals,x))
    window_range = window_range_max-window_range_min
    
    qual_err = (1/len(x))*np.sum(np.absolute(residuals)/(window_range))
    
    return qual_err

def MAPE(t, x, residuals):
    import numpy as np
    
    MAPE = (1/len(x))*np.sum(np.absolute(residuals)/(x))
    
    return MAPE

def nRMSD(t, x, residuals):
    import numpy as np
    
    y_min = np.min(x)
    y_max = np.max(x)
    window_range = y_max-y_min
    
    nRMSD = (1/window_range)*np.sqrt(np.sum(residuals**2/len(x)))
    
    return nRMSD

def runs_of_signs(residuals):
    
    import numpy as np
    
    n_d=len(residuals)
    # convert to -1, 0, or 1
    signs=np.sign(residuals)
    # remove zeroes (they don't count)
    signs = signs[signs != 0]
    # compute the sum of the offest signs (ie, element 0 + element 1, 1+2. )
    signs_d = signs[0:-1]+signs[1:]
    # if there is a crossover, or change of signs, -1 + 1 = 0. Count these events. Add one (first run always counts, but no 'crossover' for first event)
    n_runs = len(signs_d[signs_d==0])+1
    n_pos = len(signs[signs==1])
    
    mu = 2*n_pos*(n_d-n_pos)/n_d + 1
    sig_squared = (mu-1)*(mu-2)/(n_d-1)
    
    z = (n_runs-mu+.5)/np.sqrt(sig_squared)
    
    return z

def on_interval(t1,t2,LL2,UL2,dt_LL=4,dt_UL=5):
    # accepts an already truncated interval of interest (t1) and asks for a matching interval among LL2[] UL2[] on t2
    # returns if there is an interval that starts and ends on a similar time scale
    # LL and UL are paired vectors for ranges
    
    import numpy as np
    
    t1_LL=t1[0]
    t1_UL=t1[-1]
    
    t2_LL=t2[LL2]
    t2_UL=t2[UL2-1]
    
    out_LL, out_idx_trash = find_nearest(t2_LL,t1_LL)
    idx_LL_closest = np.where(LL2==np.where(t2==out_LL)[0][0])[0][0]
    
    out_UL = t2[UL2[idx_LL_closest]-1]
    best_idx=-1
    
    # is the closest LL pretty close?
    if out_LL < t1_LL+dt_LL and out_LL > t1_LL-dt_LL:
        #yes -- within dt for LL. 
        best_idx = idx_LL_closest
        # for now, don't dictate the UL has to be similar. In theory if the LL is close that's all that matters...
        
#         if out_UL < t1_UL+dt_UL and out_UL > t1_UL-dt_UL:
#             best_idx = idx_LL_closest
        
    return best_idx
    
def find_nearest(array, value):
    '''find_nearest(array, value)
    
    array : array to search for value
    value : value to search for
    
    returns value_of_array_at_nearest, idx_of_array_at_nearest
    '''
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def UTC_days_to_date(UTC_in_days):
    from datetime import datetime
    date_vector=[]
    try: 
        for utc_day in UTC_in_days:
            ts = int(utc_day*24*3600)
    #         date_vector.append(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d'))
            date_vector.append(datetime.utcfromtimestamp(ts))
    except:
        ts = int(UTC_in_days*24*3600)
        date_vector.append(datetime.utcfromtimestamp(ts))

    return date_vector

def write_results(directory,filename_prefix, df, loopfit_stats):
    import pandas as pd
    
    fname=directory+filename_prefix
    df.to_csv(fname+'_param_fits.csv')
    with open(fname+'_loopfit_stats.txt', 'w') as f:
        print(loopfit_stats, file=f)
        
    return None
        
def get_nearest_lab(lfts,ids,t0,lab='INR'):
    # convention is nearest time minus t0 (passed value), so negative = before t0, positive=after t0
    
    x, t = get_traj(lfts, ids, lab=lab)
    if len(x) == 0:
        # this lab was never obtained
        return None, None, None
    else:
        nearest_t, nearest_t_idx = find_nearest(t, value=t0)
        nearest_x=x[nearest_t_idx]

        lab_value=nearest_x
        lab_time=nearest_t
        lab_dt=nearest_t-t0
    
    return lab_value, lab_time, lab_dt

def append_nearest_lab(df_in, lfts, suffix, labs_meta_list=['ALT', 'AST', 'AKP', 'INR', 'TBILI', 'DBILI', 'CR', 'IGG']):
    ''' append_nearest_lab
    
    DESC: for AST/ALT fit dataframe, goto the t0_+suffix column to get the time of the fit-peak, then
      using lfts as a lookup database, find the nearest labs_meta_list labs and add them along with their time/dt as cols'''
    import pandas as pd
    
    # remove the 'calling' lab (suffix) from meta list
    if suffix in labs_meta_list:
        labs_meta_list.remove(suffix)
    
    df = df_in.copy()
    
    # merge-asof will lose indice values, store them here
    df['idx_peak_'+suffix] = df.index
    
    # merge_asof requires sorted values; sort on epoch_days for the input frame
    df.sort_values(by='t0_'+suffix, ascending=True, inplace=True)
    
    # these can all become inputs / parameters eventually..
    exactly_match1='id_'+suffix
    exactly_match2='EMPI'
    time1='t0_'+suffix
    
    for lab in labs_meta_list:
        # shrink the frame to only this lab
        print('Filtering for ' + lab + '...')
        this_lab = lfts[lfts.test_desc == lab].copy()
        if not this_lab.empty:
            this_lab = this_lab.rename(columns={'result': lab+'_value'})
            # make a new column for epoch time (lfts/labs start as datetime)
            time2=lab+'_epoch'
            this_lab[time2] = to_np_timeseries(this_lab.datetime, dtype='ds')
            this_lab=this_lab[['EMPI',time2,lab+'_value']].copy()
            # sort here as well for merge_asof
            this_lab.sort_values(by=time2,ascending=True, inplace=True)

            # the magic. merge_asof merges based on nearest values rather than exact values. left_on and right_on are the columns (time) to match nearest
            #  It will further only match nearest WITHIN left_by and right_by which must be exact, and here are the EMPI/ids
            print('Merging nearest ' + lab + '...')
            df = pd.merge_asof(df, this_lab, left_on=time1, right_on=time2, left_by=exactly_match1, right_by=exactly_match2, direction='nearest')
            df.drop(columns=['EMPI'], inplace=True)
            df[lab+'_dt'] = df[time2]-df[time1]
        else:
            print('No such lab in reference: ' + lab)

    return df


def append_nearest_peak(df1,suffix1,df2,suffix2,dt):
    import pandas as pd
    import numpy as np
    
    idcolname1='id_'+suffix1
    idcolname2='id_'+suffix2
    t0colname1='t0_'+suffix1
    t0colname2='t0_'+suffix2    
    
    df1_idx=df1.index.tolist()
    
    idx_to_keep1 = []
    idx_to_keep2 = []
    counter_keep = []
    dt_keep = []
    ct=0
    
    for idx in df1_idx:
        id2match=df1.loc[idx,idcolname1]
        t0match=df1.loc[idx,t0colname1]
        # screen df2 for the same value
        df2_sub=df2[df2.loc[:,idcolname2]==id2match]
        
        if not df2_sub.empty:
            # convert t0 in df2 to a numparray to ease the difficulty of finding the value
            t0_2=df2_sub.loc[:,t0colname2].to_numpy()

            # get the closest value
            val_nearest, idx_nearest = find_nearest(t0_2,t0match)
            
            # calculate difference; reject if too far apart
            diff=val_nearest-t0match
            
            if np.absolute(diff) < dt:
                # store both indices
                idx_to_keep1.append(idx)
                idx_to_keep2.append(df2_sub.index[idx_nearest])
                counter_keep.append(ct)
                dt_keep.append(diff) # by convention, df2_t0-df1_t0
                ct=ct+1
                
    # done cycling; now merge the databases and add the dt col (INNER MERGE)
    ## this is weird because have to have the same ID to join.. so we made a new id in counter_keep. BUT. this isn't on the dataframes yet.
    # create a ref df for each
    match1=pd.DataFrame({'id1': counter_keep}, index=idx_to_keep1)
    match2=pd.DataFrame({'id2': counter_keep, 't0_dt': dt_keep}, index=idx_to_keep2)
    df1n=df1.merge(match1,left_index=True,right_index=True,how='inner')
    df2n=df2.merge(match2,left_index=True,right_index=True,how='inner')
    
    # now complete the merge
    merge_df=df1n.merge(df2n, left_on='id1',right_on='id2', how='inner')
    
    return merge_df, match1, match2

def plot_fit(ids, plots_save, df, suffix, pause_for_input=True):
    '''
    ids - peak indices (not patient)'''
    
    import matplotlib.pyplot as plt
    
    inp='y'
    
    if not isinstance(ids, list):
        ids=[ids]
        
    for idx in ids:
        
        if inp =='n':
            break
        
        t=plots_save[idx]['t']
        x=plots_save[idx]['x']
        xfit=plots_save[idx]['xfit']
        t_highres=plots_save[idx]['t_highres']
        x_highres=plots_save[idx]['x_highres']
        
        C_track2_present=False
        C_track3_present=False
        try:
            C_track2 = plots_save[idx]['C_track2']
            d_assume = plots_save[idx]['d']
            C_track2_present=True
        except:
            print('no c-track2')
        try:
            C_track3 = plots_save[idx]['C_track3']
            C_track3_present=True
        except:
            print('no c-track3')
        
        # THIS IS GOING TO ERROR LATER WHEN '_' IS ADDED BACK
        d=df.loc[idx,'d'+suffix]
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.plot(t, xfit, 'rx', label='best fit')
        ax.plot(t, x, 'b.', label='data')
        ax.plot(t_highres,x_highres,'--',color='tab:gray')
        
        if C_track3_present:
            # plot the 3point running C values; print the d values 
            ax.plot(t[1:-1],C_track3,'ko')
        if C_track2_present:
            # plot the 3point running C values; print the d values 
            ax.plot(t[1:],C_track2/d_assume,'bo')
        
        ax.set_title(idx)
        d_text='d={:0.3f}'.format(d)
        ax.text(0.79, 0.95, d_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
        ax.set_ylim([0,xfit[0]*1.2])
        ax.set_xlabel('days')
        ax.set_ylabel(suffix+' U/L')
        fig.show()
    
        if pause_for_input:
            inp=input('continue?')
            if inp=='n':
                return fig, ax
                
    return None

def binned_stats(x_data,y_data,n_bins=10, statistic='mean'):
    ''' binned_stats
    DESC: takes in x_data, y_data, computes the histogram with n_bins, and takes the average for each bin
    returns bin_averages, bin_centers'''
    import numpy as np
    from scipy.stats import binned_statistic
    
    if statistic=='sem':
        bin_stat, bin_edges, binnumber = binned_statistic(x_data, y_data, statistic='std', bins=n_bins)
        bin_counts, _trash, _trash2 = binned_statistic(x_data, y_data, statistic='count', bins=n_bins)
        bin_stat=bin_stat/np.sqrt(bin_counts)
        bin_centers = bin_edges[0:-1]+(bin_edges[1:]-bin_edges[0:-1])/2
    else:
        bin_stat, bin_edges, binnumber = binned_statistic(x_data, y_data, statistic=statistic, bins=n_bins)
        bin_centers = bin_edges[0:-1]+(bin_edges[1:]-bin_edges[0:-1])/2
    
    return bin_stat, bin_centers

def find_event_within(df_input1, df_input2, datetime_col1='datetime', datetime_col2='datetime', datetime_type='timestamp', by1='MRN', by2='MRN', dt=[7,14], suffix=('_path', '_meds'), asof_direction='nearest', remove=False, save_index=False):
    ''' find_event_within(df1, df2, datetime_col1='datetime', datetime_col2='datetime', dt=[7,14])
        DESC: takes any event list (by row) in df1, and finds events with matching by1/by2 elements (e.g., MRN) WITHIN -dt[0]
         to dt[1] days of each row event
         
         df1 - the dataframe to go through row by row
         df2 - the dataframe to search for each row of df1
         datetime_col1/2 - the column with datetime elements we want to match within for df1 and df2 respectively
         datetime_type - format of the datetime column. default='timestamp', alternatively='days_since_epoch'
         by1/by2 - the EXACT match column to ensure before doing the above; MRN was the imagined use-case
         dt - a list with two elements, default = [7, 14], which means allow a match in datetime for up to 7 days before or 14 days after
          * TAKE THE CLOSEST ELEMENT IN THAT RANGE
         suffix - what we want to append to duplicated columns as (), default=('_path', '_meds')
         save_index - merge_asof will lose the index at time of merge; if true, save these to a column name
        
        NOTES/Warnings:
        - Always takes the NEAREST event in df2 to a row in df1; thus, possible for events in df2 to be duplicated in multiple matches to df1 rows
        - Use case: 
        -- take liver biopsies (df1) and a df2 of medications *already filtered for prednisone and equivalents*
        -- return a joined dataframe of df2 elements matched to df1 elements first by MRN (by1/by2) then by range in datetime columns set by dt
        - default behavior is to dump NA columns (ie, rows of df1 with no match in df2)
    '''
    import pandas as pd
    
    # copy the dataframes so we're not altering the original
    df1=df_input1.copy()
    df2=df_input2.copy()
    
    # copy the indices (these will be lost in merge_asof)
    if save_index:
        df1['orig_index'+suffix[0]] = df1.index
        df2['orig_peak'+suffix[1]] = df2.index
    
    # get num cols initial
    num_cols_df1_init=len(df1.columns.tolist())
    
    # sort values (required by merge_asof)
    df1.sort_values(by=datetime_col1, ascending=True, inplace=True)
    df2.sort_values(by=datetime_col2, ascending=True, inplace=True)
    
    if datetime_type == 'days_since_epoch':
        df1['utc'+suffix[0]] = df1[datetime_col1]
        df2['utc'+suffix[1]] = df2[datetime_col2]
    elif datetime_type == 'timestamp':
        # add utc column
        df1['utc'+suffix[0]] = to_np_timeseries(df1, datetime_col=datetime_col1)
        df2['utc'+suffix[1]] = to_np_timeseries(df2, datetime_col=datetime_col2)

    # total number of days in dt range (first element = before event, second element = after event)
    len_dt=dt[0]+dt[1]

    # time offset since timedelta tolerance for merge_asof only takes balanced +/- tolerances, but want to be able to say 7 days before up to 21 days after (for example)
    # half the total range of days
    
    # make a temp column capturing the date of the midway point
    if datetime_type=='timestamp':
        timedelta_split = pd.Timedelta(days=len_dt/2)
        df1['dt_temp'] = df1[datetime_col1]-pd.Timedelta(days=dt[0])+timedelta_split
    elif datetime_type=='days_since_epoch':
        timedelta_split = len_dt/2
        df1['dt_temp'] = df1[datetime_col1]-dt[0]+timedelta_split
    
    # get num cols
    num_cols_df1=len(df1.columns.tolist())
    
    # the merge. 
    out_df2 = pd.merge_asof(df1, df2, left_on='dt_temp', right_on=datetime_col2, left_by=by1, right_by=by2, suffixes=suffix, direction=asof_direction, tolerance=timedelta_split)

    # make sure we've found at least one match (if so, we will have added columns compared to df1)
    if len(out_df2.columns.tolist()) > num_cols_df1:
        print('removing na cols')
        
        if remove==False:
            # get rid of NA columns by using the fact that any new column should be after len(df1 columns), and default behavior of merge_asof is NA if no match.
            out_df3=out_df2[~out_df2.iloc[:,num_cols_df1].isna()].copy()
            removed = out_df2[out_df2.iloc[:,num_cols_df1].isna()].copy()
        else:
            # remove=True, meaning invert this whole process. Keep the NAs only. Ie, if there is a drug script in the time range, REMOVE those entries. 
            out_df3=out_df2[out_df2.iloc[:,num_cols_df1].isna()].copy()
            removed = out_df2[~out_df2.iloc[:,num_cols_df1].isna()].copy()
        
        # delta col name
        timediff_name='dt' + suffix[1] + suffix[0]
        out_df3[timediff_name] = out_df3.loc[:,'utc'+suffix[1]] -out_df3.loc[:,'utc'+suffix[0]]

    out_df3.drop(labels=['dt_temp','utc'+suffix[1], 'utc'+suffix[0]], axis=1, inplace=True)
    
    if remove:
        out_df3=out_df3.iloc[:, 0:num_cols_df1_init].copy()
    
    return out_df3, removed
  
    
def save_obj(save_path, obj):
    import pickle
    with open(save_path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(load_path):
    import pickle
    with open(load_path + '.pkl', 'rb') as f:
        return pickle.load(f)

# def find_two_closest_vals
# from itertools import product
# min(product(arr1, arr2), key=lambda t: abs(t[0]-t[1]))[0]
