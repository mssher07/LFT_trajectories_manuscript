
def load_RPDR_dem_multiple(dir_data, fname, delimiter='|'):
    ''' load_RPDR_dem_multiple(dir_data, fname, delimiter='\t'):
        Sequentially loads all files from RPDR data dump when output is split. 
        
        1. Starts in dir_data (should have trailing slash), grabs all sub-folders' names automatically, then sequentially loads: dir_data/[sub-folders]/fname (where fname is the name of the file)
        * Note for whatever reason, on a multiple-split file dump from RPDR the labs, demographics, etc files are all named the exact same, just in different zips
        2. Calls the traditional load path function on each file
        3. Concatenates all results and returns 1 DF
        
        See load_native_data for remainder of parameters which are passed to that function
        
        '''
    import os
    import pandas as pd
    
    # get list of subdirectories
    subdirectories = [x[0] for x in os.walk(dir_data)][1:]
    
    first=True
    # for each subdir, use the traditional load function to load data and concat
    for subdir in subdirectories:
        print(subdir)
        path_to_path=subdir+'/'+fname
        path = load_RPDR_dem(path_to_path, delimiter=delimiter)
        
        if first==True:
            concat_pd = path
            first=False
        else:
            concat_pd=pd.concat([concat_pd, path],ignore_index=True)
    
    return concat_pd


def load_RPDR_dem(path, delimiter='|'):
    
    import pandas as pd
    import os.path
    from os import path as os_path
    
    path_df = pd.read_csv(path, delimiter=delimiter)
    path_df['EMPI'] = path_df['EMPI'].astype(str)
    path_df['Date_of_Birth'] = pd.to_datetime(path_df['Date_of_Birth'], errors='ignore')
    path_df['Date_Of_Death'] = pd.to_datetime(path_df['Date_Of_Death'], errors='ignore')
    
    return path_df
