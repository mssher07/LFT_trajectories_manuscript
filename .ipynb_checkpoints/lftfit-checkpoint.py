
# FUNCTIONS

def EXP_solve_baseline(Sth, Sh, d=1.132, mid_t=True):
    ''' EXP_solve_baseline(Sth, Sh, d)
    Sth = time series as np array
    Sh = AST or ALT trajectory for time series 
    d = assumed degradation rate of mono-exponential function with constant injury H
    
    **Sth and Sh must be equal in length
    **returns H(t)/d - ie, the estimated "baseline" AST or ALT once the curve stabilizes
    
    returns: direct calculated midpoint solution assuming d, ie dL/dt = H(t) - d*L, 
     solve for H(t) where dL/dt is estimated from pairs of points, and L(t) is considered to be the midpoint L(i) and L(i+1)
     
     returns tfit , xfit  
     where tfit = Sth[1:]
     
     REFERENCES:
     d_ALT mean by std minimization : 0.47349533414857237+-0.007413096158667578 (mean+-SEM)
     95% CI for d_ALT : 0.194432767548125 to 0.9598856607249993
     99% CI for d_ALT : 0.15128599865318237 to 1.2336450643550432
     
     d_AST mean by std minimization : 1.1324928671977397+-0.060759944579000864 (mean+-SEM)
     95% CI for d_AST : 0.7388641090720738 to 1.6607386974701672
     99% CI for d_AST : 0.6505664390112067 to 1.8861412832374433 
     '''
    # midpoint or average of L(i) and L(i+1)
    midLt=(Sh[0:-1]+Sh[1:])/2
    midt=(Sth[0:-1]+Sth[1:])/2
    # vectorized delta L and delta t
    dL=Sh[1:]-Sh[0:-1]
    dt=Sth[1:]-Sth[0:-1]
    # calculate result. Note, we divide by d because H(t)/d = estimated baseline value, so units of result are IU/L of AST/ALT. 
    dLdT=((dL/dt)+d*midLt)/d
    
    if mid_t:
        return midt, dLdT
    else:
        return Sth[1:], dLdT
    
def fit_2point(t,x,fix='False',d=.3,C=8, return_best_fit_curve=False):
    '''fit_2point(t,x,fix='False',d=.3,C=8):
        Takes a pair of ALT/AST points and fits to the simple first-order kinetics model
        t=list or np array of time values
        x=list or np array of lab values (AST or ALT)
        fix= whether to fix some of the parameters. Default = 'False' (with quotes!!), fit both params
        ! need at least 3 points to fit both params
        -- other options: 'd' (fix d; make sure to set d=value); 'C' (set 'C')
        
        returns result (gmodel output)
        - access xfit by result.best_fit
        - access result.params as result.params[key].value, where keys in this model are:
        -- L0, C, d, derived_baseline_LFTs'''
    
    import lftmodels
    from lmfit import Model
    import numpy as np
    
    # t-correction
    initial_t=t[0]
    t_model=t-initial_t
    
    # create the lmfit model
    gmodel = Model(lftmodels.ALT_no_heps)
    
    #ALT_no_heps(t, L0, C, a):
    gmodel.set_param_hint('L0', value=x[0], vary=False)
    gmodel.set_param_hint('derived_baseline_LFTs', expr='C/d')
    
    if fix == 'd':
        gmodel.set_param_hint('C', value=8, min=0, vary=True)
        gmodel.set_param_hint('d', value=d, min=0, vary=False)
    elif fix == 'C':
        gmodel.set_param_hint('C', value=C, min=0, vary=False)
        gmodel.set_param_hint('d', value=.3, min=0, vary=True)
    elif fix == 'False':
        gmodel.set_param_hint('C', value=8, min=0, vary=True)
        gmodel.set_param_hint('d', value=.3, min=0, vary=True)
    params = gmodel.make_params()
    
    # fit the model
    result = gmodel.fit(x, params, t=t_model)
    
    if return_best_fit_curve:
        # add add'l datapoints
        t_highres=np.linspace(0,np.ceil(t_model[-1]),50)
        x_highres=gmodel.eval(result.params,t=t_highres)
        
        return result, t_highres+initial_t, x_highres
    
    return result

def fit_2pointBI_track(t,x,d1=.68, d2=39):
    import numpy as np
    
    fast_track = np.array([])
    slow_track = np.array([])
    for s in range(0,len(t)-1):
        p1=s
        pn=s+2
        fast_model, slow_model = fit_2pointBI(t[p1:pn],x[p1:pn],d1=d1, d2=d2)
        fast=fast_model.params['C'].value
        slow=slow_model.params['C'].value
        fast_track=np.append(fast_track,fast)
        slow_track=np.append(slow_track,slow)
    
    return t[1:], fast_track, slow_track

def fit_2pointBI(t,x,d1=.68, d2=39, return_best_fit_curve=False):
    '''fit_2pointBI(t,x,fix='False',d=.3,C=8):
        Takes a pair of ALT/AST points and fits to the simple first-order kinetics model
        t=list or np array of time values
        x=list or np array of lab values (AST or ALT)
        fix= whether to fix some of the parameters. Default = 'False' (with quotes!!), fit both params
        ! need at least 3 points to fit both params
        -- other options: 'd' (fix d; make sure to set d=value); 'C' (set 'C')
        
        returns result (gmodel output)
        - access xfit by result.best_fit
        - access result.params as result.params[key].value, where keys in this model are:
        -- L0, C, d, derived_baseline_LFTs'''
    
    import lftmodels
    from lmfit import Model
    import numpy as np
    
    # t-correction
    initial_t=t[0]
    t_model=t-initial_t
    
    # create the lmfit model - bounding with biexponential
    # UPPER LIMIT (alpha mostly fast)
    alpha_fast=.4
    BI_modelF = Model(lftmodels.biexponential)
    
    BI_modelF.set_param_hint('L0', value=x[0], vary=False)
    BI_modelF.set_param_hint('C', value=x[0], vary=True)
    BI_modelF.set_param_hint('a', value=alpha_fast, vary=False)
    BI_modelF.set_param_hint('d1', value=d1, min=0, vary=False)
    BI_modelF.set_param_hint('d2', value=d2, min=0, vary=False)
    
    params = BI_modelF.make_params()
    
    # fit the model
    result_upper = BI_modelF.fit(x, params, t=t_model)
    
    # create the lmfit model - bounding with biexponential
    # lower LIMIT (alpha mostly fast)
    alpha_slow=1
    BI_modelS = Model(lftmodels.biexponential)
    
    BI_modelS.set_param_hint('L0', value=x[0], vary=False)
    BI_modelS.set_param_hint('C', value=18, vary=True)
    BI_modelS.set_param_hint('a', value=alpha_slow, vary=False)
    BI_modelS.set_param_hint('d1', value=d1, min=0, vary=False)
    BI_modelS.set_param_hint('d2', value=d2, min=0, vary=False)
    
    params = BI_modelS.make_params()
    
    # fit the model
    result_lower = BI_modelS.fit(x, params, t=t_model)
    
    if return_best_fit_curve:
        # add add'l datapoints
        t_highres=np.linspace(0,np.ceil(t_model[-1]),50)
        x_fast=BI_modelF.eval(result.params,t=t_highres)
        x_slow=BI_modelS.eval(result.params,t=t_highres)
        
        return result_upper, result_lower, t_highres, x_fast, x_slow
    
    return result_upper, result_lower

def delta_EXP(t_in,x,d=.44, baseline_LFT=18, calc_UL_LL=True, d_UL=.8, d_LL=.35):
    '''delta_EXP(t,x,fix='False',d=.3,C=8):
        Takes a pair of ALT/AST points and calculates the difference between x2 and the expected x2* if following exponential decay according to parameters d and C
        t=pair of time values
        x=pair of AST or ALT values
        
        returns delta (which can be interpreted as *excess* AST or ALT present over what one expects given exponential decay with parameters d and C)'''
    
    import lftmodels
    from lmfit import Model
    import numpy as np
    import scipy
    from scipy.integrate import quad
    
    if len(t_in) != 2 or len(x) !=2:
        print('delta_EXP expects only a pair of t1,t2 and x1,x2 values')
        return None
    
    # calculate C from baseline
    C = d*baseline_LFT
    
    # t-correction (in case we're passed mid-trajectory or non-zeroed time)
    initial_t=t_in[0]
    t=t_in-initial_t
    
    
    # ALT_no_heps(t, L0, C, d). This is the magic. Plug in t1,t2 (in t_model, zeroed), L0=X[0]. 
#     EX_x = lftmodels.ALT_no_heps(t_model,x[0],C,d)
    # now the model x[1] is the expected x[1] at time t[1] if exponential decay alone. 
    # ..therefore, return x[1] (actual, observed) minus EX_x[1] (expected)
    #delta_EX = (x[1]-EX_x[1])/(t[1]-t[0])
    #delta_EX = x[1]-EX_x[1]

    def diff(tt,t01,x01,C,d):
        m=(x01[1]-x01[0])/(t01[1]-t01[0])
        MODEL=lftmodels.ALT_no_heps(tt, x01[0], C, d)
        return (m*(tt-t01[0])+x01[0])-MODEL
    
    int_diff, _ = quad(diff, t[0], t[1],args=(t,x,C,d))
    delta_EX=int_diff/(t[1]-t[0])
    
    if calc_UL_LL:
        # repeat using UL and LL of d
        ## UL ##
        C_UL = d_UL*baseline_LFT
#         EX_x_UL = lftmodels.ALT_no_heps(t_model,x[0],C_UL,d_UL)
#         delta_UL = (x[1]-EX_x_UL[1])/(t[1]-t[0])
        int_diff_UL, _ = quad(diff, t[0], t[1],args=(t,x,C_UL,d_UL))
        delta_UL=int_diff_UL/(t[1]-t[0])
        ## UL ##
        C_LL = d_LL*baseline_LFT
#         EX_x_LL = lftmodels.ALT_no_heps(t_model,x[0],C_LL,d_LL)
#         delta_LL = (x[1]-EX_x_LL[1])/(t[1]-t[0])
        int_diff_LL, _ = quad(diff, t[0], t[1],args=(t,x,C_LL,d_LL))
        delta_LL=int_diff_LL/(t[1]-t[0])
        
        return delta_EX, delta_LL, delta_UL
    else:
        return delta_EX

def fit_biexp(t,x):
    
    import lftmodels
    from lmfit import Model
    import numpy as np
    
    # t-correction
    initial_t=t[0]
    t_model=t-initial_t
    
    # create the lmfit model
    gmodel = Model(lftmodels.biexponential)
    
    #biexponential(t, A, B, d1, d2):
    gmodel.set_param_hint('L0', value=x[0], vary=False)
    
    gmodel.set_param_hint('C', value=22, min=0, max=100, vary=True)
    gmodel.set_param_hint('a', value=.5, min=0, max=1, vary=True)
    gmodel.set_param_hint('d1', value=1, min=0, vary=True)
    gmodel.set_param_hint('d2', value=2, min=0, vary=True)
    
    params = gmodel.make_params()
    
    # fit the model
    result = gmodel.fit(x, params, t=t_model)
    
    t_highres=np.linspace(0,np.ceil(t_model[-1]),50)
    x_highres=gmodel.eval(result.params,t=t_highres)
        
    return result, t_highres+initial_t, x_highres

def fit_simple(ids,
               lfts,
               lab='ALT',
               dates=[],
               pause_for_input=True,
               plot_figs=True,
               show_residuals=False,
               show_heps=False,
               dtau=0,
               min_peak2trough=5,
               min_peak_height=100,
               prominence=1,
               opt_loopfit_stats=False,
               eval_C_d=False,
               eval_C=False,
               d=.42):
    
    # WARNINGS / ISSUES
    # - only 2x flex in the L0 param in case we implement lookback
    
    from scipy.stats import shapiro
    from scipy.stats import normaltest
    from lmfit import Model
    import matplotlib.pyplot as plt
    import lftlib
    import lftmodels
    import numpy as np
    import pandas as pd
    
    if opt_loopfit_stats:
        print('Loopfit stats will be reported, be sure to have 3 outputs - df, plots_save, loopfit_stats')
    
    first_fit=True
    inp='t'

    # overall loop fitting stats (e.g., number that pass various filters)
    num_peaks=0
    num_meeting_traj_criteria=0
    num_fit=0
    
    
    for i in range(0,len(ids)):
        if inp =='n':
            break
            
        xalt, tseries = lftlib.get_traj(lfts, ids[i], lab=lab)
        
        if list(dates):
            # we can pass lists of dates for each MRN
            peak_dates=dates[i]
            # d of get_peaks() captures dates. Expects a list of dates; if we're only passing single value, make it a list...
            if isinstance(peak_dates,float):
                peak_dates=[peak_dates]
            peaks, n_peaks, LL, UL = lftlib.get_peaks(xalt, tseries, d=peak_dates, opt_prominence=prominence, plot_peaks=False, min_height=min_peak_height, min_datapoints=min_peak2trough, allow_edge_peaks=True, options='peak_to_min')
        else:
            # no specific dates to grab, return all peaks meeting criteria
            peaks, n_peaks, LL, UL = lftlib.get_peaks(xalt, tseries, opt_prominence=prominence, plot_peaks=False, min_height=min_peak_height, min_datapoints=min_peak2trough, allow_edge_peaks=True, options='peak_to_min')

        # make sure there IS at least one peak
        if n_peaks > 0:
            if i%100==0: print(i)
            # cycle through peaks
            for j in range(0,len(LL)):
                num_peaks=num_peaks+1
                x=xalt[LL[j]:UL[j]]

                # necessitate there are least min_peak2trough datapoints in this series and that the initial value is greater than min_peak_height
                if len(x) > min_peak2trough and x[0] > min_peak_height:
                    num_meeting_traj_criteria = num_meeting_traj_criteria+1
                    # truncate t to this particular range
                    t=tseries[LL[j]:UL[j]] # absolute time
                
                    # create the lmfit model
                    gmodel = Model(lftmodels.ALT_no_heps)

                    # auto-extract the parameters and give them initial values lft_trajz(t, C_l, C_h, a, k, z, d, baseline):

                    #ALT_no_heps(t, L0, C, a):
                    # NEW PARAMS
                    gmodel.set_param_hint('L0', value=x[0], min=x[0]*.2, max=x[0]*10, vary=True)
                    gmodel.set_param_hint('C', value=8, min=0, vary=True)
                    gmodel.set_param_hint('d', value=.3, min=0, vary=True)
                    gmodel.set_param_hint('derived_baseline_LFTs', expr='C/d')
                
                    params = gmodel.make_params()

                    # have fit test
                    has_fit=False
                    has_excess_tail=False
                    
                    # record some values for saving
                    t0_rec=t[0]
                    t0_date=lftlib.epoch_days_to_date(t0_rec)
                    tfinal_rec=t[-1]
                    tlength_rec=t[-1]-t[0]
                    x0_rec=x[0]
                    xfinal_rec=x[-1]
                    
                    # set t0 as first time point of this sim; dtau allows an offset for backward prediction
                    t=t-t[0]+dtau
                    
                    # BONUS TRACKS
                    # THREE POINT TRACK
                    # no assumptions, what is C and d throughout the time course
                    if eval_C_d == True:
                        d_track=np.array([])
                        C_track3=np.array([])
                        # sequentially fit the 3 point model on a rolling basis from 0:2, to n-3:n lab values
                        # - record d, C at each
                        for s in range(0,len(t)-2):
                            p1=s
                            pn=s+3
                            three_point = fit_2point(t[p1:pn]-t[p1],x[p1:pn],fix='False')
                            d_track=np.append(d_track,three_point.params['d'].value)
                            C_track3=np.append(C_track3,three_point.params['C'].value)
                        
                    # TWO POINT TRACK
                    # assume d=d, what is C (and/or C/d) throughout the time course
                    if eval_C == True:
                        C_track2=np.array([])
                        # sequentially fit the 2 point model on a rolling basis from 0:1, to n-1:n lab values
                        # - record C at each assuming input d=d
                        for s in range(0,len(t)-1):
                            p1=s
                            pn=s+2
                            two_point = fit_2point(t[p1:pn]-t[p1],x[p1:pn],fix='d', d=d)
                            C_track2=np.append(C_track2,two_point.params['C'].value)
                    try:
                        result = gmodel.fit(x, params, t=t)
                        has_fit=True
                    except:
                        # smart debugging, why didn't the model fit?
                        print('Troubleshooting ids index=' + str(i) + ', peak number ' + str(j))
                        # common problems:
                        # 1. Very long tail values
                        #  *solution, truncate. *Approach: find t1/2 (time for this LFT to decline by half) -- say 3 days, and multiply by some number (e.g., 10-15)
                        #    then truncate to 30-45 days, which should be enough time to capture the beginning of the tail
                        print('Checking whether there is an excessively long tail...')
                        #    first x value less than x12
                        x12=((x[0]-x[-1])/2)+x[-1]
                        x12_idx = np.argmax(x<x12)

                        # therefore, the transition in time where x12 occurs is between x12_idx and it's previous value. use linear interpolation to get the actual value
                        xpair=x[x12_idx-1:x12_idx+1] # only 2 values, the first value where x is less than x12, and the value right before it
                        tpair=t[x12_idx-1:x12_idx+1]
                        # have to flip and reorder since np.interp accepts an *x* value to interp y at, while we have a y-value (x12) to interp a t (x) at
                        t12=np.interp(x12,np.flip(xpair),np.flip(tpair)) # gets the linear interp for t at x12 between these two points

                        # calculate truncation based on number of half_lives < X, where X is number of half-lives (in theory) for x0 to become normal
                        normal_transaminases=15 # low/conservative number for ALT and AST
                        t12_to_normal=np.log(normal_transaminases/x[0])/np.log(.5) #number of half lives for initial value (x[0]) 
                        num_days_to_truncate = (t12_to_normal*2)*t12 # num half-lives X length of half-life X 2 (for add'l buffer)
                        # do the truncation with some checks
                        t_old=t
                        t=t_old[t_old<num_days_to_truncate] # this will return t if num_days_to_truncate exceeds the last element of t (all values of t<num_days will be True)
                        len_t_old=len(t_old)
                        len_t = len(t) # 0 if same length, otherwise len_diff is the number of datapoints truncated; make sure this doesn't truncate to less than minimum
                        x_old = x
                        x=x[:len_t]
                        if len_t_old-len_t > 0 and len_t>min_peak2trough:
                            # a truncation did happen:
                            print('Long tail present; truncating by ' + str(t_old[-1]-t[-1]) + ' days and ' + str(len_t_old-len_t) + ' datapoints. Attempting fit again...')
                            # truncate x as well
                            has_excess_tail=True

                            try:
                                result = gmodel.fit(x, params, t=t)
                                print('... Success! This is the truncation of tail: ')
                                if plot_figs:
                                    plt.plot(t, x, 'b.', label='final')
                                    plt.plot(t_old, x_old, 'rx', label='original')
                                    plt.show()                            
                                has_fit=True
                            except:
                                print('Truncation of excessive tail did not resolve issue, restoring full time course, but here is what couldnt be fit:')
                                plt.plot(t, x, 'b.', label='data')
                                plt.title('Trajectory with truncated excessive tail')
                                plt.show()
                                t_short=t
                                x_short=x
                                t=t_old
                                x=x_old

                    if has_fit:
                        num_fit=num_fit+1

                        # add add'l datapoints
                        t_highres=np.linspace(0,np.ceil(t[-1]),50)
                        x_highres=gmodel.eval(result.params,t=t_highres)

                        if plot_figs:
                            print(result.fit_report())
                            # plt.plot(x, result.init_fit, 'k--', label='initial fit')
                            plt.plot(t, result.best_fit, 'rx', label='best fit')
                            plt.plot(t, x, 'b.', label='data')
                            plt.plot(t_highres,x_highres,'--',color='tab:gray')
                            # really subtle: if dates is an np.array, convert to list first otherwise it will fail. otherwise referencing is fine in this code block
                            if list(dates):
                                t_date=dates[i]-t0_rec
                                plt.plot(t_date, 5, 'v')
                                plt.annotate("Point 1", (t_date, 5))
                                
                            if eval_C_d:
                                # plot the 3point running C values; print the d values 
                                plt.plot(t[1:-1],C_track3,'ko')
                                print('d values 3-point-running:')
                                print(d_track)
                            if eval_C:
                                # plot the 3point running C values; print the d values 
                                plt.plot(t[1:],C_track2/d,'bo')
                            
                            plt.legend(loc='best')
                            plt.title(label=ids[i] + ' | ' + t0_date)
                            plt.xlabel('days')
                            plt.ylabel(lab+' U/L')
                            axes = plt.gca()
        #                     axes.set_xlim([xmin,xmax])
                            axes.set_ylim([0,result.best_fit[0]*1.2])
                            plt.show()

                            # show heps
                            if show_heps:
                                x_heps=lftmodels.lft_traj_baseh(t=t_highres, H0=result.params['H0'].value, a=result.params['a'].value, bl_heps=result.params['bl_heps'].value)
                                plt.plot(t_highres, x_heps)
                                plt.show()

                            # show residuals
                            if show_residuals:
                                plt.plot(t,result.residual)
                                plt.show()
                                
                            plt.plot(t,np.log(x))
                            plt.title('semilog')
                            plt.show()

#                         stat, p = shapiro(result.residual)
#                         print('Statistics_Shapiro=%.3f, p=%.3f' % (stat, p))
#                         if len(result.residual) >= 8:
#                             stat, p = normaltest(result.residual)
#                             print('Statistics_normal_test=%.3f, p=%.3f' % (stat, p))
#                         acorr, p = lftlib.discrete_autocorr(result.residual,h=1)
#                         print('Statistics_autocorr_R1=%.3f, Z=%.3f' % (acorr, p))
                        

                        qual_error=lftlib.qualitative_error(t, x, result.residual)
                        MAPE=lftlib.MAPE(t, x, result.residual)
                        nRMSD=lftlib.nRMSD(t, x, result.residual)
                        if plot_figs:
                            print('Marc_qual_error=%.3f' % (qual_error))
                            print('MAPE=%.3f' % (MAPE) )
                            print('nRMSD=%.3f' % (nRMSD) )
                    
                        # capture this model, automatically unpack what parameters and build save structure
                        if first_fit:
                            print('yes got to first fit...')
                            # get parameter names
                            save_dict={}
                            plots_save={}
                            first_fit=False
                            for key in result.params:
                                # initialize save_dict
                                save_dict[key+'_'+lab]=[]
                                save_dict[key+'_stderr_'+lab]=[]
                            # now initialize the rest of the dict
                            save_dict['id_'+lab]=[]
                            save_dict['qerr_'+lab]=[]
                            save_dict['peak_'+lab]=[]
                            save_dict['t0_'+lab]=[]
                            save_dict['tf_'+lab]=[]
                            save_dict['tlen_'+lab]=[]
                            save_dict['x0_'+lab]=[]
                            save_dict['xf_'+lab]=[]
                            idx_save=[]
                        
                        # record values to save for this fit
                        for key in result.params:
                                # record fitted and derived parameters
                                save_dict[key+lab].append(result.params[key].value)
                                save_dict[key+'stderr'+lab].append(result.params[key].stderr)
                                
                        # record other fit parameters
                        save_dict['qerr_'+lab].append(qual_error)
                        save_dict['id_'+lab].append(ids[i])
                        save_dict['peak_'+lab].append(peaks[j]) # index of t where this peak starts
                        save_dict['t0_'+lab].append(t0_rec)
                        save_dict['tf_'+lab].append(tfinal_rec)
                        save_dict['tlen_'+lab].append(tlength_rec)
                        save_dict['x0_'+lab].append(x0_rec)
                        save_dict['xf_'+lab].append(xfinal_rec)
                        idx_save.append(ids[i]+'_'+str(peaks[j]))
                        plots_save[idx_save[-1]]={'t': t,
                                                  'x': x,
                                                  'xfit': result.best_fit,
                                                  't_highres': t_highres,
                                                  'x_highres': x_highres}
                        if eval_C:
                            plots_save[idx_save[-1]]['C_track2'] = C_track2
                            plots_save[idx_save[-1]]['d'] = d
                        if eval_C_d:
                            plots_save[idx_save[-1]]['C_track3'] = C_track3
                            plots_save[idx_save[-1]]['d_track3'] = d_track

                    else:
                        print('EXCEPTION: could not fit to this plot: ids index=' + str(i) + ', peak number ' + str(j))
                        plt.plot(t, x, 'b.', label='data')
                        plt.show()

                    if pause_for_input:
                        inp=input('continue?')
                        if inp=='n':
                            break
      
    # ie, if we didn't have a fit..
    if first_fit:
        # if we're getting here and first_fit still true, there were no fits
        if opt_loopfit_stats:
            return None, None, None
        else:
            return None, None
    else:
        df = pd.DataFrame(save_dict,index=idx_save)

        loopfit_stats={'num_peaks': num_peaks,
                       'num_meeting_traj_criteria': num_meeting_traj_criteria,
                       'num_fit': num_fit}

    # df has all the things you'd want to look at, psave has a few additional things (exact parameter fitting conditions / mins / maxes etc)
    if opt_loopfit_stats:
        return df, plots_save, loopfit_stats
    else:
        return df, plots_save

def truncate_long_tail(t,x):
    #  *Approach: find t1/2 (time for this LFT to decline by half) -- say 3 days, and multiply by some number (e.g., 10-15)
    #    then truncate to 30-45 days, which should be enough time to capture the beginning of the tail
    
    import numpy as np
    print('Checking whether there is an excessively long tail...')
    #    first x value less than x12
    x12=((x[0]-x[-1])/2)+x[-1]
    x12_idx = np.argmax(x<x12)

    # therefore, the transition in time where x12 occurs is between x12_idx and it's previous value. use linear interpolation to get the actual value
    xpair=x[x12_idx-1:x12_idx+1] # only 2 values, the first value where x is less than x12, and the value right before it
    tpair=t[x12_idx-1:x12_idx+1]
    # have to flip and reorder since np.interp accepts an *x* value to interp y at, while we have a y-value (x12) to interp a t (x) at
    t12=np.interp(x12,np.flip(xpair),np.flip(tpair)) # gets the linear interp for t at x12 between these two points
    print(t12-t[0])

    # calculate truncation based on number of half_lives < X, where X is number of half-lives (in theory) for x0 to become normal
    normal_transaminases=12 # low/conservative number for ALT and AST
    t12_to_normal=np.log(normal_transaminases/x[0])/np.log(.5) #number of half lives for initial value (x[0]) 
    print('t12_to_normal')
    print(t12_to_normal)
    num_days_to_truncate5x = (t12_to_normal*5)+t[0] # num half-lives X length of half-life X 5 (for add'l buffer)
    num_days_to_truncate10x = (t12_to_normal*10)+t[0] # num half-lives X length of half-life X 10 (for add'l buffer)
    # accept no more than 1 value between 5 and 10
    print('5x: ' + str(num_days_to_truncate5x) + ' | 10x: ' + str(num_days_to_truncate10x))
    print('t:')
    print(t)
    # do the truncation with some checks
    t_old=t
    t5=t_old[t_old<num_days_to_truncate5x] # this will return t if num_days_to_truncate exceeds the last element of t (all values of t<num_days will be True)
    t10 = t_old[t_old<num_days_to_truncate10x]
    # now, if num_days_to_truncate5x and num_days_to_truncate10x differ, ie 1+ values falls in the 5x-10x half lives range, take **1 more value** past t5
    if len(t5) < len(t10):
        t=t10[0:len(t5)+1]
    else:
        t=t5
    len_t_old=len(t_old)
    len_t = len(t) # 0 if same length, otherwise len_diff is the number of datapoints truncated; make sure this doesn't truncate to less than minimum
    x_old = x
    x=x[:len_t]
    if len_t_old-len_t > 0:
        # a truncation did happen:
        print('Long tail present; truncating by ' + str(t_old[-1]-t[-1]) + ' days and ' + str(len_t_old-len_t) + ' datapoints. Attempting fit again...')
        # truncate x as well
        has_excess_tail=True
        
    return t, x

def truncate_steady_state(t,x,frac_change_cutoff=.01):
    import numpy as np
    
    window=x[0]-x[-1]
    frac_change_vector = ((x[0:-1]-x[1:])/(t[1:]-t[0:-1]))/window
    state_change = frac_change_vector<frac_change_cutoff
    
    # make sure at least one value changed, otherwise they're all above cutoff
    if np.any(state_change):
        # plus 1 because we took delta first and lost an element. So if cutoff happens between the 1st and 2nd elements (0-based), delta is True at 1, and we should keep elements 0, 1, which means cutoff=2 (non-inclusive final indexing)
        first_element_w_less_than_frac_change = np.argmax(state_change)+1
    
        # does not include the first element with less than frac change, includes last element before that element
        x_return = x[0:first_element_w_less_than_frac_change]
        t_return = t[0:first_element_w_less_than_frac_change]

#         if len(x) != len(x_return):
#             print('Truncated at steady state by ' + str(len(x)-len(x_return)))
#             print(frac_change_vector)
    else:
        t_return=t
        x_return=x
        
    return t_return, x_return

def first_gap_present(t,max_gap=5):
    if t[1]-t[0] > max_gap:        
        return True
    else:
        return False
    
def generate_p2t_trajectories(savepath,
                                ids,
                                df_lfts,
                                dates=[],
                                lab='ALT',
                                min_peak2trough=5,
                                min_peak_height=150,
                                prominence=1, 
                                screen_first_gap=True,
                                screen_steady_state=True,
                                screen_steady_state_cutoff=.01,
                                screen_long_tail=True,
                                trajectory_type='peak_to_min',
                                opt_strict_monotonic=True):
    
    # WARNINGS / ISSUES

    import lftlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pprint
    
    ## run metadata structure
    p2t_run_meta = {}
    # initialize some
    num_peaks=0
    num_meeting_traj_criteria=0
    
    p2t_traj_meta={}
    p2t_trajectories={}
    full_trajectories={}
    first_fit=False

    # now initialize the rest of the dict
    p2t_traj_meta['id_'+lab]=[]
    p2t_traj_meta['t0_'+lab]=[]
    p2t_traj_meta['t0_date_'+lab]=[]
    p2t_traj_meta['index_xt_peak_'+lab]=[]
    p2t_traj_meta['tf_'+lab]=[]
    p2t_traj_meta['tf_date_'+lab]=[]
    p2t_traj_meta['index_xt_trough_'+lab]=[]
    p2t_traj_meta['tlen_'+lab]=[]
    p2t_traj_meta['x0_'+lab]=[]
    p2t_traj_meta['xf_'+lab]=[]
    p2t_traj_meta['xdelta_'+lab]=[]
    idx_save=[]
    
    # pre-filter for the proper lab (this should dramatically speed up get_traj)
    lfts = df_lfts[df_lfts.test_desc == lab].copy()
    
    for i in range(0,len(ids)):
        # progress bar
        if i%100==0: print(i)
        
        # extract the trajectory for this lab value
        xlab, tseries = lftlib.get_traj(lfts, ids[i], lab=lab)
        
        ####################
        if list(dates):
            # we can pass lists of dates for each MRN
            peak_dates=dates[i]
            # d of get_peaks() captures dates. Expects a list of dates; if we're only passing single value, make it a list...
            if isinstance(peak_dates,float):
                peak_dates=[peak_dates]
            peaks, n_peaks, LL, UL = lftlib.get_peaks(xlab, tseries, d=peak_dates, opt_prominence=prominence, plot_peaks=False, min_height=min_peak_height, min_datapoints=min_peak2trough, allow_edge_peaks=True, options=trajectory_type, strict_monotonic=opt_strict_monotonic)
        else:
            # no specific dates to grab, return all peaks meeting criteria
            peaks, n_peaks, LL, UL = lftlib.get_peaks(xlab, tseries, opt_prominence=prominence, plot_peaks=False, min_height=min_peak_height, min_datapoints=min_peak2trough, allow_edge_peaks=True, options=trajectory_type, strict_monotonic=opt_strict_monotonic)
        
        ####################
        
        # accepts a trajectory, and extracts the peaks 
        #peaks, n_peaks, LL, UL = lftlib.get_peaks(xlab, tseries, opt_prominence=prominence, plot_peaks=False, min_height=min_peak_height, min_datapoints=min_peak2trough, allow_edge_peaks=True, options='peak_to_min')

        # make sure there IS at least one peak
        if n_peaks > 0:
            # save full trajectories (and the peak parsing results)
            full_trajectories[ids[i]]={'tseries': tseries,
                                       'xlab': xlab,
                                       'peaks': peaks,
                                       'n_peaks': n_peaks,
                                       'LL': LL,
                                       'UL': UL}
            
            # cycle through peaks
            for j in range(0,len(LL)):
                num_peaks=num_peaks+1
                x=xlab[LL[j]:UL[j]]
                t=tseries[LL[j]:UL[j]]
                                
                # assess whether there is a large gap between the first and second observation (if so, fitting will be thrown off, plus most of the data is early in a traj)
                if screen_first_gap:
                    no_first_gap = not first_gap_present(t)
                else:
                    no_first_gap = True
                    
                # although get_peaks requires monotonically declining values, they could still be declining very slowly. That adds datapoints that aren't really contributing to fit
                if no_first_gap and screen_steady_state:
                    # here, because t and x are truncated as needed, no need for true/false; if too many values removed, the len(x) > min_peak2trough will catch
                    t, x = truncate_steady_state(t,x,frac_change_cutoff=screen_steady_state_cutoff)

                # long tails are bad. e.g., falling trajectory days 0-5 from 1200->50, and then 10 years later the next value is 25, the peak-finder would include that value way later since it's less than 50. but we don't want it.
                if no_first_gap and len(x) > min_peak2trough and screen_long_tail:
                    t, x = truncate_long_tail(t,x)
                
                print(t)
                print(x)
                
                # necessitate there are least min_peak2trough datapoints in this series and that the initial value is greater than min_peak_height
                if no_first_gap and len(x) >= min_peak2trough and x[0] >= min_peak_height:
                    num_meeting_traj_criteria = num_meeting_traj_criteria+1
                
                    # record some values for saving
                    t0_rec=t[0]
                    t0_date=lftlib.epoch_days_to_date(t0_rec)
                    tfinal_rec=t[-1]
                    tfinal_date=lftlib.epoch_days_to_date(t0_rec)
                    tlength_rec=t[-1]-t[0]
                    x0_rec=x[0]
                    xfinal_rec=x[-1]
                    xdelta=x[0]-x[-1]
                    
                    # set t0 as first time point of this sim
                    t=t-t[0]
                    
                    # record other fit parameters
                    p2t_traj_meta['id_'+lab].append(ids[i])
                    p2t_traj_meta['t0_'+lab].append(t0_rec)
                    p2t_traj_meta['t0_date_'+lab].append(t0_date)
                    p2t_traj_meta['index_xt_peak_'+lab].append(LL[j]) # index of t where this peak starts
                    p2t_traj_meta['tf_'+lab].append(tfinal_rec)
                    p2t_traj_meta['tf_date_'+lab].append(tfinal_date)
                    p2t_traj_meta['index_xt_trough_'+lab].append(UL[j]) # index of t where this peak starts, not inclusive (meant to be used like tseries[LL[j]:UL[j]])
                    p2t_traj_meta['tlen_'+lab].append(tlength_rec)
                    p2t_traj_meta['x0_'+lab].append(x0_rec)
                    p2t_traj_meta['xf_'+lab].append(xfinal_rec)
                    p2t_traj_meta['xdelta_'+lab].append(xdelta)
                    idx_save.append(ids[i]+'_'+str(peaks[j]))
                    p2t_trajectories[idx_save[-1]]={'t': t,
                                              'x': x,
                                              't0': t0_rec,
                                              't0_timestamp': str(t0_date),
                                              'tf': tfinal_rec,
                                              'tf_timestamp': str(tfinal_date),
                                              'id': ids[i],
                                              'peak_num': peaks[j]}

    # make sure we found at least one fit by checking whether idx_save is empty
    if idx_save:
        # save paths
        p2t_run_meta_path = savepath+lab+'_p2t_run_meta.txt'
        p2t_traj_meta_df_path = savepath+lab+'_p2t_traj_meta_df.csv'
        p2t_trajectories_path = savepath+lab+'_p2t_trajectories.npy'
        p2t_full_traj_path = savepath+lab+'_full_trajectories.npy'
        
        # save run data
        p2t_run_meta={'peak_finding_mode': 'peak_to_trough',
                      'lab': lab,
                      'num_input_ids': len(ids),
                      'num_peaks': num_peaks,
                      'num_traj_meeting_criteria': num_meeting_traj_criteria,
                      'min_peak2trough': min_peak2trough,
                      'min_peak_height': min_peak_height,
                      'prominence': prominence,
                      'screen_first_gap': screen_first_gap,
                      'screen_steady_state': screen_steady_state,
                      'screen_steady_state_cutoff': screen_steady_state_cutoff,
                      'screen_long_tail': screen_long_tail}
        # write the run data in dict form
        file1 = open(p2t_run_meta_path,'w')
        file1.write(pprint.pformat(p2t_run_meta, width=1))
        file1.close()
        
        # write p2t_traj_metadata
        p2t_traj_meta_df = pd.DataFrame(p2t_traj_meta,index=idx_save)
        p2t_traj_meta_df.to_csv(p2t_traj_meta_df_path)
        
        # write full and individual trajectories as dicts in numpy
        np.save(p2t_trajectories_path, p2t_trajectories) 
        np.save(p2t_full_traj_path, full_trajectories)
        
        return p2t_traj_meta_df, p2t_trajectories, full_trajectories, p2t_run_meta
    
    else:
        print('No trajectories meeting criteria')
        return None, None, None, None
    
    
def fit_models(traj,
               lab='ALT',
               pause_for_input=True,
               plot_figs=True,
               dtau=0,
               fit_simple_exponential=True,
               fit_sq_impulse=True,
               fit_biexponential=True,
               fit_full_heps=False,
               calculate_delta_EX=True,
               eval_C_d=False,
               eval_C=False,
               d=.42,
               d_LL=.22,
               d_UL=.62,
               specific_ids=False,
               ids=[]):
    
    # WARNINGS / ISSUES
    # - only 2x flex in the L0 param in case we implement lookback
    
    from scipy.stats import shapiro
    from scipy.stats import normaltest
    from lmfit import Model
    import matplotlib.pyplot as plt
    import lftlib
    import lftmodels
    import numpy as np
    import pandas as pd
    
    # copy traj for output
    traj_out=traj.copy()
    
    # prime the save structure
    save_dict={}
    save_dict['EMPI'] = []
    idx_save=[]
#     models_meta = ['EX','BI','FH']
    models_meta = []
    if fit_simple_exponential:
        models_meta.append('EX')
    if fit_biexponential:
        models_meta.append('BI')
    if fit_full_heps:
        models_meta.append('FH')
    if fit_sq_impulse:
        models_meta.append('SQ')
    model_params = {}
    model_params['EX'] = ['L0', 'C', 'd', 'drvBLFT']
    model_params['SQ'] = ['L0', 'C', 'k', 'tau', 'd']
    model_params['BI'] = ['L0', 'C', 'a', 'd1', 'd2']
    model_params['FH'] = ['H0', 'L0', 'bLFT', 'bHEP', 'a', 'd', 'drvK', 'drvZ','drvFracInjur']
    fit_meta = ['qerr', 'nRMSD', 'aic', 'bic', 'chisqr', 'redchi',]
    
    for model in models_meta:
        for param in model_params[model]:
            save_dict[param+'_'+model+'_'+lab]=[]
            save_dict[param+'_stderr_'+model+'_'+lab]=[]
        for fitstat in fit_meta:
            # initialize save_dict (which will become a pandas df)
            save_dict[fitstat+'_'+model+'_'+lab]=[]
        
    # these are custom error functions (ie not returned by lmfit) so we can't automatically iterate through them in a model.result, will add manually. But the save_dict is primed
    fit_meta.remove('nRMSD')
    fit_meta.remove('qerr')
    
    inp='t'
    num_traj=0
    
    if specific_ids:
        trajlist = ids
    else:
        trajlist = traj
    
    for idx in trajlist:
        # keep track of total traj, vs fit traj
        num_traj = num_traj+1
        # progress bar
        if num_traj%100==0: print(num_traj)
        # save ids as list for pandas df
        idx_save.append(idx)
        
        #also save EMPI (same as idx, except without peak assignment)
        save_dict['EMPI'].append(traj[idx]['id'])
        
        # extract x, t
        t = traj[idx]['t']
        x = traj[idx]['x']
        
        # calculate 'high res' t
        t_highres = np.linspace(0,np.ceil(t[-1]),50)
        # and save
        traj_out[idx]['t_highres'] = t_highres
        
        # dtau allows an offset for backward prediction (defaults 0, generally leave it there)
        t=t+dtau
        
        # models
        EX_has_fit=False
        
        if fit_simple_exponential:
                        
            # SIMPLE EXPONENTIAL WITH CONSTANT C HEPATOCYTE DEATH
            EX_model = Model(lftmodels.ALT_no_heps)

            #ALT_no_heps(t, L0, C, a):
            # NEW PARAMS
            EX_model.set_param_hint('L0', value=x[0], min=x[0]*.2, max=x[0]*10, vary=True)
            EX_model.set_param_hint('C', value=8, min=0, vary=True)
            EX_model.set_param_hint('d', value=.4, min=0, vary=True)
            EX_model.set_param_hint('drvBLFT', expr='C/d')

            EX_params = EX_model.make_params()

            try:
                EX_result = EX_model.fit(x, EX_params, t=t)
                EX_has_fit=True

            except:
                print('failed EXP fit:')
                print(idx)
                # store nan values in all elements
                for param in model_params['EX']:
                    save_dict[param+'_EX_'+lab].append(np.nan)
                    save_dict[param+'_stderr_EX_'+lab].append(np.nan)
                # custom errors
                save_dict['qerr_EX_'+lab].append(np.nan)
                save_dict['nRMSD_EX_'+lab].append(np.nan)
                # iterate through remainder of result.[error]
                for fitstat in fit_meta:
                    save_dict[fitstat+'_EX_'+lab].append(np.nan)

            if EX_has_fit:
                #highres solution
                x_highres_EX=EX_model.eval(EX_result.params,t=t_highres)
                # save to traj
                traj_out[idx]['x_EX'] = EX_result.best_fit
                traj_out[idx]['x_highres_EX'] = x_highres_EX

                EX_qual_error=lftlib.qualitative_error(t, x, EX_result.residual)
                EX_nRMSD=lftlib.nRMSD(t, x, EX_result.residual)

                # save fit results
                for param in model_params['EX']:
                    save_dict[param+'_EX_'+lab].append(EX_result.params[param].value)
                    save_dict[param+'_stderr_EX_'+lab].append(EX_result.params[param].stderr)
                # custom errors
                save_dict['qerr_EX_'+lab].append(EX_qual_error)
                save_dict['nRMSD_EX_'+lab].append(EX_nRMSD)
                # iterate through remainder of result.[error]
                for fitstat in fit_meta:
                    save_dict[fitstat+'_EX_'+lab].append(getattr(EX_result,fitstat))
        
        # models
        SQ_has_fit=False
        
        if fit_sq_impulse:
                        
            # SIMPLE SQPONENTIAL WITH CONSTANT C HEPATOCYTE DEATH
            SQ_model = Model(lftmodels.square_impulse)

            #ALT_no_heps(t, L0, C, a):
            # NEW PARAMS
            SQ_model.set_param_hint('L0', value=x[0], min=x[0]*.2, max=x[0]*10, vary=True)
            SQ_model.set_param_hint('C', value=8, min=0, vary=True)
            SQ_model.set_param_hint('d', value=.44, min=0, vary=True)
            SQ_model.set_param_hint('tau', value=1, min=0, max=t[-1], vary=True)
            SQ_model.set_param_hint('k', value=100, min=0, vary=True)

            SQ_params = SQ_model.make_params()

            try:
                SQ_result = SQ_model.fit(x, SQ_params, t=t)
                SQ_has_fit=True

            except:
                print('failed SQ fit with tau=1, changing tau to midpoint..')
                SQ_params['tau'].value=(t[-1]-t[0])/2
                try:
                    SQ_result = SQ_model.fit(x, SQ_params, t=t)
                    SQ_has_fit=True
                except:
                    print('failed SQ fit with tau=midpoint')
                    print(idx)
                    # store nan values in all elements
                    for param in model_params['SQ']:
                        save_dict[param+'_SQ_'+lab].append(np.nan)
                        save_dict[param+'_stderr_SQ_'+lab].append(np.nan)
                    # custom errors
                    save_dict['qerr_SQ_'+lab].append(np.nan)
                    save_dict['nRMSD_SQ_'+lab].append(np.nan)
                    # iterate through remainder of result.[error]
                    for fitstat in fit_meta:
                        save_dict[fitstat+'_SQ_'+lab].append(np.nan)

            if SQ_has_fit:
                #highres solution
                x_highres_SQ=SQ_model.eval(SQ_result.params,t=t_highres)
                # save to traj
                traj_out[idx]['x_SQ'] = SQ_result.best_fit
                traj_out[idx]['x_highres_SQ'] = x_highres_SQ

                SQ_qual_error=lftlib.qualitative_error(t, x, SQ_result.residual)
                SQ_nRMSD=lftlib.nRMSD(t, x, SQ_result.residual)

                # save fit results
                for param in model_params['SQ']:
                    save_dict[param+'_SQ_'+lab].append(SQ_result.params[param].value)
                    save_dict[param+'_stderr_SQ_'+lab].append(SQ_result.params[param].stderr)
                # custom errors
                save_dict['qerr_SQ_'+lab].append(SQ_qual_error)
                save_dict['nRMSD_SQ_'+lab].append(SQ_nRMSD)
                # iterate through remainder of result.[error]
                for fitstat in fit_meta:
                    save_dict[fitstat+'_SQ_'+lab].append(getattr(SQ_result,fitstat))
        
        
        BI_has_fit=False
                
        if fit_biexponential:

            # create the lmfit model
            BI_model = Model(lftmodels.biexponential)
            
            # NEW PARAMS
            BI_model.set_param_hint('L0', value=x[0], min=x[0]*.2, max=x[0]*10, vary=True)
            BI_model.set_param_hint('C', value=8, min=0, vary=True)
            BI_model.set_param_hint('a', value=.5, min=0, max=1, vary=True)
            BI_model.set_param_hint('d1', value=.5, min=0, vary=True)
            BI_model.set_param_hint('d2', value=2, min=0, vary=True)

            BI_params = BI_model.make_params()
            
            try:
                BI_result = BI_model.fit(x, BI_params, t=t)
                BI_has_fit=True
                
            except:
                print('failed BI fit:')
                print(idx)
                # store nan values in all elements
                for param in model_params['BI']:
                    save_dict[param+'_BI_'+lab].append(np.nan)
                    save_dict[param+'_stderr_BI_'+lab].append(np.nan)
                # custom errors
                save_dict['qerr_BI_'+lab].append(np.nan)
                save_dict['nRMSD_BI_'+lab].append(np.nan)
                # iterate through remainder of result.[error]
                for fitstat in fit_meta:
                    save_dict[fitstat+'_BI_'+lab].append(np.nan)
            
            if BI_has_fit:
                #highres solution
                x_highres_BI=BI_model.eval(BI_result.params,t=t_highres)
                # save to traj
                traj_out[idx]['x_BI'] = BI_result.best_fit
                traj_out[idx]['x_highres_BI'] = x_highres_BI
                
                BI_qual_error=lftlib.qualitative_error(t, x, BI_result.residual)
                BI_nRMSD=lftlib.nRMSD(t, x, BI_result.residual)
                
                # save results
                for param in model_params['BI']:
                    save_dict[param+'_BI_'+lab].append(BI_result.params[param].value)
                    save_dict[param+'_stderr_BI_'+lab].append(BI_result.params[param].stderr)
                # custom errors
                save_dict['qerr_BI_'+lab].append(BI_qual_error)
                save_dict['nRMSD_BI_'+lab].append(BI_nRMSD)
                # iterate through remainder of result.[error]
                for fitstat in fit_meta:
                    save_dict[fitstat+'_BI_'+lab].append(getattr(BI_result,fitstat))
        
                
        FH_has_fit=False
                
        if fit_full_heps:
            
            # heps factor
            hfactor=1/1000000

            # baseline estimate
            baseline_ALT = x[-1]
            baseline_heps = 927*107*(10**6)*hfactor

            init_d=0.3
            init_a=10**8*hfactor

            # create the lmfit model
            FH_model = Model(lftmodels.lft_traj_base)
            
            # NEW PARAMS
            FH_model.set_param_hint('H0', value=baseline_heps*.95, min=baseline_heps*.5, max=baseline_heps, vary=True)
            FH_model.set_param_hint('L0', value=x[0], min=x[0]*.5, max=x[0]*2, vary=True)
            FH_model.set_param_hint('bLFT', value=baseline_ALT, min=10, max=2*x[-1], vary=True)
            FH_model.set_param_hint('bHEP', value=baseline_heps, min=.75*baseline_heps, vary=False)
            FH_model.set_param_hint('a', value=init_a, min=init_a*.01, max=baseline_heps/30, vary=True)
            FH_model.set_param_hint('d', value=.3, min=0, vary=True)
            FH_model.set_param_hint('drvK', expr='a/bHEP')
            FH_model.set_param_hint('drvZ', expr='bLFT*d/a')
            FH_model.set_param_hint('drvFracInjur', expr='1-H0/bHEP')

            FH_params = FH_model.make_params()
            
            try:
                FH_result = FH_model.fit(x, FH_params, t=t)
                FH_has_fit=True
                
            except:
                print('failed FH fit, allowing baseline heps to vary...') 
                FH_params['bHEP'].vary=True
                try: 
                    FH_result = FH_model.fit(x, FH_params, t=t)
                    FH_has_fit=True
                except:
                    print('FH fit failed:')
                    print(idx)
                
                    # store nan values in all elements
                    for param in model_params['FH']:
                        save_dict[param+'_FH_'+lab].append(np.nan)
                        save_dict[param+'_stderr_FH_'+lab].append(np.nan)
                    # custom errors
                    save_dict['qerr_FH_'+lab].append(np.nan)
                    save_dict['nRMSD_FH_'+lab].append(np.nan)
                    # iterate through remainder of result.[error]
                    for fitstat in fit_meta:
                        save_dict[fitstat+'_FH_'+lab].append(np.nan)
            
            if FH_has_fit:
                #highres solution
                x_highres_FH=FH_model.eval(FH_result.params,t=t_highres)
                # save to traj
                traj_out[idx]['x_FH'] = FH_result.best_fit
                traj_out[idx]['x_highres_FH'] = x_highres_FH
                
                FH_qual_error=lftlib.qualitative_error(t, x, FH_result.residual)
                FH_nRMSD=lftlib.nRMSD(t, x, FH_result.residual)
                
                # save results
                for param in model_params['FH']:
                    save_dict[param+'_FH_'+lab].append(FH_result.params[param].value)
                    save_dict[param+'_stderr_FH_'+lab].append(FH_result.params[param].stderr)
                # custom errors
                save_dict['qerr_FH_'+lab].append(FH_qual_error)
                save_dict['nRMSD_FH_'+lab].append(FH_nRMSD)
                # iterate through remainder of result.[error]
                for fitstat in fit_meta:
                    save_dict[fitstat+'_FH_'+lab].append(getattr(FH_result,fitstat))
                
        if calculate_delta_EX:
            delta_EX_track=np.array([])
            delta_EX_LL_track=np.array([])
            delta_EX_UL_track=np.array([])
            # sequentially fit the delta_EX score on a rolling basis from 0:1, to n-1:n lab values
            # - record delta at each assuming input d=d
            for s in range(0,len(t)-1):
                p1=s
                pn=s+2
                delta_EX_i, LL, UL = delta_EXP(t[p1:pn],x[p1:pn],d=d, baseline_LFT=0, calc_UL_LL=True, d_UL=d_UL, d_LL=d_LL)
                delta_EX_track=np.append(delta_EX_track,delta_EX_i)
                delta_EX_LL_track=np.append(delta_EX_LL_track,LL)
                delta_EX_UL_track=np.append(delta_EX_UL_track,UL)
                
            traj_out[idx]['delta_EX'] = delta_EX_track
            traj_out[idx]['delta_EX_LL'] = delta_EX_LL_track
            traj_out[idx]['delta_EX_UL'] = delta_EX_UL_track
        
        # THREE POINT TRACK
        # no assumptions, what is C and d throughout the time course
        if eval_C_d:
            d_track=np.array([])
            C_track3=np.array([])
            # sequentially fit the 3 point model on a rolling basis from 0:2, to n-3:n lab values
            # - record d, C at each
            for s in range(0,len(t)-2):
                p1=s
                pn=s+3
                three_point = fit_2point(t[p1:pn]-t[p1],x[p1:pn],fix='False')
                d_track=np.append(d_track,three_point.params['d'].value)
                C_track3=np.append(C_track3,three_point.params['C'].value)

        # TWO POINT TRACK
        # assume d=d, what is C (and/or C/d) throughout the time course
        if eval_C:
            C_track2=np.array([])
            # sequentially fit the 2 point model on a rolling basis from 0:1, to n-1:n lab values
            # - record C at each assuming input d=d
            for s in range(0,len(t)-1):
                p1=s
                pn=s+2
                two_point = fit_2point(t[p1:pn]-t[p1],x[p1:pn],fix='d', d=d)
                C_track2=np.append(C_track2,two_point.params['C'].value)

        if plot_figs:
            print(EX_result.fit_report())
            # plt.plot(x, result.init_fit, 'k--', label='initial fit')
            plt.plot(t, EX_result.best_fit, 'rx', label='best fit')
            plt.plot(t, x, 'b.', label='data')
            plt.plot(t_highres,x_highres_EX,'--',color='tab:gray')
            # really subtle: if dates is an np.array, convert to list first otherwise it will fail. otherwise referencing is fine in this code block
#             if list(dates):
#                 t_date=dates[i]-t0_rec
#                 plt.plot(t_date, 5, 'v')
#                 plt.annotate("Point 1", (t_date, 5))

            if eval_C_d:
                # plot the 3point running C values; print the d values 
                plt.plot(t[1:-1],C_track3,'ko')
                print('d values 3-point-running:')
                print(d_track)
            if eval_C:
                # plot the 3point running C values; print the d values 
                plt.plot(t[1:],C_track2/d,'bo')
                
            plt.show()
                
    df = pd.DataFrame(save_dict,index=idx_save)
    
    return df, traj_out
