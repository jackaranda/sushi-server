from flask import Flask, request, make_response
import xarray as xr
import numpy as np

from scipy.stats import scoreatpercentile, percentileofscore

import glob

MONTH_NAMES = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

app = Flask(__name__)

@app.route("/sushi-server/dataset/<collection>/<dataset>/<varname>/<feature_id>/")
def dataset_timeseries(collection, dataset, varname, feature_id):

    timeagg = request.args.get('timeagg')
    anomaly = request.args.get('anomaly')
    startmonth = request.args.get('startmonth')
    endmonth = request.args.get('endmonth')

    print(anomaly)

    if not timeagg:
        timeagg = 'monthly'

    if not anomaly:
        anomaly = 'none'
    else:
        if anomaly not in ['absolute', 'relative', 'standardized']:
            anomaly = 'none'

    if not startmonth:
        startmonth = 1
    if not endmonth:
        endmonth = 12

    response = {}
    headers = {'Access-Control-Allow-Origin': '*'}

    try:
        startmonth = int(startmonth)
        endmonth = int(endmonth)
    except:
        response['error'] = f'Cannot convert startmonth and enddmonth to integers {startmonth} {endmonth}'
        return (response, headers)

    if startmonth > endmonth:
        months = list(range(startmonth, 13)) + list(range(1,endmonth+1))
    else:
        months = list(range(startmonth, endmonth+1))

    months = list(set(months))
    startmonth_name = MONTH_NAMES[startmonth-1]

    print('months', months, startmonth_name)
    print('anomaly', anomaly)

    try:
        feature_id = float(feature_id)
    except:
        response['error'] = f'Invalid feature_id {feature_id}'
        return (response, headers)

    path = f'../data/{collection}/{dataset}.nc'

    try:
        ds = xr.open_dataset(path)
    except:
        response['error'] = f'Failed to open datasets {dataset}'
        return (response, headers)

    variable = ds[varname]
    print(variable)

    if timeagg == 'monthly':
        times = list(variable.time.data)

    elif timeagg == 'seasonal':
        variable = variable.groupby(variable.time.dt.month).mean()
        times = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    elif timeagg == 'annual':
        date_filter = ds.time.dt.month.isin(months).data
        variable = variable.isel(time=date_filter)
        variable = variable.groupby(variable.time.dt.year).sum()
        times = list(variable.year.data)

    try:
        ids = list(ds['id'].data)
    except:
        response['error'] = f'No id variable in dataset {dataset}'
        return (response, headers)

    print(ds)
    loc = ids.index(feature_id)

    if loc < 0:
        response['error'] = f'Failed to find feature id {feature_id} in {dataset}'

    slices = [slice(None)] * len(variable.shape)
    slices[-1] = loc
    subset = variable[tuple(slices)]

    vals = subset.data

    vals_min = vals.min()
    vals_max = vals.max()

    if anomaly == 'absolute':
        vals = vals - vals.mean()
    elif anomaly == 'relative':
        vals = 100.0 * ((vals - vals.mean())/vals.mean())

    vals = [float(v) for v in list(vals)]
    times = [str(t) for t in times]

    response['vals'] = vals
    response['minval'] = float(vals_min)
    response['maxval'] = float(vals_max)
    response['times'] = times

    ds.close()

    return (response, headers)



@app.route("/sushi-server/forecast/seasonal/<model>/<fcst_year>/<fcst_month>/<varname>/<feature_id>/")
def forecast(model, fcst_year, fcst_month, varname, feature_id):

    lead = request.args.get('lead')
    obs_threshold = float(request.args.get('threshold'))

    if not (lead):
        lead = (0,5)
    else:
        lead = tuple(int(v) for v in lead.split(','))

    if not obs_threshold:
        obs_percentile = 50.0

    try:
        feature_id = float(feature_id)
    except:
        response['error'] = f'Invalid feature_id {feature_id}'
        return (response, headers)


    VARMAP = {'pr': {'name':'tprate'}}
    VARMAP['pr']['scale'] = 86400*1000*30.1

    obsfile = '../data/observed/pr_mon_CHG-CHIRPS2.0_hexgrid_p25.nc'

    obsds = xr.open_dataset(obsfile)
    obsvar = obsds[varname]
    print(obsvar)

    ids = list(obsds['id'].data)
    loc = ids.index(feature_id)
    slices = [slice(None)] * len(obsvar.shape)
    slices[-1] = loc
    obsvar = obsvar[tuple(slices)]
    

    path = f'../data/forecast/{model}/nc4_classic'

    response = {'model':model}
    headers = {'Access-Control-Allow-Origin': '*'}


    vals = {}
    bigbag = {'observed':[], 'ensemble':[]}

    for year in range(1981,2024):

        filesearch = f'{path}/{varname}_mon_{model}_{year}{fcst_month}-*.nc'
        filename = glob.glob(filesearch)[0]
        print(filename)

        print('bigbag', len(bigbag['ensemble']))

        try:
            ds = xr.open_dataset(filename)
        except:
            print('Error opening file')
            continue

        variable = ds[VARMAP[varname]['name']] * VARMAP[varname]['scale']

        try:
            ids = list(ds['id'].data)
        except:
            response['error'] = f'No id variable in dataset {dataset}'
            return (response, headers)

        loc = ids.index(feature_id)

        if loc < 0:
            response['error'] = f'Failed to find feature id {feature_id} in {dataset}'

        slices = [slice(None)] * len(variable.shape)
        slices[-1] = loc
        subset = variable[tuple(slices)]
        print('subset.dat a.shape', subset.data.shape)

        vals[year] = {}
        vals[year]['ensemble'] = subset.data[lead[0]:lead[1]+1,:].sum(axis=0).tolist()
        print(lead, np.sort(vals[year]['ensemble']))

        if year < 2023:

            bigbag['ensemble'] += vals[year]['ensemble']
            yearA = int(year)
            yearB = int(year)
            monthA = int(fcst_month)+lead[0]
            monthB = int(fcst_month)+lead[1]
            if monthB > 12:
                monthB -= 12
                yearB += 1

            startdate = f'{yearA}-{monthA:02d}-01'
            enddate = f'{yearB}-{monthB:02d}-28'

            print(startdate, enddate)

            obsvals = obsvar.loc[startdate: enddate].sum(axis=0).data
            print(startdate, enddate, obsvals)
            vals[year]['observed'] = obsvals.tolist()
            bigbag['observed'].append(vals[year]['observed'])

        ds.close()
    
    obsds.close()

    bigbag['observed'] = np.array(bigbag['observed'])
    bigbag['ensemble'] = np.array(bigbag['ensemble'])
    print(bigbag['ensemble'].shape, np.sort(bigbag['ensemble']))
        
    obs_percentile = percentileofscore(bigbag['observed'], obs_threshold)

    response['obs_threshold'] = obs_threshold
    response['obs_percentile'] = int(obs_percentile)

    print('obs_percentile', obs_percentile)

    ens_threshold = scoreatpercentile(bigbag['ensemble'], [obs_percentile])[0]

    response['ens_threshold'] = ens_threshold

    print('ens_threshold', ens_threshold)
    print((bigbag['ensemble'] < ens_threshold).sum() / bigbag['ensemble'].shape[0])

    hits_below = 0
    hits_above = 0
    misses_below = 0
    misses_above = 0
    falses_below = 0
    falses_above = 0

    # Calibrate threshold
    print('calibrating')
    prob_below = []
    prob_above = []

    for year in range(1981,2023):

        ens_vals = np.sort(vals[year]['ensemble'])

        prob_below.append(100.0 * (ens_vals < ens_threshold).sum()/len(ens_vals))
        prob_above.append(100.0 * (ens_vals >= ens_threshold).sum()/len(ens_vals))

        print('year', prob_below[-1], prob_above[-1])

    prob_below = np.array(prob_below)
    prob_above = np.array(prob_above)

    threshold_below = scoreatpercentile(prob_below, [100-obs_percentile])[0]
    threshold_above = scoreatpercentile(prob_above, [100-obs_percentile])[0]
    
    print('threshold_below', threshold_below)
    print('threshold above', threshold_above)

    for year in range(1981,2024):

        ens_vals = np.sort(vals[year]['ensemble'])
        print(year, ens_threshold)

        prob_below = 100.0 * (ens_vals < ens_threshold).sum()/len(ens_vals)
        prob_above = 100.0 * (ens_vals >= ens_threshold).sum()/len(ens_vals)

        fcst_below = prob_below > threshold_below
        fcst_above = prob_above > threshold_above

        if year < 2023:
            
            obs_below = vals[year]['observed'] < obs_threshold
            obs_above = vals[year]['observed'] >= obs_threshold

            print(year, prob_below, prob_above, fcst_below, fcst_above, vals[year]['observed'], obs_below, obs_above)

            if fcst_below and obs_below:
                hits_below += 1
            if fcst_above and obs_above:
                hits_above += 1

            if fcst_below and not obs_below:
                falses_below += 1
            if fcst_above and not obs_above:
                falses_above += 1

            if not fcst_below and obs_below:
                misses_below += 1
            if not fcst_above and obs_above:
                misses_above += 1
        
        else:
            response['prob_below'] = int(prob_below)
            response['prob_above'] = int(prob_above)
            response['fcst_below'] = str(fcst_below)
            response['fcst_above'] = str(fcst_above)

    response['hits_below'] = hits_below
    response['hits_above'] = hits_above

    response['misses_below'] = misses_below
    response['misses_above'] = misses_above

    response['falses_below'] = falses_below
    response['falses_above'] = falses_above


    return (response, headers)
