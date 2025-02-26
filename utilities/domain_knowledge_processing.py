import pywt
import neurokit2 as nk
import numpy as np
import logging
import math
logger = logging.getLogger(__name__)



def get_wavelet_orientation(onset, peak, offset):
    if peak > (onset+offset)/2 :
        return 1
    else:
        return -1

#R->L (correct) return 1, L->R (incorrect) return -1 LEAD I
def get_right_left_activation_leadI(p_complexes):
    if len(p_complexes) > 0:
        return cleanse_data_mean([get_wavelet_orientation(*px) for px in p_complexes])
    else:
        return 0

#Correct sinus return 1 (pwave positive), Extra sinus orign return -1 (pwave negative) on lead II or lead III or aVF
def get_vertical_orientation(p_complexes):
    if len(p_complexes) > 0:
        return cleanse_data_mean([get_wavelet_orientation(*px) for px in p_complexes])
    else:
        return 0

def get_p_complex(signals, info):
    num_peaks = len(info['ECG_P_Peaks'])
    result=[]
    for i in range(num_peaks):
        p_on = info['ECG_P_Onsets'][i]
        p = info['ECG_P_Peaks'][i]
        p_off = info['ECG_P_Offsets'][i]

        if np.isnan([p_on, p, p_off]).any():
            continue

        p_complex = [signals[p_on], signals[p], signals[p_off]]

        if np.isnan(p_complex).any():
            continue
        else:
            result.append(p_complex)
    return result



#          | R   |  L
#---------------------
# top-down | 1 1 | 1 -1
#---------------------
# down-top | -1 1 | -1 -1
def get_rhythm_origin(signalsI, infoI, signalsII, infoII):
    I_pcomplexes = get_p_complex(signalsI, infoI)
    II_pcomplexes = get_p_complex(signalsII, infoII)

    return get_vertical_orientation(II_pcomplexes), get_right_left_activation_leadI(I_pcomplexes)




#https://www.cmj.hr/1999/40/1/9933900.htm
def get_heart_axis(leadI_QRS, leadaVF_QRS):
    results=[]
    lenI = len(leadI_QRS)
    lenaVF = len(leadaVF_QRS)
    target_len = 0
    if lenI >= lenaVF:
        target_len = lenaVF
    else:
        target_len = lenI

    for i in range(target_len):
        aVF_QRS=leadaVF_QRS[i]
        I_QRS=leadI_QRS[i]
        altitudeAVF=sum(aVF_QRS)
        altitudeI=sum(I_QRS)
        results.append(math.degrees(math.atan2((2*altitudeAVF),(math.sqrt(3)*altitudeI))))
    return results



def get_QRS_from_lead(signal, info, window,  with_nans=True):
    r_peaks = get_window_only(info['ECG_R_Peaks'], window)
    q_peaks = get_window_only(info['ECG_Q_Peaks'], window)
    s_peaks = get_window_only(info['ECG_S_Peaks'], window)

    num_peaks = min([len(r_peaks), len(q_peaks), len(s_peaks)])
    result = []
    for i in range(num_peaks):
        Q = np.nan
        if i < len(q_peaks):
            Q = q_peaks[i]
        R = np.nan
        if i < len(r_peaks):
            R = r_peaks[i]
        S = np.nan
        if i < len(s_peaks):
            S = s_peaks[i]
        QRS_ts= [Q, R, S]
        if with_nans:
            ret = [0, 0, 0]
            if not np.isnan(Q):
                ret[0] = signal[Q]
            if not np.isnan(R):
                ret[1] = signal[R]
            if not np.isnan(S):
                ret[2] = signal[R]

            result.append(ret)
        else:
            if np.isnan(QRS_ts).any():
                continue

            QRS = [signal[Q], signal[R], signal[S]]
            if np.isnan(QRS).any():
                continue
            else:
                result.append(QRS)
    logger.debug(result)
    return result



def get_window_only(array, window):
    result = []
    for elem in array:
        if np.isnan(elem):
            result.append(elem)
        if window[0] <= elem <= window[1]:
            result.append(elem)
        if elem > window[1]:
            break

    return result


#Check if there are missing QRS complexes, if so we diagnose atrioventricular block
def has_missing_qrs(info, window):
    R_peaks = get_window_only(info['ECG_R_Peaks'], window)
    distances = np.diff(R_peaks)
    if len(distances) == 0:
        return -1
    quantile90=np.quantile(distances,0.9)
    quantile10=np.quantile(distances,0.1)
    outliers_removed=[d for d in distances if (d>quantile10 and d<quantile90)]
    if len(outliers_removed) > 0:
        mean_without_outliers = cleanse_data_mean(outliers_removed)
        is_missing_qrs = distances > (mean_without_outliers * 1.5)
        return any(is_missing_qrs)
    else:
        return -1

def get_R_distances(info, window):
    R_peaks = get_window_only(info['ECG_R_Peaks'], window)
    if len(R_peaks) == 0:
        return pad_array([])
    distances = np.diff(R_peaks)
    result = np.insert(distances, 0, R_peaks[0], axis=0)
    pad_array(result)
    return result

def has_missing_p(info, window):
    num_of_p = np.count_nonzero(~np.isnan(get_window_only(info['ECG_P_Peaks'], window)))
    num_of_beats = len(get_window_only(info['ECG_R_Peaks'], window))
    return num_of_p < (num_of_beats - 1)


def get_QRS_duration(signals, info, window, freq=500, with_nans=True):
    r_peaks = get_window_only(info['ECG_R_Peaks'], window)
    q_peaks = get_window_only(info['ECG_Q_Peaks'], window)
    s_peaks = get_window_only(info['ECG_S_Peaks'], window)
    if with_nans:
        num_peaks=len(r_peaks)
    else:
        num_peaks = min([len(r_peaks), len(q_peaks), len(s_peaks)])
    result = []
    for i in range(num_peaks):
        Q = np.nan
        if i < len(q_peaks):
            Q = q_peaks[i]
        R = np.nan
        if i < len(r_peaks):
            R = r_peaks[i]
        S = np.nan
        if i < len(s_peaks):
            S = s_peaks[i]
        if any(np.isnan([Q,R,S])) and not with_nans:
            continue
        if any(np.isnan([Q,R,S])) and with_nans:
            result.append(0)
        else:
            result.append((S-Q)/freq)

    return pad_array(result)

def get_S_duration(signals, info, window, freq=500, with_nans=True):
    r_peaks = get_window_only(info['ECG_R_Peaks'], window)
    s_peaks = get_window_only(info['ECG_S_Peaks'], window)
    num_peaks = len(r_peaks)
    result = []
    for i in range(num_peaks):
        s = info['ECG_S_Peaks'][i]
        r = info['ECG_R_Peaks'][i]
        if any(np.isnan([r,s])) and not with_nans:
            continue
        if any(np.isnan([r,s])) and with_nans:
            result.append(0)
        else:
            result.append((s-r)/freq)

    return pad_array(result)


def get_R_duration(signals, info, window, freq=500, with_nans=True):
    r_peaks = get_window_only(info['ECG_R_Peaks'], window)
    r_ons = get_window_only(info['ECG_R_Onsets'], window)
    r_offs = get_window_only(info['ECG_R_Offsets'], window)
    s_peaks = get_window_only(info['ECG_S_Peaks'], window)
    num_peaks = min([len(r_peaks),len(r_ons),len(r_offs), len(s_peaks)])
    result = []
    for i in range(num_peaks):
        R = r_peaks[i]
        R_on = r_ons[i]
        R_off = r_offs[i]
        S = s_peaks[i]
        if any(np.isnan([R,R_on, R_off, S])) and not with_nans:
            continue
        if any(np.isnan([R,R_on, R_off, S])) and with_nans:
            result.append(0)
        else:
            if S < R_off:
                result.append((S - R_on)/freq)
            else:
                result.append((R_off - R_on)/freq)

    return pad_array(result)

def sokolov_lyons_index(V1_info, V1_signal, V5_info, V5_signal, window):
    if "ECG_S_Peaks" in V1_info and "ECG_R_Peaks" in V5_info:
        V1_QRS = get_QRS_from_lead(V1_signal, V1_info, window, with_nans=True)
        V5_QRS = get_QRS_from_lead(V5_signal, V5_info, window,  with_nans=True)
        default_length = len(V1_QRS)
        if len(V5_QRS) < default_length:
            default_length=len(V5_QRS)
        return pad_array([abs(V1_QRS[i][2]) + abs(V5_QRS[i][1]) for i in range(default_length)])
    else:
        default_length = len(V5_info['ECG_R_Peaks'])
        if len(V1_info['ECG_R_Peaks']) < default_length:
            default_length=len(V1_info['ECG_R_Peaks'])
        return pad_array(np.zeros(default_length))

def romhilt_index(V2_info, V2_signal, V5_info, V5_signal, window):
    if "ECG_S_Peaks" in V2_info and "ECG_R_Peaks" in V5_info:
        V2_QRS = get_QRS_from_lead(V2_signal, V2_info, window, with_nans=True)
        V5_QRS = get_QRS_from_lead(V5_signal, V5_info, window, with_nans=True)
        default_length = len(V2_QRS)
        if len(V5_QRS) < default_length:
            default_length=len(V5_QRS)
        return pad_array([abs(V2_QRS[i][2]) + abs(V5_QRS[i][1]) for i in range(default_length)])
    else:
        default_length = len(V5_info['ECG_R_Peaks'])
        if len(V2_info['ECG_R_Peaks']) < default_length:
            default_length=len(V2_info['ECG_R_Peaks'])
        return pad_array(np.zeros(default_length))



def cornells_index(V3_info, V3_signal, aVL_info, aVL_signal, window):
    if "ECG_S_Peaks" in V3_info and "ECG_R_Peaks" in aVL_info:
        V3_QRS = get_QRS_from_lead(V3_signal, V3_info, window, with_nans=True)
        aVL_QRS = get_QRS_from_lead(aVL_signal, aVL_info, window, with_nans=True)
        default_length = len(V3_QRS)
        if len(aVL_QRS) < default_length:
            default_length=len(aVL_QRS)
        return pad_array([abs(V3_QRS[i][2]) + abs(aVL_QRS[i][1]) for i in range(default_length)])
    else:
        default_length = len(aVL_info['ECG_R_Peaks'])
        if len(V3_info['ECG_R_Peaks']) < default_length:
            default_length=len(V3_info['ECG_R_Peaks'])
        return pad_array(np.zeros(default_length))

def cornells_product(V3_info, V3_signal, aVL_info, aVL_signal, window):
    voltages=cornells_index(V3_info, V3_signal, aVL_info, aVL_signal, window)
    durations=get_QRS_duration(aVL_signal, aVL_info,window, with_nans=True)
    default_length = len(voltages)
    if len(durations) < default_length:
        default_length=len(durations)
    product = [voltages[i] * durations[i] for i in range(default_length)]
    return pad_array(product)



def lewis_index(III_info, III_signal, I_info, I_signal, window):
    if "ECG_S_Peaks" in III_info and "ECG_R_Peaks" in I_info:
        III_QRS = get_QRS_from_lead(III_signal, III_info, window, with_nans=True)
        I_QRS = get_QRS_from_lead(I_signal, I_info, window, with_nans=True)
        default_length = len(III_QRS)
        if len(I_QRS) < default_length:
            default_length=len(I_QRS)
        return pad_array([(III_QRS[i][2] - I_QRS[i][2]) + (I_QRS[i][1]-III_QRS[i][1]) for i in range(default_length)])
    else:
        default_length = len(I_info['ECG_R_Peaks'])
        if len(III_info['ECG_R_Peaks']) < default_length:
            default_length=len(III_info['ECG_R_Peaks'])
        return pad_array(np.zeros(default_length))

def get_max_param_from_lead(signal, info, param, window):
    qrs = get_QRS_from_lead(signal, info, window, with_nans=True)

    if len(qrs) == 0:
        return -1000000000
    if param == "q":
        return abs(max([elem[0] for elem in qrs], key=abs))
    if param == "r":
        return abs(max([elem[1] for elem in qrs], key=abs))
    if param == "s":
        return abs(max([elem[2] for elem in qrs], key=abs))



def mcphie_index(signals, infos, window):
    max_r = -1000000000
    max_s = -1000000000
    for signal, info in zip(signals, infos):
        tmp_r = get_max_param_from_lead(signal, info, "r", window)
        tmp_s = get_max_param_from_lead(signal, info, "s", window)
        if tmp_r > max_r:
            max_r = tmp_r
        if tmp_s > max_s:
            max_s = tmp_s

    result = max_r + max_s

    if result < 0:
        result=0

    return result


def get_0_crossings(biorcD, beg_qrs, end_qrs, threshold=15, show=False, **kwargs):
    bior_qrs_beg = beg_qrs // 2
    bior_qrs_end = end_qrs // 2

    biorcD_widnows_beat = [biorcD[bior_qrs_beg[i]:bior_qrs_end[i]] for i in range(len(bior_qrs_beg)) if bior_qrs_end[i] - bior_qrs_beg[i] > threshold ]
    biorcD_min_max_beat = []

    for wind in biorcD_widnows_beat :
        argmin = np.argmin(wind)
        argmax = np.argmax(wind)
        if argmax > argmin:
            biorcD_min_max_beat.append(wind[argmin:argmax])
        else:
            biorcD_min_max_beat.append(wind[argmax:argmin])

    crossing_0 = [((window[:-1] * window[1:]) < 0).sum() for window in biorcD_min_max_beat]

    if show:
        figure, axis = plt.subplots(5, 1)
        figure.set_size_inches(24,10)
        for i in range(len(biorcD_widnows_beat)):
            beat = biorcD_widnows_beat[i]
            x= range(len(beat))
            axis[i].bar(x, beat)
            axis[i].bar(x, np.zeros(len(x)), linestyle='dashed')

    return pad_array(crossing_0)




def analyse_notched_signal(signal, info, recording, peaks, window, threshold=1.5,  **kwargs):
    #list_of_qrs = get_QRS_from_lead(signal, info) #get_qrs_beginning_and_end(signal['ECG_Raw'], **kwargs)
    peaks = get_window_only(peaks, window)
    N = len(recording)

    list_of_qrs = []
    window = 150
    for peak in peaks:
        if peak < (window // 2) + 1:
            list_of_qrs.append([0, window])
        elif peak > ( N - window//2):
            list_of_qrs.append([N-window-1,N-1])
        else:
            list_of_qrs.append([(peak-window//2), (peak+window//2)])

   # list_of_qrs = [[info[beginning_key][i], info[beginning_key][i]] for i in len(info['ECG_R_Peaks'])]
    logger.debug(f"List of qrs for zero crossings: {list_of_qrs}")
    if len(list_of_qrs) ==0:
        return pad_array(list_of_qrs)

    list_of_qrs = np.array(list_of_qrs)
    logger.debug(f"Shape of qrs for zero crossings: {list_of_qrs.shape}")

    beg_qrs = list_of_qrs[:, 0]
    end_qrs = list_of_qrs[:, 1]
    (_, cD) = pywt.dwt(recording, 'bior1.1')
    crossing_0 = get_0_crossings(cD, beg_qrs, end_qrs, **kwargs)
    if len(crossing_0) > 0:
        return crossing_0
    else:
        return pad(crossing_0)






def cleanse_data_mean(array):
    if len(array) > 0 and array is not None:
        result = np.nan_to_num(array, posinf=99999, neginf=-99999)
        return  np.mean(result)
    else:
        return -1


def check_for_lead(lead_name, leads_idxs, analysed_results) -> bool:
    return lead_name in leads_idxs and analysed_results[lead_name]['info'] is not None


def pad_array(array, mhb=25):
    if len(array)>mhb:
        return array[:mhb]

    return np.pad(array, (0, mhb - len(array)), 'constant', constant_values=(0, 0))


def analyse_recording(rec, signals, infos, rates, leads_idxs,window=None, pantompkins_peaks=None, label=None,  sampling_rate=500):
    #assuming max bpm = 500 it gives us max 25 heart beats within 3s window
    mhb = 25
    logger.debug("Entering analysed_results")
    analysed_results = {}
    precordial_leads=[]
    for lead_name, idx in leads_idxs.items():
        signal = signals[lead_name]
        info = infos[lead_name] #nk.ecg_process(rec[idx], sampling_rate=sampling_rate)
        if signal is None or info is None:
            analysed_results[lead_name]={
                'signal': rec[idx],
                'info': None,
                'bpm': 0,
                'missing_qrs': -1,
                'missing_p': -1,
                'qrs_duration': pad_array([]),
                's_duration': pad_array([]),
                'notched': pad_array([]),
                'r_distances': pad_array([]),
                'r_distances_max':-1,
                'r_distances_min':-1,
                'romhilt': pad_array([]),
                'conrell':pad_array([]),
                'lewis':pad_array([]),
                'cornell-product':pad_array([]),
                'mcphie':0,
                'sokolov-lyon':pad_array([]),
                'heart_axis':pad_array([]),
                'rhythm_origin_vertical':0,
                'rhythm_origin_horizontal':0,
            }
            continue

        logger.debug(f"Peaks from pantompkings: {pantompkins_peaks}, Peaks from neurokit2: {len(info['ECG_R_Peaks'])}")

        if "V" in lead_name:
            precordial_leads.append(lead_name)
        bpm = -1
        if lead_name in rates:
            bpm = cleanse_data_mean(rates[lead_name])
        missing_qrs = has_missing_qrs(info, window)
        missing_p = has_missing_p( info, window)

        qrs_duration = get_QRS_duration(signal, info, window, with_nans=True)
        s_duration = get_S_duration(signal, info, window, with_nans=True)
        notched = analyse_notched_signal(signal,info, rec[idx], pantompkins_peaks, window)

        r_distances = pad_array(get_R_distances(info, window), mhb).tolist()
        r_dist_on_zero =[ dist for dist in r_distances if dist != 0]
        r_dist_min = -1
        r_dist_max = -1
        if len(r_dist_on_zero) > 0:
            r_dist_min = min(r_dist_on_zero)
            r_dist_max = max(r_dist_on_zero)

        analysed_results[lead_name]={
            'signal': rec[idx],
            'info': info,
            'bpm': bpm,
            'missing_qrs':missing_qrs,
            'missing_p': missing_p,
            'qrs_duration':qrs_duration,
            's_duration':s_duration,
            'notched': notched,
            'r_distances': r_distances,
            'r_distances_max': r_dist_max,
            'r_distances_min': r_dist_min,
        }

    #CrossLead indicators

    heart_axis = None
    rhythm_origin = None
    if check_for_lead('I',leads_idxs,analysed_results):
        if check_for_lead('II',leads_idxs,analysed_results):
            rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['II']['signal'], analysed_results['II']['info'])
        if check_for_lead('aVF',leads_idxs,analysed_results):
            if 'II' not in leads_idxs:
                rhythm_origin = get_rhythm_origin(analysed_results['I']['signal'], analysed_results['I']['info'], analysed_results['aVF']['signal'], analysed_results['aVF']['info'])
            heart_axis = pad_array(get_heart_axis(get_QRS_from_lead(analysed_results['I']['signal'], analysed_results['I']['info'], window, with_nans=True), get_QRS_from_lead(analysed_results['aVF']['signal'], analysed_results['aVF']['info'], window, with_nans=True)))



    #Lewis Index
    if check_for_lead('I', leads_idxs, analysed_results) and check_for_lead('III', leads_idxs, analysed_results):
        analysed_results['lewis']=lewis_index(analysed_results['III']['info'], analysed_results['III']['signal'], analysed_results['I']['info'], analysed_results['I']['signal'], window)
    else:
        analysed_results['lewis']=pad_array([])


    #McPhie
    precordial_signals=[analysed_results[l]['signal'] for l in precordial_leads]
    precordial_infos=[analysed_results[l]['info'] for l in precordial_leads]
    if len(precordial_leads) > 0:
        analysed_results['mcphie']=mcphie_index(precordial_signals, precordial_infos, window)
    else:
        analysed_results['mcphie']=0

    #Sokolov-Lyon
    if check_for_lead('V1', leads_idxs, analysed_results) and check_for_lead('V5', leads_idxs, analysed_results):
        analysed_results['sokolov-lyon']=sokolov_lyons_index(analysed_results['V1']['info'], analysed_results['V1']['signal'], analysed_results['V5']['info'], analysed_results['V5']['signal'], window)
    else:
        analysed_results['sokolov-lyon']=pad_array([])

    #Cornell
    if check_for_lead('V3', leads_idxs, analysed_results) and check_for_lead('aVL', leads_idxs, analysed_results):
        analysed_results['cornell']=cornells_index(analysed_results['V3']['info'], analysed_results['V3']['signal'], analysed_results['aVL']['info'], analysed_results['aVL']['signal'], window)
        analysed_results['cornell-product']=cornells_product(analysed_results['V3']['info'], analysed_results['V3']['signal'], analysed_results['aVL']['info'], analysed_results['aVL']['signal'], window)
    else:
        analysed_results['cornell']=pad_array([])
        analysed_results['cornell-product']=pad_array([])

    #Romhilt
    if check_for_lead('V2', leads_idxs, analysed_results) and check_for_lead('V5', leads_idxs, analysed_results):
        analysed_results['romhilt']=sokolov_lyons_index(analysed_results['V2']['info'], analysed_results['V2']['signal'], analysed_results['V5']['info'], analysed_results['V5']['signal'], window)
    else:
        analysed_results['romhilt']=pad_array([])


    if heart_axis is not None:
        analysed_results['heart_axis']=heart_axis
    else:
        analysed_results['heart_axis']=pad_array([])

    if rhythm_origin:
        analysed_results['rhythm_origin_vertical']=rhythm_origin[0]
        analysed_results['rhythm_origin_horizontal']=rhythm_origin[1]
    else:
        analysed_results['rhythm_origin_vertical']=0
        analysed_results['rhythm_origin_horizontal']=0


    analysed_results['label'] = label

    for lead_name, idx in leads_idxs.items():
        analysed_results[lead_name].pop('signal', None)

    return analysed_results

def analysis_dict_to_array(analysis_dict, leads_idxs):
    result = []
    logger.debug(analysis_dict)
    per_lead_parameters = ['bpm', 'missing_qrs', 'missing_p', 'qrs_duration', 's_duration', 'notched', 'r_distances']
    if len(leads_idxs.keys()) == 12:
        cross_lead_parameters = ['heart_axis','rhythm_origin_vertical','rhythm_origin_horizontal', 'romhilt', 'cornell', 'cornell-product', 'sokolov-lyon', 'mcphie', 'lewis']
    elif len(leads_idxs.keys()) == 6:
        cross_lead_parameters = ['heart_axis','rhythm_origin_vertical','rhythm_origin_horizontal', 'lewis']
    elif len(leads_idxs.keys()) == 4:
        cross_lead_parameters = ['heart_axis','rhythm_origin_vertical','rhythm_origin_horizontal']
    else:
        cross_lead_parameters = ['rhythm_origin_vertical','rhythm_origin_horizontal']

    for lead_name, idx in leads_idxs.items():
        tmp_result = []
        for key in per_lead_parameters:
            if key in analysis_dict[lead_name]:
                try:
                    if type(analysis_dict[lead_name][key]) == list:
                        tmp_result.extend(analysis_dict[lead_name][key])
                    elif isinstance(analysis_dict[lead_name][key], np.ndarray):
                        tmp_result.extend(analysis_dict[lead_name][key].tolist())
                    else:
                        tmp_result.append(analysis_dict[lead_name][key])
                except Exception as e:
                    logger.error(f"Key {key}, lead {lead_name}  from result: {analysis_dict[lead_name][key]}")
                    raise e
            else:
                raise Exception(f"No key {key} in results dict")
        for key in cross_lead_parameters:
            if key in analysis_dict:
                try:
                    if type(analysis_dict[key]) == list:
                        tmp_result.extend(analysis_dict[key])
                    elif isinstance(analysis_dict[key], np.ndarray):
                        tmp_result.extend(analysis_dict[key].tolist())
                    else:
                        tmp_result.append(analysis_dict[key])
                except Exception as e:
                    logger.error(f"Key {key}, array from result: {analysis_dict[key]}")
                    raise e
            else:
                logger.warn(f"No key {key} in results dict")
                tmp_result.append(0)
        result.append(tmp_result)
    logger.debug(f"Result now: {result}")
    logger.debug(f"Results length = {len(result)}")
    for i,x in enumerate(result):
        logger.debug(f"Results[i] length = {len(x)}")
 
    return np.array(result, dtype=np.float64)

