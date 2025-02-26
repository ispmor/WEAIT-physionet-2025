
def has_rsR_complex(recording, threshold=20):
    ecg, inverted = nk.ecg_invert(recording, sampling_rate=500)
    coeff_bigger_20 = []
    #For V1 with rSR there should be no inversion. If V1 is healthy inversion would occurr
    if not inverted:
        signals, info = nk.ecg_process(ecg, sampling_rate=500)
        num_peaks = len(info['ECG_R_Peaks'])
        result = []
        for i in range(num_peaks):
            r = info['ECG_R_Peaks'][i]
            r_on = info['ECG_R_Onsets'][i]
            r_off = info['ECG_R_Offsets'][i]
            p_on = info['ECG_P_Onsets'][i]
            t_off = info['ECG_T_Offsets'][i]

            if np.isnan([r, r_on, r_off, p_on, t_off]).any():
                continue
            else:
                # if signals.iloc[r_on]['ECG_Raw'] > 0:
                #     result.append(any(signals.iloc[r_on-threshold:r]['ECG_Raw'] < 0))
                # else:
                    #TODO? What if R_onset is not positive?
                    # Lets see how many times we crossed from positive to negative again, if > 0 between ron and r then its rsr
                    # pos_to_neg_changes = sum((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))<0
                    # print((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))
                pos_to_neg_changes = sum(np.diff(np.diff(signals.iloc[r_on-threshold:r]['ECG_Raw']) > 0).astype(int))
                (db_cA, db_cD) = pywt.dwt(signals.iloc[p_on: t_off]['ECG_Clean'], 'db2')
                coeff_bigger_20.append(sum(db_cD > 2.5))

                result.append(pos_to_neg_changes>5)

        return int(any(result))
    else:
        rec_clean = nk.ecg_clean(recording, method='pantompkins1985', sampling_rate=500)
        signals, info = nk.ecg_process(rec_clean, sampling_rate=500)
        num_peaks = len(info['ECG_R_Peaks'])
        result = []
        for i in range(num_peaks):
            r = info['ECG_R_Peaks'][i]
            r_on = info['ECG_R_Onsets'][i]
            r_off = info['ECG_R_Offsets'][i]
            p_on = info['ECG_P_Onsets'][i]
            t_off = info['ECG_T_Offsets'][i]
  
            if np.isnan([r, r_on, r_off, p_on, t_off]).any():
                continue
            else:
                # if signals.iloc[r_on]['ECG_Raw'] > 0:
                #     result.append(any(signals.iloc[r_on-threshold:r]['ECG_Raw'] < 0))
                # else:
                    #TODO? What if R_onset is not positive?
                    # Lets see how many times we crossed from positive to negative again, if > 0 between ron and r then its rsr
                    # pos_to_neg_changes = sum((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))<0
                    # print((np.diff(signals.iloc[r_on:r]['ECG_Raw']) > 0).astype(int))
                pos_to_neg_changes = sum(np.diff(np.diff(signals.iloc[r: r_off]['ECG_Raw']) > 0).astype(int)) 
                result.append(pos_to_neg_changes>7)
                (db_cA, db_cD) = pywt.dwt(signals.iloc[p_on: t_off]['ECG_Clean'], 'db2')
                coeff_bigger_20.append(sum(db_cD > 5) )
              
        return int(any(result))



    
def get_qrs_beginning_and_end(ecg_clean, smoothwindow=0.1, avgwindow=0.75, gradthreshweight=1.5, minlenweight=0.4, mindelay=0.3, sampling_rate=500, **kwargs):
    signal_gradient = np.gradient(ecg_clean)
    absgrad = np.abs(signal_gradient)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = nk.signal.signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = nk.signal.signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))
    qrs = smoothgrad > gradthreshold
    beg_qrs_tmp = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs_tmp = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    # Throw out QRS-ends that precede first QRS-start.
    end_qrs_tmp = end_qrs_tmp[end_qrs_tmp > beg_qrs_tmp[0]]

     # Identify R-peaks within QRS (ignore QRS that are too short).
    num_qrs = min(beg_qrs_tmp.size, end_qrs_tmp.size)
    min_len = np.mean(end_qrs_tmp[:num_qrs] - beg_qrs_tmp[:num_qrs]) * minlenweight
    beg_qrs = []
    end_qrs = []

    for i in range(num_qrs):
        beg = beg_qrs_tmp[i]
        end = end_qrs_tmp[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue
        else:
            beg_qrs.append(beg)
            end_qrs.append(end)

    return np.array(beg_qrs), np.array(end_qrs)
