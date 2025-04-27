import wfdb
import numpy as np
from matplotlib import pyplot as plt
from helper_code import *
from utilities import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

pd.options.display.max_rows = 999

data_folder = "data/micro_code_sami/"
print('Finding the Challenge data...')
records = find_records(data_folder)
num_records = len(records)

if num_records == 0:
    raise FileNotFoundError('No data were provided.')

utilObject = UtilityFunctions("cpu")
sampling_rate = 400
leads_idxs = {'I': 0, 'II': 1, 'III':2, 'aVR': 3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11}

code15_peaks_values_negative = []
code15_peaks_values_positive = []

sami_peaks_values_negative = []
sami_peaks_values_positive = []

other_peaks_values_negative = []
other_peaks_values_positive = []

# Iterate over the records.
for i in range(num_records):
    print(f"{i+1}/{num_records}")
    recording_file = os.path.join(data_folder, records[i])
    header_file = os.path.join(data_folder, get_header_file(records[i]))
    header = load_header(recording_file)
    try:
        current_label= get_label(header)
    except Exception as e:
        print("Failed to load label, assigning 0")
        current_label = 0

    try:
        signal, fields = load_signals(recording_file)
    except Exception as e:
        print(f"Skipping {header_file} and associated recording  because of {e}")
        continue

    recording_full = utilObject.load_and_equalize_recording(signal, fields, header_file, sampling_rate)


    
    leads_peaks_values = []
    for lead_name, idx in leads_idxs.items():
        signal, info = None, None
        try:
            rpeaks = nk.ecg_findpeaks(recording_full[idx], sampling_rate, method="pantompkins1985")
            signal, info =nk.ecg_delineate(recording_full[idx], rpeaks=rpeaks, sampling_rate=sampling_rate, method='dwt')
            info.update(rpeaks)
        except Exception as e:
            print(f"Exception in {header_file}") 
            
        points = {}
        for point in ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']:
            if info is not None and info[point] is not None :
                indices = [indice for indice in info[point] if not np.isnan(indice)]
                values = recording_full[idx][indices]  # Original signal, not cleaned
                points[point] = values
            else:
                points[point] = np.array([])
        
        leads_peaks_values.append(points)


    
    splitted = header.split("\n")
    source_info = [x for x in splitted if "Source" in x]
    if len(source_info) > 0:
        if "SaMi-Trop" in source_info[0]:
            if current_label==0:
                sami_peaks_values_negtive.append(leads_peaks_values)
            else:
                sami_peaks_values_positive.append(leads_peaks_values)
        elif "CODE" in source_info[0]:
            if current_label==0:
                code15_peaks_values_negative.append(leads_peaks_values)
            else:
                code15_peaks_values_positive.append(leads_peaks_values)
        else:
            if current_label==0:
                other_peaks_values_negtive.append(leads_peaks_values)
            else:
                other_peaks_values_positive.append(leads_peaks_values)    







# ======================================
# Step 1: Organize data into one big DataFrame
# ======================================
def create_dataframe(all_peaks_values, label):
    rows = []
    for sample_idx, sample in enumerate(all_peaks_values):
        for lead_idx, lead_dict in enumerate(sample):
            for wave_name, values in lead_dict.items():
                for v in values:
                    rows.append({
                        "Sample": sample_idx,
                        "Lead": lead_idx,
                        "Wave": wave_name.replace('ECG_', '').replace('_Peaks', ''),
                        "Amplitude": v,
                        "Class": label
                    })
    return pd.DataFrame(rows)

def analyse(positive_class_dataset, negative_class_dataset, experiment_name):
    save_path = "plots/data_analysis/"

    # Build DataFrames for both classes
    df_pos = create_dataframe(positive_class_dataset, label=True)
    df_neg = create_dataframe(negative_class_dataset, label=False)
    
    # Concatenate into one DataFrame
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    print("Dataframe head:")
    print(df.head())
    
    # ======================================
    # Step 2: Descriptive statistics
    # ======================================
    print("\n=== Descriptive statistics by Wave and Class ===")
    desc_stats_wave_class = df.groupby(["Wave", "Class"])["Amplitude"].describe()
    print(desc_stats_wave_class)
    
    print("\n=== Descriptive statistics by Lead, Wave and Class ===")
    desc_stats_lead_wave_class = df.groupby(["Lead", "Wave", "Class"])["Amplitude"].describe()
    print(desc_stats_lead_wave_class)
    
    # ======================================
    # Step 3: Percentile analysis
    # ======================================
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    
    print("\n=== Percentiles by Wave and Class ===")
    percentile_wave_class = df.groupby(["Wave", "Class"])["Amplitude"].quantile(np.array(percentiles)/100).unstack()
    print(percentile_wave_class)
    
    print("\n=== Percentiles by Lead, Wave and Class ===")
    percentile_lead_wave_class = df.groupby(["Lead", "Wave", "Class"])["Amplitude"].quantile(np.array(percentiles)/100).unstack()
    print(percentile_lead_wave_class)
    
    # ======================================
    # Step 4: EDA Global - Compare Classes
    # ======================================
    
    # 4a. Histograms by Class
    plt.figure(figsize=(15, 8))
    sns.histplot(data=df, x="Amplitude", hue="Class", element="step", kde=True, stat="density", common_norm=False, palette="Set1")
    plt.title("Global Amplitude Distribution - Positive vs Negative Classes")
    plt.xlabel("Amplitude")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"{experiment_name}_global_histogram_classes.png"), bbox_inches="tight")
    plt.close()
    
    # 4b. Boxplot by Class
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df, x="Wave", y="Amplitude", hue="Class", palette="Set2")
    plt.title("Boxplot of Amplitudes by ECG Wave - Positive vs Negative Classes")
    plt.grid(True)
    plt.legend(title="Class", labels=["Negative", "Positive"])
    plt.savefig(os.path.join(save_path, f"{experiment_name}_boxplot_wave_classes.png"), bbox_inches="tight")
    plt.close()
    
    # 4c. Violin Plot by Class
    plt.figure(figsize=(15, 6))
    sns.violinplot(data=df, x="Wave", y="Amplitude", hue="Class", split=True, inner="quartile", palette="Set2")
    plt.title("Violin Plot of Amplitudes by ECG Wave - Positive vs Negative Classes")
    plt.grid(True)
    plt.legend(title="Class", labels=["Negative", "Positive"])
    plt.savefig(os.path.join(save_path, f"{experiment_name}_violinplot_wave_classes.png"), bbox_inches="tight")
    plt.close()
    
    # 4d. Standardized Strip Plot
    scaler = StandardScaler()
    df['Amplitude_Standardized'] = scaler.fit_transform(df[['Amplitude']])
    
    plt.figure(figsize=(15, 6))
    sns.stripplot(data=df, x="Wave", y="Amplitude_Standardized", hue="Class", dodge=True, jitter=True, alpha=0.5, palette="Set1")
    plt.title("Standardized Amplitudes - Stripplot by ECG Wave and Class")
    plt.grid(True)
    plt.legend(title="Class", labels=["Negative", "Positive"])
    plt.savefig(os.path.join(save_path, f"{experiment_name}_stripplot_wave_classes.png"), bbox_inches="tight")
    plt.close()
    
    # ======================================
    # Step 5: Per-Lead EDA - Compare Classes
    # ======================================
    
    # 5a. Per-Lead Boxplots
    plt.figure(figsize=(20, 12))
    sns.boxplot(data=df, x="Wave", y="Amplitude", hue="Class", palette="Set3")
    plt.title("Boxplot of Amplitudes by ECG Wave and Lead - Class Comparison")
    plt.grid(True)
    plt.legend(title="Class", labels=["Negative", "Positive"])
    plt.savefig(os.path.join(save_path, f"{experiment_name}_boxplot_wave_lead_classes.png"), bbox_inches="tight")
    plt.close()
    
    # 5b. Per-Lead Violin plots
    g = sns.catplot(
        data=df, kind="violin", x="Wave", y="Amplitude", hue="Class", split=True,
        col="Lead", col_wrap=4, height=4, aspect=1, palette="Set2", sharey=False
    )
    g.fig.suptitle("Violin Plots by Lead - Positive vs Negative Classes", y=1.02)
    plt.grid(True)
    g.savefig(os.path.join(save_path, f"{experiment_name}_violinplot_per_lead_classes.png"), bbox_inches="tight")
    plt.close()
    
    # 5c. Histograms per Lead
    for lead in sorted(df['Lead'].unique()):
        plt.figure(figsize=(15, 6))
        sns.histplot(
            data=df[df['Lead'] == lead], x="Amplitude", hue="Class", element="step",
            kde=True, stat="density", common_norm=False, palette="Set1"
        )
        plt.title(f"Amplitude Distribution for Each Class - Lead {lead}")
        plt.xlabel("Amplitude")
        plt.ylabel("Density")
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"histogram_lead_{lead}.png"), bbox_inches="tight")
        plt.close()
    
    # 5d. Standardized Scatter by Lead
    for lead in sorted(df['Lead'].unique()):
        plt.figure(figsize=(15, 6))
        sns.stripplot(
            data=df[df['Lead'] == lead],
            x="Wave", y="Amplitude_Standardized", hue="Class",
            dodge=True, jitter=True, alpha=0.5, palette="Set1"
        )
        plt.title(f"Standardized Amplitudes - Scatter by Wave and Class - Lead {lead}")
        plt.grid(True)
        plt.legend(title="Class", labels=["Negative", "Positive"])
        plt.savefig(os.path.join(save_path, f"stripplot_lead_{lead}.png"), bbox_inches="tight")
        plt.close()


analyse(sami_peaks_values_positive, code15_peaks_values_negative, "sami_vs_code_neg")
analyse(code15_peaks_values_positive, code15_peaks_values_negative, "code_vs_code_neg")