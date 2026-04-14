import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, accuracy_score

def get_features(sig, time_values):
    """Extracts physical and statistical features from the time-domain signal."""
    rms = np.sqrt(np.mean(sig**2))
    kurtosis = pd.Series(sig).kurtosis()
    peak = np.max(np.abs(sig))
    p2p = np.ptp(sig)

    threshold = 0.15 * peak
    arrival = np.where(np.abs(sig) > threshold)[0]
    tof = time_values[arrival[0]] if len(arrival) > 0 else 0

    fft_vals = np.fft.rfft(sig)
    fft_freq = np.fft.rfftfreq(len(sig), d=(time_values[1]-time_values[0]))
    peak_freq = fft_freq[np.argmax(np.abs(fft_vals))]

    return pd.Series({
        'Peak': peak, 'P2P': p2p, 'RMS': rms, 'Kurtosis': kurtosis,
        'Peak_Freq': peak_freq, 'ToF': tof
    })

def main():
    print("--- [Step 1] Loading Data & Augmentation ---")
    filename = 'gpr_training_dataset_final.csv'

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'. Ensure it is in the same directory as this script.")
        return

    header_row = 0
    for i, line in enumerate(lines):
        if 'Time (s)' in line:
            header_row = i
            break

    df_wide = pd.read_csv(filename, skiprows=header_row)
    time_values = df_wide['% Time (s)'].values

    cracked_data = []
    target_point = 14

    for col in df_wide.columns:
        if f'Point: {target_point}' in col and 'crack_x=' in col:
            match = re.search(r'crack_x=([\d\.]+)', col)
            if match:
                crack_x = float(match.group(1))
                voltage = df_wide[col].values
                cracked_data.append({'signal': voltage, 'crack_x': crack_x, 'label': 1})

    cracked_data.sort(key=lambda x: x['crack_x'])
    print(f"Loaded {len(cracked_data)} Cracked Samples.")

    template_sig = cracked_data[0]['signal']
    healthy_data = []

    print("Generating robust healthy baseline (with Random Augmentation)...")
    for i in range(35):
        healed_sig = np.roll(template_sig, -5)
        amp_scale = np.random.uniform(0.90, 1.10)
        noise = np.random.normal(0, 2e-6, len(healed_sig))
        final_sig = (healed_sig * amp_scale) + noise
        healthy_data.append({'signal': final_sig, 'crack_x': 0, 'label': 0})

    print(f"Generated {len(healthy_data)} Augmented Healthy Samples.")
    all_samples = healthy_data + cracked_data
    df_all = pd.DataFrame(all_samples)

    print("\n--- [Step 2] Extracting Enhanced Features ---")
    # Apply feature extraction, passing time_values explicitly
    X = df_all['signal'].apply(lambda sig: get_features(sig, time_values))
    y_label = df_all['label']
    y_loc = df_all['crack_x']

    print("\n--- [Objective 1] Training Classifier (Random Split) ---")
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X, y_label, test_size=0.2, random_state=None)

    scaler_cls = StandardScaler()
    X_cls_train_s = scaler_cls.fit_transform(X_cls_train)
    X_cls_test_s = scaler_cls.transform(X_cls_test)

    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True, random_state=None), param_grid, cv=3)
    grid.fit(X_cls_train_s, y_cls_train)

    best_svm = grid.best_estimator_
    acc = accuracy_score(y_cls_test, best_svm.predict(X_cls_test_s))
    print(f"Optimized SVM Accuracy: {acc*100:.2f}% (Varies per run)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaler_cls.transform(X))

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y_label==1, 0], X_pca[y_label==1, 1], c='red', label='Cracked Data Distribution', edgecolors='k', alpha=0.7)
    plt.scatter(X_pca[y_label==0, 0], X_pca[y_label==0, 1], c='blue', label='Healthy Baseline', edgecolors='k', alpha=0.7)
    plt.title('Objective 1: PCA Feature Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    print(">>> Note: Close the plot window to continue script execution. <<<")
    plt.show()

    print("\n--- [Objective 2] Training GPR (Random Optimizer) ---")
    mask_crack = y_label == 1
    X_reg = X[mask_crack]
    y_reg = y_loc[mask_crack]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=None)

    scaler_reg = StandardScaler()
    X_reg_train_s = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_s = scaler_reg.transform(X_reg_test)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, normalize_y=True, random_state=None)
    gpr.fit(X_reg_train_s, y_reg_train)

    y_pred, y_std = gpr.predict(X_reg_test_s, return_std=True)
    mae = mean_absolute_error(y_reg_test, y_pred)
    print(f"Optimized GPR MAE: {mae:.2f} mm (Varies per run)")

    plt.figure(figsize=(8, 6))
    plt.errorbar(y_reg_test, y_pred, yerr=1.96*y_std, fmt='o', c='blue', ecolor='gray', label='Predictions (95% CI)')
    plt.plot([20, 180], [20, 180], 'r--', label='Ideal Perfect Fit')
    plt.title(f'Objective 2: Localization (MAE: {mae:.2f} mm)')
    plt.xlabel('Actual Crack Location (mm)')
    plt.ylabel('Predicted Crack Location (mm)')
    plt.legend()
    plt.grid(True)
    print(">>> Note: Close the plot window to continue script execution. <<<")
    plt.show()

    print("\n--- [Objective 3] Dynamic Health Monitoring Simulation ---")
    monitor_timeline = healthy_data[:10] + cracked_data[:15]
    di_values = []

    for i, sample in enumerate(monitor_timeline):
        original_sig = sample['signal']
        live_noise = np.random.normal(0, 1e-6, len(original_sig))
        live_sig = original_sig + live_noise

        feats_series = get_features(live_sig, time_values)
        feats_df = pd.DataFrame([feats_series])
        feats_s = scaler_cls.transform(feats_df)

        prob_healthy = best_svm.predict_proba(feats_s)[0][0]
        di_values.append(1 - prob_healthy)

    plt.figure(figsize=(10, 5))
    plt.plot(di_values, 'b-o', linewidth=2, label='Live Sensor Reading')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Alarm Threshold')
    plt.axvspan(0, 9.5, color='green', alpha=0.1, label='Healthy Phase')
    plt.axvspan(9.5, 25, color='red', alpha=0.1, label='Cracked Phase')

    plt.title('Objective 3: Dynamic Health Monitoring')
    plt.xlabel('Time (Measurement Sequence)')
    plt.ylabel('Damage Probability')
    plt.legend()
    plt.grid(True)
    print(">>> Note: Close the final plot window to end script. <<<")
    plt.show()

if __name__ == "__main__":
    main()