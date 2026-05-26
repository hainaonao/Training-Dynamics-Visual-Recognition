import os
import glob
import numpy as np
import pandas as pd

def run_prospective_volatility_test(csv_path, psi_threshold=0.35, window=3):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    
    # Compute rolling standard deviation (Ψ-volatility)
    df['Psi_volatility'] = df['Psi'].rolling(window=window).std()
    
    # Simulate prospective early stopping trigger
    psi_stop_epoch = None
    consecutive_count = 0
    
    for idx, row in df.iterrows():
        vol = row['Psi_volatility']
        if not np.isnan(vol) and vol < psi_threshold:
            consecutive_count += 1
        else:
            consecutive_count = 0
            
        if consecutive_count >= window:
            psi_stop_epoch = int(row['epoch'])
            break
            
    actual_stop_epoch = int(df['epoch'].max())
    max_val_acc = df['val_acc'].max()
    
    if psi_stop_epoch is not None:
        acc_at_psi_stop = df[df['epoch'] == psi_stop_epoch]['val_acc'].values[0]
        epochs_saved = actual_stop_epoch - psi_stop_epoch
        acc_delta = max_val_acc - acc_at_psi_stop
        status = "TRIGGERED"
    else:
        psi_stop_epoch = actual_stop_epoch
        acc_at_psi_stop = df['val_acc'].iloc[-1]
        epochs_saved = 0
        acc_delta = max_val_acc - acc_at_psi_stop
        status = "FAILED"
        
    return {
        "filename": filename,
        "status": status,
        "actual_stop_epoch": actual_stop_epoch,
        "psi_stop_epoch": psi_stop_epoch,
        "epochs_saved": epochs_saved,
        "acc_at_psi_stop": acc_at_psi_stop,
        "acc_delta_vs_max": acc_delta
    }

def main():
    PSI_THRESHOLD = 0.35  
    WINDOW_SIZE = 3       
    
    # Auto-scan directory for logs files
    csv_files = glob.glob("results_*.csv")
    results_list = []
    for file_path in csv_files:
        res = run_prospective_volatility_test(file_path, psi_threshold=PSI_THRESHOLD, window=WINDOW_SIZE)
        if res:
            results_list.append(res)
            
    # Print clean summary table
    print("=" * 115)
    print(f"🔬 PROSPECTIVE THRESHOLD EXPERIMENT (Rule: Stop if Rolling_Std(Psi, w={WINDOW_SIZE}) < {PSI_THRESHOLD})")
    print("=" * 115)
    print(f"{'Log File Name':<32} | {'Status':<11} | {'Stop Epoch (Loss)':<18} | {'Stop Epoch (Ψ)':<14} | {'Saved Ep.':<10} | {'Acc at Ψ (%)':<14} | {'Δ Acc vs Max':<12}")
    print("-" * 115)
    
    for r in results_list:
        print(f"{r['filename']:<32} | {r['status']:<11} | {r['actual_stop_epoch']:<18d} | {r['psi_stop_epoch']:<14d} | {r['epochs_saved']:<10d} | {r['acc_at_psi_stop']:<14.2f} | {r['acc_delta_vs_max']:<12.2f}")
        
    print("=" * 115)

if __name__ == '__main__':
    main()