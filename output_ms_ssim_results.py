"""
Output MS-SSIM evaluation results from CSV files
Reads bpp, MS-SSIM, and PSNR values and displays in a formatted table
"""

import pandas as pd
import os

# Define paths to evaluation results
RESULTS_DIR = "CA_Entropy_Model/Evaluation Results/[MS-SSIM optimized]"

def load_results():
    """Load evaluation results from CSV files"""
    bpp_file = os.path.join(RESULTS_DIR, "bpp.csv")
    msssim_file = os.path.join(RESULTS_DIR, "MS-SSIM.csv")
    psnr_file = os.path.join(RESULTS_DIR, "PSNR.csv")
    
    # Read CSV files
    bpp_df = pd.read_csv(bpp_file, index_col=0)
    msssim_df = pd.read_csv(msssim_file, index_col=0)
    psnr_df = pd.read_csv(psnr_file, index_col=0)
    
    return bpp_df, msssim_df, psnr_df

def print_summary_stats(bpp_df, msssim_df, psnr_df):
    """Print summary statistics for all metrics"""
    print("=" * 80)
    print("MS-SSIM OPTIMIZED MODEL - EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n### BITS PER PIXEL (BPP) ###")
    print("\nAverage BPP by Lambda:")
    bpp_means = bpp_df.mean()
    print(bpp_means)
    
    print("\nAverage BPP by Image:")
    print(bpp_df.mean(axis=1))
    
    print("\n### MS-SSIM ###")
    print("\nAverage MS-SSIM by Lambda:")
    msssim_means = msssim_df.mean()
    print(msssim_means)
    
    print("\nAverage MS-SSIM by Image:")
    print(msssim_df.mean(axis=1))
    
    print("\n### PSNR (dB) ###")
    print("\nAverage PSNR by Lambda:")
    psnr_means = psnr_df.mean()
    print(psnr_means)
    
    print("\nAverage PSNR by Image:")
    print(psnr_df.mean(axis=1))

def print_detailed_results(bpp_df, msssim_df, psnr_df):
    """Print detailed results for each lambda value"""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY LAMBDA VALUE")
    print("=" * 80)
    
    lambdas = bpp_df.columns
    
    for lam in lambdas:
        print(f"\n--- Lambda = {lam} ---")
        print(f"\n{'Image':<12} {'BPP':<12} {'MS-SSIM':<12} {'PSNR (dB)':<12}")
        print("-" * 48)
        
        for image in bpp_df.index:
            bpp_val = bpp_df.loc[image, lam]
            msssim_val = msssim_df.loc[image, lam]
            psnr_val = psnr_df.loc[image, lam]
            print(f"{image:<12} {bpp_val:<12.6f} {msssim_val:<12.6f} {psnr_val:<12.6f}")

def export_combined_results(bpp_df, msssim_df, psnr_df):
    """Export combined results to a single CSV for easy analysis"""
    output_file = "ms_ssim_combined_results.csv"
    
    # Create combined dataframe by lambda
    combined_data = []
    
    for lam in bpp_df.columns:
        for image in bpp_df.index:
            combined_data.append({
                'Lambda': lam,
                'Image': image,
                'BPP': bpp_df.loc[image, lam],
                'MS-SSIM': msssim_df.loc[image, lam],
                'PSNR': psnr_df.loc[image, lam]
            })
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined results exported to: {output_file}")

def main():
    """Main function"""
    try:
        bpp_df, msssim_df, psnr_df = load_results()
        
        # Print summary statistics
        print_summary_stats(bpp_df, msssim_df, psnr_df)
        
        # Print detailed results
        print_detailed_results(bpp_df, msssim_df, psnr_df)
        
        # Export combined results
        export_combined_results(bpp_df, msssim_df, psnr_df)
        
        print("\n" + "=" * 80)
        print("Results output complete!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find evaluation results files. {e}")
        print(f"Expected directory: {RESULTS_DIR}")
    except Exception as e:
        print(f"Error processing results: {e}")

if __name__ == "__main__":
    main()
